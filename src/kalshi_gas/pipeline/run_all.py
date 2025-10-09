"""End-to-end pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from kalshi_gas.backtest.evaluate import run_backtest
from kalshi_gas.config import PipelineConfig, load_config
from kalshi_gas.data.assemble import assemble_dataset
from kalshi_gas.etl.pipeline import run_all_etl
from kalshi_gas.models.ensemble import EnsembleModel
from kalshi_gas.models.posterior import PosteriorDistribution, compute_sensitivity
from kalshi_gas.models.prior import MarketPriorCDF
from kalshi_gas.models.structural import fit_structural_pass_through
from kalshi_gas.reporting.report_builder import ReportBuilder
from kalshi_gas.reporting.visuals import plot_calibration, plot_price_forecast
from kalshi_gas.viz.plots import plot_fundamentals_dashboard
from kalshi_gas.risk.gates import evaluate_risk


def _load_prior_bins(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Kalshi bins file at {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    thresholds = np.asarray(payload.get("thresholds", []), dtype=float)
    probabilities = np.asarray(payload.get("probabilities", []), dtype=float)
    if len(thresholds) == 0 or len(probabilities) != len(thresholds):
        raise ValueError("Invalid Kalshi bins configuration")
    return thresholds, probabilities


def _prior_cdf_factory(model: MarketPriorCDF):
    def prior_cdf(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        transformed = model._iso.transform(arr)
        return np.clip(transformed, 0.0, 1.0)

    return prior_cdf


def _select_beta(
    delta: float,
    params: Dict[str, float | None],
    beta_up_scale: float = 1.0,
    beta_dn_scale: float = 1.0,
) -> float:
    raw_beta = params.get("beta")
    beta = float(raw_beta) if raw_beta is not None else 0.0
    beta_up = params.get("beta_up")
    beta_dn = params.get("beta_dn")
    if delta >= 0 and beta_up is not None:
        return float(beta_up) * beta_up_scale
    if delta < 0 and beta_dn is not None:
        return float(beta_dn) * beta_dn_scale
    scale = beta_up_scale if delta >= 0 else beta_dn_scale
    return beta * scale


def run_pipeline(config_path: str | None = None) -> Dict[str, object]:
    cfg: PipelineConfig = load_config(config_path)
    cfg.data.build_dir.mkdir(parents=True, exist_ok=True)

    etl_outputs = run_all_etl(cfg)
    dataset = assemble_dataset(cfg)

    ensemble_weights = cfg.model.ensemble_weights
    ensemble_bt = EnsembleModel(weights=ensemble_weights)
    backtest = run_backtest(dataset, ensemble_bt)

    risk = evaluate_risk(dataset, cfg)

    wpsr_details = risk.details.get("wpsr", {})
    nhc_details = risk.details.get("nhc", {})

    stocks_draw = float(wpsr_details.get("gasoline_stocks_draw", 0.0))
    refinery_util_pct = float(wpsr_details.get("refinery_util_pct", 100.0))
    tightness_flag = stocks_draw > 3.0 and refinery_util_pct < 90.0

    nhc_flag = bool(risk.nhc_alert) or bool(nhc_details.get("analyst_flag", False))

    beta_up_scale = 1.0
    beta_dn_scale = 1.0
    alpha_shift = 0.0
    drift_bump = 0.0
    risk_adjustments: List[str] = []

    if tightness_flag:
        beta_up_scale = max(beta_up_scale, 1.25)
        beta_dn_scale = min(beta_dn_scale, 0.95)
        alpha_shift += 0.03
        drift_bump = max(drift_bump, 0.02)
        risk_adjustments.append(
            "WPSR tightness: boosted upside pass-through and +3¢ alpha lift"
        )

    if nhc_flag:
        beta_up_scale = max(beta_up_scale, 1.2)
        alpha_shift += 0.02
        drift_bump = max(drift_bump, 0.03)
        risk_adjustments.append(
            "NHC risk: widened nowcast drift and tilted upside beta"
        )

    risk_flags = {
        "tightness": {
            "active": tightness_flag,
            "gasoline_stocks_draw": stocks_draw,
            "refinery_util_pct": refinery_util_pct,
            "product_supplied_mbd": float(
                wpsr_details.get("product_supplied_mbd", 0.0)
            ),
        },
        "nhc": {
            "active": nhc_flag,
            "nhc_alert": bool(risk.nhc_alert),
            "analyst_flag": bool(nhc_details.get("analyst_flag", False)),
        },
        "adjustments": risk_adjustments,
    }

    live_ensemble = EnsembleModel(weights=ensemble_weights)
    if drift_bump > 0:
        lower, upper = live_ensemble.nowcast.drift_bounds
        live_ensemble.nowcast.drift_bounds = (lower, upper + drift_bump)
    live_ensemble.fit(dataset)
    nowcast_sim = live_ensemble.nowcast.simulate()

    structural = fit_structural_pass_through(dataset, asymmetry=True)

    bins_path = Path("data_raw/kalshi_bins.yml")
    thresholds, probabilities = _load_prior_bins(bins_path)
    prior_model = MarketPriorCDF.fit(thresholds, probabilities)
    prior_fn = _prior_cdf_factory(prior_model)

    base_samples = nowcast_sim.samples
    prior_weight = cfg.model.prior_weight

    def posterior_factory(
        rbob_delta: float, alpha_delta: float
    ) -> PosteriorDistribution:
        beta_effect = _select_beta(
            rbob_delta,
            structural,
            beta_up_scale=beta_up_scale,
            beta_dn_scale=beta_dn_scale,
        )
        adjusted_samples = (
            base_samples + alpha_shift + alpha_delta + beta_effect * rbob_delta
        )
        return PosteriorDistribution(
            samples=adjusted_samples,
            prior_cdf=prior_fn,
            prior_weight=prior_weight,
        )

    base_posterior = posterior_factory(0.0, 0.0)
    sensitivity = compute_sensitivity(posterior_factory, thresholds=thresholds)

    data_proc_dir = Path("data_proc")
    data_proc_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_path = data_proc_dir / "sensitivity.csv"
    sensitivity.to_csv(sensitivity_path, index=False)

    posterior_summary = base_posterior.summary()
    for threshold in thresholds:
        posterior_summary[f"prob_ge_{threshold:.2f}"] = base_posterior.prob_above(
            float(threshold)
        )

    figures_dir = cfg.data.build_dir / "figures"
    memo_dir = cfg.data.build_dir / "memo"
    figures_dir.mkdir(parents=True, exist_ok=True)
    memo_dir.mkdir(parents=True, exist_ok=True)

    forecast_fig = figures_dir / "forecast_vs_actual.png"
    calibration_fig = figures_dir / "calibration.png"
    plot_price_forecast(backtest.test_frame, forecast_fig)
    plot_calibration(backtest.calibration, calibration_fig)

    fundamentals_fig = figures_dir / "fundamentals_dashboard.png"
    plot_fundamentals_dashboard(dataset, risk_flags, fundamentals_fig)

    results_csv = memo_dir / "forecast_results.csv"
    backtest.test_frame.to_csv(results_csv, index=False)

    builder = ReportBuilder()
    report_path = memo_dir / "report.md"
    builder.build(
        metrics=backtest.metrics,
        risk=risk,
        calibration=backtest.calibration,
        figures={
            "forecast": Path("..") / "figures" / forecast_fig.name,
            "calibration": Path("..") / "figures" / calibration_fig.name,
            "fundamentals": Path("..") / "figures" / fundamentals_fig.name,
        },
        posterior=posterior_summary,
        sensitivity=sensitivity,
        risk_flags=risk_flags,
        output_path=report_path,
    )

    return {
        "etl_outputs": etl_outputs,
        "report_path": report_path,
        "results_csv": results_csv,
        "sensitivity_path": sensitivity_path,
        "posterior_summary": posterior_summary,
        "structural": structural,
        "risk_flags": risk_flags,
        "figures": {
            "forecast": forecast_fig,
            "calibration": calibration_fig,
            "fundamentals": fundamentals_fig,
        },
    }
