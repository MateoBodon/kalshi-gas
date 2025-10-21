"""End-to-end pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from kalshi_gas.backtest.evaluate import run_backtest
from kalshi_gas.config import PipelineConfig, load_config
from kalshi_gas.data.assemble import assemble_dataset
from kalshi_gas.data.provenance import write_meta
from kalshi_gas.etl.pipeline import run_all_etl
from kalshi_gas.models.ensemble import EnsembleModel
from kalshi_gas.models.posterior import PosteriorDistribution, compute_sensitivity
from kalshi_gas.models.prior import MarketPriorCDF
from kalshi_gas.models.structural import fit_structural_pass_through
from kalshi_gas.reporting.report_builder import ReportBuilder
from kalshi_gas.reporting.visuals import (
    plot_calibration,
    plot_price_forecast,
    plot_risk_box,
    plot_sensitivity_bars,
)
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

    etl_results = run_all_etl(cfg)
    etl_outputs = {
        name: str(result.output_path) for name, result in etl_results.items()
    }
    etl_provenance = {
        name: result.provenance.serialize() if result.provenance else None
        for name, result in etl_results.items()
    }
    metadata_dir = cfg.data.build_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = metadata_dir / "data_provenance.json"
    provenance_path.write_text(json.dumps(etl_provenance, indent=2), encoding="utf-8")
    dataset = assemble_dataset(cfg)
    dataset_as_of = None
    if not dataset.empty:
        max_date = dataset["date"].max()
        if max_date is not None and not pd.isna(max_date):
            dataset_as_of = pd.Timestamp(max_date).normalize().date().isoformat()

    ensemble_weights = cfg.model.ensemble_weights
    ensemble_bt = EnsembleModel(weights=ensemble_weights)
    backtest = run_backtest(dataset, ensemble_bt)
    calibrated_prior_weight = (
        backtest.calibrated_prior_weight
        if backtest.calibrated_prior_weight is not None
        else cfg.model.prior_weight
    )

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
    prior_weight = float(calibrated_prior_weight)

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
    base_rows = sensitivity[
        (sensitivity["rbob_delta"].abs() < 1e-9)
        & (sensitivity["alpha_delta"].abs() < 1e-9)
    ]
    base_lookup = {
        float(row.threshold): float(row.prob_above) for row in base_rows.itertuples()
    }
    sensitivity_bars = (
        sensitivity.groupby("threshold", observed=True)["prob_above"]
        .agg(["min", "max"])
        .rename(columns={"min": "prob_min", "max": "prob_max"})
        .reset_index()
    )
    sensitivity_bars["prob_base"] = sensitivity_bars["threshold"].map(base_lookup)
    sensitivity_bars_path = data_proc_dir / "sensitivity_bars.csv"
    sensitivity_bars.to_csv(sensitivity_bars_path, index=False)

    posterior_summary = base_posterior.summary()
    posterior_summary["prior_weight"] = prior_weight
    if backtest.calibrated_prior_weight is not None:
        posterior_summary["prior_weight_source"] = "calibrated"
    else:
        posterior_summary["prior_weight_source"] = "config"
    for threshold in thresholds:
        posterior_summary[f"prob_ge_{threshold:.2f}"] = base_posterior.prob_above(
            float(threshold)
        )

    metrics = backtest.metrics
    benchmarks: list[dict[str, float | str | None]] = [
        {
            "model": "Ensemble",
            "brier": metrics.get("brier_score"),
            "brier_se": metrics.get("brier_score_se"),
            "rmse": metrics.get("rmse"),
        },
        {
            "model": "Posterior",
            "brier": metrics.get("posterior_brier"),
            "brier_se": metrics.get("posterior_brier_se"),
        },
        {
            "model": "Carry Forward",
            "brier": metrics.get("brier_carry"),
            "rmse": metrics.get("carry_rmse"),
        },
        {
            "model": "Kalshi Prior",
            "brier": metrics.get("brier_prior"),
        },
    ]
    benchmarks = [
        row
        for row in benchmarks
        if any(value is not None for key, value in row.items() if key != "model")
    ]

    provenance_records: list[dict[str, object]] = []
    for source, prov in etl_provenance.items():
        if isinstance(prov, dict):
            record = {"source": source, **prov}
            provenance_records.append(record)
    provenance_records.sort(key=lambda entry: str(entry.get("source") or ""))
    source_names = sorted(
        {
            str(entry.get("source"))
            for entry in provenance_records
            if entry.get("source") is not None
        }
    )
    source_summary = ", ".join(source_names)

    meta_dir = Path("data_proc") / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_paths: list[Path] = []
    for record in provenance_records:
        source_name = str(record.get("source") or "source")
        meta_path = write_meta(meta_dir / f"{source_name}.json", record)
        meta_paths.append(meta_path)
    dataset_meta_path = meta_dir / "dataset.json"
    if dataset_meta_path.exists():
        meta_paths.append(dataset_meta_path)

    figures_dir = cfg.data.build_dir / "figures"
    memo_dir = cfg.data.build_dir / "memo"
    figures_dir.mkdir(parents=True, exist_ok=True)
    memo_dir.mkdir(parents=True, exist_ok=True)

    model_source = "Kalshi Gas ensemble model"
    data_source = source_summary or "Kalshi Gas data sources"

    forecast_fig = figures_dir / "forecast_vs_actual.png"
    calibration_fig = figures_dir / "calibration.png"
    plot_price_forecast(
        backtest.test_frame,
        forecast_fig,
        as_of=dataset_as_of,
        source=model_source,
    )
    plot_calibration(
        backtest.calibration,
        calibration_fig,
        as_of=dataset_as_of,
        source=model_source,
    )

    fundamentals_fig = figures_dir / "fundamentals_dashboard.png"
    plot_fundamentals_dashboard(dataset, risk_flags, fundamentals_fig)

    risk_box_fig = figures_dir / "risk_box.png"
    plot_risk_box(
        risk_flags,
        metrics,
        risk_box_fig,
        as_of=dataset_as_of,
        source=data_source,
    )

    sensitivity_fig = figures_dir / "sensitivity_bars.png"
    plot_sensitivity_bars(
        sensitivity_bars,
        sensitivity_fig,
        as_of=dataset_as_of,
        source=model_source,
    )

    results_csv = memo_dir / "forecast_results.csv"
    backtest.test_frame.to_csv(results_csv, index=False)

    builder = ReportBuilder()
    report_path = memo_dir / "report.md"
    figures_relative = {
        key: Path("..") / "figures" / value.name
        for key, value in {
            "forecast": forecast_fig,
            "calibration": calibration_fig,
            "fundamentals": fundamentals_fig,
            "risk_box": risk_box_fig,
            "sensitivity": sensitivity_fig,
        }.items()
    }
    builder.build(
        metrics=backtest.metrics,
        risk=risk,
        calibration=backtest.calibration,
        figures=figures_relative,
        posterior=posterior_summary,
        sensitivity=sensitivity,
        risk_flags=risk_flags,
        provenance=provenance_records,
        benchmarks=benchmarks,
        sensitivity_bars=sensitivity_bars,
        meta_files=[str(path) for path in meta_paths],
        output_path=report_path,
    )

    headline_threshold = float(thresholds[0]) if len(thresholds) > 0 else None
    headline_probability = (
        posterior_summary.get(f"prob_ge_{headline_threshold:.2f}")
        if headline_threshold is not None
        else None
    )
    deck_dir = cfg.data.build_dir / "deck"
    deck_dir.mkdir(parents=True, exist_ok=True)
    deck_path = deck_dir / "deck.md"
    builder.build_deck(
        posterior=posterior_summary,
        risk_flags=risk_flags,
        benchmarks=benchmarks,
        figures=figures_relative,
        provenance=provenance_records,
        sensitivity_bars=sensitivity_bars.to_dict(orient="records"),
        headline_threshold=headline_threshold,
        headline_probability=headline_probability,
        output_path=deck_path,
    )

    artifacts_manifest = {
        "report": str(report_path),
        "deck": str(deck_path),
        "figures": {
            key: str(path)
            for key, path in {
                "forecast": forecast_fig,
                "calibration": calibration_fig,
                "fundamentals": fundamentals_fig,
                "risk_box": risk_box_fig,
                "sensitivity": sensitivity_fig,
            }.items()
        },
        "data": {
            "results_csv": str(results_csv),
            "sensitivity_grid": str(sensitivity_path),
            "sensitivity_bars": str(sensitivity_bars_path),
            "provenance": str(provenance_path),
        },
    }
    artifacts_path = metadata_dir / "artifacts.json"
    artifacts_path.write_text(
        json.dumps(artifacts_manifest, indent=2), encoding="utf-8"
    )

    return {
        "etl_outputs": etl_outputs,
        "etl_provenance": etl_provenance,
        "report_path": report_path,
        "results_csv": results_csv,
        "sensitivity_path": sensitivity_path,
        "sensitivity_bars_path": sensitivity_bars_path,
        "posterior_summary": posterior_summary,
        "structural": structural,
        "risk_flags": risk_flags,
        "figures": {
            "forecast": forecast_fig,
            "calibration": calibration_fig,
            "fundamentals": fundamentals_fig,
            "risk_box": risk_box_fig,
            "sensitivity": sensitivity_fig,
        },
        "provenance_path": provenance_path,
        "prior_weight": prior_weight,
        "sensitivity_bars": sensitivity_bars,
        "benchmarks": benchmarks,
        "provenance_records": provenance_records,
        "deck_path": deck_path,
        "artifacts_path": artifacts_path,
        "meta_files": [str(path) for path in meta_paths],
    }
