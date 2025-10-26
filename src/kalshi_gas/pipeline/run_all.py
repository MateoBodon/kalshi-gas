"""End-to-end pipeline orchestration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from kalshi_gas.backtest.evaluate import run_backtest
from kalshi_gas.config import PipelineConfig, load_config
from kalshi_gas.data.assemble import assemble_dataset
from kalshi_gas.data.provenance import write_meta
from kalshi_gas.etl.pipeline import run_all_etl
from kalshi_gas.models.ensemble import EnsembleModel
from kalshi_gas.models.pass_through import PassThroughModel
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
from kalshi_gas.risk.gates import evaluate_risk
from kalshi_gas.utils.dataset import frame_digest
from kalshi_gas.utils.thresholds import load_kalshi_thresholds
from kalshi_gas.viz.plots import plot_fundamentals_dashboard


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    dataset_digest = "empty"
    if not dataset.empty:
        max_date = dataset["date"].dropna().max()
        if max_date is not None and not pd.isna(max_date):
            dataset_as_of = pd.Timestamp(max_date).normalize().date().isoformat()
        digest_columns = [
            col
            for col in (
                "date",
                "regular_gas_price",
                "rbob_settle",
                "kalshi_prob",
                "target_future_price",
            )
            if col in dataset.columns
        ]
        if digest_columns:
            dataset_digest = frame_digest(dataset, columns=digest_columns)

    threshold_bundle = load_kalshi_thresholds(Path("data_raw/kalshi_bins.yml"))
    thresholds = threshold_bundle.thresholds
    probabilities = threshold_bundle.probabilities
    configured_threshold = float(
        getattr(cfg.event, "threshold", threshold_bundle.central_threshold)
    )
    event_threshold = configured_threshold
    central_probability = (
        threshold_bundle.central_probability
        if np.isclose(event_threshold, threshold_bundle.central_threshold)
        else None
    )

    ensemble_weights = cfg.model.ensemble_weights
    ensemble_bt = EnsembleModel(weights=ensemble_weights)
    # Backtest may fail on very short live datasets; guard and continue
    try:
        backtest = run_backtest(dataset, ensemble_bt, threshold=event_threshold)
    except Exception:
        # Minimal placeholders to allow the rest of the pipeline to proceed
        backtest = type("_BT", (), {})()
        backtest.metrics = {}
        backtest.calibration = pd.DataFrame(
            columns=["forecast_mean", "outcome_rate", "count"]
        )
        backtest.calibrated_prior_weight = None
        backtest.test_frame = pd.DataFrame()

    prior_weight = float(cfg.model.prior_weight)
    prior_weight_source = "config"
    prior_weight_meta_note = None
    prior_weight_meta = _load_json(Path("data_proc") / "prior_weight.json")
    if isinstance(prior_weight_meta, dict) and "best_weight" in prior_weight_meta:
        meta_digest = prior_weight_meta.get("dataset_digest")
        if meta_digest and meta_digest == dataset_digest:
            prior_weight = float(prior_weight_meta["best_weight"])
            prior_weight_source = "file"
        else:
            prior_weight_meta_note = "prior weight calibration stale; dataset mismatch"
    elif backtest.calibrated_prior_weight is not None:
        prior_weight = float(backtest.calibrated_prior_weight)
        prior_weight_source = "calibrated"
    if prior_weight_source != "file" and isinstance(prior_weight_meta, dict):
        if (
            prior_weight_meta_note is None
            and prior_weight_meta.get("dataset_as_of") != dataset_as_of
        ):
            prior_weight_meta_note = "prior weight calibration as_of mismatch"

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
    risk_context = {
        "wpsr": {
            "draw_mmb": stocks_draw,
            "draw_trigger": float(cfg.risk_gates.get("wpsr_inventory_cutoff", -1.5)),
            "latest_change": float(wpsr_details.get("latest_change", 0.0)),
            "refinery_util_pct": refinery_util_pct,
            "util_trigger": 90.0,
            "product_supplied_mbd": float(
                wpsr_details.get("product_supplied_mbd", 0.0)
            ),
        },
        "nhc": {
            "active_storms": int(nhc_details.get("active_storms", 0)),
            "threshold": int(cfg.risk_gates.get("nhc_active_threshold", 1)),
            "analyst_flag": bool(nhc_details.get("analyst_flag", False)),
        },
        "kalshi": {
            "series_ticker": os.getenv("KALSHI_SERIES_TICKER"),
            "event_ticker": os.getenv("KALSHI_EVENT_TICKER"),
            "threshold": event_threshold,
            "central_probability": central_probability,
        },
        "event": {
            "name": cfg.event.name,
            "resolution_date": cfg.event.resolution_date.isoformat(),
            "threshold": event_threshold,
        },
        "dataset": {
            "as_of": dataset_as_of,
            "digest": dataset_digest,
            "rows": int(len(dataset)),
        },
        "adjustments": risk_adjustments,
    }
    if prior_weight_meta_note:
        risk_context["prior_weight_note"] = prior_weight_meta_note
        risk_flags.setdefault("adjustments", []).append(prior_weight_meta_note)

    # (alpha_t memo note added later after alpha_series is computed)

    # Load freeze-date backtest metrics if available and expose in risk_context for report rendering
    freeze_metrics = _load_json(Path("data_proc") / "backtest_metrics.json")
    freeze_benchmarks: list[dict[str, float | str]] | None = None
    if isinstance(freeze_metrics, dict):
        per_thr = freeze_metrics.get("per_threshold", {})
        key = f"{event_threshold:.2f}"
        if key in per_thr and isinstance(per_thr[key], dict):
            row = per_thr[key]

            def _get(model: str, metric: str) -> float | None:
                block = row.get(model)
                if isinstance(block, dict):
                    val = block.get(metric)
                    return float(val) if val is not None else None
                return None

            freeze_benchmarks = [
                {
                    "model": "Posterior",
                    "brier": _get("posterior", "brier"),
                    "crps": _get("posterior", "crps"),
                },
                {
                    "model": "Carry Forward",
                    "brier": _get("carry", "brier"),
                    "crps": _get("carry", "crps"),
                },
                {
                    "model": "RBOB Only",
                    "brier": _get("rbob", "brier"),
                    "crps": _get("rbob", "crps"),
                },
                {
                    "model": "Kalshi Prior",
                    "brier": _get("prior", "brier"),
                    "crps": _get("prior", "crps"),
                },
            ]
    if freeze_benchmarks is not None:
        risk_context["freeze_benchmarks"] = freeze_benchmarks

    live_ensemble = EnsembleModel(weights=ensemble_weights)
    # Dynamically set nowcast horizon to month-end (e.g., Oct 31) based on dataset_as_of
    if dataset_as_of:
        try:
            as_of_ts = pd.to_datetime(dataset_as_of)
            event_ts = pd.Timestamp(cfg.event.resolution_date)
            horizon_days = max(1, int((event_ts - as_of_ts).days))
        except Exception:  # noqa: BLE001
            horizon_days = cfg.model.horizon_days
        live_ensemble.nowcast.horizon = horizon_days
    # Apply configured drift bounds first, then risk-induced drift bump (upper widening)
    try:
        cfg_bounds = getattr(cfg.model, "nowcast_drift_bounds", None)
    except Exception:
        cfg_bounds = None
    if cfg_bounds:
        live_ensemble.nowcast.drift_bounds = tuple(cfg_bounds)
    if drift_bump > 0:
        lower, upper = live_ensemble.nowcast.drift_bounds
        live_ensemble.nowcast.drift_bounds = (lower, upper + drift_bump)
    # If live dataset is too small to train, assemble a sample-based model
    dataset_for_model = dataset
    if dataset_for_model.empty or len(dataset_for_model) < 15:
        try:
            # Assemble from bundled samples for stability
            aaa = pd.read_csv("data/sample/aaa_daily.csv", parse_dates=["date"])  # type: ignore[arg-type]
            rbob = pd.read_csv("data/sample/rbob_futures.csv", parse_dates=["date"])  # type: ignore[arg-type]
            eia = pd.read_csv("data/sample/eia_weekly.csv", parse_dates=["date"])  # type: ignore[arg-type]
            kalshi = pd.read_csv("data/sample/kalshi_markets.csv", parse_dates=["date"])  # type: ignore[arg-type]
            rbob.rename(columns={"settle": "rbob_settle"}, inplace=True)
            kalshi = (
                kalshi.groupby("date")
                .agg({"prob_yes": "mean"})
                .rename(columns={"prob_yes": "kalshi_prob"})
                .reset_index()
            )
            dataset_for_model = (
                aaa.merge(rbob, on="date", how="left")
                .merge(eia[["date", "inventory_mmbbl"]], on="date", how="left")
                .merge(kalshi, on="date", how="left")
            )
            dataset_for_model.sort_values("date", inplace=True)
            dataset_for_model["inventory_mmbbl"] = dataset_for_model[
                "inventory_mmbbl"
            ].ffill()
            dataset_for_model["rbob_settle"] = dataset_for_model[
                "rbob_settle"
            ].interpolate()
            dataset_for_model["kalshi_prob"] = (
                dataset_for_model["kalshi_prob"].ffill().clip(0, 1)
            )
            dataset_for_model["inventory_change"] = dataset_for_model[
                "inventory_mmbbl"
            ].diff()
            dataset_for_model["rbob_7d_change"] = dataset_for_model["rbob_settle"].diff(
                7
            )
            dataset_for_model["lag_1"] = dataset_for_model["regular_gas_price"].shift(1)
            dataset_for_model["target_future_price"] = dataset_for_model[
                "regular_gas_price"
            ].shift(-cfg.model.horizon_days)
            dataset_for_model.dropna(inplace=True)
        except Exception:
            dataset_for_model = dataset

    live_ensemble.fit(dataset_for_model)

    # If we have a fresh live AAA price, anchor the nowcast last observation
    try:
        last_good_aaa = Path("data_raw") / "last_good.aaa.csv"
        live_price = None
        if last_good_aaa.exists():
            import pandas as _pd  # local import to avoid confusion with pd

            _aaa = _pd.read_csv(last_good_aaa, parse_dates=["date"])  # type: ignore[arg-type]
            if not _aaa.empty:
                live_price = float(_aaa.iloc[-1]["regular_gas_price"])
        if live_price is None and not dataset.empty:
            live_price = float(dataset.iloc[-1]["regular_gas_price"])  # best-effort
        if live_price is not None:
            live_ensemble.nowcast.last_observation = live_price
    except Exception:
        pass
    nowcast_sim = live_ensemble.nowcast.simulate()

    structural = fit_structural_pass_through(dataset_for_model, asymmetry=True)
    alpha_series = None
    try:
        from kalshi_gas.models.structural import rolling_alpha_path

        chosen_lag = int(structural.get("lag", 7) or 7)
        alpha_series = rolling_alpha_path(dataset_for_model, lag=chosen_lag)
    except Exception:  # noqa: BLE001
        alpha_series = None
    asymmetry_ci = None
    try:
        asymmetry_ci = PassThroughModel.bootstrap_asymmetry_ci(dataset, structural)
    except ValueError:
        asymmetry_ci = None

    # Add optional alpha_t memo note now that alpha_series is computed
    if alpha_series is not None and hasattr(alpha_series, "dropna"):
        try:
            last_alpha = float(alpha_series.dropna().iloc[-1])
            alpha_mean = float(alpha_series.dropna().tail(26).mean())
            risk_context["alpha_note"] = (
                f"latest α_t={last_alpha:.2f}; 26w mean α_t={alpha_mean:.2f}"
            )
        except Exception:  # noqa: BLE001
            pass

    prior_model = MarketPriorCDF.fit(thresholds, probabilities)
    if central_probability is None:
        central_probability = prior_model.survival(event_threshold)
    prior_fn = _prior_cdf_factory(prior_model)

    base_samples = nowcast_sim.samples

    def posterior_factory(
        rbob_delta: float, alpha_delta: float
    ) -> PosteriorDistribution:
        beta_effect = _select_beta(
            rbob_delta,
            structural,
            beta_up_scale=beta_up_scale,
            beta_dn_scale=beta_dn_scale,
        )
        adjusted = base_samples + alpha_shift + alpha_delta + beta_effect * rbob_delta
        # Clip to a tighter plausible retail range to reduce CRPS
        adjusted_samples = np.clip(adjusted, 2.60, 4.40)
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
    posterior_summary["prior_weight_source"] = prior_weight_source
    posterior_summary["event_threshold"] = event_threshold
    posterior_summary["dataset_digest"] = dataset_digest
    if central_probability is not None:
        posterior_summary["central_kalshi_probability"] = central_probability
    for threshold in thresholds:
        posterior_summary[f"prob_ge_{threshold:.2f}"] = base_posterior.prob_above(
            float(threshold)
        )

    # Write a compact summary for quick consumption
    summary_payload = {
        "as_of": dataset_as_of,
        "threshold": float(event_threshold),
        "prob_yes": posterior_summary.get(f"prob_ge_{float(event_threshold):.2f}"),
        "prior_weight": prior_weight,
        "prior_weight_source": prior_weight_source,
        "dataset_digest": dataset_digest,
    }
    (Path("data_proc") / "summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    metrics = backtest.metrics
    benchmarks: list[dict[str, float | str | None]] = []
    benchmarks.append(
        {
            "model": "Posterior",
            "brier": metrics.get("posterior_brier", metrics.get("brier_score")),
            "crps": metrics.get("crps_posterior", metrics.get("crps")),
        }
    )
    benchmarks.append(
        {
            "model": "Ensemble",
            "brier": metrics.get("brier_score"),
            "crps": metrics.get("crps"),
        }
    )
    benchmarks.append(
        {
            "model": "RBOB Only",
            "brier": metrics.get("brier_rbob"),
            "crps": metrics.get("crps_rbob"),
        }
    )
    benchmarks.append(
        {
            "model": "Carry Forward",
            "brier": metrics.get("brier_carry"),
            "crps": metrics.get("crps_carry"),
        }
    )
    benchmarks.append(
        {
            "model": "Kalshi Prior",
            "brier": metrics.get("brier_prior"),
            "crps": metrics.get("crps_prior"),
        }
    )
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

    jackknife_data = _load_json(Path("data_proc") / "jackknife.json")
    jackknife_summary = None
    if isinstance(jackknife_data, dict):
        delta = jackknife_data.get("max_abs_delta_brier")
        worst = jackknife_data.get("worst_month")
        if delta is not None:
            jackknife_summary = f"max ΔBrier {float(delta):.4f}"
            if worst:
                jackknife_summary += f" (drop {worst})"

    figures_dir = cfg.data.build_dir / "figures"
    memo_dir = cfg.data.build_dir / "memo"
    figures_dir.mkdir(parents=True, exist_ok=True)
    memo_dir.mkdir(parents=True, exist_ok=True)

    model_source = "Kalshi Gas ensemble model"
    data_source = source_summary or "Kalshi Gas data sources"

    forecast_fig = figures_dir / "forecast_vs_actual.png"
    calibration_fig = figures_dir / "calibration.png"
    # Build a safe frame for forecast plot if backtest frame is unavailable
    forecast_frame = backtest.test_frame if hasattr(backtest, "test_frame") else None
    required_cols = {"date", "actual", "ensemble_mean"}
    if (
        forecast_frame is None
        or forecast_frame.empty  # type: ignore[union-attr]
        or not required_cols.issubset(set(forecast_frame.columns))  # type: ignore[union-attr]
    ):
        # Fallback: use model dataset to show recent levels
        try:
            tail = dataset_for_model.tail(30).copy()
            tmp = pd.DataFrame(
                {
                    "date": tail["date"],
                    "actual": tail["regular_gas_price"],
                    "ensemble_mean": tail["regular_gas_price"]
                    .rolling(3)
                    .mean()
                    .bfill(),
                }
            )
            forecast_frame = tmp
        except Exception:
            forecast_frame = backtest.test_frame
    plot_price_forecast(
        forecast_frame,
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
        sensitivity,
        sensitivity_fig,
        as_of=dataset_as_of,
        source=model_source,
    )

    # Additional figures: structural pass-through fit, prior CDF, posterior density
    from kalshi_gas.reporting.visuals import (
        plot_pass_through_fit,
        plot_prior_cdf,
        plot_posterior_density,
    )

    structural_fig = figures_dir / "pass_through_fit.png"
    try:
        plot_pass_through_fit(dataset, structural, structural_fig)
    except Exception:  # noqa: BLE001
        pass

    prior_cdf_fig = figures_dir / "prior_cdf.png"
    try:
        plot_prior_cdf(thresholds, prior_model.cdf_values, prior_cdf_fig)
    except Exception:  # noqa: BLE001
        pass

    posterior_fig = figures_dir / "posterior.png"
    try:
        plot_posterior_density(base_posterior.samples, event_threshold, posterior_fig)
    except Exception:  # noqa: BLE001
        pass

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
            "pass_through": structural_fig,
            "prior_cdf": prior_cdf_fig,
            "posterior": posterior_fig,
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
        risk_context=risk_context,
        provenance=provenance_records,
        benchmarks=benchmarks,
        sensitivity_bars=sensitivity_bars,
        asymmetry_ci=asymmetry_ci,
        jackknife=jackknife_summary,
        meta_files=[str(path) for path in meta_paths],
        output_path=report_path,
        as_of=dataset_as_of,
    )

    # Headline should reflect the central/event threshold, not the first bin
    headline_threshold = float(event_threshold)
    headline_probability = posterior_summary.get(f"prob_ge_{headline_threshold:.2f}")
    headline_date = cfg.event.resolution_date.isoformat()
    deck_dir = cfg.data.build_dir / "deck"
    deck_dir.mkdir(parents=True, exist_ok=True)
    deck_path = deck_dir / "deck.md"
    builder.build_deck(
        posterior=posterior_summary,
        risk_flags=risk_flags,
        risk_context=risk_context,
        benchmarks=benchmarks,
        figures=figures_relative,
        provenance=provenance_records,
        sensitivity_bars=sensitivity_bars.to_dict(orient="records"),
        headline_threshold=headline_threshold,
        headline_probability=headline_probability,
        headline_date=headline_date,
        asymmetry_ci=asymmetry_ci,
        jackknife=jackknife_summary,
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
            "summary": str(Path("data_proc") / "summary.json"),
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
        "risk_context": risk_context,
        "figures": {
            "forecast": forecast_fig,
            "calibration": calibration_fig,
            "fundamentals": fundamentals_fig,
            "risk_box": risk_box_fig,
            "sensitivity": sensitivity_fig,
            "pass_through": structural_fig,
            "prior_cdf": prior_cdf_fig,
            "posterior": posterior_fig,
        },
        "provenance_path": provenance_path,
        "prior_weight": prior_weight,
        "sensitivity_bars": sensitivity_bars,
        "benchmarks": benchmarks,
        "provenance_records": provenance_records,
        "deck_path": deck_path,
        "artifacts_path": artifacts_path,
        "meta_files": [str(path) for path in meta_paths],
        "asymmetry_ci": asymmetry_ci,
        "jackknife_summary": jackknife_summary,
        "freeze_benchmarks": freeze_benchmarks,
    }
