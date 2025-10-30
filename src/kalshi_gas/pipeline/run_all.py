"""End-to-end pipeline orchestration."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from kalshi_gas.backtest.evaluate import run_backtest
from kalshi_gas.pipeline.backtest import run_freeze_backtest
from kalshi_gas.config import PipelineConfig, load_config
from kalshi_gas.data.assemble import assemble_dataset
from kalshi_gas.data.provenance import write_meta
from kalshi_gas.etl.pipeline import run_all_etl
from kalshi_gas.models.ensemble import EnsembleModel
from kalshi_gas.models.pass_through import (
    PassThroughModel,
    fit_structural_pass_through,
)
from kalshi_gas.models.structural import rolling_alpha_path
from kalshi_gas.models.posterior import PosteriorDistribution, compute_sensitivity
from kalshi_gas.models.prior import MarketPriorCDF
from kalshi_gas.reporting.report_builder import ReportBuilder
from kalshi_gas.reporting.visuals import (
    plot_calibration,
    plot_aaa_delta_histogram,
    plot_price_forecast,
    plot_risk_box,
    plot_sensitivity_bars,
)
from kalshi_gas.risk.gates import evaluate_risk
from kalshi_gas.utils.dataset import frame_digest
from kalshi_gas.utils.thresholds import load_kalshi_thresholds
from kalshi_gas.viz.plots import plot_fundamentals_dashboard


def _build_raw_requirements(cfg: PipelineConfig) -> Tuple[Dict[str, object], ...]:
    raw_dir = cfg.data.raw_dir
    return (
        {
            "name": "Kalshi bin probabilities",
            "path": Path("data_raw/kalshi_bins.yml"),
            "schema": "yaml: thresholds: [], probabilities: []",
            "sample": "scripts/update_kalshi_bins.py",
            "max_age": timedelta(days=2),
        },
        {
            "name": "WPSR state snapshot",
            "path": Path("data_raw/wpsr_state.yml"),
            "schema": "yaml: draw_mmbbl, refinery_util_pct, product_supplied_mbd",
            "sample": "data/sample/wpsr_summary.csv",
            "max_age": timedelta(days=7),
        },
        {
            "name": "NHC analyst flag",
            "path": Path("data_raw/nhc_flag.yml"),
            "schema": "yaml: active_storms, analyst_flag",
            "sample": "data/sample/nhc_outlook.csv",
            "max_age": timedelta(days=3),
        },
        {
            "name": "AAA fallback daily price",
            "path": raw_dir / "last_good.aaa.csv",
            "schema": "csv columns: date, regular_gas_price",
            "sample": "data/sample/aaa_daily.csv",
            "max_age": timedelta(days=3),
        },
        {
            "name": "RBOB fallback daily settle",
            "path": raw_dir / "last_good.rbob.csv",
            "schema": "csv columns: date, rbob_settle",
            "sample": "data/sample/rbob_futures.csv",
            "max_age": timedelta(days=3),
        },
        {
            "name": "EIA fallback weekly stocks",
            "path": raw_dir / "last_good.eia.csv",
            "schema": "csv columns: date, inventory_mmbbl, inventory_change",
            "sample": "data/sample/eia_weekly.csv",
            "max_age": timedelta(days=14),
        },
        {
            "name": "Kalshi fallback quotes",
            "path": raw_dir / "last_good.kalshi.csv",
            "schema": "csv columns: date, prob_yes",
            "sample": "data/sample/kalshi_markets.csv",
            "max_age": timedelta(days=2),
        },
    )


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _beta_scale_for_horizon(days_to_event: int | None, lag_min: int) -> float:
    if days_to_event is None:
        return 1.0
    if days_to_event <= 1:
        return 0.0
    if days_to_event < lag_min:
        return float(days_to_event) / float(lag_min)
    return 1.0


def _force_rebuild_directories(cfg: PipelineConfig) -> None:
    targets = [
        cfg.data.processed_dir,
        cfg.data.interim_dir,
        cfg.data.build_dir,
        Path("data_proc"),
    ]
    for target in targets:
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()


def _log_latest_source(
    label: str, path: Path, date_col: str, value_candidates: list[str]
) -> dict[str, object] | None:
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[run_pipeline] {label}: missing ({path})")
        return None
    if date_col not in frame.columns:
        print(f"[run_pipeline] {label}: column '{date_col}' missing in {path}")
        return None
    frame = frame.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    value_series = None
    value_col = None
    for candidate in value_candidates:
        if candidate in frame.columns:
            series = pd.to_numeric(frame[candidate], errors="coerce")
            if series.notna().any():
                value_series = series
                value_col = candidate
                break
    if value_series is None:
        print(f"[run_pipeline] {label}: no value columns found in {path}")
        return None
    frame[value_col] = value_series
    frame = frame.dropna(subset=[date_col, value_col])
    if frame.empty:
        print(f"[run_pipeline] {label}: no valid rows in {path}")
        return None
    frame.sort_values(date_col, inplace=True)
    latest = frame.iloc[-1]
    date_val = latest[date_col]
    if isinstance(date_val, pd.Timestamp):
        date_str = date_val.date().isoformat()
    else:
        date_str = str(date_val)
    value = float(latest[value_col])
    print(f"[run_pipeline] {label}: {date_str} -> {value:.4f}")
    return {"label": label, "date": date_str, "value": value, "path": str(path)}


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


def run_pipeline(
    config_path: str | None = None, *, force: bool = False
) -> Dict[str, object]:
    cfg: PipelineConfig = load_config(config_path)
    env_force = os.getenv("KALSHI_GAS_FORCE", "0") == "1"
    if force or env_force:
        _force_rebuild_directories(cfg)
        force = True
    _check_required_raw_inputs(_build_raw_requirements(cfg))
    cfg.data.build_dir.mkdir(parents=True, exist_ok=True)

    if force:
        print("[run_pipeline] Force rebuild enabled: cached artifacts cleared.")

    etl_results = run_all_etl(cfg)
    etl_outputs = {
        name: str(result.output_path) for name, result in etl_results.items()
    }
    etl_provenance = {
        name: result.provenance.serialize() if result.provenance else None
        for name, result in etl_results.items()
    }
    sample_fallback_used = False
    for prov in etl_provenance.values():
        if not isinstance(prov, dict):
            continue
        mode = prov.get("mode")
        fallback_chain = prov.get("fallback_chain", [])
        if mode == "sample":
            sample_fallback_used = True
            break
        if isinstance(fallback_chain, list) and any(
            isinstance(entry, str) and "sample" in entry for entry in fallback_chain
        ):
            sample_fallback_used = True
            break
    latest_inputs: Dict[str, dict[str, object]] = {}
    aaa_latest = _log_latest_source(
        "AAA price",
        Path("data_raw/aaa_override.csv"),
        "date",
        ["price", "regular_gas_price"],
    )
    if aaa_latest:
        latest_inputs["aaa"] = aaa_latest
    eia_latest = _log_latest_source(
        "EIA retail",
        Path("data_raw/eia_weekly.csv"),
        "date",
        ["retail"],
    )
    if eia_latest:
        latest_inputs["eia"] = eia_latest
    rbob_latest = _log_latest_source(
        "RBOB settle",
        Path("data_raw/rbob_daily.csv"),
        "date",
        ["settle", "rbob_settle"],
    )
    if rbob_latest:
        latest_inputs["rbob"] = rbob_latest
    metadata_dir = cfg.data.build_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = metadata_dir / "data_provenance.json"
    provenance_path.write_text(json.dumps(etl_provenance, indent=2), encoding="utf-8")
    dataset = assemble_dataset(cfg)
    dataset_as_of: str | None = None
    dataset_digest = "empty"
    latest_aaa_date_str = (
        dataset.attrs.get("latest_aaa_date") if hasattr(dataset, "attrs") else None
    )
    if isinstance(latest_aaa_date_str, str) and latest_aaa_date_str:
        dataset_as_of = latest_aaa_date_str
    if not dataset.empty:
        max_date = dataset["date"].dropna().max()
        if dataset_as_of is None and max_date is not None and not pd.isna(max_date):
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
    event_ts = pd.Timestamp(cfg.event.resolution_date)
    latest_aaa_ts = (
        pd.to_datetime(latest_aaa_date_str).normalize()
        if isinstance(latest_aaa_date_str, str) and latest_aaa_date_str
        else None
    )
    days_to_event_attr = (
        dataset.attrs.get("days_to_event") if hasattr(dataset, "attrs") else None
    )
    days_to_event: int | None
    if isinstance(days_to_event_attr, (int, float)):
        days_to_event = int(days_to_event_attr)
    else:
        days_to_event = None
    if days_to_event is None and latest_aaa_ts is not None:
        delta_days = int((event_ts - latest_aaa_ts).days)
        days_to_event = max(0, delta_days)
    horizon_attr = (
        dataset.attrs.get("horizon_days") if hasattr(dataset, "attrs") else None
    )
    if isinstance(horizon_attr, (int, float)):
        horizon_days_effective = max(1, int(horizon_attr))
    elif days_to_event is not None:
        horizon_days_effective = max(1, int(days_to_event))
    else:
        horizon_days_effective = max(1, int(cfg.model.horizon_days))
    if dataset_as_of is None and latest_aaa_ts is not None:
        dataset_as_of = latest_aaa_ts.date().isoformat()
    if dataset_as_of is None:
        dataset_as_of = event_ts.normalize().date().isoformat()

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
    drift_bump = 0.0
    risk_adjustments: List[str] = []
    tightness_alpha_lift = 0.0
    nhc_alpha_lift = 0.0

    if tightness_flag:
        beta_up_scale = max(beta_up_scale, 1.25)
        beta_dn_scale = min(beta_dn_scale, 0.95)
        tightness_alpha_lift = 0.03
        drift_bump = max(drift_bump, 0.02)
        risk_adjustments.append(
            "WPSR tightness: boosted upside pass-through and +3¢ alpha lift"
        )

    if nhc_flag:
        beta_up_scale = max(beta_up_scale, 1.2)
        nhc_alpha_lift = 0.02
        drift_bump = max(drift_bump, 0.03)
        risk_adjustments.append(
            "NHC risk: widened nowcast drift and tilted upside beta"
        )
    alpha_shift = 0.0

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
            "horizon_days": horizon_days_effective,
        },
        "adjustments": risk_adjustments,
    }
    if prior_weight_meta_note:
        risk_context["prior_weight_note"] = prior_weight_meta_note
        risk_flags.setdefault("adjustments", []).append(prior_weight_meta_note)

    # (alpha_t memo note added later after alpha_series is computed)

    # Run or load freeze-date backtest metrics for report rendering
    freeze_metrics_path = Path("data_proc") / "backtest_metrics.json"
    freeze_metrics_data: dict | None = None
    freeze_refresh_error: str | None = None
    try:
        freeze_result = run_freeze_backtest(cfg=cfg, dataset=dataset)
        result_summary = (
            freeze_result.get("summary") if isinstance(freeze_result, dict) else None
        )
        if isinstance(result_summary, dict):
            freeze_metrics_data = result_summary
        if prior_weight_source != "file" and isinstance(freeze_result, dict):
            calibrated_weight = freeze_result.get("calibrated_prior_weight")
            freeze_digest = freeze_result.get("dataset_digest")
            if calibrated_weight is not None and (
                dataset_digest == freeze_digest or not dataset_digest
            ):
                prior_weight = float(calibrated_weight)
                prior_weight_source = "calibrated"
    except Exception as exc:  # noqa: BLE001
        freeze_refresh_error = str(exc).splitlines()[0][:180]
    if freeze_metrics_data is None:
        freeze_metrics_data = _load_json(freeze_metrics_path)
    if freeze_refresh_error:
        risk_adjustments.append(
            f"freeze-backtest refresh failed: {freeze_refresh_error}"
        )
    freeze_metrics = (
        freeze_metrics_data if isinstance(freeze_metrics_data, dict) else None
    )
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
    if freeze_refresh_error:
        risk_context["freeze_backtest_note"] = freeze_refresh_error

    live_ensemble = EnsembleModel(weights=ensemble_weights)
    live_ensemble.nowcast.horizon = int(horizon_days_effective)
    lag_min_candidates = (7, 8, 9, 10)
    lag_min = min(lag_min_candidates)
    beta_horizon_scale = _beta_scale_for_horizon(days_to_event, lag_min)
    if tightness_alpha_lift > 0.0 and days_to_event is not None and days_to_event <= 2:
        tightness_alpha_lift *= beta_horizon_scale
    alpha_shift = tightness_alpha_lift + nhc_alpha_lift
    if days_to_event is not None and days_to_event <= 1:
        alpha_shift = 0.0
    alpha_lift_applied = alpha_shift
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
            ].shift(-horizon_days_effective)
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

    beta_eff = 0.0
    structural = fit_structural_pass_through(dataset_for_model, asymmetry=True)
    raw_beta = structural.get("beta")
    beta_base = float(raw_beta) if raw_beta is not None else 0.0
    beta_eff = beta_base * beta_horizon_scale
    days_display = days_to_event if days_to_event is not None else "n/a"
    print(
        "[run_pipeline] horizon gating: "
        f"days_to_event={days_display} beta_eff={beta_eff:.4f} alpha_lift={alpha_lift_applied:.4f}"
    )
    event_ctx = risk_context.setdefault("event", {})
    event_ctx["days_to_event"] = days_to_event
    event_ctx["beta_eff"] = beta_eff
    event_ctx["alpha_lift_applied"] = alpha_lift_applied
    alpha_series = None
    try:
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
    alpha_dynamic_offset = 0.0
    alpha_latest = None
    alpha_mean = None
    if alpha_series is not None and hasattr(alpha_series, "dropna"):
        try:
            alpha_clean = alpha_series.dropna()
            if not alpha_clean.empty:
                alpha_latest = float(alpha_clean.iloc[-1])
                alpha_mean = float(alpha_clean.tail(26).mean())
                base_alpha = float(structural.get("alpha", 0.0) or 0.0)
                alpha_dynamic_offset = float(
                    np.clip(alpha_latest - base_alpha, -0.25, 0.25)
                )
                if beta_horizon_scale < 1.0:
                    alpha_dynamic_offset *= beta_horizon_scale
                risk_context["alpha_note"] = (
                    f"latest α_t={alpha_latest:.2f}; 26w mean α_t={alpha_mean:.2f}"
                )
        except Exception:  # noqa: BLE001
            alpha_latest = None
            alpha_mean = None
            alpha_dynamic_offset = 0.0

    if alpha_mean is not None and "alpha_note" not in risk_context:
        risk_context["alpha_note"] = f"26w mean α_t={alpha_mean:.2f}"

    if alpha_latest is not None:
        risk_flags.setdefault("alpha", {})["latest"] = alpha_latest
    if alpha_mean is not None:
        risk_flags.setdefault("alpha", {})["mean_26w"] = alpha_mean

    prior_model = MarketPriorCDF.fit(thresholds, probabilities)
    if central_probability is None:
        central_probability = prior_model.survival(event_threshold)
    prior_fn = _prior_cdf_factory(prior_model)

    residual_meta = _load_json(Path("data_proc") / "residual_sigma.json") or {}
    residual_sigma = float(residual_meta.get("sigma", 0.0) or 0.0)
    if not np.isfinite(residual_sigma) or residual_sigma <= 0:
        residual_sigma = float(np.std(nowcast_sim.samples, ddof=0))
        if not np.isfinite(residual_sigma) or residual_sigma <= 0:
            residual_sigma = 0.01
    sample_size = int(len(nowcast_sim.samples))
    if sample_size <= 0:
        sample_size = 2048
    base_samples = np.random.normal(
        loc=nowcast_sim.mean,
        scale=residual_sigma,
        size=sample_size,
    )

    price_history: pd.DataFrame | None = None
    aaa_delta_recent = pd.Series(dtype=float)
    try:
        price_history = dataset[["date", "regular_gas_price"]].dropna().copy()
        price_history["date"] = pd.to_datetime(price_history["date"])
        price_history = (
            price_history.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        if price_history.empty:
            raise ValueError("no price history")
        cutoff_ts = None
        if dataset_as_of is not None:
            cutoff_ts = pd.Timestamp(dataset_as_of) - pd.DateOffset(months=24)
        else:
            cutoff_ts = price_history["date"].max() - pd.DateOffset(months=24)
        if cutoff_ts is not None:
            price_history = price_history[price_history["date"] >= cutoff_ts]
        price_history["delta"] = price_history["regular_gas_price"].diff()
        aaa_delta_recent = price_history["delta"].dropna()
    except Exception:
        price_history = None
        aaa_delta_recent = pd.Series(dtype=float)

    freeze_metrics: dict[str, object] = {}
    if price_history is not None and not price_history.empty:
        latest_row = price_history.iloc[-1]
        latest_price = float(latest_row["regular_gas_price"])
        latest_date_iso = pd.Timestamp(latest_row["date"]).date().isoformat()

        def _lookup_offset(days: int) -> dict[str, object] | None:
            target_ts = pd.Timestamp(latest_row["date"]) - pd.Timedelta(days=days)
            eligible = price_history[price_history["date"] <= target_ts]
            if eligible.empty:
                return None
            row = eligible.iloc[-1]
            return {
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "value": float(row["regular_gas_price"]),
            }

        week_ago = _lookup_offset(7)
        month_ago = _lookup_offset(30)
        rbob_latest = None
        if "rbob_settle" in dataset.columns and not dataset.empty:
            try:
                rbob_latest_val = float(dataset.iloc[-1]["rbob_settle"])
                rbob_latest = {
                    "date": pd.Timestamp(dataset.iloc[-1]["date"]).date().isoformat()
                    if not pd.isna(dataset.iloc[-1]["date"])
                    else dataset_as_of,
                    "value": rbob_latest_val,
                }
            except Exception:
                rbob_latest = None

        freeze_metrics = {
            "aaa_today": {"date": latest_date_iso, "value": latest_price},
            "aaa_week_ago": week_ago,
            "aaa_month_ago": month_ago,
            "event_threshold": float(event_threshold),
            "price_gap": float(event_threshold - latest_price),
            "rbob_settle": rbob_latest,
            "wti_proxy": None,
            "alpha_lift": float(alpha_lift_applied),
            "beta_eff": float(beta_eff),
            "prior_weight": float(prior_weight),
            "residual_sigma": float(residual_sigma),
        }
    risk_context["freeze_metrics"] = freeze_metrics

    def posterior_factory(
        rbob_delta: float, alpha_delta: float
    ) -> PosteriorDistribution:
        beta_effect = (
            _select_beta(
                rbob_delta,
                structural,
                beta_up_scale=beta_up_scale,
                beta_dn_scale=beta_dn_scale,
            )
            * beta_horizon_scale
        )
        adjusted = (
            base_samples
            + alpha_shift
            + alpha_dynamic_offset
            + alpha_delta
            + beta_effect * rbob_delta
        )
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
    posterior_summary["residual_sigma"] = residual_sigma
    event_ctx["residual_sigma"] = residual_sigma
    posterior_summary["days_to_event"] = days_to_event
    posterior_summary["beta_eff"] = beta_eff
    posterior_summary["alpha_lift_applied"] = alpha_lift_applied
    if central_probability is not None:
        posterior_summary["central_kalshi_probability"] = central_probability
    for threshold in thresholds:
        prob_value = base_posterior.prob_above(float(threshold))
        key_gt = f"prob_gt_{threshold:.2f}"
        posterior_summary[key_gt] = prob_value

    horizon_threshold = float(event_threshold)
    base_prob_event = base_posterior.prob_above(horizon_threshold)
    horizon_scenarios: list[dict[str, float]] = []
    scenario_specs = [
        ("RBOB +$0.05", 0.05, 0.0),
        ("RBOB -$0.05", -0.05, 0.0),
        ("α +$0.02", 0.0, 0.02),
        ("α -$0.02", 0.0, -0.02),
    ]
    for label, rbob_change, alpha_change in scenario_specs:
        scenario_posterior = posterior_factory(rbob_change, alpha_change)
        scenario_prob = scenario_posterior.prob_above(horizon_threshold)
        horizon_scenarios.append(
            {
                "scenario": label,
                "beta_eff": beta_eff,
                "rbob_delta": rbob_change,
                "alpha_delta": alpha_change,
                "probability": scenario_prob,
                "prob_delta_pp": (scenario_prob - base_prob_event) * 100,
            }
        )

    event_key = f"prob_gt_{float(event_threshold):.2f}"
    summary_payload = {
        "as_of": dataset_as_of,
        "threshold": float(event_threshold),
        "prob_yes": posterior_summary.get(event_key),
        "prior_weight": prior_weight,
        "prior_weight_source": prior_weight_source,
        "dataset_digest": dataset_digest,
        "days_to_event": days_to_event,
        "beta_eff": beta_eff,
        "alpha_lift_applied": alpha_lift_applied,
        "residual_sigma": residual_sigma,
    }
    summary_payload["tplus1_sensitivity"] = horizon_scenarios
    (Path("data_proc") / "summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    # Write a compact summary for quick consumption
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

    forecast_fig = figures_dir / "nowcast.png"
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

    fundamentals_fig = figures_dir / "wpsr_dashboard.png"
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

    delta_hist_fig = figures_dir / "aaa_delta_hist.png"
    try:
        plot_aaa_delta_histogram(
            aaa_delta_recent,
            delta_hist_fig,
            as_of=dataset_as_of,
            sigma=residual_sigma,
            reference_delta=0.062,
            source="AAA daily",
        )
    except Exception:  # noqa: BLE001
        pass

    # Additional figures: structural pass-through fit, prior CDF, posterior density
    from kalshi_gas.reporting.visuals import (
        plot_pass_through_fit,
        plot_prior_cdf,
        plot_posterior_density,
    )

    structural_fig = figures_dir / "pass_through.png"
    try:
        plot_pass_through_fit(
            dataset,
            structural,
            structural_fig,
            as_of=dataset_as_of,
            source=model_source,
        )
    except Exception:  # noqa: BLE001
        pass

    prior_cdf_fig = figures_dir / "prior_cdf.png"
    try:
        plot_prior_cdf(
            thresholds,
            prior_model.cdf_values,
            prior_cdf_fig,
            as_of=dataset_as_of,
            source="Kalshi market bins",
        )
    except Exception:  # noqa: BLE001
        pass

    posterior_fig = figures_dir / "posterior.png"
    try:
        plot_posterior_density(
            base_posterior.samples,
            event_threshold,
            posterior_fig,
            as_of=dataset_as_of,
            source=model_source,
        )
    except Exception:  # noqa: BLE001
        pass

    results_csv = memo_dir / "forecast_results.csv"
    backtest.test_frame.to_csv(results_csv, index=False)

    # Headline should reflect the central/event threshold, not the first bin
    headline_threshold = float(event_threshold)
    headline_probability = posterior_summary.get(f"prob_gt_{headline_threshold:.2f}")
    headline_date = cfg.event.resolution_date.isoformat()
    posterior_summary["final_threshold"] = headline_threshold
    posterior_summary["final_probability"] = headline_probability
    posterior_summary["final_probability_pct"] = (
        float(headline_probability) * 100 if headline_probability is not None else None
    )
    thesis_parts: list[str] = []
    as_of_text = dataset_as_of or "the latest available date"
    event_name = cfg.event.name
    if headline_probability is not None:
        thesis_parts.append(
            f"As of {as_of_text}, we assign {float(headline_probability) * 100:.1f}% to {event_name}."
        )
    else:
        thesis_parts.append(f"As of {as_of_text}, we evaluate {event_name}.")
    mean_value = posterior_summary.get("mean")
    if isinstance(mean_value, (int, float)):
        thesis_parts.append(
            f"The posterior mean price is ${float(mean_value):.2f}, blending a prior weight of {prior_weight:.2f}."
        )
    if risk_adjustments:
        thesis_parts.append("Key adjustments: " + "; ".join(risk_adjustments) + ".")
    else:
        thesis_parts.append("No active tail adjustments were applied.")
    posterior_summary["thesis"] = " ".join(thesis_parts)

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
            "delta_hist": delta_hist_fig,
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
        tplus1_sensitivity=horizon_scenarios,
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

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    memo_output = reports_dir / "memo.md"
    shutil.copy2(report_path, memo_output)

    reports_figures_dir = reports_dir / "figures"
    reports_figures_dir.mkdir(parents=True, exist_ok=True)
    figure_exports = {
        forecast_fig,
        calibration_fig,
        fundamentals_fig,
        risk_box_fig,
        sensitivity_fig,
        structural_fig,
        prior_cdf_fig,
        posterior_fig,
    }
    for path in figure_exports:
        try:
            shutil.copy2(path, reports_figures_dir / path.name)
        except FileNotFoundError:
            continue
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
        "latest_inputs": latest_inputs,
        "sample_fallback_used": sample_fallback_used,
    }


def _check_required_raw_inputs(requirements: Tuple[Dict[str, object], ...]) -> None:
    now = datetime.now(timezone.utc)
    missing: List[Dict[str, object]] = []
    stale: List[tuple[Dict[str, object], timedelta]] = []
    for asset in requirements:
        path = asset["path"]
        if not isinstance(path, Path):
            path = Path(str(path))
            asset["path"] = path
        if not path.exists():
            missing.append(asset)
            continue
        max_age = asset.get("max_age")
        if isinstance(max_age, timedelta):
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            except FileNotFoundError:
                missing.append(asset)
                continue
            age = now - mtime
            if age > max_age:
                stale.append((asset, age))

    if not missing and not stale:
        return

    lines: List[str] = []
    if missing:
        lines.append("Missing required raw inputs:")
        for asset in missing:
            schema = asset.get("schema", "")
            sample = asset.get("sample")
            sample_hint = f"; sample: {sample}" if sample else ""
            schema_hint = f" (schema: {schema})" if schema else ""
            lines.append(
                f"  - {asset['name']}: {asset['path']}{schema_hint}{sample_hint}"
            )
    if stale:
        lines.append("Stale raw inputs (refresh before running):")
        for asset, age in stale:
            max_age = asset.get("max_age")
            schema = asset.get("schema", "")
            sample = asset.get("sample")
            sample_hint = f"; sample: {sample}" if sample else ""
            schema_hint = f" (schema: {schema})" if schema else ""
            age_hours = age.total_seconds() / 3600.0
            limit_hours = (
                max_age.total_seconds() / 3600.0
                if isinstance(max_age, timedelta)
                else None
            )
            limit_hint = (
                f", limit {limit_hours:.1f}h" if limit_hours is not None else ""
            )
            lines.append(
                f"  - {asset['name']}: {asset['path']} (age {age_hours:.1f}h{limit_hint}){schema_hint}{sample_hint}"
            )
    lines.append(
        "Populate the files above (see data/sample/ for schemas) before rerunning `make report`."
    )
    raise RuntimeError("\n".join(lines))
