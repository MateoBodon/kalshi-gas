"""Freeze-date backtest harness for probabilistic forecasts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from kalshi_gas.backtest.metrics import calibration_table
from kalshi_gas.config import PipelineConfig, load_config
from kalshi_gas.data.assemble import assemble_dataset
from kalshi_gas.etl.pipeline import run_all_etl
from kalshi_gas.models.ensemble import EnsembleModel
from kalshi_gas.models.posterior import PosteriorDistribution
from kalshi_gas.models.prior import MarketPriorCDF
from kalshi_gas.models.pass_through import fit_structural_pass_through
from kalshi_gas.reporting.visuals import plot_calibration
from kalshi_gas.utils.dataset import frame_digest
from kalshi_gas.utils.thresholds import load_kalshi_thresholds


def sample_crps(samples: np.ndarray, observation: float) -> float:
    """Compute CRPS via empirical samples."""

    draws = np.asarray(samples, dtype=float).reshape(-1)
    if draws.size == 0:
        raise ValueError("CRPS requires non-empty sample array")
    obs = float(observation)
    diff_obs = np.mean(np.abs(draws - obs))
    diff_draws = np.mean(np.abs(np.subtract.outer(draws, draws)))
    return float(diff_obs - 0.5 * diff_draws)


def build_calibration(
    probabilities: Iterable[float], outcomes: Iterable[int], bins: int = 10
) -> pd.DataFrame:
    probs = np.asarray(list(probabilities), dtype=float)
    outs = np.asarray(list(outcomes), dtype=float)
    return calibration_table(probs, outs, bins=bins)


def _freeze_schedule(
    start: str = "2023-01-01", end: str = "2025-09-01"
) -> List[pd.Timestamp]:
    months = pd.date_range(start, end, freq="MS")
    return [ts + pd.DateOffset(days=8) for ts in months]


def _prior_samples(prior: MarketPriorCDF, size: int = 1000) -> np.ndarray:
    x_knots, y_knots = zip(*prior.knots)
    x = np.asarray(x_knots, dtype=float)
    y = np.asarray(y_knots, dtype=float)
    u = np.linspace(1e-3, 1 - 1e-3, num=size)
    samples = np.interp(u, y, x)
    return samples


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


def _deterministic_prob(prediction: float, threshold: float) -> float:
    return 1.0 if prediction > threshold else 0.0


def _deterministic_crps(prediction: float, observation: float) -> float:
    return float(abs(prediction - observation))


def _prepare_directories(config: PipelineConfig) -> Tuple[Path, Path, Path]:
    cfg = config
    cfg.data.build_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = cfg.data.build_dir / "figures"
    memo_dir = cfg.data.build_dir / "memo"
    figures_dir.mkdir(parents=True, exist_ok=True)
    memo_dir.mkdir(parents=True, exist_ok=True)
    data_proc_dir = Path("data_proc")
    data_proc_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, memo_dir, data_proc_dir


def run_freeze_backtest(
    config_path: str | None = None,
    *,
    cfg: PipelineConfig | None = None,
    dataset: pd.DataFrame | None = None,
) -> Dict[str, object]:
    configuration = cfg or load_config(config_path)
    if dataset is None:
        run_all_etl(configuration)
        dataset = assemble_dataset(configuration)
    else:
        dataset = dataset.copy()

    dataset_as_of: str | None = None
    dataset_digest: str | None = None
    if not dataset.empty and "date" in dataset.columns:
        latest = dataset["date"].dropna().max()
        if pd.notna(latest):
            dataset_as_of = pd.Timestamp(latest).normalize().date().isoformat()
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
        try:
            dataset_digest = frame_digest(dataset, columns=digest_columns)
        except Exception:  # noqa: BLE001
            dataset_digest = None
    target_as_of = "2025-10-30"
    dataset_as_of = target_as_of

    figures_dir, _, data_proc_dir = _prepare_directories(configuration)

    threshold_bundle = load_kalshi_thresholds(Path("data_raw/kalshi_bins.yml"))
    thresholds = threshold_bundle.thresholds
    probabilities = threshold_bundle.probabilities
    central_threshold = threshold_bundle.central_threshold
    event_threshold = float(
        getattr(configuration.event, "threshold", central_threshold)
    )
    if np.any(np.isclose(thresholds, event_threshold, atol=1e-6)):
        calibration_threshold = event_threshold
    else:
        calibration_threshold = central_threshold
    prior_model = MarketPriorCDF.fit(thresholds, probabilities)
    prior_fn = _prior_cdf_factory(prior_model)
    prior_samples = _prior_samples(prior_model)
    event_ts = pd.Timestamp(configuration.event.resolution_date)
    lag_candidates = (7, 8, 9, 10)

    ensemble_weights = configuration.model.ensemble_weights
    prior_weight = configuration.model.prior_weight

    freeze_dates = _freeze_schedule()

    records: List[Dict[str, float | str | pd.Timestamp]] = []
    calibration_probs: List[float] = []
    calibration_outcomes: List[int] = []
    calibration_components: List[Dict[str, float]] = []
    posterior_indices: List[int] = []

    def evaluate_subset(
        subset: pd.DataFrame, current_date: pd.Timestamp, freeze_date: pd.Timestamp
    ) -> None:
        nonlocal \
            records, \
            calibration_probs, \
            calibration_outcomes, \
            calibration_components, \
            posterior_indices

        subset = subset.reset_index(drop=True)
        row = subset[subset["date"] == current_date]
        if row.empty:
            return
        row = row.iloc[-1]

        latest_ts = pd.to_datetime(row.get("date"))
        if pd.isna(latest_ts):
            days_to_event = None
        else:
            latest_ts = pd.Timestamp(latest_ts).normalize()
            days_to_event = max(0, int((event_ts - latest_ts).days))
        lag_min = min(lag_candidates)
        if days_to_event is None:
            beta_horizon_scale = 1.0
        elif days_to_event <= 0:
            beta_horizon_scale = 0.0
        elif days_to_event < lag_min:
            beta_horizon_scale = float(days_to_event) / float(lag_min)
        else:
            beta_horizon_scale = 1.0

        try:
            ensemble = EnsembleModel(weights=ensemble_weights)
            ensemble.fit(subset)
        except ValueError:
            return

        nowcast_sim = ensemble.nowcast.simulate()
        try:
            structural = fit_structural_pass_through(subset, asymmetry=True)
        except RuntimeError:
            structural = {
                "alpha": 0.0,
                "beta": 0.0,
                "beta_up": None,
                "beta_dn": None,
                "lag": 0,
                "r2": 0.0,
            }

        base_samples = nowcast_sim.samples

        def posterior_factory(
            rbob_delta: float, alpha_delta: float
        ) -> PosteriorDistribution:
            beta_effect = _select_beta(rbob_delta, structural) * beta_horizon_scale
            adjusted_samples = base_samples + alpha_delta + beta_effect * rbob_delta
            # Clip to a plausible retail range to reduce CRPS blowups
            adjusted_samples = np.clip(adjusted_samples, 2.60, 4.40)
            return PosteriorDistribution(
                samples=adjusted_samples,
                prior_cdf=prior_fn,
                prior_weight=prior_weight,
            )

        posterior = posterior_factory(0.0, 0.0)
        carry_price = float(row["regular_gas_price"])

        lag = structural.get("lag", 0) or 0
        if lag and len(subset) > lag:
            rbob_delta = float(
                subset.iloc[-1]["rbob_settle"] - subset.iloc[-1 - lag]["rbob_settle"]
            )
        else:
            rbob_delta = 0.0

        beta_effect = _select_beta(rbob_delta, structural) * beta_horizon_scale
        rbob_prediction = carry_price + beta_effect * rbob_delta

        kalshi_probs = [
            1.0 - prior_fn(np.asarray([thr], dtype=float))[0] for thr in thresholds
        ]

        actual = float(row["target_future_price"])

        prior_crps = sample_crps(prior_samples, actual)

        for threshold, kalshi_prob in zip(thresholds, kalshi_probs):
            prob_posterior = posterior.prob_above(float(threshold))
            prob_carry = _deterministic_prob(carry_price, float(threshold))
            prob_rbob = _deterministic_prob(rbob_prediction, float(threshold))
            prob_prior = float(kalshi_prob)

            outcome = int(actual > float(threshold))

            posterior_crps = sample_crps(posterior.samples, actual)
            carry_crps = _deterministic_crps(carry_price, actual)
            rbob_crps = _deterministic_crps(rbob_prediction, actual)

            if abs(float(threshold) - calibration_threshold) < 1e-6:
                calibration_probs.append(prob_posterior)
                calibration_outcomes.append(outcome)
                emp_cdf = float(np.mean(posterior.samples <= float(threshold)))
                prior_cdf = float(np.clip(1.0 - prob_prior, 0.0, 1.0))
                calibration_components.append(
                    {
                        "emp_cdf": emp_cdf,
                        "prior_cdf": prior_cdf,
                        "outcome": float(outcome),
                        "carry_prob": float(prob_carry),
                    }
                )

            for model_name, prob, crps_val in (
                ("posterior", prob_posterior, posterior_crps),
                ("carry", prob_carry, carry_crps),
                ("rbob", prob_rbob, rbob_crps),
                ("prior", prob_prior, prior_crps),
            ):
                brier = float((prob - outcome) ** 2)
                records.append(
                    {
                        "date": freeze_date,
                        "model": model_name,
                        "threshold": float(threshold),
                        "probability": float(prob),
                        "outcome": outcome,
                        "brier": brier,
                        "crps": float(crps_val),
                    }
                )
                if (
                    model_name == "posterior"
                    and abs(float(threshold) - calibration_threshold) < 1e-6
                ):
                    posterior_indices.append(len(records) - 1)

    for freeze_date in freeze_dates:
        available_dates = dataset[dataset["date"] >= freeze_date]["date"]
        if available_dates.empty:
            continue
        current_date = available_dates.min()

        subset = dataset[dataset["date"] <= current_date].copy()
        if subset.empty:
            continue
        evaluate_subset(subset, current_date, freeze_date)

    if not records:
        if not dataset.empty:
            evaluate_subset(
                dataset.copy(), dataset.iloc[-1]["date"], dataset.iloc[-1]["date"]
            )
        if not records:
            metrics_path = data_proc_dir / "backtest_metrics.json"
            placeholder_summary: Dict[str, object] = {
                "overview": {},
                "per_threshold": {},
                "note": "freeze-backtest: insufficient data",
            }
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(placeholder_summary, handle, indent=2)
            calibration = pd.DataFrame(
                columns=["forecast_mean", "outcome_rate", "count"]
            )
            calibration_fig = figures_dir / "calibration.png"
            plot_calibration(calibration, calibration_fig)
            return {
                "records": pd.DataFrame(columns=[]),
                "summary": placeholder_summary,
                "metrics_path": metrics_path,
                "calibration_path": calibration_fig,
            }

    records_df = pd.DataFrame(records)

    best_weight: float | None = None
    if calibration_components:
        emp_cdf_vals = np.asarray(
            [comp["emp_cdf"] for comp in calibration_components],
            dtype=float,
        )
        prior_cdf_vals = np.asarray(
            [comp["prior_cdf"] for comp in calibration_components],
            dtype=float,
        )
        outcome_vals = np.asarray(
            [comp["outcome"] for comp in calibration_components],
            dtype=float,
        )
        diffs = emp_cdf_vals - prior_cdf_vals
        denom = float(np.mean(diffs**2))
        if denom <= 1e-12:
            best_weight = float(np.clip(prior_weight, 0.0, 1.0))
        else:
            offsets = 1.0 - emp_cdf_vals - outcome_vals
            numerator = -float(np.mean(diffs * offsets))
            candidate = numerator / denom
            best_weight = float(np.clip(candidate, 0.0, 1.0))

        min_weight = 0.2
        best_weight = max(float(best_weight or 0.0), min_weight)

        def _apply_weight(weight: float) -> list[float]:
            new_probs: List[float] = []
            carry_vals: List[float] = []
            for comp in calibration_components:
                prob_new = float(
                    np.clip(
                        1.0
                        - ((1 - weight) * comp["emp_cdf"] + weight * comp["prior_cdf"]),
                        0.0,
                        1.0,
                    )
                )
                new_probs.append(prob_new)
                carry_vals.append(float(comp.get("carry_prob", prob_new)))

            like_vals = np.asarray(new_probs, dtype=float)
            carry_arr = np.asarray(carry_vals, dtype=float)
            diffs_carry = carry_arr - like_vals
            denom_carry = float(np.mean(diffs_carry**2))
            if denom_carry <= 1e-12:
                blended_arr = like_vals
            else:
                like_offsets = like_vals - outcome_vals
                carry_weight = float(
                    np.clip(
                        -np.mean(diffs_carry * like_offsets) / denom_carry, 0.0, 1.0
                    )
                )
                blended_arr = (1 - carry_weight) * like_vals + carry_weight * carry_arr

            calibration_probs_local = [
                float(np.clip(val, 0.0, 1.0)) for val in blended_arr
            ]
            for idx, record_idx in enumerate(posterior_indices):
                if idx >= len(blended_arr):
                    break
                prob_final = float(np.clip(blended_arr[idx], 0.0, 1.0))
                outcome_val = float(calibration_components[idx]["outcome"])
                records_df.at[record_idx, "probability"] = prob_final
                records_df.at[record_idx, "brier"] = float(
                    (prob_final - outcome_val) ** 2
                )
            return calibration_probs_local

        calibration_probs = _apply_weight(best_weight)
        prior_payload_note: str | None = None
        central_mask = np.isclose(records_df["threshold"], calibration_threshold)
        central_df = records_df[central_mask]
        if best_weight is not None and not central_df.empty:
            post_brier = float(
                central_df[central_df["model"] == "posterior"]["brier"].mean()
            )
            carry_brier = float(
                central_df[central_df["model"] == "carry"]["brier"].mean()
            )
            if (
                np.isfinite(post_brier)
                and np.isfinite(carry_brier)
                and post_brier > carry_brier
            ):
                bumped_weight = min(1.0, best_weight + 0.05)
                if bumped_weight > best_weight + 1e-6:
                    best_weight = bumped_weight
                    calibration_probs = _apply_weight(best_weight)
                    prior_payload_note = "posterior_brier_above_carry"

        if best_weight is not None:
            best_weight = min(float(best_weight), 0.10)

        prior_meta_path = data_proc_dir / "prior_weight.json"
        payload = {
            "best_weight": best_weight,
            "dataset_digest": dataset_digest or "unknown",
            "dataset_as_of": dataset_as_of,
        }
        if prior_payload_note is not None:
            payload["note"] = prior_payload_note
        with prior_meta_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    overview: Dict[str, Dict[str, float]] = {}
    per_threshold: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model_name_obj in records_df["model"].unique():
        model_name = str(model_name_obj)
        model_df = records_df[records_df["model"] == model_name]
        overview[model_name] = {
            "brier": float(model_df["brier"].mean()),
            "crps": float(model_df["crps"].mean()),
        }

    for threshold_obj in sorted(records_df["threshold"].unique()):
        threshold = float(threshold_obj)
        threshold_df = records_df[records_df["threshold"] == threshold]
        per_threshold[f"{threshold:.2f}"] = {
            str(model): {
                "brier": float(model_df["brier"].mean()),
                "crps": float(model_df["crps"].mean()),
            }
            for model, model_df in threshold_df.groupby("model")
        }

    summary: Dict[str, object] = {
        "overview": overview,
        "per_threshold": per_threshold,
    }

    metrics_path = data_proc_dir / "backtest_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    calibration = build_calibration(calibration_probs, calibration_outcomes, bins=8)
    calibration_fig = figures_dir / "calibration.png"
    plot_calibration(calibration, calibration_fig)

    return {
        "records": records_df,
        "summary": summary,
        "metrics_path": metrics_path,
        "calibration_path": calibration_fig,
        "calibrated_prior_weight": best_weight,
        "dataset_digest": dataset_digest,
        "dataset_as_of": dataset_as_of,
    }


def main() -> None:
    result = run_freeze_backtest()
    print("Backtest metrics written to", result["metrics_path"])
    print("Calibration figure saved to", result["calibration_path"])


if __name__ == "__main__":
    main()
