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
from kalshi_gas.models.structural import fit_structural_pass_through
from kalshi_gas.pipeline.run_all import _prior_cdf_factory, _select_beta
from kalshi_gas.reporting.visuals import plot_calibration
from kalshi_gas.utils.kalshi_bins import load_kalshi_bins, select_central_threshold


def sample_crps(samples: np.ndarray, observation: float) -> float:
    """Compute CRPS via empirical samples."""

    draws = np.asarray(samples, dtype=float)
    if draws.size == 0:
        raise ValueError("CRPS requires non-empty sample array")
    obs = float(observation)
    diff_obs = np.mean(np.abs(draws - obs))
    diff_draws = np.mean(np.abs(draws[:, None] - draws[None, :]))
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


def _deterministic_prob(prediction: float, threshold: float) -> float:
    return 1.0 if prediction >= threshold else 0.0


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


def run_freeze_backtest(config_path: str | None = None) -> Dict[str, object]:
    cfg = load_config(config_path)
    run_all_etl(cfg)
    dataset = assemble_dataset(cfg)

    figures_dir, _, data_proc_dir = _prepare_directories(cfg)

    bins_path = Path("data_raw/kalshi_bins.yml")
    thresholds, probabilities = load_kalshi_bins(bins_path)
    central_threshold, _ = select_central_threshold(thresholds, probabilities)
    prior_model = MarketPriorCDF.fit(thresholds, probabilities)
    prior_fn = _prior_cdf_factory(prior_model)
    prior_samples = _prior_samples(prior_model)

    ensemble_weights = cfg.model.ensemble_weights
    prior_weight = cfg.model.prior_weight

    freeze_dates = _freeze_schedule()

    records: List[Dict[str, float | str | pd.Timestamp]] = []
    calibration_probs: List[float] = []
    calibration_outcomes: List[int] = []

    def evaluate_subset(
        subset: pd.DataFrame, current_date: pd.Timestamp, freeze_date: pd.Timestamp
    ) -> None:
        nonlocal records, calibration_probs, calibration_outcomes

        subset = subset.reset_index(drop=True)
        row = subset[subset["date"] == current_date]
        if row.empty:
            return
        row = row.iloc[-1]

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
            beta_effect = _select_beta(rbob_delta, structural)
            adjusted_samples = base_samples + alpha_delta + beta_effect * rbob_delta
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

        beta_effect = _select_beta(rbob_delta, structural)
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

            outcome = int(actual >= float(threshold))

            posterior_crps = sample_crps(posterior.samples, actual)
            carry_crps = _deterministic_crps(carry_price, actual)
            rbob_crps = _deterministic_crps(rbob_prediction, actual)

            if abs(float(threshold) - central_threshold) < 1e-9:
                calibration_probs.append(prob_posterior)
                calibration_outcomes.append(outcome)

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
            raise RuntimeError("No backtest records generated")

    records_df = pd.DataFrame(records)

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
    }


def main() -> None:
    result = run_freeze_backtest()
    print("Backtest metrics written to", result["metrics_path"])
    print("Calibration figure saved to", result["calibration_path"])


if __name__ == "__main__":
    main()
