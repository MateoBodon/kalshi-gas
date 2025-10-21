"""Backtesting utilities for the ensemble."""

from __future__ import annotations

from dataclasses import dataclass

from math import erf

import numpy as np
import pandas as pd

from kalshi_gas.backtest.metrics import brier_score, calibration_table, crps_gaussian
from kalshi_gas.models.ensemble import EnsembleModel


@dataclass
class BacktestResult:
    test_frame: pd.DataFrame
    metrics: dict
    calibration: pd.DataFrame
    calibrated_prior_weight: float | None = None


def compute_event_probabilities(
    mean_forecast: pd.Series,
    sigma: float,
    threshold: float,
) -> pd.Series:
    sigma = max(sigma, 1e-6)
    z = (threshold - mean_forecast) / sigma
    vec_erf = np.vectorize(erf)
    probs = 1 - 0.5 * (1 + vec_erf(z / np.sqrt(2)))
    return probs.clip(0, 1)


def calibrate_mixture_weight(
    likelihood: np.ndarray,
    prior: np.ndarray,
    outcomes: np.ndarray,
    weight_grid: np.ndarray | None = None,
) -> float | None:
    """Select prior weight minimizing Brier score on historical outcomes."""
    mask = np.isfinite(likelihood) & np.isfinite(prior) & np.isfinite(outcomes)
    if mask.sum() < 5:
        return None
    like = likelihood[mask]
    prior_probs = prior[mask]
    outs = outcomes[mask]

    if weight_grid is None:
        weight_grid = np.linspace(0.0, 1.0, num=21)

    best_weight = None
    best_score = float("inf")
    for weight in weight_grid:
        mixture = (1 - weight) * like + weight * prior_probs
        score = brier_score(mixture, outs)
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


def jackknife_brier(probabilities: np.ndarray, outcomes: np.ndarray) -> float | None:
    """Return jackknife standard error for the Brier score."""
    n = len(probabilities)
    if n <= 1:
        return None
    estimates = []
    for idx in range(n):
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        estimates.append(brier_score(probabilities[mask], outcomes[mask]))
    pseudo = np.asarray(estimates, dtype=float)
    pseudo_mean = np.mean(pseudo)
    variance = (n - 1) / n * np.sum((pseudo - pseudo_mean) ** 2)
    if variance < 0:
        return None
    return float(np.sqrt(variance))


def run_backtest(
    dataset: pd.DataFrame,
    ensemble: EnsembleModel,
    threshold: float = 3.5,
    test_fraction: float = 0.3,
) -> BacktestResult:
    if dataset.empty:
        raise ValueError("Dataset empty")
    split_idx = int(len(dataset) * (1 - test_fraction))
    train = dataset.iloc[:split_idx]
    test = dataset.iloc[split_idx:].copy()

    ensemble.fit(train)

    test_preds = ensemble.predict(test)

    sigma = getattr(ensemble, "residual_std", 0.1)

    test["ensemble_mean"] = test_preds["ensemble"]
    test["actual"] = test["target_future_price"]
    test["event_outcome"] = (test["actual"] >= threshold).astype(int)
    test["event_probability"] = compute_event_probabilities(
        test["ensemble_mean"],
        sigma=sigma,
        threshold=threshold,
    )

    metrics = {
        "brier_score": brier_score(
            test["event_probability"].values, test["event_outcome"].values
        ),
        "crps": crps_gaussian(
            mu=test["ensemble_mean"].values,
            sigma=np.full(len(test), sigma),
            observation=test["actual"].values,
        ),
        "rmse": float(
            np.sqrt(
                np.mean((test["ensemble_mean"].values - test["actual"].values) ** 2)
            )
        ),
    }
    metrics["brier_score_se"] = jackknife_brier(
        test["event_probability"].values, test["event_outcome"].values
    )

    prior_probs: np.ndarray | None = None

    if "regular_gas_price" in test.columns:
        carry_prob = (test["regular_gas_price"].values >= threshold).astype(float)
        metrics["brier_carry"] = brier_score(carry_prob, test["event_outcome"].values)
        metrics["carry_rmse"] = float(
            np.sqrt(
                np.mean((test["regular_gas_price"].values - test["actual"].values) ** 2)
            )
        )
    if "kalshi_prob" in test.columns:
        prior_probs = test["kalshi_prob"].to_numpy(dtype=float)
        metrics["brier_prior"] = brier_score(prior_probs, test["event_outcome"].values)

    calib = calibration_table(
        probabilities=test["event_probability"].values,
        outcomes=test["event_outcome"].values,
        bins=10,
    )

    calibrated_weight = None
    if prior_probs is not None:
        likelihood = test["event_probability"].to_numpy(dtype=float)
        outcomes = test["event_outcome"].to_numpy(dtype=float)
        calibrated_weight = calibrate_mixture_weight(
            likelihood=likelihood,
            prior=prior_probs,
            outcomes=outcomes,
        )
        if calibrated_weight is not None:
            posterior_probs = (
                1 - calibrated_weight
            ) * likelihood + calibrated_weight * prior_probs
            test["posterior_event_probability"] = posterior_probs
            metrics["posterior_brier"] = brier_score(posterior_probs, outcomes)
            metrics["prior_weight_calibrated"] = calibrated_weight
            metrics["posterior_brier_se"] = jackknife_brier(posterior_probs, outcomes)

    if "posterior_event_probability" not in test.columns:
        test["posterior_event_probability"] = test["event_probability"]

    return BacktestResult(
        test_frame=test,
        metrics=metrics,
        calibration=calib,
        calibrated_prior_weight=calibrated_weight,
    )
