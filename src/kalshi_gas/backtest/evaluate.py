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

    train_preds = ensemble.predict(train)
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
        "brier_score": brier_score(test["event_probability"].values, test["event_outcome"].values),
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

    calib = calibration_table(
        probabilities=test["event_probability"].values,
        outcomes=test["event_outcome"].values,
        bins=10,
    )

    return BacktestResult(test_frame=test, metrics=metrics, calibration=calib)
