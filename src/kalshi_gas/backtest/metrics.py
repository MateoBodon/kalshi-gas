"""Scoring rules and calibration utilities."""

from __future__ import annotations

from math import erf

import numpy as np
import pandas as pd


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute the Brier score."""
    return float(np.mean((probabilities - outcomes) ** 2))


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    vec_erf = np.vectorize(erf)
    return 0.5 * (1 + vec_erf(z / np.sqrt(2)))


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, observation: np.ndarray) -> float:
    """Closed-form CRPS for Gaussian forecasts."""
    sigma = np.maximum(sigma, 1e-6)
    z = (observation - mu) / sigma
    pdf = _norm_pdf(z)
    cdf = _norm_cdf(z)
    crps_values = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    return float(np.mean(crps_values))


def calibration_table(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    """Return calibration table summarizing forecast reliability."""
    df = pd.DataFrame({"prob": probabilities, "outcome": outcomes})
    df = df.dropna()
    df["bin"] = pd.cut(df["prob"], bins=bins, labels=False, include_lowest=True)
    grouped = df.groupby("bin", observed=True)
    table = grouped.agg(
        forecast_mean=("prob", "mean"),
        outcome_rate=("outcome", "mean"),
        count=("outcome", "size"),
    ).reset_index(drop=True)
    return table
