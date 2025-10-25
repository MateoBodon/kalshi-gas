"""Single source for Kalshi market threshold configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import yaml
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class ThresholdBundle:
    thresholds: np.ndarray
    probabilities: np.ndarray
    central_threshold: float
    central_probability: float | None


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing Kalshi bins file at {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload


def _compute_central(
    thresholds: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float | None]:
    if thresholds.size == 0:
        raise ValueError("Cannot select central threshold from empty list")
    sort_idx = np.argsort(thresholds)
    sorted_thresholds = thresholds[sort_idx]
    median_idx = len(sorted_thresholds) // 2
    central_threshold = float(sorted_thresholds[median_idx])

    central_probability = None
    if probabilities.size == thresholds.size and probabilities.size > 0:
        sorted_probs = probabilities[sort_idx]
        central_probability = float(sorted_probs[median_idx])

    return central_threshold, central_probability


def load_kalshi_thresholds(path: Path | None = None) -> ThresholdBundle:
    resolved = Path(path) if path is not None else Path("data_raw/kalshi_bins.yml")
    payload = _load_yaml(resolved)

    thresholds = np.asarray(payload.get("thresholds", []), dtype=float)
    probabilities = np.asarray(payload.get("probabilities", []), dtype=float)

    if thresholds.size == 0 or probabilities.size != thresholds.size:
        raise ValueError("Invalid Kalshi bins configuration")

    unique_thresholds = np.unique(thresholds)
    if len(unique_thresholds) != len(thresholds):
        raise ValueError("Kalshi thresholds must be unique")

    # Optional isotonic smoothing of provided CDF points to enforce monotonicity
    # Set KALSHI_GAS_SMOOTH_BINS=1 to enable
    if os.getenv("KALSHI_GAS_SMOOTH_BINS", "0") == "1":
        # Preserve original order; fit on sorted x
        order = np.argsort(thresholds)
        x_sorted = thresholds[order]
        y_sorted = probabilities[order]
        iso = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        y_fit = iso.fit_transform(x_sorted, y_sorted)
        # Map fitted values back to original order
        inv_order = np.argsort(order)
        probabilities = y_fit[inv_order]

    central_threshold, central_probability = _compute_central(thresholds, probabilities)

    return ThresholdBundle(
        thresholds=thresholds,
        probabilities=probabilities,
        central_threshold=central_threshold,
        central_probability=central_probability,
    )
