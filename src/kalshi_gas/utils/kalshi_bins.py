"""Helpers for working with Kalshi bin configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import yaml


def load_kalshi_bins(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Kalshi bins file at {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    thresholds = np.asarray(payload.get("thresholds", []), dtype=float)
    probabilities = np.asarray(payload.get("probabilities", []), dtype=float)
    if thresholds.size == 0 or probabilities.size != thresholds.size:
        raise ValueError("Invalid Kalshi bins configuration")
    return thresholds, probabilities


def select_central_threshold(
    thresholds: np.ndarray, probabilities: np.ndarray | None = None
) -> Tuple[float, float | None]:
    if thresholds.size == 0:
        raise ValueError("Cannot select threshold from empty array")
    sorted_idx = np.argsort(thresholds)
    sorted_thresholds = thresholds[sorted_idx]
    median_idx = len(sorted_thresholds) // 2
    central_threshold = float(sorted_thresholds[median_idx])

    central_probability = None
    if probabilities is not None and probabilities.size == thresholds.size:
        sorted_probs = probabilities[sorted_idx]
        central_probability = float(sorted_probs[median_idx])

    return central_threshold, central_probability
