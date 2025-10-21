"""Single source for Kalshi market threshold configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


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

    central_threshold, central_probability = _compute_central(thresholds, probabilities)

    return ThresholdBundle(
        thresholds=thresholds,
        probabilities=probabilities,
        central_threshold=central_threshold,
        central_probability=central_probability,
    )
