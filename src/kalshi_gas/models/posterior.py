"""Posterior utilities blending prior and likelihood with sensitivity grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

CDFCallable = Callable[[np.ndarray], np.ndarray]


def _validate_prior_weight(weight: float) -> float:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("prior_weight must be within [0, 1]")
    return float(weight)


def _ensure_array(values: Sequence[float] | float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


@dataclass
class PosteriorDistribution:
    """Mixture posterior combining empirical samples and a prior CDF."""

    samples: np.ndarray
    prior_cdf: CDFCallable
    prior_weight: float = 0.35

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=float)
        if self.samples.size == 0:
            raise ValueError("PosteriorDistribution requires non-empty samples")
        self.sorted_samples = np.sort(self.samples)
        self.empirical_size = float(len(self.sorted_samples))
        self.prior_weight = _validate_prior_weight(self.prior_weight)

    def _empirical_cdf(self, thresholds: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(self.sorted_samples, thresholds, side="right")
        return indices / self.empirical_size

    def cdf(self, thresholds: Sequence[float] | float) -> np.ndarray | float:
        values = _ensure_array(thresholds)
        empirical = self._empirical_cdf(values)
        prior = np.clip(self.prior_cdf(values), 0.0, 1.0)
        mixture = (1 - self.prior_weight) * empirical + self.prior_weight * prior
        mixture = np.clip(mixture, 0.0, 1.0)
        if np.ndim(thresholds) == 0:
            return float(mixture[0])
        return mixture

    def prob_above(self, threshold: float) -> float:
        return float(1 - self.cdf(threshold))

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def variance(self) -> float:
        return float(np.var(self.samples))

    def summary(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "variance": self.variance,
        }


def compute_sensitivity(
    posterior_fn: Callable[[float, float], PosteriorDistribution],
    thresholds: Iterable[float] | None = None,
    rbob_deltas: Iterable[float] | None = None,
    alpha_deltas: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Evaluate posterior probability sensitivity across deltas."""

    thresholds = list(thresholds) if thresholds is not None else [3.05, 3.10, 3.15]
    rbob_deltas = list(rbob_deltas) if rbob_deltas is not None else [-0.05, 0.0, 0.05]
    alpha_deltas = (
        list(alpha_deltas) if alpha_deltas is not None else [-0.05, 0.0, 0.05]
    )

    records: list[dict[str, float]] = []
    for rbob_delta in rbob_deltas:
        for alpha_delta in alpha_deltas:
            posterior = posterior_fn(rbob_delta, alpha_delta)
            for threshold in thresholds:
                prob = posterior.prob_above(threshold)
                records.append(
                    {
                        "threshold": float(threshold),
                        "rbob_delta": float(rbob_delta),
                        "alpha_delta": float(alpha_delta),
                        "prob_above": prob,
                    }
                )

    frame = pd.DataFrame(records)
    frame.sort_values(["threshold", "rbob_delta", "alpha_delta"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame
