"""Market-implied prior construction from Kalshi bin probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _validate_thresholds(thresholds: Sequence[float]) -> np.ndarray:
    if not (3 <= len(thresholds) <= 5):
        raise ValueError("Thresholds count must be between 3 and 5")
    arr = np.asarray(thresholds, dtype=float)
    if np.any(np.isnan(arr)):
        raise ValueError("Thresholds contain NaN")
    if np.any(np.diff(arr) <= 0):
        raise ValueError("Thresholds must be strictly increasing")
    return arr


def _validate_probabilities(probabilities: Sequence[float], size: int) -> np.ndarray:
    if len(probabilities) != size:
        raise ValueError("Probabilities length must match thresholds")
    probs = np.asarray(probabilities, dtype=float)
    if np.any(np.isnan(probs)):
        raise ValueError("Probabilities contain NaN")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("Probabilities must lie within [0, 1]")
    return probs


@dataclass
class MarketPriorCDF:
    """Isotonic regression CDF fitted to Kalshi market bins."""

    thresholds: np.ndarray
    cdf_values: np.ndarray
    _iso: IsotonicRegression

    @classmethod
    def fit(
        cls,
        thresholds: Sequence[float],
        probabilities: Sequence[float],
    ) -> "MarketPriorCDF":
        x = _validate_thresholds(thresholds)
        y = _validate_probabilities(probabilities, len(x))

        model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        fitted = model.fit_transform(x, y)

        return cls(thresholds=x, cdf_values=fitted, _iso=model)

    def evaluate(self, threshold: float) -> float:
        """Return the CDF evaluated at the supplied threshold."""
        value = float(self._iso.transform([threshold])[0])
        return float(np.clip(value, 0.0, 1.0))

    def survival(self, threshold: float) -> float:
        return 1.0 - self.evaluate(threshold)

    @property
    def knots(self) -> list[tuple[float, float]]:
        x_knots: Iterable[float] = getattr(self._iso, "X_thresholds_", self.thresholds)
        y_knots: Iterable[float] = getattr(self._iso, "y_thresholds_", self.cdf_values)
        return list(zip(map(float, x_knots), map(float, y_knots)))

    def is_monotone(self) -> bool:
        diffs = np.diff([value for _, value in self.knots])
        return bool(np.all(diffs >= -1e-12))
