import numpy as np
import pandas as pd

from kalshi_gas.backtest.calibrate_prior import (
    SWEEP_VALUES,
    _evaluate_weights,
)
from kalshi_gas.utils.kalshi_bins import select_central_threshold


def test_prior_dominates_when_likelihood_is_poor() -> None:
    preds = pd.Series(
        np.concatenate([np.full(50, 0.2), np.full(50, 0.8)]),
        dtype=float,
    )
    prior = pd.Series(
        np.concatenate([np.full(50, 0.98), np.full(50, 0.02)]),
        dtype=float,
    )
    outcomes = pd.Series(
        np.concatenate([np.ones(50), np.zeros(50)]),
        dtype=int,
    )

    sweep = _evaluate_weights(SWEEP_VALUES, preds, outcomes, prior)
    best_weight = float(sweep.loc[sweep["log_score"].idxmax(), "prior_weight"])

    assert best_weight >= 0.95


def test_select_central_threshold_sorts_inputs() -> None:
    thresholds = np.array([3.2, 3.0, 3.1], dtype=float)
    probabilities = np.array([0.8, 0.2, 0.5], dtype=float)

    median_threshold, median_probability = select_central_threshold(
        thresholds, probabilities
    )

    assert median_threshold == 3.1
    assert median_probability == 0.5
