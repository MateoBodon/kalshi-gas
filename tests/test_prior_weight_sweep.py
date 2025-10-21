import numpy as np
import pandas as pd

from kalshi_gas.backtest.calibrate_prior import (
    SWEEP_VALUES,
    _evaluate_weights,
)
from kalshi_gas.utils.thresholds import load_kalshi_thresholds


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


def test_thresholds_unique_and_central_tracked() -> None:
    bundle = load_kalshi_thresholds()
    thresholds = bundle.thresholds
    used_thresholds = set(float(value) for value in thresholds)
    used_thresholds.add(float(bundle.central_threshold))

    assert len(used_thresholds) == len(thresholds)
