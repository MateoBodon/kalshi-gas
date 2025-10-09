import numpy as np
import pytest

from kalshi_gas.backtest.metrics import brier_score, calibration_table, crps_gaussian


def test_brier_score_basic() -> None:
    probs = np.array([0.1, 0.9, 0.5])
    outcomes = np.array([0, 1, 1])
    score = brier_score(probs, outcomes)
    assert score == pytest.approx((0.01 + 0.01 + 0.25) / 3)


def test_crps_gaussian_zero_error() -> None:
    mu = np.zeros(10)
    obs = np.zeros(10)
    sigma = np.ones(10)
    score = crps_gaussian(mu, sigma, obs)
    expected = (np.sqrt(2 / np.pi) - 1 / np.sqrt(np.pi))
    assert score == pytest.approx(expected, rel=1e-4)


def test_calibration_table_counts() -> None:
    probs = np.linspace(0.1, 0.9, num=9)
    outcomes = (probs > 0.5).astype(int)
    table = calibration_table(probs, outcomes, bins=3)
    assert table["count"].sum() == len(probs)
