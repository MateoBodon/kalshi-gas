import numpy as np
import pytest

from kalshi_gas.pipeline.backtest import build_calibration, sample_crps


def test_sample_crps_zero_for_perfect_forecast() -> None:
    samples = np.full(100, 3.5)
    assert sample_crps(samples, 3.5) == pytest.approx(0.0, abs=1e-9)


def test_calibration_monotone_bins() -> None:
    probabilities = np.linspace(0.1, 0.9, num=9)
    outcomes = (probabilities > 0.5).astype(int)
    table = build_calibration(probabilities, outcomes, bins=3)
    forecast_means = table["forecast_mean"].to_numpy()
    assert np.all(np.diff(forecast_means) >= -1e-9)
