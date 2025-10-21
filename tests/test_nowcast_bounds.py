import numpy as np
import pandas as pd

from kalshi_gas.models.nowcast import NowcastModel


def make_flat_series(length: int = 90, value: float = 3.0) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=length, freq="D")
    data = np.full(length, value, dtype=float)
    return pd.Series(data, index=index)


def make_trending_series(length: int = 120) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=length, freq="D")
    trend = np.linspace(0.0, 1.5, num=length)
    rng = np.random.default_rng(123)
    noise = rng.normal(scale=0.05, size=length)
    data = 2.5 + trend + noise
    return pd.Series(data, index=index)


def percentile_width(samples: np.ndarray, lower: float, upper: float) -> float:
    lower_q = np.percentile(samples, lower)
    upper_q = np.percentile(samples, upper)
    return float(upper_q - lower_q)


def test_flat_series_mean_stays_near_last_level() -> None:
    np.random.seed(42)
    series = make_flat_series()
    model = NowcastModel(horizon=5, simulations=2000, drift_bounds=(-0.2, 0.2))
    result = model.predict(series)
    last_level = float(series.iloc[-1])
    assert abs(result.mean - last_level) < 0.05


def test_tight_drift_bounds_shrink_confidence_interval() -> None:
    series = make_trending_series()
    wide = NowcastModel(horizon=10, simulations=4000, drift_bounds=(-0.6, 0.6))
    narrow = NowcastModel(horizon=10, simulations=4000, drift_bounds=(-0.2, 0.2))

    wide.fit(series)
    np.random.seed(101)
    wide_sim = wide.simulate()

    narrow.fit(series)
    np.random.seed(101)
    narrow_sim = narrow.simulate()

    wide_width = percentile_width(wide_sim.samples, 2.5, 97.5)
    narrow_width = percentile_width(narrow_sim.samples, 2.5, 97.5)

    assert narrow_width < wide_width


def test_variance_grows_with_forecast_horizon() -> None:
    series = make_trending_series()
    model = NowcastModel(horizon=10, simulations=3000, drift_bounds=(-0.3, 0.3))
    model.fit(series)

    np.random.seed(7)
    one_step = model.simulate(steps=1)
    np.random.seed(7)
    long_step = model.simulate(steps=10)

    assert long_step.variance > one_step.variance
