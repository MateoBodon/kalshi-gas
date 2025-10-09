import numpy as np
import pandas as pd

from kalshi_gas.models.nowcast import NowcastModel


def generate_series(length: int = 120) -> pd.Series:
    rng = np.random.default_rng(42)
    trend = np.linspace(0, 0.5, num=length)
    noise = rng.normal(scale=0.05, size=length)
    data = 3.0 + trend + noise
    index = pd.date_range("2024-01-01", periods=length, freq="D")
    return pd.Series(data, index=index)


def test_nowcast_variance_grows_with_horizon() -> None:
    series = generate_series()
    model = NowcastModel(horizon=7, simulations=1000, drift_bounds=(2.5, 4.5))
    model.fit(series)
    short_sim = model.simulate(steps=1)
    long_sim = model.simulate(steps=7)
    assert long_sim.variance >= short_sim.variance


def test_nowcast_samples_length_matches_simulations() -> None:
    series = generate_series()
    sims = 750
    model = NowcastModel(horizon=5, simulations=sims, drift_bounds=(2.0, 5.0))
    result = model.predict(series)
    assert len(result.samples) == sims
    assert result.samples.dtype.kind == "f"
