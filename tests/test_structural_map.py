import numpy as np
import pandas as pd

from kalshi_gas.models.structural import fit_structural_pass_through


def build_synthetic(
    lag: int = 8, beta_up: float = 0.6, beta_dn: float = -0.2
) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n = 200
    rbob_changes = rng.normal(0, 0.03, size=n)
    rbob_prices = 2.0 + np.cumsum(rbob_changes)

    price_changes = np.zeros(n)
    noise = rng.normal(0, 0.01, size=n)
    for t in range(lag, n):
        source = rbob_changes[t - lag]
        contribution = beta_up * max(source, 0) + beta_dn * min(source, 0)
        price_changes[t] = contribution + noise[t]

    gas_prices = 3.0 + np.cumsum(price_changes)
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "regular_gas_price": gas_prices,
            "rbob_settle": rbob_prices,
        },
        index=index,
    )


def test_structural_selects_correct_lag() -> None:
    data = build_synthetic(lag=8, beta_up=0.5, beta_dn=-0.1)
    result = fit_structural_pass_through(data, lags=(7, 8, 9), asymmetry=True)
    assert result["lag"] == 8
    assert result["beta_up"] is not None
    assert result["beta_dn"] is not None
    assert result["beta_up"] > 0
    assert result["beta_dn"] < 0


def test_structural_symmetric_branch() -> None:
    data = build_synthetic(lag=7)
    result = fit_structural_pass_through(data, lags=(7,), asymmetry=False)
    assert result["lag"] == 7
    assert result["beta_up"] is None
    assert result["beta_dn"] is None
    assert isinstance(result["beta"], float)
