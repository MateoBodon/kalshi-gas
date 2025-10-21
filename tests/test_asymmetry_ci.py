import numpy as np
import pandas as pd

from kalshi_gas.models.pass_through import PassThroughModel


def _make_synthetic(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rbob = np.cumsum(rng.normal(0, 0.02, size=n)) + 2.5
    noise = rng.normal(0, 0.03, size=n)
    price = 2.8 + 0.6 * rbob + noise
    return pd.DataFrame(
        {
            "date": dates,
            "regular_gas_price": price,
            "rbob_settle": rbob,
            "rbob_7d_change": pd.Series(rbob).diff(7),
            "inventory_change": rng.normal(0, 0.1, size=n),
        }
    )


def test_bootstrap_asymmetry_ci_returns_interval() -> None:
    dataset = _make_synthetic()
    structural_result = {
        "beta_up": 0.55,
        "beta_dn": 0.45,
        "lag": 7,
    }
    point, lower, upper = PassThroughModel.bootstrap_asymmetry_ci(
        dataset,
        structural_result,
        samples=50,
        seed=0,
    )
    assert lower <= point <= upper
    assert upper - lower >= 0.0
