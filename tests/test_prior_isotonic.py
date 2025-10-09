import numpy as np
import pytest

from kalshi_gas.models.prior import MarketPriorCDF


@pytest.mark.parametrize(
    "thresholds, probabilities",
    [
        ([3.0, 3.1, 3.2], [0.2, 0.5, 0.8]),
        ([3.0, 3.05, 3.1, 3.15, 3.2], [0.1, 0.25, 0.5, 0.7, 0.9]),
    ],
)
def test_cdf_monotone_and_bounded(thresholds, probabilities) -> None:
    prior = MarketPriorCDF.fit(thresholds, probabilities)
    knots = prior.knots
    values = [value for _, value in knots]
    assert all(values[idx] <= values[idx + 1] + 1e-9 for idx in range(len(values) - 1))

    samples = np.linspace(thresholds[0] - 0.5, thresholds[-1] + 0.5, num=21)
    cdf_values = [prior.evaluate(val) for val in samples]
    assert all(0.0 <= val <= 1.0 for val in cdf_values)


def test_survival_complements_cdf() -> None:
    prior = MarketPriorCDF.fit([3.0, 3.1, 3.2], [0.3, 0.6, 0.85])
    for point in [2.95, 3.05, 3.15, 3.25]:
        cdf = prior.evaluate(point)
        survival = prior.survival(point)
        assert np.isclose(cdf + survival, 1.0, atol=1e-9)


def test_invalid_threshold_count_raises() -> None:
    with pytest.raises(ValueError):
        MarketPriorCDF.fit([3.0, 3.1], [0.2, 0.5])
