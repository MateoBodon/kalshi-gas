import numpy as np

from kalshi_gas.backtest.evaluate import calibrate_mixture_weight
from kalshi_gas.models.posterior import PosteriorDistribution


def test_posterior_summary_includes_asymmetric_cis() -> None:
    samples = np.linspace(2.5, 3.5, 501)

    def zero_prior(values: np.ndarray) -> np.ndarray:
        return np.zeros_like(values)

    posterior = PosteriorDistribution(samples=samples, prior_cdf=zero_prior)
    summary = posterior.summary()
    assert "ci_5" in summary and "ci_95" in summary
    assert summary["ci_5"] <= summary["mean"] <= summary["ci_95"]
    assert summary["ci_lower_span"] <= summary["ci_upper_span"]


def test_calibrate_mixture_weight_prefers_prior_when_informative() -> None:
    likelihood = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=float)
    prior = np.array([0.9, 0.85, 0.82, 0.18, 0.12, 0.1], dtype=float)
    outcomes = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    weight = calibrate_mixture_weight(likelihood, prior, outcomes)
    assert weight is not None
    assert 0.0 <= weight <= 1.0
    assert weight > 0.3
