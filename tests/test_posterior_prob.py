import numpy as np

from kalshi_gas.models.posterior import PosteriorDistribution


def linear_prior(values: np.ndarray) -> np.ndarray:
    return np.clip((values - 2.5) / 1.0, 0.0, 1.0)


def create_posterior(prior_weight: float = 0.3) -> PosteriorDistribution:
    rng = np.random.default_rng(0)
    samples = 3.0 + rng.normal(scale=0.05, size=500)
    return PosteriorDistribution(
        samples=samples, prior_cdf=linear_prior, prior_weight=prior_weight
    )


def test_cdf_monotone() -> None:
    posterior = create_posterior(0.4)
    thresholds = np.linspace(2.9, 3.2, num=8)
    cdf_values = posterior.cdf(thresholds)
    assert np.all(np.diff(cdf_values) >= -1e-9)


def test_prob_above_decreases_with_threshold() -> None:
    posterior = create_posterior(0.2)
    probs = [posterior.prob_above(x) for x in [3.0, 3.05, 3.1]]
    assert probs[0] >= probs[1] >= probs[2]
