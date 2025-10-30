"""Prior gating behavior for T+1 horizon."""

from __future__ import annotations

import numpy as np

from kalshi_gas.models.posterior import PosteriorDistribution
from kalshi_gas.pipeline.run_all import _effective_prior_weight


def test_prior_weight_collapses_at_tplus1() -> None:
    rng = np.random.default_rng(seed=7)
    samples = rng.normal(loc=3.03, scale=0.008, size=8192)
    nowcast_mean = float(np.mean(samples))
    prior_weight = 0.10
    threshold = 3.10
    residual_sigma = 0.008

    prior_weight_eff, z_score = _effective_prior_weight(
        prior_weight,
        days_to_event=1,
        event_threshold=threshold,
        nowcast_mean=nowcast_mean,
        residual_sigma=residual_sigma,
    )

    assert prior_weight_eff == 0.0
    assert z_score > 0.0

    posterior = PosteriorDistribution(
        samples=samples,
        prior_cdf=lambda values: np.zeros_like(values, dtype=float),
        prior_weight=prior_weight_eff,
    )
    likelihood_tail = float(np.mean(samples > threshold))
    assert abs(posterior.prob_above(threshold) - likelihood_tail) < 1e-6
