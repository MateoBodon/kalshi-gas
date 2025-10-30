"""Smoke checks for T+1 horizon discipline."""

from __future__ import annotations

import numpy as np

from kalshi_gas.models.posterior import PosteriorDistribution
from kalshi_gas.pipeline.run_all import _beta_scale_for_horizon


def test_posterior_mean_matches_nowcast_when_prior_zero() -> None:
    nowcast_mean = 3.045
    residual_sigma = 0.01
    rng = np.random.default_rng(seed=42)
    samples = rng.normal(loc=nowcast_mean, scale=residual_sigma, size=8192)
    posterior = PosteriorDistribution(
        samples=samples,
        prior_cdf=lambda values: np.zeros_like(values, dtype=float),
        prior_weight=0.0,
    )
    assert abs(posterior.mean - nowcast_mean) < 1e-3


def test_beta_scale_zero_when_one_day() -> None:
    beta_eff = 0.25 * _beta_scale_for_horizon(days_to_event=1, lag_min=7)
    assert beta_eff <= 1e-3
