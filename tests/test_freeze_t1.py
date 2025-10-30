"""Smoke checks for T+1 horizon discipline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kalshi_gas.models.posterior import PosteriorDistribution
from kalshi_gas.pipeline.run_all import _beta_scale_for_horizon
from kalshi_gas.utils.sigma import load_residual_sigma


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


def test_sigma_loader_prefers_file(tmp_path: Path) -> None:
    payload = {
        "sigma": 0.0123,
        "dataset_as_of": "2025-10-30",
        "window_months": 24,
    }
    target = tmp_path / "residual_sigma.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

    sigma, meta = load_residual_sigma(target, fallback=0.5)

    assert abs(sigma - 0.0123) < 1e-6
    assert meta.get("dataset_as_of") == "2025-10-30"
