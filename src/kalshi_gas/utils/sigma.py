"""Helpers for loading residual volatility estimates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Tuple

DEFAULT_FALLBACK_SIGMA = 0.01


def load_residual_sigma(
    path: str | Path = Path("data_proc/residual_sigma.json"),
    *,
    fallback: float | None = None,
) -> Tuple[float, dict]:
    """Return residual sigma (USD/gal) and metadata, preferring on-disk estimate."""

    sigma = fallback
    meta: dict = {}
    sigma_path = Path(path)
    if sigma_path.exists():
        try:
            with sigma_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                meta = data
                candidate = data.get("sigma")
                if isinstance(candidate, (int, float)):
                    sigma = float(candidate)
        except json.JSONDecodeError:
            meta = {}

    if sigma is None or not math.isfinite(sigma) or sigma <= 0:
        sigma = DEFAULT_FALLBACK_SIGMA

    return sigma, meta


__all__ = ["load_residual_sigma", "DEFAULT_FALLBACK_SIGMA"]
