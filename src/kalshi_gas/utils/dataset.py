"""Dataset utilities for reproducibility checks."""

from __future__ import annotations

import hashlib
from typing import Sequence

import pandas as pd


def _normalize_columns(
    frame: pd.DataFrame, columns: Sequence[str] | None
) -> pd.DataFrame:
    if columns is None:
        return frame
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"Missing columns for digest: {sorted(missing)}")
    return frame.loc[:, list(columns)]


def frame_digest(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    float_precision: int = 6,
) -> str:
    """Return a stable SHA256 digest for a dataframe.

    Args:
        frame: DataFrame to hash.
        columns: Optional subset of columns to include (preserves order).
        float_precision: Decimal precision for floating point serialization.
    """
    if frame.empty:
        return "empty"
    subset = _normalize_columns(frame, columns)
    csv_bytes = subset.to_csv(index=False, float_format=f"%.{float_precision}f").encode(
        "utf-8"
    )
    return hashlib.sha256(csv_bytes).hexdigest()


__all__ = ["frame_digest"]
