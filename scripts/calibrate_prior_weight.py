"""Calibrate prior weight for short horizons using Brier minimisation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from kalshi_gas.backtest.evaluate import compute_event_probabilities
from kalshi_gas.backtest.metrics import brier_score
from kalshi_gas.config import load_config
from kalshi_gas.data.assemble import assemble_dataset
from kalshi_gas.etl.pipeline import run_all_etl
from kalshi_gas.utils.dataset import frame_digest
from kalshi_gas.utils.thresholds import load_kalshi_thresholds
from kalshi_gas.utils.sigma import load_residual_sigma

WEIGHT_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
TARGET_HORIZONS: tuple[int, ...] = (1, 2)


@dataclass
class CalibrationResult:
    best_weight: float
    dataset_digest: str | None
    dataset_as_of: str | None
    dataset_rows: int
    event_threshold: float
    horizons: tuple[int, ...]


def _prepare_dataset(config_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    run_all_etl(cfg)
    dataset = assemble_dataset(cfg)
    if dataset.empty:
        raise RuntimeError("assembled dataset empty")
    return dataset


def _horizon_mask(frame: pd.DataFrame, event_date: pd.Timestamp) -> pd.Series:
    day_counts = (event_date - pd.to_datetime(frame["date"]).dt.normalize()).dt.days
    return day_counts.isin(TARGET_HORIZONS)


def _compute_brier_by_weight(
    weights: Iterable[float],
    likelihood: np.ndarray,
    prior_probs: np.ndarray,
    outcomes: np.ndarray,
) -> dict[float, float]:
    scores: dict[float, float] = {}
    for weight in weights:
        mixture = (1 - weight) * likelihood + weight * prior_probs
        score = brier_score(mixture, outcomes)
        scores[float(weight)] = float(score)
    return scores


def calibrate_prior_weight(config_path: str | None = None) -> CalibrationResult:
    dataset = _prepare_dataset(config_path)
    cfg = load_config(config_path)
    threshold_bundle = load_kalshi_thresholds(Path("data_raw/kalshi_bins.yml"))
    event_threshold = float(cfg.event.threshold or threshold_bundle.central_threshold)
    event_date = pd.Timestamp(cfg.event.resolution_date)

    mask = _horizon_mask(dataset, event_date)
    subset = dataset.loc[mask].copy()
    if subset.empty:
        raise RuntimeError("No observations within target horizons (1-2 days)")

    fallback_sigma = float(subset["regular_gas_price"].diff().std(ddof=1))
    if not np.isfinite(fallback_sigma) or fallback_sigma <= 0:
        fallback_sigma = 0.01
    sigma, _meta = load_residual_sigma(fallback=fallback_sigma)

    subset_preds = subset["regular_gas_price"].astype(float)
    likelihood = compute_event_probabilities(
        subset_preds,
        sigma=sigma,
        threshold=event_threshold,
    )
    likelihood = np.asarray(likelihood, dtype=float)

    prior_probs = (
        subset["kalshi_prob"].astype(float).ffill().bfill().to_numpy(dtype=float)
    )
    outcomes = (
        (subset["target_future_price"] >= event_threshold).astype(float).to_numpy()
    )

    scores = _compute_brier_by_weight(WEIGHT_GRID, likelihood, prior_probs, outcomes)
    best_weight = min(scores, key=lambda w: (scores[w], w))
    best_weight = min(best_weight, 0.10)

    digest_columns = [
        col
        for col in (
            "date",
            "regular_gas_price",
            "rbob_settle",
            "kalshi_prob",
            "target_future_price",
        )
        if col in dataset.columns
    ]
    dataset_digest = frame_digest(dataset, columns=digest_columns)
    dataset_as_of = None
    if "date" in dataset.columns and not dataset["date"].empty:
        latest = dataset["date"].dropna().max()
        if pd.notna(latest):
            dataset_as_of = pd.Timestamp(latest).normalize().date().isoformat()

    return CalibrationResult(
        best_weight=best_weight,
        dataset_digest=dataset_digest,
        dataset_as_of=dataset_as_of,
        dataset_rows=int(len(dataset)),
        event_threshold=event_threshold,
        horizons=TARGET_HORIZONS,
    )


def write_output(result: CalibrationResult, output_dir: Path | None = None) -> Path:
    target_dir = output_dir or Path("data_proc")
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_weight": result.best_weight,
        "event_threshold": result.event_threshold,
        "dataset_digest": result.dataset_digest,
        "dataset_as_of": result.dataset_as_of,
        "dataset_rows": result.dataset_rows,
        "horizons": list(result.horizons),
        "generated_at": date.today().isoformat(),
    }
    output_path = target_dir / "prior_weight.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate prior weight for T+1")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    result = calibrate_prior_weight(args.config)
    path = write_output(result)
    print(json.dumps({"best_weight": result.best_weight, "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
