"""Sweep prior weight hyperparameter using log score selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kalshi_gas.config import load_config
from kalshi_gas.data.assemble import assemble_dataset
from kalshi_gas.etl.pipeline import run_all_etl
from kalshi_gas.models.ensemble import EnsembleModel
from kalshi_gas.backtest.evaluate import compute_event_probabilities
from kalshi_gas.utils.dataset import frame_digest
from kalshi_gas.utils.thresholds import load_kalshi_thresholds

SWEEP_VALUES = np.arange(0.0, 1.0001, 0.05)


def _log_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.clip(probabilities, 1e-6, 1 - 1e-6)
    outs = outcomes.astype(int)
    log_terms = outs * np.log(probs) + (1 - outs) * np.log(1 - probs)
    return float(np.mean(log_terms))


def _evaluate_weights(
    weights: Iterable[float],
    predictions: pd.Series,
    outcomes: pd.Series,
    prior_probs: pd.Series,
) -> pd.DataFrame:
    rows = []
    like = np.asarray(predictions, dtype=float)
    prior = np.asarray(prior_probs, dtype=float)
    outcome = np.asarray(outcomes, dtype=float)

    for weight in weights:
        posterior = (1 - weight) * like + weight * prior
        log_score = _log_score(posterior, outcome)
        rows.append(
            {
                "prior_weight": weight,
                "log_score": log_score,
            }
        )
    frame = pd.DataFrame(rows)
    frame.sort_values("prior_weight", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def sweep_prior_weights() -> (
    tuple[pd.DataFrame, float, float, float | None, str | None, str | None, int]
):
    cfg = load_config()
    run_all_etl(cfg)
    dataset = assemble_dataset(cfg)
    if dataset.empty:
        raise RuntimeError("Dataset empty")

    threshold_bundle = load_kalshi_thresholds(Path("data_raw/kalshi_bins.yml"))
    event_threshold = threshold_bundle.central_threshold
    event_probability = threshold_bundle.central_probability

    ensemble = EnsembleModel(weights=cfg.model.ensemble_weights)
    ensemble.fit(dataset)
    predictions = ensemble.predict(dataset)["ensemble"]

    sigma = getattr(ensemble, "residual_std", 0.1)
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = float(np.nanstd(dataset["target_future_price"] - predictions))
        if not np.isfinite(sigma) or sigma < 1e-6:
            sigma = 0.1

    likelihood = compute_event_probabilities(
        predictions,
        sigma=sigma,
        threshold=event_threshold,
    )
    outcomes = (dataset["target_future_price"] >= event_threshold).astype(int)
    prior_probs = dataset["kalshi_prob"].clip(0, 1)

    sweep = _evaluate_weights(SWEEP_VALUES, likelihood, outcomes, prior_probs)
    best_row = sweep.loc[sweep["log_score"].idxmax()]
    best_weight = float(best_row["prior_weight"])
    dataset_as_of = None
    if pd.api.types.is_datetime64_any_dtype(dataset["date"]):
        latest = dataset["date"].dropna().max()
        if pd.notna(latest):
            dataset_as_of = pd.Timestamp(latest).date().isoformat()
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
    dataset_fingerprint = frame_digest(dataset, columns=digest_columns)
    return (
        sweep,
        best_weight,
        event_threshold,
        event_probability,
        dataset_as_of,
        dataset_fingerprint,
        len(dataset),
    )


def write_outputs(
    sweep: pd.DataFrame,
    best_weight: float,
    event_threshold: float,
    event_probability: float | None,
    dataset_as_of: str | None,
    dataset_digest: str | None,
    dataset_rows: int,
) -> None:
    data_proc = Path("data_proc")
    data_proc.mkdir(parents=True, exist_ok=True)
    sweep_path = data_proc / "prior_weight.csv"
    sweep.to_csv(sweep_path, index=False)

    meta_path = data_proc / "prior_weight.json"
    payload = {
        "best_weight": best_weight,
        "generated_at": sweep_path.stat().st_mtime,
        "event_threshold": event_threshold,
        "dataset_as_of": dataset_as_of,
        "dataset_digest": dataset_digest,
        "dataset_rows": dataset_rows,
    }
    if event_probability is not None:
        payload["central_probability"] = event_probability
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    figures_dir = Path("build/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep["prior_weight"], sweep["log_score"], marker="o", color="#1f77b4")
    ax.axvline(
        best_weight, color="#d62728", linestyle="--", label=f"Best: {best_weight:.2f}"
    )
    ax.set_xlabel("Prior Weight")
    ax.set_ylabel("Mean Log Score")
    ax.set_title("Prior Weight Calibration Sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "prior_weight_sweep.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    (
        sweep,
        best_weight,
        event_threshold,
        event_probability,
        dataset_as_of,
        dataset_digest,
        dataset_rows,
    ) = sweep_prior_weights()
    write_outputs(
        sweep,
        best_weight,
        event_threshold,
        event_probability,
        dataset_as_of,
        dataset_digest,
        dataset_rows,
    )
    print(
        json.dumps(
            {
                "best_weight": best_weight,
                "event_threshold": event_threshold,
                "dataset_as_of": dataset_as_of,
                "dataset_rows": dataset_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
