"""Jackknife stability check dropping one calendar month at a time."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_gas.backtest.evaluate import run_backtest
from kalshi_gas.config import load_config
from kalshi_gas.data.assemble import assemble_dataset
from kalshi_gas.etl.pipeline import run_all_etl
from kalshi_gas.models.ensemble import EnsembleModel


def _month_series(dataset: pd.DataFrame) -> pd.Series:
    return dataset["date"].dt.to_period("M")


def _run_backtest(dataset: pd.DataFrame, weights: dict[str, float]) -> float:
    ensemble = EnsembleModel(weights=weights)
    result = run_backtest(dataset, ensemble)
    return float(result.metrics.get("brier_score", np.nan))


def jackknife_month_drop(
    dataset: pd.DataFrame, weights: dict[str, float]
) -> dict[str, object]:
    dataset = dataset.copy()
    dataset.sort_values("date", inplace=True)
    baseline_brier = _run_backtest(dataset, weights)

    months = _month_series(dataset).unique()
    deltas: list[tuple[str, float]] = []

    for period in months:
        mask = _month_series(dataset) != period
        subset = dataset.loc[mask]
        if subset.empty:
            continue
        brier = _run_backtest(subset, weights)
        delta = brier - baseline_brier
        deltas.append((str(period), float(delta)))

    if not deltas:
        return {
            "baseline_brier": baseline_brier,
            "max_abs_delta_brier": 0.0,
            "worst_month": None,
        }

    worst_month, worst_delta = max(deltas, key=lambda item: abs(item[1]))
    return {
        "baseline_brier": baseline_brier,
        "max_abs_delta_brier": abs(worst_delta),
        "worst_month": worst_month,
    }


def main() -> None:
    cfg = load_config()
    run_all_etl(cfg)
    dataset = assemble_dataset(cfg)
    if dataset.empty:
        raise RuntimeError("Dataset empty for jackknife analysis")
    summary = jackknife_month_drop(dataset, cfg.model.ensemble_weights)
    data_proc = Path("data_proc")
    data_proc.mkdir(parents=True, exist_ok=True)
    output_path = data_proc / "jackknife.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
