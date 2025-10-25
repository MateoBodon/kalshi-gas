"""Grid sweep of ensemble weights optimizing log-score and CRPS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from kalshi_gas.config import load_config
from kalshi_gas.data.assemble import assemble_dataset
from kalshi_gas.etl.pipeline import run_all_etl
from kalshi_gas.models.ensemble import EnsembleModel
from kalshi_gas.backtest.evaluate import run_backtest
from kalshi_gas.utils.thresholds import load_kalshi_thresholds


@dataclass
class SweepResult:
    weights: Dict[str, float]
    log_score: float
    crps: float
    brier: float


def _grid(step: float = 0.1) -> Iterable[Tuple[float, float, float]]:
    values = [round(x, 10) for x in np.arange(0.0, 1.0 + 1e-9, step)]
    for w1, w2 in product(values, values):
        w3 = 1.0 - (w1 + w2)
        if w3 < -1e-9:
            continue
        w3 = max(0.0, w3)
        # Normalize to sum to 1.0 in case of rounding
        total = w1 + w2 + w3
        if total <= 0:
            continue
        yield (w1 / total, w2 / total, w3 / total)


def _log_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    outs = outcomes.astype(int)
    return float(np.mean(outs * np.log(probs) + (1 - outs) * np.log(1 - probs)))


def sweep(step: float = 0.1) -> Tuple[pd.DataFrame, SweepResult, SweepResult]:
    cfg = load_config()
    run_all_etl(cfg)
    dataset = assemble_dataset(cfg)
    if dataset.empty:
        raise RuntimeError("Dataset empty for ensemble sweep")

    threshold_bundle = load_kalshi_thresholds(Path("data_raw/kalshi_bins.yml"))
    event_threshold = threshold_bundle.central_threshold

    records: List[Dict[str, float]] = []
    best_log: Tuple[float, Dict[str, float], float, float] | None = None
    best_crps: Tuple[float, Dict[str, float], float, float] | None = None

    for nowcast_w, pass_w, prior_w in _grid(step):
        weights = {
            "nowcast": float(nowcast_w),
            "pass_through": float(pass_w),
            "market_prior": float(prior_w),
        }
        ensemble = EnsembleModel(weights=weights)
        bt = run_backtest(dataset, ensemble, threshold=float(event_threshold))
        test = bt.test_frame
        probs = test["event_probability"].to_numpy(dtype=float)
        outs = test["event_outcome"].to_numpy(dtype=float)
        log_s = _log_score(probs, outs)
        crps = float(bt.metrics.get("crps", np.nan))
        brier = float(bt.metrics.get("brier_score", np.nan))

        records.append(
            {
                "nowcast": weights["nowcast"],
                "pass_through": weights["pass_through"],
                "market_prior": weights["market_prior"],
                "log_score": log_s,
                "crps": crps,
                "brier": brier,
            }
        )

        if best_log is None or log_s > best_log[0]:
            best_log = (log_s, weights, crps, brier)
        if best_crps is None or (np.isfinite(crps) and crps < best_crps[0]):
            best_crps = (crps, weights, log_s, brier)

    frame = pd.DataFrame(records)
    frame.sort_values(["log_score"], ascending=False, inplace=True)
    frame.reset_index(drop=True, inplace=True)

    best_log_res = SweepResult(
        weights=best_log[1], log_score=best_log[0], crps=best_log[2], brier=best_log[3]
    )
    best_crps_res = SweepResult(
        weights=best_crps[1],
        log_score=best_crps[2],
        crps=best_crps[0],
        brier=best_crps[3],
    )
    return frame, best_log_res, best_crps_res


def write_outputs(
    frame: pd.DataFrame, best_log: SweepResult, best_crps: SweepResult
) -> None:
    data_proc = Path("data_proc")
    data_proc.mkdir(parents=True, exist_ok=True)
    csv_path = data_proc / "ensemble_weight_sweep.csv"
    frame.to_csv(csv_path, index=False)

    meta = {
        "best_by_log_score": {
            "weights": best_log.weights,
            "log_score": best_log.log_score,
            "crps": best_log.crps,
            "brier": best_log.brier,
        },
        "best_by_crps": {
            "weights": best_crps.weights,
            "log_score": best_crps.log_score,
            "crps": best_crps.crps,
            "brier": best_crps.brier,
        },
    }
    (data_proc / "ensemble_weight_sweep.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def main() -> None:
    frame, best_log, best_crps = sweep(step=0.1)
    write_outputs(frame, best_log, best_crps)
    print(
        json.dumps(
            {
                "best_by_log_score": best_log.weights,
                "best_by_crps": best_crps.weights,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
