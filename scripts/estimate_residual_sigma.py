"""Estimate residual sigma from AAA daily change history."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from kalshi_gas.config import load_config


@dataclass
class SigmaEstimate:
    sigma: float
    dataset_as_of: str | None
    window_months: int
    sample_size: int


def _load_aaa_series(config_path: str | None) -> pd.DataFrame:
    cfg = load_config(config_path)
    aaa_path = cfg.data.processed_dir / "aaa_daily.csv"
    if not aaa_path.exists():
        raise FileNotFoundError(f"Processed AAA series missing: {aaa_path}")
    frame = pd.read_csv(aaa_path, parse_dates=["date"])
    frame.sort_values("date", inplace=True)
    return frame.reset_index(drop=True)


def estimate_sigma(
    config_path: str | None = None,
    *,
    months: int = 24,
) -> SigmaEstimate:
    series = _load_aaa_series(config_path)
    if series.empty:
        raise RuntimeError("AAA series empty — cannot estimate sigma")

    latest = series["date"].dropna().max()
    if pd.isna(latest):
        raise RuntimeError("AAA series lacks valid dates")
    cutoff = pd.Timestamp(latest) - pd.DateOffset(months=months)
    recent = series[series["date"] >= cutoff].copy()
    if recent.empty:
        recent = series.copy()
    recent["delta"] = recent["regular_gas_price"].diff()
    deltas = recent["delta"].dropna()
    if deltas.empty:
        raise RuntimeError("Not enough observations to estimate sigma")

    sigma = float(deltas.std(ddof=1))
    if not pd.notna(sigma) or sigma <= 0:
        sigma = float(deltas.abs().mean())
    if sigma <= 0:
        raise RuntimeError("Residual sigma estimation failed")

    dataset_as_of = pd.Timestamp(latest).normalize().date().isoformat()
    return SigmaEstimate(
        sigma=sigma,
        dataset_as_of=dataset_as_of,
        window_months=months,
        sample_size=int(len(deltas)),
    )


def write_output(estimate: SigmaEstimate, output_dir: Path | None = None) -> Path:
    target_dir = output_dir or Path("data_proc")
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sigma": estimate.sigma,
        "dataset_as_of": estimate.dataset_as_of,
        "window_months": estimate.window_months,
        "samples": estimate.sample_size,
        "generated_at": date.today().isoformat(),
    }
    output_path = target_dir / "residual_sigma.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate AAA residual sigma")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--months", type=int, default=24)
    args = parser.parse_args()
    estimate = estimate_sigma(config_path=args.config, months=args.months)
    path = write_output(estimate)
    print(json.dumps({"sigma": estimate.sigma, "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
