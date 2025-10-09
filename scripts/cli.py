"""Utility CLI wrapper for dev scaffolding tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import Timestamp

from kalshi_gas.cli import run_report
from kalshi_gas.config import load_config
from kalshi_gas.data.assemble import assemble_dataset


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    return obj


def init_raw_templates() -> None:
    """Run report build and surface artifacts in standard locations."""
    run_report()

    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    data_proc_dir = Path("data_proc")

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_proc_dir.mkdir(parents=True, exist_ok=True)

    memo_src = Path("build/memo/report.md")
    if memo_src.exists():
        memo_dst = reports_dir / "memo.md"
        memo_dst.write_text(memo_src.read_text(encoding="utf-8"), encoding="utf-8")

    forecast_src = Path("build/figures/forecast_vs_actual.png")
    if forecast_src.exists():
        forecast_dst = figures_dir / "posterior.png"
        forecast_dst.write_bytes(forecast_src.read_bytes())

    cfg = load_config()
    dataset = assemble_dataset(cfg)
    summary = dataset.describe(include="all").reset_index().to_dict(orient="records")
    summary = [
        {key: _to_serializable(value) for key, value in row.items()} for row in summary
    ]
    summary_path = data_proc_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Artifacts staged in reports/ and data_proc/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Developer scaffolding CLI.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "init-raw-templates", help="Prepare report artifacts and summaries."
    )

    args = parser.parse_args()
    if args.command == "init-raw-templates":
        init_raw_templates()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
