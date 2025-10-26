"""Bootstrap last_good snapshots from offline CSVs.

This helper promotes sample or user-supplied CSV files into the raw
`data_raw/last_good.*` snapshots that the ETL pipeline prefers over the
bundled samples. Use it to get the repo out of sample mode quickly or to
stage curated data pulls without enabling live HTTP requests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from kalshi_gas.config import load_config
from kalshi_gas.etl.utils import save_snapshot


DATASETS = {
    "aaa": {
        "filename": "last_good.aaa.csv",
        "default": Path("data/sample/aaa_daily.csv"),
        "parse_dates": ["date"],
    },
    "rbob": {
        "filename": "last_good.rbob.csv",
        "default": Path("data/sample/rbob_futures.csv"),
        "parse_dates": ["date"],
    },
    "eia": {
        "filename": "last_good.eia.csv",
        "default": Path("data/sample/eia_weekly.csv"),
        "parse_dates": ["date"],
    },
    "kalshi": {
        "filename": "last_good.kalshi.csv",
        "default": Path("data/sample/kalshi_markets.csv"),
        "parse_dates": ["date"],
    },
}


def _as_of(frame: pd.DataFrame, column: str = "date") -> str | None:
    if column not in frame.columns:
        return None
    series = pd.to_datetime(frame[column], errors="coerce")
    if series.notna().any():
        latest = series.max()
        if pd.notna(latest):
            return pd.Timestamp(latest).date().isoformat()
    return None


def _load_frame(path: Path, parse_dates: Iterable[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=list(parse_dates))
    frame.sort_values(list(parse_dates), inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def _bootstrap_dataset(
    name: str,
    source: Path,
    target: Path,
    *,
    parse_dates: Iterable[str],
    overwrite: bool,
) -> str | None:
    if target.exists() and not overwrite:
        return None
    frame = _load_frame(source, parse_dates)
    as_of = _as_of(frame)
    metadata = {
        "source": name,
        "mode": "bootstrap",
        "as_of": as_of,
        "seed_path": str(source),
    }
    save_snapshot(frame, target, metadata=metadata)
    return as_of


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote offline CSVs into last_good snapshots."
    )
    parser.add_argument(
        "--aaa",
        type=Path,
        help="CSV path with AAA national average daily prices.",
    )
    parser.add_argument(
        "--rbob",
        type=Path,
        help="CSV path with RBOB futures settlements.",
    )
    parser.add_argument(
        "--eia",
        type=Path,
        help="CSV path with EIA weekly inventory data.",
    )
    parser.add_argument(
        "--kalshi",
        type=Path,
        help="CSV path with Kalshi market probabilities.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing last_good snapshots.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    overrides = {
        key: getattr(args, key)
        for key in ("aaa", "rbob", "eia", "kalshi")
        if getattr(args, key) is not None
    }

    cfg = load_config()
    raw_dir = cfg.data.raw_dir

    for name, cfg_ds in DATASETS.items():
        target = raw_dir / cfg_ds["filename"]
        source = Path(overrides.get(name) or cfg_ds["default"])
        if not source.exists():
            parser.error(f"Source file for {name} not found: {source}")
        as_of = _bootstrap_dataset(
            name=name,
            source=source,
            target=target,
            parse_dates=cfg_ds["parse_dates"],
            overwrite=args.overwrite,
        )
        if as_of:
            print(f"Bootstrapped {target} from {source} (as_of={as_of})")
        else:
            print(f"Skipped {target} (exists; use --overwrite to refresh)")


if __name__ == "__main__":
    main()
