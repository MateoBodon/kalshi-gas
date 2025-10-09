"""Append RBOB daily settlements to the offline CSV cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data_raw/rbob_daily.csv")
COLUMNS = ["date", "settle", "source"]


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, parse_dates=["date"])


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["settle"] = frame["settle"].astype(float)
    frame["source"] = frame["source"].astype(str)
    frame.sort_values("date", inplace=True)
    frame.drop_duplicates(subset="date", keep="last", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def append_row(path: Path, date: str, settle: float, source: str) -> pd.DataFrame:
    existing = load_existing(path)
    new_entry = pd.DataFrame(
        [{"date": pd.to_datetime(date), "settle": float(settle), "source": source}]
    )
    combined = pd.concat([existing, new_entry], ignore_index=True)
    normalized = normalize_frame(combined)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(path, index=False, date_format="%Y-%m-%d")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update RBOB settlement CSV cache.")
    parser.add_argument("--date", required=True, help="Settlement date (YYYY-MM-DD)")
    parser.add_argument(
        "--settle", required=True, type=float, help="Settlement price in USD/gal"
    )
    parser.add_argument("--source", required=True, help="Data provenance string")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="CSV path to update (default: data_raw/rbob_daily.csv)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    frame = append_row(args.path, args.date, args.settle, args.source)
    latest = frame.iloc[-1]
    print(
        f"Updated {args.path} with {latest['date'].date()} @ {latest['settle']:.4f} ({latest['source']})"
    )


if __name__ == "__main__":
    main()
