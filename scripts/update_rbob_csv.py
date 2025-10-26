"""Append RBOB daily settlements to the offline CSV cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data_raw/rbob_daily.csv")
COLUMNS = [
    "date",
    "settle",
    "second_month",
    "spread",
    "settle_change",
    "second_change",
    "source",
]


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, parse_dates=["date"])


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "settle" in frame:
        frame["settle"] = pd.to_numeric(frame["settle"], errors="coerce")
    optional_cols = {
        "second_month": float,
        "spread": float,
        "settle_change": float,
        "second_change": float,
    }
    for col, dtype in optional_cols.items():
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = float("nan")
    frame["source"] = frame.get("source", "manual").astype(str)
    frame.sort_values("date", inplace=True)
    frame.drop_duplicates(subset="date", keep="last", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def append_row(
    path: Path,
    date: str,
    settle: float,
    source: str,
    *,
    second_month: float | None = None,
    spread: float | None = None,
    settle_change: float | None = None,
    second_change: float | None = None,
) -> pd.DataFrame:
    existing = load_existing(path)
    new_entry = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(date),
                "settle": float(settle),
                "second_month": float(second_month)
                if second_month is not None
                else pd.NA,
                "spread": float(spread) if spread is not None else pd.NA,
                "settle_change": float(settle_change)
                if settle_change is not None
                else pd.NA,
                "second_change": float(second_change)
                if second_change is not None
                else pd.NA,
                "source": source,
            }
        ]
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
        "--second-month",
        type=float,
        help="Optional second-month settlement in USD/gal",
    )
    parser.add_argument(
        "--spread",
        type=float,
        help="Optional front-second spread in USD/gal",
    )
    parser.add_argument(
        "--settle-change",
        type=float,
        help="Optional day-over-day change for front month (USD/gal)",
    )
    parser.add_argument(
        "--second-change",
        type=float,
        help="Optional day-over-day change for second month (USD/gal)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="CSV path to update (default: data_raw/rbob_daily.csv)",
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        help="Optional CSV file with historical settlements to merge "
        "(expects columns date, settle, optional source).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.from_csv:
        bulk = pd.read_csv(args.from_csv)
        if "date" not in bulk or "settle" not in bulk:
            parser.error(
                f"Bulk CSV missing required columns ['date','settle']: {args.from_csv}"
            )
        default_source = args.source or args.from_csv.stem
        for _, row in bulk.iterrows():
            source_value = row.get("source") if "source" in row else None
            if isinstance(source_value, str) and source_value.strip():
                source = source_value.strip()
            else:
                source = default_source
            append_row(
                args.path,
                str(row["date"]),
                float(row["settle"]),
                source,
                second_month=row.get("second_month"),
                spread=row.get("spread"),
                settle_change=row.get("settle_change"),
                second_change=row.get("second_change"),
            )
        frame = load_existing(args.path)
        latest = frame["date"].max().date().isoformat() if not frame.empty else "n/a"
        print(
            f"Merged {len(bulk)} rows from {args.from_csv} into {args.path} "
            f"(latest={latest})"
        )
    else:
        frame = append_row(
            args.path,
            args.date,
            args.settle,
            args.source,
            second_month=args.second_month,
            spread=args.spread,
            settle_change=args.settle_change,
            second_change=args.second_change,
        )
        latest = frame.iloc[-1]
        print(
            f"Updated {args.path} with {latest['date'].date()} @ {latest['settle']:.4f} ({latest['source']})"
        )


if __name__ == "__main__":
    main()
