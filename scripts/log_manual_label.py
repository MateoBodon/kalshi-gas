"""Append manual label observations (AAA/EIA/GasBuddy) to the audit log."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

LOG_PATH = Path("data_raw/manual_logs.csv")
COLUMNS = [
    "timestamp_iso",
    "date",
    "aaa_regular",
    "eia_regular",
    "gasbuddy_regular",
    "operator",
    "notes",
]


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(columns=COLUMNS)
        frame.to_csv(path, index=False)
        return frame
    return pd.read_csv(path)


def append_log(
    *,
    timestamp: datetime,
    aaa: float | None,
    eia: float | None,
    gasbuddy: float | None,
    operator: str,
    notes: str | None,
    path: Path = LOG_PATH,
) -> pd.DataFrame:
    frame = _load_existing(path)
    entry = {
        "timestamp_iso": timestamp.isoformat(timespec="seconds"),
        "date": timestamp.date().isoformat(),
        "aaa_regular": float(aaa) if aaa is not None else pd.NA,
        "eia_regular": float(eia) if eia is not None else pd.NA,
        "gasbuddy_regular": float(gasbuddy) if gasbuddy is not None else pd.NA,
        "operator": operator,
        "notes": notes or "",
    }
    frame = pd.concat([frame, pd.DataFrame([entry])], ignore_index=True)
    frame.to_csv(path, index=False)
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append a manual AAA/EIA/GasBuddy observation to the audit log."
    )
    parser.add_argument(
        "--time",
        type=str,
        help="ISO timestamp for the observation (default: current UTC time)",
    )
    parser.add_argument(
        "--aaa",
        type=float,
        help="AAA national regular gasoline price (USD/gal)",
    )
    parser.add_argument(
        "--eia",
        type=float,
        help="EIA 'Today in Energy' AAA retail price (USD/gal)",
    )
    parser.add_argument(
        "--gasbuddy",
        type=float,
        help="Optional GasBuddy Fuel Insights national price (USD/gal)",
    )
    parser.add_argument(
        "--operator",
        required=True,
        help="Initials or identifier of the person logging the value",
    )
    parser.add_argument(
        "--notes",
        help="Optional notes (data source URL, anomalies, etc.)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=LOG_PATH,
        help="Override manual log CSV path (default: data_raw/manual_logs.csv)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    timestamp = datetime.fromisoformat(args.time) if args.time else datetime.utcnow()

    frame = append_log(
        timestamp=timestamp,
        aaa=args.aaa,
        eia=args.eia,
        gasbuddy=args.gasbuddy,
        operator=args.operator,
        notes=args.notes,
        path=args.path,
    )
    latest = frame.iloc[-1]
    print(
        "Logged manual observation at",
        latest["timestamp_iso"],
        "AAA=",
        latest["aaa_regular"],
        "EIA=",
        latest["eia_regular"],
        "GasBuddy=",
        latest["gasbuddy_regular"],
    )


if __name__ == "__main__":
    main()
