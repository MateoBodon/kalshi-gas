"""Fetch CME RBOB front/second settlements from Nasdaq Data Link (CHRIS).

Usage:

    DATA_LINK_API_KEY=... python3 scripts/rbob_fetch_chris.py \
        --start-date 2018-01-01 \
        --output data_raw/rbob_chris.csv

After the file is created, load it into the pipeline with::

    python3 scripts/update_rbob_csv.py --from-csv data_raw/rbob_chris.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import pandas as pd
import requests

CHRIS_BASE: Final[str] = "https://data.nasdaq.com/api/v3/datasets"
RB_DATASETS: Final[dict[str, str]] = {
    "rb1": "CHRIS/CME_RB1",
    "rb2": "CHRIS/CME_RB2",
}


class DataLinkError(RuntimeError):
    """Raised when the Data Link API returns an error response."""


def _get_api_key() -> str:
    key = os.getenv("DATA_LINK_API_KEY")
    if not key:
        raise RuntimeError(
            "DATA_LINK_API_KEY not set. Export your Nasdaq Data Link API key before running."
        )
    return key


def _fetch_dataset(code: str, api_key: str, start_date: str | None) -> pd.DataFrame:
    params: dict[str, str] = {"api_key": api_key, "order": "asc"}
    if start_date:
        params["start_date"] = start_date
    url = f"{CHRIS_BASE}/{code}.json"
    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        raise DataLinkError(f"Data Link error {response.status_code}: {response.text}")
    payload = response.json().get("dataset")
    if not payload:
        raise DataLinkError("Unexpected response format from Data Link")
    data = payload.get("data", [])
    columns = payload.get("column_names", [])
    frame = pd.DataFrame(data, columns=columns)
    if frame.empty:
        raise DataLinkError(f"No data returned for {code}")
    frame.rename(columns={"Date": "date"}, inplace=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


@dataclass
class CombinedSeries:
    frame: pd.DataFrame

    @classmethod
    def from_frames(cls, rb1: pd.DataFrame, rb2: pd.DataFrame) -> "CombinedSeries":
        needed_col = "Settle"
        for name, frame in {"rb1": rb1, "rb2": rb2}.items():
            if needed_col not in frame.columns:
                raise DataLinkError(f"Column '{needed_col}' missing in {name} dataset")

        merged = (
            rb1[["date", "Settle"]]
            .rename(columns={"Settle": "rb1_settle"})
            .merge(
                rb2[["date", "Settle"]].rename(columns={"Settle": "rb2_settle"}),
                on="date",
                how="inner",
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        merged["rb_spread"] = merged["rb1_settle"] - merged["rb2_settle"]
        merged["d_rb1"] = merged["rb1_settle"].diff()
        merged["d_rb2"] = merged["rb2_settle"].diff()
        merged["d_spread"] = merged["rb_spread"].diff()
        return cls(frame=merged)

    def to_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        export = self.frame.copy()
        export.rename(
            columns={
                "rb1_settle": "settle",
                "rb2_settle": "second_month",
                "d_rb1": "settle_change",
                "d_rb2": "second_change",
                "rb_spread": "spread",
            },
            inplace=True,
        )
        # Persist only the columns the downstream CSV merger expects.
        columns = [
            "date",
            "settle",
            "second_month",
            "spread",
            "settle_change",
            "second_change",
        ]
        export = export[columns]
        export.to_csv(
            path,
            index=False,
            date_format="%Y-%m-%d",
            float_format="%.6f",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch CME RBOB settlements from Nasdaq Data Link (CHRIS)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Fetch data starting from this date (YYYY-MM-DD). Defaults to earliest available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_raw/rbob_chris.csv"),
        help="Destination CSV file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = _get_api_key()

    start_date = args.start_date
    if start_date:
        # Validate format eagerly for clearer error messages
        datetime.strptime(start_date, "%Y-%m-%d")

    rb1 = _fetch_dataset(RB_DATASETS["rb1"], api_key, start_date)
    rb2 = _fetch_dataset(RB_DATASETS["rb2"], api_key, start_date)

    combined = CombinedSeries.from_frames(rb1, rb2)
    combined.to_csv(args.output)
    print(f"Saved {len(combined.frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
