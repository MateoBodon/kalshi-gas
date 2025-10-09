"""Utility classes shared across ETL tasks."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import requests

log = logging.getLogger(__name__)


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class CSVLoader:
    """Persist a DataFrame to CSV."""

    output_path: Path
    index: bool = False

    def load(self, frame: pd.DataFrame) -> Path:
        ensure_directory(self.output_path)
        frame.to_csv(self.output_path, index=self.index)
        return self.output_path


def read_csv_with_date(path: Path, parse_dates: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing fallback dataset at {path}")
    return pd.read_csv(path, parse_dates=list(parse_dates))


def fetch_json(url: str, headers: dict[str, str] | None = None) -> dict:
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def read_env(name: str) -> str | None:
    value = os.getenv(name)
    if not value:
        log.debug("Environment variable %s missing", name)
    return value


def use_live_data() -> bool:
    return os.getenv("KALSHI_GAS_USE_LIVE", "0") == "1"


def safe_request(
    request_fn: Callable[[], pd.DataFrame],
    fallback_fn: Callable[[], pd.DataFrame],
    source_name: str,
) -> pd.DataFrame:
    try:
        frame = request_fn()
        if frame.empty:
            raise ValueError("Received empty frame")
        return frame
    except Exception as exc:  # noqa: BLE001
        log.warning("Falling back to local %s data due to %s", source_name, exc)
        return fallback_fn()
