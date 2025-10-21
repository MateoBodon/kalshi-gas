"""Utility classes shared across ETL tasks."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence

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


def utcnow_iso() -> str:
    """Return current UTC timestamp in ISO format without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class SourceSelection:
    """Describe chosen data source for ETL."""

    mode: str
    path: Path | None


def get_source(
    name: str,
    *,
    sample_path: Path,
    suffix: str | None = None,
    allow_live: bool = True,
) -> SourceSelection:
    """Return source selection respecting env-controlled live fallbacks."""
    suffix = suffix or sample_path.suffix or ".csv"
    last_good_path = Path("data_raw") / f"last_good.{name}{suffix}"

    if allow_live and use_live_data():
        return SourceSelection(mode="live", path=None)

    if last_good_path.exists():
        return SourceSelection(mode="last_good", path=last_good_path)

    return SourceSelection(mode="sample", path=sample_path)


def snapshot_meta_path(snapshot_path: Path) -> Path:
    return snapshot_path.parent / f"{snapshot_path.name}.meta.json"


def save_snapshot(
    frame: pd.DataFrame,
    snapshot_path: Path,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Persist raw snapshot alongside provenance metadata."""
    ensure_directory(snapshot_path)
    frame.to_csv(snapshot_path, index=False)
    meta = dict(metadata or {})
    meta.setdefault("fetched_at", utcnow_iso())
    meta["records"] = int(len(frame))
    meta["columns"] = list(frame.columns)
    meta_path = snapshot_meta_path(snapshot_path)
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def load_snapshot(
    snapshot_path: Path,
    parse_dates: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any] | None]:
    """Load snapshot frame and metadata."""
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found at {snapshot_path}")
    frame = pd.read_csv(
        snapshot_path, parse_dates=list(parse_dates) if parse_dates else None
    )
    meta_path = snapshot_meta_path(snapshot_path)
    metadata: Dict[str, Any] | None = None
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return frame, metadata


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp):
        return parsed.to_pydatetime()
    return datetime.fromisoformat(str(parsed))


def snapshot_is_fresh(
    metadata: Dict[str, Any] | None,
    freshness: timedelta,
    now: datetime | None = None,
) -> bool:
    if metadata is None:
        return False
    fetched_at = parse_timestamp(metadata.get("fetched_at"))
    if fetched_at is None:
        return False
    now = now or datetime.now(timezone.utc)
    return now - fetched_at <= freshness


def infer_as_of(
    frame: pd.DataFrame,
    date_columns: Sequence[str] = ("date",),
) -> str | None:
    for column in date_columns:
        if column not in frame.columns:
            continue
        series = pd.to_datetime(frame[column], errors="coerce", utc=True)
        if series.notna().any():
            latest = series.max()
            if pd.isna(latest):
                continue
            timestamp = pd.Timestamp(latest).to_pydatetime().astimezone(timezone.utc)
            return timestamp.date().isoformat()
    return None


def freshness_age_hours(
    metadata: Dict[str, Any] | None,
    now: datetime | None = None,
) -> float | None:
    if metadata is None:
        return None
    fetched_at = parse_timestamp(metadata.get("fetched_at"))
    if fetched_at is None:
        return None
    now = now or datetime.now(timezone.utc)
    delta = now - fetched_at
    return delta.total_seconds() / 3600.0
