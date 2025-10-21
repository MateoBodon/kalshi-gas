#!/usr/bin/env python3
"""Verify data freshness for core feeds with offline fallbacks."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SOURCE_MAX_AGE = {
    "aaa": timedelta(hours=48),
    "rbob": timedelta(hours=48),
    "eia": timedelta(days=14),
    "kalshi": timedelta(hours=24),
}


def _load_meta(name: str) -> tuple[dict, Path] | tuple[None, Path]:
    path = Path("data_proc/meta") / f"{name}.json"
    if not path.exists():
        return None, path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle), path


def _parse_as_of(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        try:
            parsed_date = datetime.fromisoformat(f"{value}T00:00:00+00:00")
            return parsed_date
        except ValueError:
            return None


def main() -> int:
    now = datetime.now(timezone.utc)
    status = 0

    for source, max_age in SOURCE_MAX_AGE.items():
        meta, path = _load_meta(source)
        if meta is None:
            print(f"[ERROR] missing provenance for {source}: {path}", file=sys.stderr)
            status = 1
            continue

        mode = str(meta.get("mode", "")).lower()
        as_of_raw = meta.get("as_of")
        as_of_ts = _parse_as_of(as_of_raw)
        fresh_flag = bool(meta.get("fresh", True))

        if mode == "sample":
            print(
                f"[INFO] {source}: using bundled sample data (freshness check skipped)"
            )
            continue

        if as_of_ts is None:
            print(
                f"[ERROR] {source}: invalid or missing as_of in {path}", file=sys.stderr
            )
            status = 1
            continue

        age = now - as_of_ts
        stale = age > max_age or not fresh_flag
        age_hours = age.total_seconds() / 3600.0

        if stale:
            print(
                f"[ERROR] {source}: stale ({age_hours:.1f}h old, limit {max_age.total_seconds()/3600:.1f}h)",
                file=sys.stderr,
            )
            status = 1
        else:
            print(
                f"[OK] {source}: {age_hours:.1f}h old (limit {max_age.total_seconds()/3600:.1f}h)"
            )

    return status


if __name__ == "__main__":
    sys.exit(main())
