"""Helpers for writing provenance sidecar metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_meta(path: Path, payload: Mapping[str, Any]) -> Path:
    """Persist provenance payload to JSON, ensuring parent directories exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    return path
