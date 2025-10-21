"""Base ETL abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Protocol

import pandas as pd


@dataclass
class DataProvenance:
    """Capture provenance metadata for a data extraction."""

    source: str
    mode: str
    path: Path | None = None
    fetched_at: str | None = None
    as_of: str | None = None
    fresh: bool = True
    records: int | None = None
    details: Dict[str, Any] = field(default_factory=dict)
    fallback_chain: list[str] = field(default_factory=list)

    def serialize(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "mode": self.mode,
            "path": str(self.path) if self.path else None,
            "fetched_at": self.fetched_at,
            "as_of": self.as_of,
            "fresh": self.fresh,
            "records": self.records,
            "details": self.details,
            "fallback_chain": list(self.fallback_chain),
        }


@dataclass
class ExtractorResult:
    """Bundle frame output with provenance metadata."""

    frame: pd.DataFrame
    provenance: DataProvenance | None = None


@dataclass
class ETLRunResult:
    """Return value for ETL tasks including saved path and provenance."""

    output_path: Path
    provenance: DataProvenance | None = None


class Extractor(Protocol):
    def extract(self) -> pd.DataFrame | ExtractorResult: ...


class Transformer(Protocol):
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame: ...


class Loader(Protocol):
    def load(self, frame: pd.DataFrame) -> Path: ...


@dataclass
class ETLTask:
    extractor: Extractor
    transformer: Transformer
    loader: Loader

    def run(self) -> ETLRunResult:
        """Execute ETL flow."""
        raw = self.extractor.extract()
        provenance = None
        if isinstance(raw, ExtractorResult):
            frame = raw.frame
            provenance = raw.provenance
        else:
            frame = raw
            provenance = getattr(self.extractor, "provenance", None)
        transformed = self.transformer.transform(frame)
        output = self.loader.load(transformed)
        return ETLRunResult(output_path=output, provenance=provenance)
