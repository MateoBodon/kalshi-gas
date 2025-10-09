"""Base ETL abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd


class Extractor(Protocol):
    def extract(self) -> pd.DataFrame:
        ...


class Transformer(Protocol):
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        ...


class Loader(Protocol):
    def load(self, frame: pd.DataFrame) -> Path:
        ...


@dataclass
class ETLTask:
    extractor: Extractor
    transformer: Transformer
    loader: Loader

    def run(self) -> Path:
        """Execute ETL flow."""
        raw = self.extractor.extract()
        transformed = self.transformer.transform(raw)
        return self.loader.load(transformed)
