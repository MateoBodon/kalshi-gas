"""EIA weekly petroleum status ETL."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kalshi_gas.config import PipelineConfig
from kalshi_gas.etl.base import ETLTask
from kalshi_gas.etl.utils import (
    CSVLoader,
    read_csv_with_date,
    read_env,
    safe_request,
    use_live_data,
)

EIA_BASE_URL = "https://api.eia.gov/series/"
EIA_INVENTORY_SERIES = "PET.WGTSTUS1.W"  # Finished motor gasoline stocks, thousand barrels


class EIAExtractor:
    def __init__(self, fallback_path: Path, series_id: str = EIA_INVENTORY_SERIES):
        self.fallback_path = fallback_path
        self.series_id = series_id

    def extract(self) -> pd.DataFrame:
        def _remote() -> pd.DataFrame:
            if not use_live_data():
                raise RuntimeError("Live data disabled")
            from kalshi_gas.etl.utils import fetch_json

            api_key = read_env("EIA_API_KEY")
            if not api_key:
                raise RuntimeError("EIA_API_KEY not configured")

            url = f"{EIA_BASE_URL}?api_key={api_key}&series_id={self.series_id}"
            payload = fetch_json(url)
            series_meta = payload["series"][0]
            frame = pd.DataFrame(series_meta["data"], columns=["period", "inventory"])
            frame["date"] = pd.to_datetime(frame["period"])
            frame["inventory_mmbbl"] = frame["inventory"].astype(float) / 1000.0
            frame.drop(columns=["period", "inventory"], inplace=True)
            frame.sort_values("date", inplace=True)
            frame.reset_index(drop=True, inplace=True)
            frame["production_mbd"] = pd.NA
            return frame

        def _fallback() -> pd.DataFrame:
            return read_csv_with_date(self.fallback_path, parse_dates=["date"])

        return safe_request(_remote, _fallback, "EIA")


class EIATransformer:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["inventory_mmbbl"] = frame["inventory_mmbbl"].astype(float)
        if "production_mbd" not in frame:
            frame["production_mbd"] = pd.NA
        frame["production_mbd"] = frame["production_mbd"].astype(float)
        frame = frame.dropna(subset=["date", "inventory_mmbbl"])
        frame.sort_values("date", inplace=True)
        frame["inventory_change"] = frame["inventory_mmbbl"].diff()
        return frame.reset_index(drop=True)


def build_eia_etl(config: PipelineConfig) -> ETLTask:
    output_path = config.data.processed_dir / "eia_weekly.csv"
    fallback_path = Path("data/sample/eia_weekly.csv")
    extractor = EIAExtractor(fallback_path=fallback_path)
    transformer = EIATransformer()
    loader = CSVLoader(output_path=output_path)
    return ETLTask(extractor=extractor, transformer=transformer, loader=loader)
