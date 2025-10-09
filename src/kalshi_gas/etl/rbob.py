"""RBOB futures ETL."""

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

EIA_RBOB_SERIES = "PET.RBRTWD.W"  # Reformulated gasoline spot price, $/gallon


class RBOBExtractor:
    def __init__(self, fallback_path: Path, series_id: str = EIA_RBOB_SERIES):
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

            url = f"https://api.eia.gov/series/?api_key={api_key}&series_id={self.series_id}"
            payload = fetch_json(url)
            series_meta = payload["series"][0]
            frame = pd.DataFrame(series_meta["data"], columns=["period", "settle"])
            frame["date"] = pd.to_datetime(frame["period"])
            frame["settle"] = frame["settle"].astype(float)
            frame.drop(columns=["period"], inplace=True)
            return frame

        def _fallback() -> pd.DataFrame:
            return read_csv_with_date(self.fallback_path, parse_dates=["date"])

        return safe_request(_remote, _fallback, "RBOB")


class RBOBTransformer:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["settle"] = frame["settle"].astype(float)
        frame = frame.dropna(subset=["date", "settle"])
        frame.sort_values("date", inplace=True)
        frame["settle"] = frame["settle"].round(4)
        frame.rename(columns={"settle": "rbob_price"}, inplace=True)
        return frame.reset_index(drop=True)


def build_rbob_etl(config: PipelineConfig) -> ETLTask:
    output_path = config.data.processed_dir / "rbob_prices.csv"
    fallback_path = Path("data/sample/rbob_futures.csv")
    extractor = RBOBExtractor(fallback_path=fallback_path)
    transformer = RBOBTransformer()
    loader = CSVLoader(output_path=output_path)
    return ETLTask(extractor=extractor, transformer=transformer, loader=loader)
