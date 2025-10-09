"""AAA daily gasoline price ETL."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kalshi_gas.config import PipelineConfig
from kalshi_gas.etl.base import ETLTask
from kalshi_gas.etl.utils import CSVLoader, read_csv_with_date, safe_request, use_live_data

AAA_DAILY_AVG_URL = "https://gasprices.aaa.com/wp-json/aaa-api/v1/daily-national-average"


class AAAExtractor:
    def __init__(self, fallback_path: Path):
        self.fallback_path = fallback_path

    def extract(self) -> pd.DataFrame:
        def _remote() -> pd.DataFrame:
            if not use_live_data():
                raise RuntimeError("Live data disabled")
            from kalshi_gas.etl.utils import fetch_json

            data = fetch_json(AAA_DAILY_AVG_URL)
            frame = pd.DataFrame([data])
            frame["date"] = pd.to_datetime(frame["date"])
            frame.rename(columns={"price": "regular_gas_price"}, inplace=True)
            return frame[["date", "regular_gas_price"]]

        def _fallback() -> pd.DataFrame:
            return read_csv_with_date(self.fallback_path, parse_dates=["date"])

        return safe_request(_remote, _fallback, "AAA")


class AAATransformer:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["regular_gas_price"] = frame["regular_gas_price"].astype(float)
        frame = frame.dropna(subset=["date", "regular_gas_price"])
        frame = frame.sort_values("date")
        frame["regular_gas_price"] = frame["regular_gas_price"].round(3)
        return frame.reset_index(drop=True)


def build_aaa_etl(config: PipelineConfig) -> ETLTask:
    output_path = config.data.processed_dir / "aaa_daily.csv"
    fallback_path = Path("data/sample/aaa_daily.csv")
    extractor = AAAExtractor(fallback_path=fallback_path)
    transformer = AAATransformer()
    loader = CSVLoader(output_path=output_path)
    return ETLTask(extractor=extractor, transformer=transformer, loader=loader)
