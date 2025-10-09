"""Kalshi market probability ETL."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from kalshi_gas.config import PipelineConfig
from kalshi_gas.etl.base import ETLTask
from kalshi_gas.etl.utils import (
    CSVLoader,
    read_csv_with_date,
    read_env,
    safe_request,
    use_live_data,
)

KALSHI_BASE_URL = "https://trading-api.kalshi.com/v1"


class KalshiExtractor:
    def __init__(self, fallback_path: Path, market_prefix: str = "GAS_PRICE"):
        self.fallback_path = fallback_path
        self.market_prefix = market_prefix

    def extract(self) -> pd.DataFrame:
        def _remote() -> pd.DataFrame:
            if not use_live_data():
                raise RuntimeError("Live data disabled")
            email = read_env("KALSHI_EMAIL")
            password = read_env("KALSHI_PASSWORD")
            if not email or not password:
                raise RuntimeError("Kalshi credentials missing")

            session = requests.Session()
            login_resp = session.post(
                f"{KALSHI_BASE_URL}/auth/login",
                json={"email": email, "password": password},
                timeout=30,
            )
            login_resp.raise_for_status()
            token = login_resp.json()["token"]
            session.headers.update({"Authorization": f"Bearer {token}"})

            events_resp = session.get(
                f"{KALSHI_BASE_URL}/markets?category=energy",
                timeout=30,
            )
            events_resp.raise_for_status()
            payload = events_resp.json()
            markets = payload.get("markets", [])
            records = []
            for market in markets:
                ticker = market.get("ticker", "")
                if self.market_prefix not in ticker:
                    continue
                prob_yes = market.get("probability_yes", None)
                if prob_yes is None:
                    continue
                date = pd.to_datetime(market.get("listed_date"))
                records.append({"date": date, "market": ticker, "prob_yes": prob_yes / 100})
            if not records:
                raise ValueError("No matching Kalshi markets retrieved")
            return pd.DataFrame(records)

        def _fallback() -> pd.DataFrame:
            return read_csv_with_date(self.fallback_path, parse_dates=["date"])

        return safe_request(_remote, _fallback, "Kalshi")


class KalshiTransformer:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["prob_yes"] = frame["prob_yes"].astype(float)
        frame = frame.dropna(subset=["date", "market", "prob_yes"])
        frame.sort_values(["market", "date"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        return frame


def build_kalshi_etl(config: PipelineConfig) -> ETLTask:
    output_path = config.data.processed_dir / "kalshi_markets.csv"
    fallback_path = Path("data/sample/kalshi_markets.csv")
    extractor = KalshiExtractor(fallback_path=fallback_path)
    transformer = KalshiTransformer()
    loader = CSVLoader(output_path=output_path)
    return ETLTask(extractor=extractor, transformer=transformer, loader=loader)
