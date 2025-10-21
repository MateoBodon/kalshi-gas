"""Kalshi market probability ETL."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from kalshi_gas.config import PipelineConfig
from kalshi_gas.etl.base import DataProvenance, ETLTask, ExtractorResult
from kalshi_gas.etl.utils import (
    CSVLoader,
    freshness_age_hours,
    infer_as_of,
    load_snapshot,
    read_csv_with_date,
    read_env,
    save_snapshot,
    snapshot_is_fresh,
    use_live_data,
    utcnow_iso,
)

KALSHI_BASE_URL = "https://trading-api.kalshi.com/v1"

log = logging.getLogger(__name__)


class KalshiExtractor:
    def __init__(
        self,
        snapshot_path: Path,
        sample_path: Path,
        market_prefix: str = "GAS_PRICE",
        freshness_hours: int = 24,
    ):
        self.snapshot_path = snapshot_path
        self.sample_path = sample_path
        self.market_prefix = market_prefix
        self.freshness = timedelta(hours=freshness_hours)
        self.provenance: DataProvenance | None = None

    def _fetch_live(self) -> pd.DataFrame:
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
            prob_yes = market.get("probability_yes")
            listed_date = market.get("listed_date")
            if prob_yes is None or listed_date is None:
                continue
            date = pd.to_datetime(listed_date)
            records.append(
                {
                    "date": date,
                    "market": ticker,
                    "prob_yes": float(prob_yes) / 100.0,
                }
            )
        if not records:
            raise ValueError("No matching Kalshi markets retrieved")
        return pd.DataFrame(records)

    def extract(self) -> ExtractorResult:
        fallback_chain: list[str] = []
        now = datetime.now(timezone.utc)

        if use_live_data():
            try:
                frame = self._fetch_live()
                as_of = infer_as_of(frame, ("date",))
                fetched_at = utcnow_iso()
                metadata = {
                    "source": "kalshi",
                    "mode": "live",
                    "market_prefix": self.market_prefix,
                    "as_of": as_of,
                    "fetched_at": fetched_at,
                }
                save_snapshot(frame, self.snapshot_path, metadata=metadata)
                self.provenance = DataProvenance(
                    source="kalshi",
                    mode="live",
                    path=self.snapshot_path,
                    fetched_at=fetched_at,
                    as_of=as_of,
                    fresh=True,
                    records=int(len(frame)),
                    details={
                        "market_prefix": self.market_prefix,
                        "snapshot_path": str(self.snapshot_path),
                    },
                    fallback_chain=fallback_chain,
                )
                return ExtractorResult(frame=frame, provenance=self.provenance)
            except Exception as exc:  # noqa: BLE001
                log.warning("Kalshi live fetch failed, using backups: %s", exc)
                fallback_chain.append(f"live_error:{exc.__class__.__name__}")
        else:
            fallback_chain.append("live_disabled")

        snapshot_meta = None
        if self.snapshot_path.exists():
            try:
                frame, snapshot_meta = load_snapshot(
                    self.snapshot_path,
                    parse_dates=["date"],
                )
                as_of = infer_as_of(frame, ("date",))
                is_fresh = snapshot_is_fresh(snapshot_meta, self.freshness, now=now)
                age_hours = freshness_age_hours(snapshot_meta, now=now)
                if is_fresh:
                    fetched_at = (
                        snapshot_meta.get("fetched_at") if snapshot_meta else None
                    )
                    provenance_as_of = as_of
                    if provenance_as_of is None and snapshot_meta:
                        provenance_as_of = snapshot_meta.get("as_of")
                    self.provenance = DataProvenance(
                        source="kalshi",
                        mode="snapshot",
                        path=self.snapshot_path,
                        fetched_at=fetched_at,
                        as_of=provenance_as_of,
                        fresh=True,
                        records=int(len(frame)),
                        details={
                            "market_prefix": self.market_prefix,
                            "age_hours": age_hours,
                            "snapshot_path": str(self.snapshot_path),
                        },
                        fallback_chain=fallback_chain,
                    )
                    return ExtractorResult(frame=frame, provenance=self.provenance)
                fallback_chain.append("snapshot_stale")
            except FileNotFoundError:
                fallback_chain.append("snapshot_missing")
            except Exception as exc:  # noqa: BLE001
                log.warning("Kalshi snapshot load failed: %s", exc)
                fallback_chain.append("snapshot_error")

        sample_frame = read_csv_with_date(self.sample_path, parse_dates=["date"])
        as_of = infer_as_of(sample_frame, ("date",))
        age_hours = (
            freshness_age_hours(snapshot_meta, now=now) if snapshot_meta else None
        )
        self.provenance = DataProvenance(
            source="kalshi",
            mode="sample",
            path=self.sample_path,
            fetched_at=None,
            as_of=as_of,
            fresh=False,
            records=int(len(sample_frame)),
            details={
                "market_prefix": self.market_prefix,
                "snapshot_path": str(self.snapshot_path),
                "snapshot_age_hours": age_hours,
            },
            fallback_chain=fallback_chain,
        )
        return ExtractorResult(frame=sample_frame, provenance=self.provenance)


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
    sample_path = Path("data/sample/kalshi_markets.csv")
    snapshot_path = config.data.raw_dir / "kalshi_markets_snapshot.csv"
    extractor = KalshiExtractor(snapshot_path=snapshot_path, sample_path=sample_path)
    transformer = KalshiTransformer()
    loader = CSVLoader(output_path=output_path)
    return ETLTask(extractor=extractor, transformer=transformer, loader=loader)
