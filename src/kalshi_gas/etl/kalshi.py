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
    get_source,
    infer_as_of,
    load_snapshot,
    read_csv_with_date,
    read_env,
    save_snapshot,
    snapshot_is_fresh,
    utcnow_iso,
)

KALSHI_BASE_URL_V1 = "https://trading-api.kalshi.com/v1"
KALSHI_BASE_URL_V2_DEFAULT = "https://api.elections.kalshi.com"

log = logging.getLogger(__name__)


class KalshiExtractor:
    def __init__(
        self,
        last_good_path: Path,
        sample_path: Path,
        market_prefix: str = "GAS_PRICE",
        freshness_hours: int = 24,
    ):
        self.last_good_path = last_good_path
        self.sample_path = sample_path
        self.market_prefix = market_prefix
        self.freshness = timedelta(hours=freshness_hours)
        self.provenance: DataProvenance | None = None

    def _fetch_live_v1(self) -> pd.DataFrame:
        email = read_env("KALSHI_EMAIL")
        password = read_env("KALSHI_PASSWORD")
        if not email or not password:
            raise RuntimeError("Kalshi credentials missing (v1)")
        session = requests.Session()
        login_resp = session.post(
            f"{KALSHI_BASE_URL_V1}/auth/login",
            json={"email": email, "password": password},
            timeout=30,
        )
        login_resp.raise_for_status()
        token = login_resp.json()["token"]
        session.headers.update({"Authorization": f"Bearer {token}"})
        events_resp = session.get(
            f"{KALSHI_BASE_URL_V1}/markets?category=energy", timeout=30
        )
        events_resp.raise_for_status()
        payload = events_resp.json()
        return self._markets_to_frame(payload.get("markets", []))

    def _fetch_live_v2(self) -> pd.DataFrame:
        key_id = read_env("KALSHI_API_KEY_ID")
        pem_path = read_env("KALSHI_PRIVATE_KEY_PATH")
        api_base = read_env("KALSHI_API_BASE") or KALSHI_BASE_URL_V2_DEFAULT
        if not key_id or not pem_path:
            raise RuntimeError("Kalshi API key missing (v2)")
        # Local import to avoid hard dependency during tests
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        import base64
        import time

        def _load_private_key(path: str):
            with open(path, "rb") as f:
                return serialization.load_pem_private_key(f.read(), password=None)

        private_key = _load_private_key(pem_path)
        series_ticker = read_env("KALSHI_SERIES_TICKER")
        event_ticker = read_env("KALSHI_EVENT_TICKER")
        path = "/trade-api/v2/markets?limit=200"
        if series_ticker:
            path += f"&series_ticker={series_ticker}"
        if event_ticker:
            path += f"&event_ticker={event_ticker}"

        def _headers(include_query: bool = True, ts_ms: bool = True) -> dict:
            ts_val = int(time.time() * 1000) if ts_ms else int(time.time())
            to_sign = (
                str(ts_val) + "GET" + (path if include_query else path.split("?", 1)[0])
            )
            signature = private_key.sign(
                to_sign.encode("utf-8"),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            sig_b64 = base64.b64encode(signature).decode("utf-8")
            return {
                "KALSHI-ACCESS-KEY": key_id,
                "KALSHI-ACCESS-TIMESTAMP": str(ts_val),
                "KALSHI-ACCESS-SIGNATURE": sig_b64,
            }

        # Try a few header variants for compatibility
        for include_query in (True, False):
            for ts_ms in (True, False):
                url = f"{api_base}{path}"
                headers = _headers(include_query=include_query, ts_ms=ts_ms)
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 200:
                    payload = resp.json()
                    return self._markets_to_frame(payload.get("markets", []))
        raise RuntimeError("Kalshi v2 request failed after header variants")

    def _markets_to_frame(self, markets: list[dict]) -> pd.DataFrame:
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
            try:
                p = float(prob_yes) / 100.0
            except Exception:
                continue
            records.append({"date": date, "market": ticker, "prob_yes": p})
        if not records:
            raise ValueError("No matching Kalshi markets retrieved")
        return pd.DataFrame(records)

    def extract(self) -> ExtractorResult:
        fallback_chain: list[str] = []
        now = datetime.now(timezone.utc)

        selection = get_source(
            "kalshi",
            sample_path=self.sample_path,
            suffix=self.sample_path.suffix or ".csv",
        )
        if selection.mode == "live":
            try:
                # Prefer v2 RSA if available; else try v1 session login
                try:
                    frame = self._fetch_live_v2()
                except Exception:
                    frame = self._fetch_live_v1()
                as_of = infer_as_of(frame, ("date",))
                fetched_at = utcnow_iso()
                metadata = {
                    "source": "kalshi",
                    "mode": "live",
                    "market_prefix": self.market_prefix,
                    "as_of": as_of,
                    "fetched_at": fetched_at,
                }
                save_snapshot(frame, self.last_good_path, metadata=metadata)
                self.provenance = DataProvenance(
                    source="kalshi",
                    mode="live",
                    path=self.last_good_path,
                    fetched_at=fetched_at,
                    as_of=as_of,
                    fresh=True,
                    records=int(len(frame)),
                    details={
                        "market_prefix": self.market_prefix,
                        "last_good_path": str(self.last_good_path),
                    },
                    fallback_chain=fallback_chain,
                )
                return ExtractorResult(frame=frame, provenance=self.provenance)
            except Exception as exc:  # noqa: BLE001
                log.warning("Kalshi live fetch failed, using backups: %s", exc)
                fallback_chain.append(f"live_error:{exc.__class__.__name__}")
                selection = get_source(
                    "kalshi",
                    sample_path=self.sample_path,
                    suffix=self.sample_path.suffix or ".csv",
                    allow_live=False,
                )

        last_good_meta = None
        if selection.mode == "last_good" and self.last_good_path.exists():
            try:
                frame, last_good_meta = load_snapshot(
                    self.last_good_path,
                    parse_dates=["date"],
                )
                as_of = infer_as_of(frame, ("date",))
                is_fresh = snapshot_is_fresh(last_good_meta, self.freshness, now=now)
                age_hours = freshness_age_hours(last_good_meta, now=now)
                if is_fresh:
                    fetched_at = (
                        last_good_meta.get("fetched_at") if last_good_meta else None
                    )
                    provenance_as_of = as_of
                    if provenance_as_of is None and last_good_meta:
                        provenance_as_of = last_good_meta.get("as_of")
                    self.provenance = DataProvenance(
                        source="kalshi",
                        mode="last_good",
                        path=self.last_good_path,
                        fetched_at=fetched_at,
                        as_of=provenance_as_of,
                        fresh=True,
                        records=int(len(frame)),
                        details={
                            "market_prefix": self.market_prefix,
                            "age_hours": age_hours,
                            "last_good_path": str(self.last_good_path),
                        },
                        fallback_chain=fallback_chain,
                    )
                    return ExtractorResult(frame=frame, provenance=self.provenance)
                fallback_chain.append("last_good_stale")
            except FileNotFoundError:
                fallback_chain.append("last_good_missing")
            except Exception as exc:  # noqa: BLE001
                log.warning("Kalshi last-good load failed: %s", exc)
                fallback_chain.append("last_good_error")

        sample_frame = read_csv_with_date(self.sample_path, parse_dates=["date"])
        as_of = infer_as_of(sample_frame, ("date",))
        age_hours = (
            freshness_age_hours(last_good_meta, now=now) if last_good_meta else None
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
                "last_good_path": str(self.last_good_path),
                "last_good_age_hours": age_hours,
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
    last_good_path = config.data.raw_dir / "last_good.kalshi.csv"
    extractor = KalshiExtractor(last_good_path=last_good_path, sample_path=sample_path)
    transformer = KalshiTransformer()
    loader = CSVLoader(output_path=output_path)
    return ETLTask(extractor=extractor, transformer=transformer, loader=loader)
