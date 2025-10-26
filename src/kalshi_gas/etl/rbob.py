"""RBOB futures ETL."""

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

EIA_RBOB_SERIES = "PET.RBRTWD.W"  # retained for provenance metadata
# NY Harbor conventional gasoline spot price (weekly, $/gal)
EIA_RBOB_SERIES_V2 = "EER_EPMRU_PF4_Y35NY_DPG"

log = logging.getLogger(__name__)


class RBOBExtractor:
    def __init__(
        self,
        last_good_path: Path,
        sample_path: Path,
        series_id: str = EIA_RBOB_SERIES,
        freshness_hours: int = 48,
    ):
        self.last_good_path = last_good_path
        self.sample_path = sample_path
        self.series_id = series_id
        self.freshness = timedelta(hours=freshness_hours)
        self.provenance: DataProvenance | None = None

    def _fetch_live(self) -> pd.DataFrame:
        api_key = read_env("EIA_API_KEY")
        if not api_key:
            raise RuntimeError("EIA_API_KEY not configured")

        params = {
            "api_key": api_key,
            "frequency": "weekly",
            "data[0]": "value",
            "facets[series][]": EIA_RBOB_SERIES_V2,
            "length": 5000,
        }
        response = requests.get(
            "https://api.eia.gov/v2/petroleum/pri/spt/data/",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        entries = payload.get("response", {}).get("data", [])
        if not entries:
            raise RuntimeError(
                "EIA v2 returned no spot price data for the configured series"
            )

        frame = pd.DataFrame(entries)
        if "period" not in frame or "value" not in frame:
            raise RuntimeError("Unexpected EIA spot price payload structure")
        frame["date"] = pd.to_datetime(frame["period"])
        frame["settle"] = frame["value"].astype(float)
        frame = frame[["date", "settle"]]
        return frame

    def extract(self) -> ExtractorResult:
        fallback_chain: list[str] = []
        now = datetime.now(timezone.utc)
        selection = get_source(
            "rbob",
            sample_path=self.sample_path,
            suffix=self.sample_path.suffix or ".csv",
        )
        if selection.mode == "live":
            try:
                frame = self._fetch_live()
                as_of = infer_as_of(frame, ("date",))
                fetched_at = utcnow_iso()
                metadata = {
                    "source": "rbob",
                    "mode": "live",
                    "series_id": self.series_id,
                    "as_of": as_of,
                    "fetched_at": fetched_at,
                }
                save_snapshot(frame, self.last_good_path, metadata=metadata)
                self.provenance = DataProvenance(
                    source="rbob",
                    mode="live",
                    path=self.last_good_path,
                    fetched_at=fetched_at,
                    as_of=as_of,
                    fresh=True,
                    records=int(len(frame)),
                    details={
                        "series_id": self.series_id,
                        "last_good_path": str(self.last_good_path),
                    },
                    fallback_chain=fallback_chain,
                )
                return ExtractorResult(frame=frame, provenance=self.provenance)
            except Exception as exc:  # noqa: BLE001
                log.warning("RBOB live fetch failed, using backups: %s", exc)
                fallback_chain.append(f"live_error:{exc.__class__.__name__}")
                selection = get_source(
                    "rbob",
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
                        source="rbob",
                        mode="last_good",
                        path=self.last_good_path,
                        fetched_at=fetched_at,
                        as_of=provenance_as_of,
                        fresh=True,
                        records=int(len(frame)),
                        details={
                            "series_id": self.series_id,
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
                log.warning("RBOB last-good load failed: %s", exc)
                fallback_chain.append("last_good_error")

        sample_frame = read_csv_with_date(self.sample_path, parse_dates=["date"])
        as_of = infer_as_of(sample_frame, ("date",))
        age_hours = (
            freshness_age_hours(last_good_meta, now=now) if last_good_meta else None
        )
        self.provenance = DataProvenance(
            source="rbob",
            mode="sample",
            path=self.sample_path,
            fetched_at=None,
            as_of=as_of,
            fresh=False,
            records=int(len(sample_frame)),
            details={
                "series_id": self.series_id,
                "last_good_path": str(self.last_good_path),
                "last_good_age_hours": age_hours,
            },
            fallback_chain=fallback_chain,
        )
        return ExtractorResult(frame=sample_frame, provenance=self.provenance)


class RBOBTransformer:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["settle"] = frame["settle"].astype(float)
        optional_cols = {
            "second_month": "rbob_second",
            "spread": "rbob_spread",
            "settle_change": "rbob_settle_change",
            "second_change": "rbob_second_change",
        }
        for source_col, target_col in optional_cols.items():
            if source_col in frame:
                frame[target_col] = pd.to_numeric(frame[source_col], errors="coerce")
        frame.sort_values("date", inplace=True)
        frame["settle"] = frame["settle"].round(4)
        frame.rename(columns={"settle": "rbob_price"}, inplace=True)
        return frame.reset_index(drop=True)


def build_rbob_etl(config: PipelineConfig) -> ETLTask:
    output_path = config.data.processed_dir / "rbob_prices.csv"
    sample_path = Path("data/sample/rbob_futures.csv")
    last_good_path = config.data.raw_dir / "last_good.rbob.csv"
    extractor = RBOBExtractor(last_good_path=last_good_path, sample_path=sample_path)
    transformer = RBOBTransformer()
    loader = CSVLoader(output_path=output_path)
    return ETLTask(extractor=extractor, transformer=transformer, loader=loader)
