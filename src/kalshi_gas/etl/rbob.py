"""RBOB futures ETL."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from kalshi_gas.config import PipelineConfig
from kalshi_gas.etl.base import DataProvenance, ETLTask, ExtractorResult
from kalshi_gas.etl.utils import (
    CSVLoader,
    fetch_json,
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

EIA_RBOB_SERIES = "PET.RBRTWD.W"  # Reformulated gasoline spot price, $/gallon

log = logging.getLogger(__name__)


class RBOBExtractor:
    def __init__(
        self,
        snapshot_path: Path,
        sample_path: Path,
        series_id: str = EIA_RBOB_SERIES,
        freshness_hours: int = 48,
    ):
        self.snapshot_path = snapshot_path
        self.sample_path = sample_path
        self.series_id = series_id
        self.freshness = timedelta(hours=freshness_hours)
        self.provenance: DataProvenance | None = None

    def _fetch_live(self) -> pd.DataFrame:
        api_key = read_env("EIA_API_KEY")
        if not api_key:
            raise RuntimeError("EIA_API_KEY not configured")

        url = (
            f"https://api.eia.gov/series/?api_key={api_key}&series_id={self.series_id}"
        )
        payload = fetch_json(url)
        series_meta = payload["series"][0]
        frame = pd.DataFrame(series_meta["data"], columns=["period", "settle"])
        frame["date"] = pd.to_datetime(frame["period"])
        frame["settle"] = frame["settle"].astype(float)
        frame.drop(columns=["period"], inplace=True)
        return frame

    def extract(self) -> ExtractorResult:
        fallback_chain: list[str] = []
        now = datetime.now(timezone.utc)

        if use_live_data():
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
                save_snapshot(frame, self.snapshot_path, metadata=metadata)
                self.provenance = DataProvenance(
                    source="rbob",
                    mode="live",
                    path=self.snapshot_path,
                    fetched_at=fetched_at,
                    as_of=as_of,
                    fresh=True,
                    records=int(len(frame)),
                    details={
                        "series_id": self.series_id,
                        "snapshot_path": str(self.snapshot_path),
                    },
                    fallback_chain=fallback_chain,
                )
                return ExtractorResult(frame=frame, provenance=self.provenance)
            except Exception as exc:  # noqa: BLE001
                log.warning("RBOB live fetch failed, using backups: %s", exc)
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
                        source="rbob",
                        mode="snapshot",
                        path=self.snapshot_path,
                        fetched_at=fetched_at,
                        as_of=provenance_as_of,
                        fresh=True,
                        records=int(len(frame)),
                        details={
                            "series_id": self.series_id,
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
                log.warning("RBOB snapshot load failed: %s", exc)
                fallback_chain.append("snapshot_error")

        sample_frame = read_csv_with_date(self.sample_path, parse_dates=["date"])
        as_of = infer_as_of(sample_frame, ("date",))
        age_hours = (
            freshness_age_hours(snapshot_meta, now=now) if snapshot_meta else None
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
                "snapshot_path": str(self.snapshot_path),
                "snapshot_age_hours": age_hours,
            },
            fallback_chain=fallback_chain,
        )
        return ExtractorResult(frame=sample_frame, provenance=self.provenance)


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
    sample_path = Path("data/sample/rbob_futures.csv")
    snapshot_path = config.data.raw_dir / "rbob_futures_snapshot.csv"
    extractor = RBOBExtractor(snapshot_path=snapshot_path, sample_path=sample_path)
    transformer = RBOBTransformer()
    loader = CSVLoader(output_path=output_path)
    return ETLTask(extractor=extractor, transformer=transformer, loader=loader)
