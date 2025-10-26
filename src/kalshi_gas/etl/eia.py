"""EIA weekly petroleum status ETL."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag
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

EIA_BASE_URL = "https://api.eia.gov/series/"
EIA_INVENTORY_SERIES = (
    "PET.WGTSTUS1.W"  # Finished motor gasoline stocks, thousand barrels
)
PRICE_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)")
PERCENT_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")

log = logging.getLogger(__name__)


class EIAExtractor:
    def __init__(
        self,
        last_good_path: Path,
        sample_path: Path,
        series_id: str = EIA_INVENTORY_SERIES,
        freshness_hours: int = 240,
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

        url = f"https://api.eia.gov/v2/seriesid/{self.series_id}"
        params = {
            "api_key": api_key,
            "offset": 0,
            "length": 5000,
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        series_data = payload.get("response", {}).get("data", [])
        if not series_data:
            raise RuntimeError(f"EIA series {self.series_id} returned no data")

        frame = pd.DataFrame(series_data)
        if "period" not in frame or "value" not in frame:
            raise RuntimeError("Unexpected EIA payload structure for inventory series")
        frame["date"] = pd.to_datetime(frame["period"])
        frame["inventory_mmbbl"] = frame["value"].astype(float) / 1000.0
        frame["production_mbd"] = pd.NA
        frame = frame[["date", "inventory_mmbbl", "production_mbd"]]
        frame.sort_values("date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        return frame

    def extract(self) -> ExtractorResult:
        fallback_chain: list[str] = []
        now = datetime.now(timezone.utc)

        selection = get_source(
            "eia",
            sample_path=self.sample_path,
            suffix=self.sample_path.suffix or ".csv",
        )
        if selection.mode == "live":
            try:
                frame = self._fetch_live()
                as_of = infer_as_of(frame, ("date",))
                fetched_at = utcnow_iso()
                metadata = {
                    "source": "eia",
                    "mode": "live",
                    "series_id": self.series_id,
                    "as_of": as_of,
                    "fetched_at": fetched_at,
                }
                save_snapshot(frame, self.last_good_path, metadata=metadata)
                self.provenance = DataProvenance(
                    source="eia",
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
                log.warning("EIA live fetch failed, using backups: %s", exc)
                fallback_chain.append(f"live_error:{exc.__class__.__name__}")
                selection = get_source(
                    "eia",
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
                fetched_at = (
                    last_good_meta.get("fetched_at") if last_good_meta else None
                )
                provenance_as_of = as_of
                if provenance_as_of is None and last_good_meta:
                    provenance_as_of = last_good_meta.get("as_of")
                self.provenance = DataProvenance(
                    source="eia",
                    mode="last_good",
                    path=self.last_good_path,
                    fetched_at=fetched_at,
                    as_of=provenance_as_of,
                    fresh=is_fresh,
                    records=int(len(frame)),
                    details={
                        "series_id": self.series_id,
                        "age_hours": age_hours,
                        "last_good_path": str(self.last_good_path),
                    },
                    fallback_chain=fallback_chain,
                )
                return ExtractorResult(frame=frame, provenance=self.provenance)
            except FileNotFoundError:
                fallback_chain.append("last_good_missing")
            except Exception as exc:  # noqa: BLE001
                log.warning("EIA last-good load failed: %s", exc)
                fallback_chain.append("last_good_error")

        sample_frame = read_csv_with_date(self.sample_path, parse_dates=["date"])
        as_of = infer_as_of(sample_frame, ("date",))
        age_hours = (
            freshness_age_hours(last_good_meta, now=now) if last_good_meta else None
        )
        self.provenance = DataProvenance(
            source="eia",
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
    sample_path = Path("data/sample/eia_weekly.csv")
    last_good_path = config.data.raw_dir / "last_good.eia.csv"
    extractor = EIAExtractor(last_good_path=last_good_path, sample_path=sample_path)
    transformer = EIATransformer()
    loader = CSVLoader(output_path=output_path)
    return ETLTask(extractor=extractor, transformer=transformer, loader=loader)


def _find_weekly_table(soup: BeautifulSoup) -> Tag | None:
    for table in soup.find_all("table"):
        header_cells = table.find_all("th")
        header_text = [cell.get_text(" ", strip=True).lower() for cell in header_cells]
        if not header_text:
            continue
        header_blob = " ".join(header_text)
        if re.search(r"(week|date)", header_blob) and re.search(
            r"(regular|retail|price)", header_blob
        ):
            return table
    return None


def _parse_week_date(text: str) -> pd.Timestamp | None:
    cleaned = re.sub(r"\bweek\s+of\b", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(.*?\)", "", cleaned)
    cleaned = re.sub(r"[*†‡]", "", cleaned)
    cleaned = re.sub(r"(?<=\b[a-zA-Z]{3})\.", "", cleaned)
    cleaned = cleaned.strip(" :;-")
    if not cleaned:
        return None
    parsed = pd.to_datetime(cleaned, errors="coerce")
    if pd.isna(parsed):
        return None
    timestamp = pd.Timestamp(parsed).normalize()
    monday = timestamp - pd.Timedelta(days=int(timestamp.weekday()))
    return monday


def _extract_price_value(node: Tag) -> float | None:
    text = node.get_text(" ", strip=True)
    match = PRICE_PATTERN.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_weekly_series(html: str) -> pd.DataFrame:
    """Parse the EIA weekly retail HTML table into a tidy DataFrame."""
    soup = BeautifulSoup(html, "html.parser")
    table = _find_weekly_table(soup)
    if table is None:
        raise ValueError("EIA weekly retail table not found")

    records: list[tuple[pd.Timestamp, float]] = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        date_text = cells[0].get_text(" ", strip=True)
        if not date_text:
            continue
        parsed_date = _parse_week_date(date_text)
        if parsed_date is None:
            continue

        price_value = None
        for cell in cells[1:]:
            price_value = _extract_price_value(cell)
            if price_value is not None:
                break

        if price_value is None:
            continue

        records.append((parsed_date, price_value))

    if not records:
        raise ValueError("No weekly retail rows parsed from EIA markup")

    frame = pd.DataFrame(records, columns=["date", "retail"])
    frame = frame.drop_duplicates(subset="date")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["retail"] = frame["retail"].astype(float)
    frame.sort_values("date", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def parse_wpsr_summary(html: str) -> dict[str, object]:
    """Parse key metrics from the WPSR summary table."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    target_table: Tag | None = None

    for table in tables:
        text = " ".join(
            header.get_text(" ", strip=True).lower() for header in table.find_all("th")
        )
        if re.search(r"gasoline", text) or re.search(r"utilization", text):
            target_table = table
            break

    if target_table is None:
        raise ValueError("WPSR summary table not found")

    stocks_value: float | None = None
    utilization_value: float | None = None
    supplied_value: float | None = None
    week_ending_value: pd.Timestamp | None = None

    for row in target_table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        label = cells[0].get_text(" ", strip=True).lower()
        values_text = " ".join(cell.get_text(" ", strip=True) for cell in cells[1:])

        if week_ending_value is None and re.search(r"week\s+ending", label):
            parsed = pd.to_datetime(values_text, errors="coerce")
            if not pd.isna(parsed):
                week_ending_value = pd.Timestamp(parsed).normalize()
            continue

        if stocks_value is None and re.search(r"gasoline\s+stocks", label):
            match = PRICE_PATTERN.search(values_text)
            if match:
                stocks_value = float(match.group(1))
            continue

        if utilization_value is None and re.search(r"refinery\s+util", label):
            match = PERCENT_PATTERN.search(values_text)
            if match:
                utilization_value = float(match.group(1))
            continue

        if supplied_value is None and re.search(r"product\s+supplied|demand", label):
            match = PRICE_PATTERN.search(values_text)
            if match:
                supplied_value = float(match.group(1))
            continue

    if week_ending_value is None:
        raise ValueError("WPSR summary missing week ending date")
    if stocks_value is None:
        raise ValueError("WPSR summary missing gasoline stocks")
    if utilization_value is None:
        raise ValueError("WPSR summary missing refinery utilization")
    if supplied_value is None:
        raise ValueError("WPSR summary missing product supplied")

    return {
        "week_ending": week_ending_value,
        "gasoline_stocks_mmb": stocks_value,
        "refinery_util_pct": utilization_value,
        "product_supplied_mbd": supplied_value,
    }
