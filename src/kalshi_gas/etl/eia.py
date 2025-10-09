"""EIA weekly petroleum status ETL."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag

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
EIA_INVENTORY_SERIES = (
    "PET.WGTSTUS1.W"  # Finished motor gasoline stocks, thousand barrels
)
PRICE_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


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
