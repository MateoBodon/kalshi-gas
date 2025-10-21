"""AAA daily gasoline price ETL and HTML parsing helpers."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, TypedDict

import pandas as pd
from bs4 import BeautifulSoup, Tag
from dateutil import parser as date_parser

from kalshi_gas.config import PipelineConfig
from kalshi_gas.etl.base import DataProvenance, ETLTask, ExtractorResult
from kalshi_gas.etl.utils import (
    CSVLoader,
    fetch_json,
    freshness_age_hours,
    get_source,
    infer_as_of,
    load_snapshot,
    read_csv_with_date,
    save_snapshot,
    snapshot_is_fresh,
    utcnow_iso,
)

AAA_DAILY_AVG_URL = (
    "https://gasprices.aaa.com/wp-json/aaa-api/v1/daily-national-average"
)

PRICE_PATTERN = re.compile(r"\$?\s*([0-9]+(?:\.[0-9]{1,3})?)")
AS_OF_PATTERN = re.compile(r"price\s+as\s+of\s+([A-Za-z0-9.,\s]+)", re.IGNORECASE)
COMPONENT_PATTERNS = {
    "current": [r"\bcurrent\b", r"\btoday\b", r"\bnational\s+average\b"],
    "yesterday": [r"\byesterday\b", r"\bday\s+ago\b"],
    "week_ago": [r"\bweek\s+ago\b", r"\b7[-\s]?day\b"],
    "month_ago": [r"\bmonth\s+ago\b", r"\b30[-\s]?day\b"],
    "year_ago": [r"\byear\s+ago\b", r"\b365[-\s]?day\b"],
}
COMPONENT_ALIAS_KEYWORDS = {
    "current": ("current", "today", "now", "national", "avg"),
    "yesterday": ("yesterday", "day_ago", "prev_day"),
    "week_ago": ("week", "7day", "week_ago", "seven"),
    "month_ago": ("month", "30", "month_ago", "thirty"),
    "year_ago": ("year", "365", "year_ago"),
}
VALUE_ATTRS = (
    "data-price",
    "data-value",
    "data-amount",
    "data-price-usd",
    "data-price-value",
    "data-national-average",
)

log = logging.getLogger(__name__)


class AAAComponents(TypedDict):
    current: float | None
    yesterday: float | None
    week_ago: float | None
    month_ago: float | None
    year_ago: float | None


class AAANationalPayload(TypedDict):
    price: float
    as_of_date: str | None
    components: AAAComponents


def _ensure_text(source: str | BeautifulSoup | Tag) -> str:
    if isinstance(source, (BeautifulSoup, Tag)):
        return source.get_text(" ", strip=True)
    return str(source)


def _text_to_float(text: str) -> float | None:
    match = PRICE_PATTERN.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _value_from_node(node: Tag) -> float | None:
    for attr in VALUE_ATTRS:
        raw_value = node.attrs.get(attr)
        if raw_value is None:
            continue
        candidates: list[str] = []
        if isinstance(raw_value, str):
            candidates.append(raw_value)
        elif isinstance(raw_value, (list, tuple)):
            candidates.extend(str(item) for item in raw_value if item)
        else:
            candidates.append(str(raw_value))
        for candidate in candidates:
            try:
                return float(candidate)
            except ValueError:
                continue
    return _text_to_float(node.get_text(" ", strip=True))


def _select_container(soup: BeautifulSoup) -> Tag | None:
    selectors = [
        "[data-national-average-container]",
        "#nationalAverage",
        "section.national-average",
        "div.national-average",
        "section[class*='national']",
    ]
    for selector in selectors:
        match = soup.select_one(selector)
        if match:
            return match
    headline = soup.find(string=re.compile(r"National\s+Average", re.IGNORECASE))
    if headline and headline.parent:
        return headline.parent
    return None


def _normalise_component_key(raw: str) -> str | None:
    key = raw.lower()
    for canonical, keywords in COMPONENT_ALIAS_KEYWORDS.items():
        if any(keyword in key for keyword in keywords):
            return canonical
    return None


def _extract_price(root: BeautifulSoup | Tag) -> float | None:
    selectors = [
        "[data-national-average-price]",
        "[data-national-average]",
        ".national-average .price-value",
        ".national-average-value",
        ".price-value",
        ".money",
        ".average-price",
        "#nationalAveragePrice",
    ]
    for selector in selectors:
        node = (
            root.select_one(selector)
            if isinstance(root, (BeautifulSoup, Tag))
            else None
        )
        if isinstance(node, Tag):
            value = _value_from_node(node)
            if value is not None:
                return value
    if isinstance(root, Tag):
        attr_value = _value_from_node(root)
        if attr_value is not None:
            return attr_value
    for label in (
        root.find_all(string=re.compile(r"National\s+Average", re.IGNORECASE))
        if isinstance(root, (BeautifulSoup, Tag))
        else []
    ):
        parent = label.parent
        if isinstance(parent, Tag):
            candidate = _value_from_node(parent)
            if candidate is not None:
                return candidate
            sibling = parent.find_next(string=PRICE_PATTERN)
            if sibling:
                value = _text_to_float(sibling)
                if value is not None:
                    return value
    text = _ensure_text(root)
    value = _text_to_float(text)
    return value


def _extract_components(root: BeautifulSoup | Tag) -> Dict[str, float]:
    results: Dict[str, float] = {}
    tags = root.find_all(True) if isinstance(root, (BeautifulSoup, Tag)) else []
    for node in tags:
        raw_attrs = [
            node.attrs.get("data-component"),
            node.attrs.get("data-compare"),
            node.attrs.get("data-metric"),
        ]
        attr_candidates: list[str] = []
        for attr in raw_attrs:
            if isinstance(attr, str):
                attr_candidates.append(attr)
            elif isinstance(attr, (list, tuple)):
                attr_candidates.extend(
                    str(item) for item in attr if isinstance(item, str)
                )
        for attr in attr_candidates:
            component_key = _normalise_component_key(attr)
            if component_key and component_key not in results:
                value = _value_from_node(node)
                if value is not None:
                    results[component_key] = value
        text = node.get_text(" ", strip=True)
        if not text:
            continue
        numeric_value = _text_to_float(text)
        if numeric_value is None:
            continue
        for key, patterns in COMPONENT_PATTERNS.items():
            if key in results:
                continue
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                results[key] = numeric_value
                break
    return results


def parse_aaa_asof_date(source: str | BeautifulSoup | Tag) -> str | None:
    text = _ensure_text(source)
    match = AS_OF_PATTERN.search(text)
    if not match:
        return None
    raw_date = match.group(1).strip(" :")
    raw_date = re.sub(r"\s*\(.*?\)", "", raw_date)
    raw_date = re.sub(r"\s+\|.*$", "", raw_date)
    raw_date = raw_date.replace("Sept.", "September")
    raw_date = re.sub(r"(?<=\b\w)\.", "", raw_date)
    try:
        parsed = date_parser.parse(raw_date, fuzzy=True)
    except (ValueError, TypeError):
        return None
    return parsed.strftime("%Y-%m-%d")


def parse_aaa_national(html: str) -> AAANationalPayload:
    soup = BeautifulSoup(html, "html.parser")
    container = _select_container(soup)
    root = container or soup
    price = _extract_price(root)
    if price is None:
        raise ValueError("AAA national markup missing price")
    as_of_date = parse_aaa_asof_date(root)
    component_values = _extract_components(root)
    components: AAAComponents = {
        "current": component_values.get("current"),
        "yesterday": component_values.get("yesterday"),
        "week_ago": component_values.get("week_ago"),
        "month_ago": component_values.get("month_ago"),
        "year_ago": component_values.get("year_ago"),
    }
    if components["current"] is None:
        components["current"] = price
    return {
        "price": price,
        "as_of_date": as_of_date,
        "components": components,
    }


class AAAExtractor:
    def __init__(
        self,
        last_good_path: Path,
        sample_path: Path,
        freshness_hours: int = 36,
    ):
        self.last_good_path = last_good_path
        self.sample_path = sample_path
        self.freshness = timedelta(hours=freshness_hours)
        self.provenance: DataProvenance | None = None

    def extract(self) -> ExtractorResult:
        fallback_chain: list[str] = []
        now = datetime.now(timezone.utc)

        selection = get_source("aaa", sample_path=self.sample_path)
        if selection.mode == "live":
            try:
                payload = fetch_json(AAA_DAILY_AVG_URL)
                frame = pd.DataFrame([payload])
                frame["date"] = pd.to_datetime(frame["date"])
                frame.rename(columns={"price": "regular_gas_price"}, inplace=True)
                as_of = infer_as_of(frame, ("date",))
                fetched_at = utcnow_iso()
                metadata = {
                    "source": "aaa",
                    "mode": "live",
                    "as_of": as_of,
                    "fetched_at": fetched_at,
                    "url": AAA_DAILY_AVG_URL,
                }
                save_snapshot(frame, self.last_good_path, metadata=metadata)
                self.provenance = DataProvenance(
                    source="aaa",
                    mode="live",
                    path=self.last_good_path,
                    fetched_at=fetched_at,
                    as_of=as_of,
                    fresh=True,
                    records=int(len(frame)),
                    details={
                        "url": AAA_DAILY_AVG_URL,
                        "last_good_path": str(self.last_good_path),
                    },
                    fallback_chain=fallback_chain,
                )
                return ExtractorResult(
                    frame=frame[["date", "regular_gas_price"]],
                    provenance=self.provenance,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("AAA live fetch failed, will try fallbacks: %s", exc)
                fallback_chain.append(f"live_error:{exc.__class__.__name__}")
                selection = get_source(
                    "aaa", sample_path=self.sample_path, allow_live=False
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
                    source="aaa",
                    mode="last_good",
                    path=self.last_good_path,
                    fetched_at=fetched_at,
                    as_of=provenance_as_of,
                    fresh=is_fresh,
                    records=int(len(frame)),
                    details={
                        "age_hours": age_hours,
                        "last_good_path": str(self.last_good_path),
                    },
                    fallback_chain=fallback_chain,
                )
                return ExtractorResult(
                    frame=frame[["date", "regular_gas_price"]],
                    provenance=self.provenance,
                )
            except FileNotFoundError:
                fallback_chain.append("last_good_missing")
            except Exception as exc:  # noqa: BLE001
                log.warning("AAA last-good load failed: %s", exc)
                fallback_chain.append("last_good_error")

        sample_frame = read_csv_with_date(self.sample_path, parse_dates=["date"])
        as_of = infer_as_of(sample_frame, ("date",))
        age_hours = (
            freshness_age_hours(last_good_meta, now=now) if last_good_meta else None
        )
        self.provenance = DataProvenance(
            source="aaa",
            mode="sample",
            path=self.sample_path,
            fetched_at=None,
            as_of=as_of,
            fresh=False,
            records=int(len(sample_frame)),
            details={
                "last_good_path": str(self.last_good_path),
                "last_good_age_hours": age_hours,
            },
            fallback_chain=fallback_chain,
        )
        return ExtractorResult(
            frame=sample_frame[["date", "regular_gas_price"]],
            provenance=self.provenance,
        )


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
    sample_path = Path("data/sample/aaa_daily.csv")
    last_good_path = config.data.raw_dir / "last_good.aaa.csv"
    extractor = AAAExtractor(last_good_path=last_good_path, sample_path=sample_path)
    transformer = AAATransformer()
    loader = CSVLoader(output_path=output_path)
    return ETLTask(extractor=extractor, transformer=transformer, loader=loader)
