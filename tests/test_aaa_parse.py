from pathlib import Path
from typing import cast

import pytest

from kalshi_gas.etl.aaa import (
    AAANationalPayload,
    AAAComponents,
    parse_aaa_asof_date,
    parse_aaa_national,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parse_v1_price_and_components() -> None:
    html = _load_fixture("aaa_national_v1.html")
    parsed: AAANationalPayload = parse_aaa_national(html)
    assert parsed["price"] == pytest.approx(3.453)
    assert parsed["as_of_date"] == "2025-01-15"
    components = cast(AAAComponents, parsed["components"])
    assert isinstance(components, dict)
    assert isinstance(components["current"], float)
    assert isinstance(components["yesterday"], float)
    assert isinstance(components["week_ago"], float)
    assert isinstance(components["month_ago"], float)
    assert isinstance(components["year_ago"], float)


def test_parse_v2_handles_alternate_dom() -> None:
    html = _load_fixture("aaa_national_v2.html")
    parsed: AAANationalPayload = parse_aaa_national(html)
    assert parsed["price"] == pytest.approx(3.612)
    assert parsed["as_of_date"] == "2025-02-03"
    components = cast(AAAComponents, parsed["components"])
    assert components["current"] == pytest.approx(3.612)
    assert components["yesterday"] == pytest.approx(3.598)
    assert components["week_ago"] == pytest.approx(3.577)
    assert components["month_ago"] == pytest.approx(3.412)
    assert components["year_ago"] == pytest.approx(3.103)


def test_parse_asof_date_missing_returns_none() -> None:
    assert parse_aaa_asof_date("No price as of marker here") is None


def test_parse_raises_when_price_missing() -> None:
    with pytest.raises(ValueError, match="missing price"):
        parse_aaa_national("<html><body><p>No price here</p></body></html>")
