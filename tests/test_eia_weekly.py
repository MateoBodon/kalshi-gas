from pathlib import Path

import pandas as pd
import pytest

from kalshi_gas.etl.eia import parse_weekly_series

FIXTURE = Path(__file__).parent / "fixtures" / "eia_weekly_series.html"


def test_eia_weekly_parse_returns_expected_frame() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    frame = parse_weekly_series(html)

    assert list(frame.columns) == ["date", "retail"]
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])
    assert frame["retail"].dtype.kind == "f"

    latest = frame.iloc[-1]
    assert latest["date"] == pd.Timestamp("2024-10-07")
    assert latest["retail"] == pytest.approx(3.549)

    sept_30 = frame.loc[frame["date"] == pd.Timestamp("2024-09-30"), "retail"].item()
    assert sept_30 == pytest.approx(3.521)

    sep_16 = frame.loc[frame["date"] == pd.Timestamp("2024-09-16"), "retail"].item()
    assert sep_16 == pytest.approx(3.476)
