from pathlib import Path

import pandas as pd

from kalshi_gas.etl.eia import parse_wpsr_summary

FIXTURE = Path(__file__).parent / "fixtures" / "wpsr_summary.html"


def test_parse_wpsr_summary_extracts_metrics() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    summary = parse_wpsr_summary(html)

    assert set(summary.keys()) == {
        "week_ending",
        "gasoline_stocks_mmb",
        "refinery_util_pct",
        "product_supplied_mbd",
    }

    assert summary["week_ending"] == pd.Timestamp("2024-10-04")
    assert summary["gasoline_stocks_mmb"] == 226.5
    assert summary["refinery_util_pct"] == 89.6
    assert summary["product_supplied_mbd"] == 9.1
