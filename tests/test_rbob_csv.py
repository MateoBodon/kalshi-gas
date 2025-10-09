from pathlib import Path

import pandas as pd

from scripts.update_rbob_csv import append_row, normalize_frame


def test_normalize_frame_sorts_and_dedupes(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"date": "2024-01-03", "settle": "2.10", "source": "api"},
            {"date": "2024-01-01", "settle": "2.08", "source": "api"},
            {"date": "2024-01-03", "settle": "2.11", "source": "manual"},
        ]
    )
    normalized = normalize_frame(frame)
    assert list(normalized["date"].dt.strftime("%Y-%m-%d")) == [
        "2024-01-01",
        "2024-01-03",
    ]
    assert (
        normalized.loc[
            normalized["date"] == pd.Timestamp("2024-01-03"), "settle"
        ].item()
        == 2.11
    )


def test_append_row_persists_sorted_csv(tmp_path: Path) -> None:
    path = tmp_path / "rbob_daily.csv"
    append_row(path, "2024-01-02", 2.05, "manual")
    append_row(path, "2024-01-01", 2.00, "manual")
    append_row(path, "2024-01-02", 2.06, "api")

    frame = pd.read_csv(path)
    assert list(frame["date"]) == ["2024-01-01", "2024-01-02"]
    assert frame.loc[1, "settle"] == 2.06
    assert frame.loc[1, "source"] == "api"
