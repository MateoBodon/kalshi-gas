from pathlib import Path

import pytest

from kalshi_gas.pipeline import run_pipeline


def test_run_pipeline_smoke_artifacts_exist() -> None:
    pytest.importorskip("matplotlib.pyplot")
    result = run_pipeline()

    report_path = Path(result["report_path"])
    deck_path = Path(result["deck_path"])
    sensitivity_path = Path(result["sensitivity_path"])
    figures = result.get("figures", {})

    assert report_path.exists() and report_path.stat().st_size > 0
    assert deck_path.exists() and deck_path.stat().st_size > 0
    assert sensitivity_path.exists() and sensitivity_path.stat().st_size > 0

    assert isinstance(figures, dict) and figures
    for figure_path in figures.values():
        path_obj = Path(figure_path)
        assert path_obj.exists() and path_obj.stat().st_size > 0
