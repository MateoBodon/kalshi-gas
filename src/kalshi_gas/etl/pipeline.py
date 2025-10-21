"""Run ETL tasks for all data sources."""

from __future__ import annotations

from typing import Dict

from kalshi_gas.config import PipelineConfig, load_config
from kalshi_gas.etl.aaa import build_aaa_etl
from kalshi_gas.etl.base import ETLRunResult
from kalshi_gas.etl.eia import build_eia_etl
from kalshi_gas.etl.kalshi import build_kalshi_etl
from kalshi_gas.etl.rbob import build_rbob_etl


def run_all_etl(config: PipelineConfig | None = None) -> Dict[str, ETLRunResult]:
    """Execute ETL for all configured sources."""
    cfg = config or load_config()
    tasks = {
        "aaa": build_aaa_etl(cfg),
        "eia": build_eia_etl(cfg),
        "rbob": build_rbob_etl(cfg),
        "kalshi": build_kalshi_etl(cfg),
    }
    outputs: Dict[str, ETLRunResult] = {}
    for name, task in tasks.items():
        result = task.run()
        outputs[name] = result
    return outputs
