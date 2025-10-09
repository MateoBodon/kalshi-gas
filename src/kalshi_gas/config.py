"""Configuration helpers for the kalshi_gas package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class DataPaths:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    external_dir: Path
    build_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    ensemble_weights: Dict[str, float]
    calibration_bins: int
    horizon_days: int


@dataclass(frozen=True)
class PipelineConfig:
    data: DataPaths
    model: ModelConfig
    risk_gates: Dict[str, Any]


DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "raw_dir": "data/raw",
        "interim_dir": "data/interim",
        "processed_dir": "data/processed",
        "external_dir": "data/external",
        "build_dir": "build",
    },
    "model": {
        "ensemble_weights": {
            "nowcast": 0.4,
            "pass_through": 0.35,
            "market_prior": 0.25,
        },
        "calibration_bins": 10,
        "horizon_days": 7,
    },
    "risk_gates": {
        "nhc_active_threshold": 1,
        "wpsr_inventory_cutoff": -1.5,
    },
}


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load configuration from YAML or use defaults."""
    config_data = DEFAULT_CONFIG
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            config_data = yaml.safe_load(handle)

    project_root = Path(".").resolve()
    data_cfg = config_data["data"]
    model_cfg = config_data["model"]
    risk_cfg = config_data["risk_gates"]

    data_paths = DataPaths(
        raw_dir=(project_root / data_cfg["raw_dir"]).resolve(),
        interim_dir=(project_root / data_cfg["interim_dir"]).resolve(),
        processed_dir=(project_root / data_cfg["processed_dir"]).resolve(),
        external_dir=(project_root / data_cfg["external_dir"]).resolve(),
        build_dir=(project_root / data_cfg["build_dir"]).resolve(),
    )

    return PipelineConfig(
        data=data_paths,
        model=ModelConfig(
            ensemble_weights=model_cfg["ensemble_weights"],
            calibration_bins=int(model_cfg["calibration_bins"]),
            horizon_days=int(model_cfg["horizon_days"]),
        ),
        risk_gates=risk_cfg,
    )
