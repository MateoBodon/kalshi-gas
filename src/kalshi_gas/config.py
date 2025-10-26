"""Configuration helpers for the kalshi_gas package."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
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
    prior_weight: float
    nowcast_drift_bounds: tuple[float, float] | None = None


@dataclass(frozen=True)
class EventConfig:
    name: str
    resolution_date: date
    threshold: float


@dataclass(frozen=True)
class PipelineConfig:
    data: DataPaths
    model: ModelConfig
    risk_gates: Dict[str, Any]
    event: EventConfig


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
        "prior_weight": 0.35,
        # Autumn drift prior bounds (USD/gal per day)
        "nowcast_drift_bounds": [-0.004, 0.0],
    },
    "risk_gates": {
        "nhc_active_threshold": 1,
        # Align with competition plan: draw > 3.0 mmbbl (i.e., change <= -3.0)
        "wpsr_inventory_cutoff": -3.0,
    },
    "event": {
        "name": "AAA National Average Regular > $3.10 on Oct 31, 2025",
        "resolution_date": "2025-10-31",
        "threshold": 3.10,
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

    drift_bounds = model_cfg.get("nowcast_drift_bounds")
    nowcast_drift_bounds: tuple[float, float] | None
    if isinstance(drift_bounds, (list, tuple)) and len(drift_bounds) == 2:
        nowcast_drift_bounds = (float(drift_bounds[0]), float(drift_bounds[1]))
    else:
        nowcast_drift_bounds = None

    event_cfg = config_data.get("event", {})
    event_name = str(event_cfg.get("name", DEFAULT_CONFIG["event"]["name"]))
    raw_resolution = event_cfg.get(
        "resolution_date", DEFAULT_CONFIG["event"]["resolution_date"]
    )
    if isinstance(raw_resolution, (datetime, date)):
        resolution_date = (
            raw_resolution.date()
            if isinstance(raw_resolution, datetime)
            else raw_resolution
        )
    elif isinstance(raw_resolution, str):
        resolution_date = datetime.strptime(raw_resolution, "%Y-%m-%d").date()
    else:
        raise ValueError("event.resolution_date must be YYYY-MM-DD or date instance")
    event_threshold = float(
        event_cfg.get("threshold", DEFAULT_CONFIG["event"]["threshold"])
    )

    return PipelineConfig(
        data=data_paths,
        model=ModelConfig(
            ensemble_weights=model_cfg["ensemble_weights"],
            calibration_bins=int(model_cfg["calibration_bins"]),
            horizon_days=int(model_cfg["horizon_days"]),
            prior_weight=float(model_cfg.get("prior_weight", 0.35)),
            nowcast_drift_bounds=nowcast_drift_bounds,
        ),
        risk_gates=risk_cfg,
        event=EventConfig(
            name=event_name,
            resolution_date=resolution_date,
            threshold=event_threshold,
        ),
    )
