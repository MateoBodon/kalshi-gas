"""Risk gating logic for operational overrides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kalshi_gas.config import PipelineConfig


@dataclass
class RiskGateResult:
    nhc_alert: bool
    wpsr_alert: bool
    details: dict


def load_nhc_activity(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    frame.sort_values("date", inplace=True)
    return frame


def nhc_gate(config: PipelineConfig) -> tuple[bool, dict]:
    threshold = int(config.risk_gates.get("nhc_active_threshold", 1))
    fallback_path = Path("data/sample/nhc_outlook.csv")
    frame = load_nhc_activity(fallback_path)
    latest = frame.iloc[-1]
    alert = bool(latest["active_storms"] >= threshold)
    return alert, {
        "latest_date": latest["date"],
        "active_storms": int(latest["active_storms"]),
        "threshold": threshold,
    }


def wpsr_gate(
    dataset: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[bool, dict]:
    threshold = float(config.risk_gates.get("wpsr_inventory_cutoff", -1.5))
    latest_change = float(dataset["inventory_change"].iloc[-1])
    alert = latest_change <= threshold
    return alert, {
        "latest_change": latest_change,
        "threshold": threshold,
    }


def evaluate_risk(dataset: pd.DataFrame, config: PipelineConfig) -> RiskGateResult:
    nhc_alert, nhc_details = nhc_gate(config)
    wpsr_alert, wpsr_details = wpsr_gate(dataset, config)
    return RiskGateResult(
        nhc_alert=nhc_alert,
        wpsr_alert=wpsr_alert,
        details={"nhc": nhc_details, "wpsr": wpsr_details},
    )
