"""Risk gating logic for operational overrides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import yaml

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


def load_nhc_flag(path: Path) -> bool:
    if not path.exists():
        return False
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return bool(payload.get("flag", False))


def load_wpsr_state(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload


def nhc_gate(config: PipelineConfig) -> tuple[bool, dict]:
    threshold = int(config.risk_gates.get("nhc_active_threshold", 1))
    fallback_path = Path("data/sample/nhc_outlook.csv")
    flag_path = Path("data_raw/nhc_flag.yml")

    frame = load_nhc_activity(fallback_path)
    latest = frame.iloc[-1]

    analyst_flag = load_nhc_flag(flag_path)

    alert = bool(latest["active_storms"] >= threshold)
    alert = alert or analyst_flag
    return alert, {
        "latest_date": latest["date"],
        "active_storms": int(latest["active_storms"]),
        "threshold": threshold,
        "analyst_flag": analyst_flag,
    }


def wpsr_gate(
    dataset: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[bool, dict]:
    threshold = float(config.risk_gates.get("wpsr_inventory_cutoff", -1.5))
    latest_change = float(dataset["inventory_change"].iloc[-1])
    draw = max(0.0, -latest_change)

    state_path = Path("data_raw/wpsr_state.yml")
    state = load_wpsr_state(state_path)
    refinery_util = float(state.get("refinery_util_pct", 92.0))
    product_supplied = float(state.get("product_supplied_mbd", 8.8))

    alert = latest_change <= threshold
    return alert, {
        "latest_change": latest_change,
        "threshold": threshold,
        "gasoline_stocks_draw": draw,
        "refinery_util_pct": refinery_util,
        "product_supplied_mbd": product_supplied,
    }


def evaluate_risk(dataset: pd.DataFrame, config: PipelineConfig) -> RiskGateResult:
    nhc_alert, nhc_details = nhc_gate(config)
    wpsr_alert, wpsr_details = wpsr_gate(dataset, config)
    return RiskGateResult(
        nhc_alert=nhc_alert,
        wpsr_alert=wpsr_alert,
        details={"nhc": nhc_details, "wpsr": wpsr_details},
    )
