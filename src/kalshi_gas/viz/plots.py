"""Plotting helpers for dashboards."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _format_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_fundamentals_dashboard(
    dataset: pd.DataFrame,
    risk_flags: dict,
    output_path: Path,
) -> Path:
    """Render fundamentals dashboard with stocks/utilization/product supplied."""

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

    stocks_ax, util_ax, demand_ax = axes

    if "date" in dataset and "inventory_mmbbl" in dataset:
        stocks_ax.plot(
            dataset["date"],
            dataset["inventory_mmbbl"],
            color="#0B5394",
            linewidth=2,
            label="Gasoline Stocks",
        )
        stocks_ax.set_ylabel("MMbbl")
        stocks_ax.set_title("Motor Gasoline Stocks")
        stocks_ax.legend(loc="upper left")
        stocks_ax.grid(alpha=0.2)
    _format_axes(stocks_ax)

    tightness = risk_flags.get("tightness", {}) if isinstance(risk_flags, dict) else {}
    nhc = risk_flags.get("nhc", {}) if isinstance(risk_flags, dict) else {}

    refinery_util = float(tightness.get("refinery_util_pct", np.nan))
    util_ax.bar(
        ["Refinery Utilization"],
        [refinery_util],
        color="#38761D",
    )
    util_ax.set_ylim(0, 100)
    util_ax.set_ylabel("Percent")
    util_ax.set_title("Refinery Utilization")
    util_ax.grid(axis="y", alpha=0.2)
    util_ax.bar_label(util_ax.containers[0], fmt="%.1f%%")
    _format_axes(util_ax)

    product_supplied = float(tightness.get("product_supplied_mbd", np.nan))
    demand_ax.bar(
        ["Product Supplied"],
        [product_supplied],
        color="#CC0000",
    )
    demand_ax.set_ylabel("Mb/d")
    demand_ax.set_title("Finished Motor Gasoline Product Supplied")
    demand_ax.grid(axis="y", alpha=0.2)
    demand_ax.bar_label(demand_ax.containers[0], fmt="%.2f")
    _format_axes(demand_ax)

    alerts = []
    if tightness.get("active"):
        alerts.append("Tightness flag active")
    if nhc.get("active"):
        alerts.append("NHC flag active")
    if alerts:
        fig.suptitle("Fundamentals Dashboard (Risk Adjusted)", fontsize=14)
        fig.text(0.01, 0.96, " / ".join(alerts), fontsize=9, color="#990000")
    else:
        fig.suptitle("Fundamentals Dashboard", fontsize=14)

    fig.text(
        0.5,
        0.01,
        "Sources: AAA Daily Survey, EIA Weekly Petroleum Status Report, CME RBOB Futures, Kalshi",
        ha="center",
        fontsize=8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
