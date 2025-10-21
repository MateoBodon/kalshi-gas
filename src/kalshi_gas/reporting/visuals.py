"""Visualization helpers for report generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _footer_text(as_of: str | None, source: str | None) -> str | None:
    parts: list[str] = []
    if as_of:
        parts.append(f"As of: {as_of}")
    if source:
        parts.append(f"Source: {source}")
    if not parts:
        return None
    return " • ".join(parts)


def _draw_footer(fig, as_of: str | None, source: str | None) -> None:
    footer = _footer_text(as_of, source)
    if footer:
        fig.text(
            0.99,
            0.02,
            footer,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )


def plot_price_forecast(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    as_of: str | None = None,
    source: str | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frame["date"], frame["actual"], label="Actual future price")
    ax.plot(frame["date"], frame["ensemble_mean"], label="Ensemble forecast")
    ax.set_ylabel("USD per gallon")
    ax.set_title("Ensemble Forecast vs Actual")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _draw_footer(fig, as_of, source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_calibration(
    calibration: pd.DataFrame,
    output_path: Path,
    *,
    as_of: str | None = None,
    source: str | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        calibration["forecast_mean"],
        calibration["outcome_rate"],
        marker="o",
        label="Observed",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _draw_footer(fig, as_of, source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_risk_box(
    risk_flags: dict,
    metrics: dict,
    output_path: Path,
    *,
    as_of: str | None = None,
    source: str | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.axis("off")
    tight = risk_flags.get("tightness", {}) if isinstance(risk_flags, dict) else {}
    nhc = risk_flags.get("nhc", {}) if isinstance(risk_flags, dict) else {}
    adjustments = (
        risk_flags.get("adjustments", []) if isinstance(risk_flags, dict) else []
    )

    lines = [
        f"Tightness: {'ON' if tight.get('active') else 'OFF'} | Draw {tight.get('gasoline_stocks_draw', 'n/a')} mmbbl",
        f"Refinery Util: {tight.get('refinery_util_pct', 'n/a')}%",
        f"Product Supplied: {tight.get('product_supplied_mbd', 'n/a')} Mb/d",
        f"NHC Flag: {'ON' if nhc.get('active') else 'OFF'} | Alert {'ON' if nhc.get('nhc_alert') else 'OFF'} | Analyst {'ON' if nhc.get('analyst_flag') else 'OFF'}",
    ]
    if adjustments:
        lines.append("Tail Adj: " + "; ".join(str(adj) for adj in adjustments))
    prior_weight = (
        metrics.get("prior_weight_calibrated") if isinstance(metrics, dict) else None
    )
    if prior_weight is not None:
        lines.append(f"Prior Weight: {prior_weight:.2f} (calibrated)")
    brier = metrics.get("brier_score") if isinstance(metrics, dict) else None
    if brier is not None:
        lines.append(f"Ensemble Brier: {brier:.3f}")

    text = "\n".join(lines)
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=10, wrap=True)
    _draw_footer(fig, as_of, source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sensitivity_bars(
    sensitivity_bars: pd.DataFrame,
    output_path: Path,
    *,
    as_of: str | None = None,
    source: str | None = None,
) -> Path:
    frame = sensitivity_bars.copy()
    if frame.empty:
        frame = pd.DataFrame(
            {"threshold": [], "prob_min": [], "prob_max": [], "prob_base": []}
        )
    frame = frame.sort_values("threshold").reset_index(drop=True)
    y_positions = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, row in frame.iterrows():
        y = y_positions[idx]
        prob_min = row.get("prob_min")
        prob_max = row.get("prob_max")
        prob_base = row.get("prob_base")
        if pd.notna(prob_min) and pd.notna(prob_max):
            ax.hlines(y, prob_min, prob_max, color="#1f77b4", linewidth=3, alpha=0.7)
        if pd.notna(prob_base):
            ax.scatter(prob_base, y, color="#d62728", zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"≥ {row:.2f}" for row in frame["threshold"]])
    ax.set_xlabel("Probability")
    ax.set_title("Posterior Sensitivity Range")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _draw_footer(fig, as_of, source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
