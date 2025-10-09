"""Visualization helpers for report generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_price_forecast(frame: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frame["date"], frame["actual"], label="Actual future price")
    ax.plot(frame["date"], frame["ensemble_mean"], label="Ensemble forecast")
    ax.set_ylabel("USD per gallon")
    ax.set_title("Ensemble Forecast vs Actual")
    ax.legend()
    fig.autofmt_xdate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_calibration(calibration: pd.DataFrame, output_path: Path) -> Path:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
