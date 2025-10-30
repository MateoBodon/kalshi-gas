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
    sensitivity: pd.DataFrame,
    output_path: Path,
    *,
    as_of: str | None = None,
    source: str | None = None,
) -> Path:
    if sensitivity.empty:
        frame = pd.DataFrame(columns=["threshold", "scenario", "delta"])
    else:
        base = sensitivity[
            (sensitivity["rbob_delta"].abs() < 1e-9)
            & (sensitivity["alpha_delta"].abs() < 1e-9)
        ].set_index("threshold")["prob_above"]
        thresholds = sorted(base.index.unique())
        scenarios = [
            ("RBOB -5¢", -0.05, 0.0),
            ("RBOB +5¢", 0.05, 0.0),
            ("Alpha -5¢", 0.0, -0.05),
            ("Alpha +5¢", 0.0, 0.05),
        ]
        records: list[dict[str, float | str]] = []
        for threshold in thresholds:
            base_prob = float(base.get(threshold, np.nan))
            for label, rbob_delta, alpha_delta in scenarios:
                mask = (
                    np.isclose(sensitivity["threshold"], threshold)
                    & np.isclose(sensitivity["rbob_delta"], rbob_delta)
                    & np.isclose(sensitivity["alpha_delta"], alpha_delta)
                )
                if not mask.any():
                    continue
                prob = float(sensitivity.loc[mask, "prob_above"].iloc[0])
                records.append(
                    {
                        "threshold": threshold,
                        "scenario": label,
                        "delta": prob - base_prob,
                    }
                )
        frame = pd.DataFrame(records)

    if frame.empty:
        frame = pd.DataFrame(
            {
                "threshold": [],
                "scenario": [],
                "delta": [],
            }
        )

    scenarios_order = ["RBOB -5¢", "RBOB +5¢", "Alpha -5¢", "Alpha +5¢"]
    colors = {
        "RBOB -5¢": "#1f77b4",
        "RBOB +5¢": "#1f77b4",
        "Alpha -5¢": "#ff7f0e",
        "Alpha +5¢": "#ff7f0e",
    }

    unique_thresholds = sorted(frame["threshold"].unique())
    fig, ax = plt.subplots(figsize=(6.5, 4))

    if not unique_thresholds:
        ax.text(0.5, 0.5, "No sensitivity data", ha="center", va="center")
        ax.axis("off")
    else:
        y_positions: list[float] = []
        y_labels: list[str] = []
        spacing = len(scenarios_order) + 1
        for idx, threshold in enumerate(unique_thresholds):
            block = frame[frame["threshold"] == threshold]
            base_y = idx * spacing
            for offset, scenario in enumerate(scenarios_order):
                row = block[block["scenario"] == scenario]
                if row.empty:
                    continue
                y = base_y + offset
                delta = float(row["delta"].iloc[0])
                ax.barh(
                    y,
                    delta,
                    color=colors.get(scenario, "#333333"),
                    alpha=0.85,
                    height=0.8,
                )
                y_positions.append(y)
                y_labels.append(f"> {threshold:.2f} – {scenario}")

        ax.axvline(0, color="#444444", linewidth=1, alpha=0.6)
        if y_positions:
            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels, fontsize=8)
            ax.set_ylim(min(y_positions) - 1, max(y_positions) + 1)
        ax.grid(axis="x", alpha=0.2)

    ax.set_xlabel("Δ Probability")
    ax.set_title("Sensitivity: ΔP(Yes) for ±$0.05 RBOB / ±5¢ α")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _draw_footer(fig, as_of, source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pass_through_fit(
    data: pd.DataFrame,
    structural: dict,
    output_path: Path,
    *,
    as_of: str | None = None,
    source: str | None = None,
) -> Path:
    """Scatter ΔRetail vs lagged ΔRBOB with fitted beta annotations."""
    lag = int(structural.get("lag", 0) or 0)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if lag <= 0 or len(data) <= lag:
        ax.text(0.5, 0.5, "No structural fit available", ha="center", va="center")
        ax.axis("off")
    else:
        df = data.copy()
        df["delta_price"] = df["regular_gas_price"].astype(float).diff()
        df["delta_rbob_lag"] = df["rbob_settle"].astype(float).diff().shift(lag)
        df = df.dropna()
        ax.scatter(
            df["delta_rbob_lag"], df["delta_price"], s=12, alpha=0.6, color="#1f77b4"
        )
        beta = structural.get("beta")
        bu = structural.get("beta_up")
        bd = structural.get("beta_dn")
        if beta is not None:
            xs = np.linspace(
                df["delta_rbob_lag"].min(), df["delta_rbob_lag"].max(), 100
            )
            ax.plot(xs, float(beta) * xs, color="#d62728", label=f"β={float(beta):.3f}")
        if bu is not None and bd is not None:
            ax.legend(title=f"β↑={float(bu):.3f}, β↓={float(bd):.3f} (L={lag}d)")
        ax.set_xlabel(f"Δ RBOB (lag {lag}d)")
        ax.set_ylabel("Δ Retail (USD/gal)")
        ax.set_title("Pass-through: ΔRetail vs lagged ΔRBOB")
        ax.grid(alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _draw_footer(fig, as_of, source)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_prior_cdf(
    thresholds: np.ndarray | list[float],
    cdf_values: np.ndarray | list[float],
    output_path: Path,
    *,
    as_of: str | None = None,
    source: str | None = None,
) -> Path:
    """Render market-implied prior CDF over thresholds."""
    x = np.asarray(thresholds, dtype=float)
    y = np.asarray(cdf_values, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o", color="#9467bd")
    ax.set_xlabel("Price threshold (USD/gal)")
    ax.set_ylabel("CDF")
    ax.set_title("Market-implied Prior CDF")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _draw_footer(fig, as_of, source)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_posterior_density(
    samples: np.ndarray | list[float],
    threshold: float | None,
    output_path: Path,
    *,
    as_of: str | None = None,
    source: str | None = None,
) -> Path:
    """Plot posterior sample density with event threshold marker."""
    draws = np.asarray(samples, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(draws, ax=ax, fill=True, alpha=0.5, color="#2ca02c")
    ax.set_xlabel("Price (USD/gal)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Density")
    if threshold is not None:
        ax.axvline(
            float(threshold),
            color="#d62728",
            linestyle="--",
            label=f"> {float(threshold):.2f}",
        )
        ax.legend()
    ax.grid(alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _draw_footer(fig, as_of, source)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
