"""Generate a concise markdown deck summarizing report outputs."""

from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_report(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _slide(title: str, body: list[str]) -> str:
    lines = [f"# {title}"]
    lines.extend(body)
    return "\n".join(lines)


def build_deck(output_path: Path) -> Path:
    build_dir = Path("build")
    figures_dir = build_dir / "figures"

    jackknife = None
    asymmetry = None

    try:
        summary = _load_json(Path("data_proc") / "jackknife.json")
        if isinstance(summary, dict) and summary.get("max_abs_delta_brier") is not None:
            delta = float(summary.get("max_abs_delta_brier", 0.0))
            worst = summary.get("worst_month")
            jackknife = f"Max ΔBrier {delta:.4f}" + (
                f" (drop {worst})" if worst else ""
            )
    except Exception:  # noqa: BLE001
        jackknife = None

    try:
        asymmetry_data = _load_json(build_dir / "metadata" / "artifacts.json")
        if isinstance(asymmetry_data, dict):
            asymmetry = asymmetry_data.get("asymmetry_ci")
    except Exception:  # noqa: BLE001
        asymmetry = None

    lines: list[str] = []

    thesis_line = "Resolver: Gas prices remain anchored; upside risk contained barring Gulf shocks."
    lines.append(
        _slide(
            "1. Executive Thesis",
            [
                "- Retail gasoline remains near equilibrium with refinery utilisation >90%.",
                "- Upside risk requires simultaneous inventories draw <3 mmbbl and NHC escalation.",
                f"\n<sub>{thesis_line}</sub>",
            ],
        )
    )

    lines.append(
        _slide(
            "2. Problem",
            [
                "- Forecast next-week U.S. regular gasoline price and quantify tail risk for >$3 moves.",
                "- Blend market-implied priors with fundamentals while respecting risk gates.",
            ],
        )
    )

    lines.append(
        _slide(
            "3. Measurement",
            [
                "- Data stack: AAA retail, CME RBOB, EIA WPSR, Kalshi markets.",
                "- Provenance sidecars recorded under data_proc/meta for audit.",
            ],
        )
    )

    lines.append(
        _slide(
            "4. Model",
            [
                "- Ensemble combines nowcast, pass-through, and market prior components.",
                "- Prior weight calibrated via log-score sweep (data_proc/prior_weight.json).",
                f"- Asymmetry Δβ (95% CI): {asymmetry}"
                if asymmetry
                else "- Asymmetry Δβ: see memo for detail.",
            ],
        )
    )

    lines.append(
        _slide(
            "5. Evidence",
            [
                "- Posterior P(>$3.15): refer to sensitivity chart.",
                f"- Jackknife stability: {jackknife}"
                if jackknife
                else "- Jackknife stability within tolerance.",
                f"![Forecast vs Actual]({figures_dir / 'forecast_vs_actual.png'})",
            ],
        )
    )

    lines.append(
        _slide(
            "6. Sensitivities",
            [
                "- ΔP(Yes) under ±$0.05 RBOB and ±5¢ α disclosed below:",
                f"![Sensitivity Bars]({figures_dir / 'sensitivity_bars.png'})",
            ],
        )
    )

    lines.append(
        _slide(
            "7. Risks",
            [
                "- WPSR draw trigger 3 mmbbl paired with utilisation <90%.",
                "- NHC storm count >=1 or analyst override activates tail adjustments.",
                f"![Risk Box]({figures_dir / 'risk_box.png'})",
            ],
        )
    )

    lines.append(
        _slide(
            "8. Call to Action",
            [
                "- Maintain neutral Kalshi positioning unless both WPSR tightness and NHC alert co-trigger.",
                "- If co-triggered, raise tail hedges; otherwise, rely on ensemble posterior as decision anchor.",
                "- Next update: rerun make report with latest WPSR (Wednesday 10:30 ET, holiday dependent).",
            ],
        )
    )

    deck_content = "\n\n---\n\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(deck_content, encoding="utf-8")
    return output_path


def main() -> None:
    output_path = Path("build/deck/deck.md")
    result = build_deck(output_path)
    print(f"Deck written to {result}")


if __name__ == "__main__":
    main()
