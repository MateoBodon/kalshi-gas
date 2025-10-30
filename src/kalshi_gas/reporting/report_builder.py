"""Create markdown report with forecast outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import subprocess
from pathlib import Path
from typing import Dict

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from kalshi_gas.risk.gates import RiskGateResult


class ReportBuilder:
    def __init__(self, template_dir: Path | None = None):
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(default_for_string=False, default=False),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    @staticmethod
    def _git_sha() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    def build(
        self,
        metrics: Dict[str, float],
        risk: RiskGateResult,
        calibration: pd.DataFrame,
        figures: Dict[str, str],
        posterior: Dict[str, float],
        sensitivity: pd.DataFrame,
        risk_flags: Dict[str, object],
        risk_context: Dict[str, object] | None,
        provenance: list[dict[str, object]] | None,
        benchmarks: list[dict[str, object]] | None,
        sensitivity_bars: pd.DataFrame | None,
        asymmetry_ci: tuple[float, float, float] | None,
        jackknife: str | None,
        meta_files: list[str] | None,
        output_path: Path,
        as_of: str | None = None,
    ) -> Path:
        template = self.env.get_template("report.md.j2")
        figures = {key: str(value) for key, value in figures.items()}
        metrics_table = {
            key: value for key, value in metrics.items() if value is not None
        }
        provenance_records = provenance or []
        benchmark_rows = benchmarks or []
        sensitivity_bar_rows = (
            sensitivity_bars.to_dict(orient="records")
            if isinstance(sensitivity_bars, pd.DataFrame)
            else []
        )
        as_of_label = as_of or "n/a"
        figure_footer = (
            f"As of {as_of_label} • Sources: AAA (daily), EIA WPSR (Wed 10:30 ET)"
        )
        submission_sha = self._git_sha()
        content = template.render(
            metrics=metrics_table,
            risk=risk,
            calibration=calibration.fillna(0).to_dict(orient="records"),
            posterior=posterior,
            sensitivity=sensitivity.to_dict(orient="records"),
            risk_flags=risk_flags,
            risk_context=risk_context or {},
            provenance=provenance_records,
            benchmarks=benchmark_rows,
            sensitivity_bars=sensitivity_bar_rows,
            asymmetry_ci=asymmetry_ci,
            jackknife=jackknife,
            meta_files=meta_files or [],
            figures=figures,
            figure_footer=figure_footer,
            as_of_label=as_of_label,
            metadata={
                "generated_at": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            },
            submission_sha=submission_sha,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return output_path

    def build_deck(
        self,
        posterior: Dict[str, float],
        risk_flags: Dict[str, object],
        risk_context: Dict[str, object] | None,
        benchmarks: list[dict[str, object]] | None,
        figures: Dict[str, str],
        provenance: list[dict[str, object]] | None,
        sensitivity_bars: list[dict[str, object]] | None,
        headline_threshold: float | None,
        headline_probability: float | None,
        headline_date: str | None,
        asymmetry_ci: tuple[float, float, float] | None,
        jackknife: str | None,
        output_path: Path,
    ) -> Path:
        template = self.env.get_template("deck.md.j2")
        figures = {key: str(value) for key, value in figures.items()}
        submission_sha = self._git_sha()
        content = template.render(
            metadata={
                "generated_at": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            },
            posterior=posterior,
            risk_flags=risk_flags,
            risk_context=risk_context or {},
            benchmarks=benchmarks or [],
            figures=figures,
            provenance=provenance or [],
            sensitivity_bars=sensitivity_bars or [],
            headline_threshold=headline_threshold,
            headline_probability=headline_probability,
            headline_date=headline_date,
            asymmetry_ci=asymmetry_ci,
            jackknife=jackknife,
            submission_sha=submission_sha,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return output_path
