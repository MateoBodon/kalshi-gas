"""Create markdown report with forecast outputs."""

from __future__ import annotations

from datetime import datetime
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

    def build(
        self,
        metrics: Dict[str, float],
        risk: RiskGateResult,
        calibration: pd.DataFrame,
        figures: Dict[str, str],
        posterior: Dict[str, float],
        sensitivity: pd.DataFrame,
        risk_flags: Dict[str, object],
        output_path: Path,
    ) -> Path:
        template = self.env.get_template("report.md.j2")
        figures = {key: str(value) for key, value in figures.items()}
        content = template.render(
            metrics=metrics,
            risk=risk,
            calibration=calibration.fillna(0).to_dict(orient="records"),
            posterior=posterior,
            sensitivity=sensitivity.to_dict(orient="records"),
            risk_flags=risk_flags,
            figures=figures,
            metadata={"generated_at": datetime.utcnow().isoformat()},
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return output_path
