"""Command-line interface for the kalshi_gas pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from kalshi_gas.backtest.evaluate import run_backtest
from kalshi_gas.config import load_config
from kalshi_gas.data.assemble import assemble_dataset
from kalshi_gas.etl.pipeline import run_all_etl
from kalshi_gas.models.ensemble import EnsembleModel
from kalshi_gas.reporting.report_builder import ReportBuilder
from kalshi_gas.reporting.visuals import plot_calibration, plot_price_forecast
from kalshi_gas.risk.gates import evaluate_risk


def run_report(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    cfg.data.build_dir.mkdir(parents=True, exist_ok=True)

    etl_outputs = run_all_etl(cfg)

    dataset = assemble_dataset(cfg)
    ensemble = EnsembleModel(weights=cfg.model.ensemble_weights)
    backtest = run_backtest(dataset, ensemble)
    risk = evaluate_risk(dataset, cfg)

    figures_dir = cfg.data.build_dir / "figures"
    memo_dir = cfg.data.build_dir / "memo"
    figures_dir.mkdir(parents=True, exist_ok=True)
    memo_dir.mkdir(parents=True, exist_ok=True)

    forecast_fig = figures_dir / "forecast_vs_actual.png"
    calibration_fig = figures_dir / "calibration.png"
    plot_price_forecast(backtest.test_frame, forecast_fig)
    plot_calibration(backtest.calibration, calibration_fig)

    results_csv = memo_dir / "forecast_results.csv"
    backtest.test_frame.to_csv(results_csv, index=False)

    builder = ReportBuilder()
    report_path = memo_dir / "report.md"
    builder.build(
        metrics=backtest.metrics,
        risk=risk,
        calibration=backtest.calibration,
        figures={
            "forecast": str(Path("..") / "figures" / forecast_fig.name),
            "calibration": str(Path("..") / "figures" / calibration_fig.name),
        },
        output_path=report_path,
    )

    print("ETL outputs:", etl_outputs)
    print("Forecast results saved to", results_csv)
    print("Report generated at", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi gas analytics pipeline")
    subparsers = parser.add_subparsers(dest="command")

    report_parser = subparsers.add_parser("report", help="Run end-to-end report build")
    report_parser.add_argument("--config", type=str, help="Path to YAML config", default=None)

    args = parser.parse_args()
    if args.command == "report":
        run_report(args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
