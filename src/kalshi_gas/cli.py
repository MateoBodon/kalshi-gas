"""Command-line interface for the kalshi_gas pipeline."""

from __future__ import annotations

import argparse
from kalshi_gas.pipeline import run_pipeline


def run_report(config_path: str | None = None) -> None:
    result = run_pipeline(config_path=config_path)
    etl_outputs = result.get("etl_outputs", {})
    provenance = result.get("etl_provenance", {})
    print("ETL outputs:")
    for source, path in etl_outputs.items():
        prov = provenance.get(source) if isinstance(provenance, dict) else None
        mode = prov.get("mode") if isinstance(prov, dict) else "unknown"
        as_of = prov.get("as_of") if isinstance(prov, dict) else None
        fresh = prov.get("fresh") if isinstance(prov, dict) else None
        freshness = "fresh" if fresh else "stale" if fresh is not None else "n/a"
        as_of_display = as_of or "n/a"
        print(f"  - {source}: {path} [{mode}, as_of={as_of_display}, {freshness}]")
    print("Forecast results saved to", result["results_csv"])
    print("Sensitivity grid saved to", result["sensitivity_path"])
    if result.get("sensitivity_bars_path"):
        print("Sensitivity bars saved to", result["sensitivity_bars_path"])
    risk_flags = result.get("risk_flags", {})
    adjustments = (
        risk_flags.get("adjustments", []) if isinstance(risk_flags, dict) else []
    )
    if adjustments:
        print("Risk adjustments:", "; ".join(adjustments))
    else:
        print("Risk adjustments: none")
    print("Report generated at", result["report_path"])
    if result.get("deck_path"):
        print("Deck generated at", result["deck_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi gas analytics pipeline")
    subparsers = parser.add_subparsers(dest="command")

    report_parser = subparsers.add_parser("report", help="Run end-to-end report build")
    report_parser.add_argument(
        "--config", type=str, help="Path to YAML config", default=None
    )

    args = parser.parse_args()
    if args.command == "report":
        run_report(args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
