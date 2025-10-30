"""Command-line interface for the kalshi_gas pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from kalshi_gas.pipeline import run_pipeline
from kalshi_gas.pipeline.backtest import run_freeze_backtest


def run_report(config_path: str | None = None, *, force: bool = False) -> None:
    result = run_pipeline(config_path=config_path, force=force)
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

    dataset_ctx = (
        result.get("risk_context", {}).get("dataset", {})
        if isinstance(result.get("risk_context"), dict)
        else {}
    )
    dataset_as_of = dataset_ctx.get("as_of") or "n/a"
    posterior = result.get("posterior_summary", {}) or {}
    event_threshold = posterior.get("event_threshold")
    prob_key = None
    if isinstance(event_threshold, (int, float)):
        prob_key = f"prob_gt_{float(event_threshold):.2f}"
    final_prob = posterior.get(prob_key) if prob_key else None
    prior_weight = result.get("prior_weight")
    latest_inputs = result.get("latest_inputs", {}) or {}

    def _fmt_latest(key: str) -> str:
        entry = latest_inputs.get(key)
        if not isinstance(entry, dict):
            return "n/a"
        date_val = entry.get("date", "n/a")
        value_val = entry.get("value")
        if isinstance(value_val, (int, float)):
            return f"{date_val} @ {value_val:.4f}"
        return f"{date_val} @ {value_val}"

    sample_fallback_used = bool(result.get("sample_fallback_used", False))
    memo_path = Path("reports") / "memo.md"
    figures_dir = Path("reports") / "figures"

    print("\nFinal Summary:")
    print(f"- As-of date: {dataset_as_of}")
    print(f"- Latest AAA: {_fmt_latest('aaa')}")
    print(f"- Latest EIA: {_fmt_latest('eia')}")
    print(f"- Latest RBOB: {_fmt_latest('rbob')}")
    if isinstance(prior_weight, (int, float)):
        print(f"- Prior weight: {prior_weight:.4f}")
    else:
        print("- Prior weight: n/a")
    if isinstance(final_prob, (int, float)) and isinstance(
        event_threshold, (int, float)
    ):
        print(f"- Final P(AAA > {event_threshold:.2f} on 2025-10-31): {final_prob:.2%}")
    else:
        print("- Final probability: n/a")
    print(f"- Sample fallback used: {'true' if sample_fallback_used else 'false'}")
    print(f"- Memo path: {memo_path}")
    print(f"- Figures directory: {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi gas analytics pipeline")
    subparsers = parser.add_subparsers(dest="command")

    report_parser = subparsers.add_parser("report", help="Run end-to-end report build")
    report_parser.add_argument(
        "--config", type=str, help="Path to YAML config", default=None
    )
    report_parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of processed artifacts from data_raw inputs",
    )

    freeze_parser = subparsers.add_parser(
        "freeze-backtest", help="Run freeze-date backtest snapshot"
    )
    freeze_parser.add_argument(
        "--config", type=str, help="Path to YAML config", default=None
    )

    args = parser.parse_args()
    if args.command == "report":
        run_report(args.config, force=getattr(args, "force", False))
    elif args.command == "freeze-backtest":
        result = run_freeze_backtest(config_path=args.config)
        print("Backtest metrics written to", result["metrics_path"])
        print("Calibration figure saved to", result["calibration_path"])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
