"""Update the analyst WPSR state YAML used for risk gating."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml

STATE_PATH = Path("data_raw/wpsr_state.yml")


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if math.isnan(parsed):
        raise argparse.ArgumentTypeError("value must be a number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh the WPSR analyst state used by the risk gate."
    )
    parser.add_argument(
        "--latest-change",
        type=positive_float,
        required=True,
        help="Weekly inventory change in MMbbl (negative for draws).",
    )
    parser.add_argument(
        "--refinery-util",
        type=positive_float,
        required=True,
        help="Refinery utilisation percentage.",
    )
    parser.add_argument(
        "--product-supplied",
        type=positive_float,
        required=True,
        help="Finished motor gasoline product supplied (Mb/d).",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        help="Optional ISO date string for the data pull.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=STATE_PATH,
        help="Destination YAML path (default: data_raw/wpsr_state.yml).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    draw = max(0.0, -args.latest_change)
    payload = {
        "latest_change": args.latest_change,
        "gasoline_stocks_draw": draw,
        "refinery_util_pct": args.refinery_util,
        "product_supplied_mbd": args.product_supplied,
    }
    if args.as_of:
        payload["as_of"] = args.as_of
    args.path.parent.mkdir(parents=True, exist_ok=True)
    args.path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(
        f"Updated {args.path} (change={args.latest_change:.2f} mmbbl, "
        f"util={args.refinery_util:.1f}%, supplied={args.product_supplied:.2f} Mb/d)"
    )


if __name__ == "__main__":
    main()
