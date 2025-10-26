"""Update the NHC analyst override flag for risk gating."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

FLAG_PATH = Path("data_raw/nhc_flag.yml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set the NHC operational and analyst flags for the risk gate."
    )
    parser.add_argument(
        "--flag",
        action="store_true",
        help="Mark the analyst override flag as true (default false).",
    )
    parser.add_argument(
        "--active-storms",
        type=int,
        default=None,
        help="Optional count of active storms used in dashboards.",
    )
    parser.add_argument(
        "--note",
        type=str,
        help="Optional free-form note to include with the flag.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=FLAG_PATH,
        help="YAML file to update (default: data_raw/nhc_flag.yml).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    payload = {"flag": bool(args.flag)}
    if args.active_storms is not None:
        payload["active_storms"] = int(args.active_storms)
    if args.note:
        payload["note"] = args.note

    args.path.parent.mkdir(parents=True, exist_ok=True)
    args.path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(
        f"Updated {args.path} (flag={'ON' if args.flag else 'OFF'}, "
        f"active_storms={payload.get('active_storms', 'n/a')})"
    )


if __name__ == "__main__":
    main()
