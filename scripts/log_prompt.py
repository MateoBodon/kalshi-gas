"""Interactive prompt for manual AAA/EIA/GasBuddy logging."""

from __future__ import annotations

from datetime import datetime

from scripts.log_manual_label import append_log


def _prompt_float(label: str) -> float | None:
    raw = input(f"{label} (blank to skip): ").strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        print("Invalid number, skipping.")
        return None


def main() -> None:
    print("Manual resolver logging — values should come from human observation.")
    timestamp_str = input(
        "Timestamp ISO (blank for current UTC, e.g. 2025-10-26T12:45:00): "
    ).strip()
    timestamp = (
        datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.utcnow()
    )
    aaa = _prompt_float("AAA national regular (USD/gal)")
    eia = _prompt_float("EIA Today-in-Energy AAA (USD/gal)")
    gasbuddy = _prompt_float("GasBuddy national (USD/gal)")
    operator = input("Operator initials: ").strip() or "UNKNOWN"
    notes = input("Notes (e.g., AM check, link): ").strip()

    append_log(
        timestamp=timestamp,
        aaa=aaa,
        eia=eia,
        gasbuddy=gasbuddy,
        operator=operator,
        notes=notes,
    )
    print("Entry logged.")


if __name__ == "__main__":
    main()
