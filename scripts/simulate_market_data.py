"""Generate synthetic but structured market data for offline experimentation."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kalshi_gas.config import load_config
from kalshi_gas.etl.utils import save_snapshot


def _date_range(start: date, end: date) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="D")


def _trend(values: tuple[float, float, float], length: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length)
    return np.interp(t, [0.0, 0.55, 1.0], values)


def simulate_aaa_prices(
    start: date, end: date, *, seed: int, event_threshold: float
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _date_range(start, end)
    trend = _trend((3.42, 3.02, event_threshold + 0.04), len(dates))
    seasonal = 0.025 * np.sin(2 * np.pi * np.linspace(0, 6, len(dates)))
    noise = rng.normal(0.0, 0.012, len(dates))
    prices = trend + seasonal + noise
    prices = np.clip(prices, 2.65, 4.25)
    return pd.DataFrame(
        {
            "date": dates,
            "regular_gas_price": np.round(prices, 3),
        }
    )


def simulate_rbob(aaa: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    settle = 0.58 * aaa["regular_gas_price"].to_numpy() - 0.45
    settle += rng.normal(0.0, 0.018, len(settle))
    settle = np.clip(settle, 1.35, 2.35)
    return pd.DataFrame(
        {
            "date": aaa["date"],
            "settle": np.round(settle, 4),
            "source": "simulated",
        }
    )


def simulate_eia(start: date, end: date, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="W-WED")
    base_inventory = 222 + 4 * np.sin(np.linspace(0, 3, len(dates)))
    inventory = base_inventory + rng.normal(0, 0.6, len(dates))
    production = 9100 + 250 * np.cos(np.linspace(0, 3, len(dates)))
    production += rng.normal(0, 60, len(dates))
    frame = pd.DataFrame(
        {
            "date": dates,
            "inventory_mmbbl": np.round(inventory, 3),
            "production_mbd": np.round(production, 0),
        }
    )
    return frame


def simulate_kalshi(
    aaa: pd.DataFrame,
    *,
    seed: int,
    threshold: float,
    event_ticker: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = aaa["regular_gas_price"].to_numpy()
    bias = np.linspace(-0.25, 0.2, len(prices))
    logits = 10 * (prices - threshold) + bias
    noise = rng.normal(0.0, 0.4, len(prices))
    probs = 1.0 / (1.0 + np.exp(-(logits + noise)))
    probs = np.clip(probs, 0.02, 0.97)
    return pd.DataFrame(
        {
            "date": aaa["date"],
            "market": event_ticker,
            "prob_yes": np.round(probs, 4),
        }
    )


def write_kalshi_bins(path: Path, *, threshold: float, final_price: float) -> None:
    delta = 0.05
    thresholds = [
        threshold - 0.10,
        threshold - delta,
        threshold,
        threshold + delta,
    ]
    slope = 15
    logits = [slope * (final_price - thr) for thr in thresholds]
    probabilities = [float(1.0 / (1.0 + np.exp(-logit))) for logit in logits]
    payload = {
        "thresholds": thresholds,
        "probabilities": probabilities,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate offline market datasets.")
    parser.add_argument(
        "--start",
        type=lambda s: date.fromisoformat(s),
        default=date(2024, 1, 1),
        help="Start date for the synthetic series (default: 2024-01-01).",
    )
    parser.add_argument(
        "--end",
        type=lambda s: date.fromisoformat(s),
        default=date(2025, 10, 24),
        help="End date for the synthetic series (default: 2025-10-24).",
    )
    parser.add_argument(
        "--event-date",
        type=lambda s: date.fromisoformat(s),
        default=date(2025, 10, 31),
        help="Event resolution date (default: 2025-10-31).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.10,
        help="Kalshi event threshold in USD/gal (default: 3.10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=314159,
        help="Random seed for reproducibility (default: 314159).",
    )
    parser.add_argument(
        "--event-ticker",
        type=str,
        default="GAS_PRICE_ABOVE_3_10_2025OCT31",
        help="Synthetic Kalshi market ticker identifier.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start >= args.end:
        raise SystemExit("start date must be earlier than end date")

    cfg = load_config()
    raw_dir = cfg.data.raw_dir

    aaa = simulate_aaa_prices(
        args.start, args.end, seed=args.seed, event_threshold=args.threshold
    )
    rbob = simulate_rbob(aaa, seed=args.seed + 7)
    eia = simulate_eia(args.start, args.end, seed=args.seed + 17)
    kalshi = simulate_kalshi(
        aaa,
        seed=args.seed + 29,
        threshold=args.threshold,
        event_ticker=args.event_ticker,
    )

    # Persist last_good snapshots
    save_snapshot(
        aaa,
        raw_dir / "last_good.aaa.csv",
        metadata={
            "source": "aaa",
            "mode": "simulated",
            "as_of": aaa["date"].max().date().isoformat(),
        },
    )
    save_snapshot(
        rbob.drop(columns=["source"]),
        raw_dir / "last_good.rbob.csv",
        metadata={
            "source": "rbob",
            "mode": "simulated",
            "as_of": rbob["date"].max().date().isoformat(),
        },
    )
    save_snapshot(
        eia,
        raw_dir / "last_good.eia.csv",
        metadata={
            "source": "eia",
            "mode": "simulated",
            "as_of": eia["date"].max().date().isoformat(),
        },
    )
    save_snapshot(
        kalshi,
        raw_dir / "last_good.kalshi.csv",
        metadata={
            "source": "kalshi",
            "mode": "simulated",
            "as_of": kalshi["date"].max().date().isoformat(),
        },
    )

    # Update editable caches for manual workflows
    rbob_path = Path("data_raw/rbob_daily.csv")
    rbob_path.parent.mkdir(parents=True, exist_ok=True)
    rbob.to_csv(rbob_path, index=False, date_format="%Y-%m-%d")

    kalshi_bins_path = Path("data_raw/kalshi_bins.yml")
    write_kalshi_bins(
        kalshi_bins_path,
        threshold=args.threshold,
        final_price=float(aaa["regular_gas_price"].iloc[-1]),
    )

    # Risk state defaults
    latest_change = float(
        eia["inventory_mmbbl"].diff().dropna().iloc[-1] if len(eia) > 1 else -3.2
    )
    with Path("data_raw/wpsr_state.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "latest_change": latest_change,
                "gasoline_stocks_draw": max(0.0, -latest_change),
                "refinery_util_pct": float(88.5),
                "product_supplied_mbd": float(8.45),
                "as_of": eia["date"].max().date().isoformat(),
            },
            handle,
            sort_keys=False,
        )
    with Path("data_raw/nhc_flag.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"flag": False, "note": "simulated baseline"},
            handle,
            sort_keys=False,
        )

    print(
        "Simulated datasets written to data_raw/. Run `python3 -m kalshi_gas.cli report` "
        "to rebuild the report with the new offline state."
    )


if __name__ == "__main__":
    main()
