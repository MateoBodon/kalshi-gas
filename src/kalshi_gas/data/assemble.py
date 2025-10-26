"""Combine processed datasets into a modeling table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kalshi_gas.config import PipelineConfig
from kalshi_gas.data.provenance import write_meta


def load_processed_frame(path: Path, date_col: str = "date") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset missing: {path}")
    return pd.read_csv(path, parse_dates=[date_col])


def assemble_dataset(config: PipelineConfig) -> pd.DataFrame:
    """Return merged time-series for modeling with nearest-week joins.

    - Left anchor on AAA daily.
    - Merge RBOB (often weekly from EIA) via asof backward within 10 days.
    - Merge EIA weekly via asof backward within 10 days.
    - Merge Kalshi daily (mean per day), asof within 2 days.
    """
    aaa_path = config.data.processed_dir / "aaa_daily.csv"
    eia_path = config.data.processed_dir / "eia_weekly.csv"
    rbob_path = config.data.processed_dir / "rbob_prices.csv"
    kalshi_path = config.data.processed_dir / "kalshi_markets.csv"

    aaa = load_processed_frame(aaa_path)
    rbob = load_processed_frame(rbob_path)
    eia = load_processed_frame(eia_path)
    kalshi = load_processed_frame(kalshi_path)

    # Normalize column names & sort
    aaa = aaa.sort_values("date")
    rbob = rbob.sort_values("date").rename(columns={"rbob_price": "rbob_settle"})
    eia = eia.sort_values("date")
    kalshi = (
        kalshi.groupby("date")
        .agg({"prob_yes": "mean"})
        .rename(columns={"prob_yes": "kalshi_prob"})
        .reset_index()
        .sort_values("date")
    )

    # As-of merges: RBOB and EIA to AAA
    # Ensure datetime dtype
    for df in (aaa, rbob, eia, kalshi):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    merged = aaa.copy()

    # RBOB join (tolerance 10 days)
    merged = pd.merge_asof(
        merged.sort_values("date"),
        rbob.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=10),
    )

    # EIA join (inventory & change), tolerance 10 days
    merged = pd.merge_asof(
        merged.sort_values("date"),
        eia[["date", "inventory_mmbbl", "inventory_change"]].sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=10),
    )

    # Kalshi join (tolerance 2 days)
    merged = pd.merge_asof(
        merged.sort_values("date"),
        kalshi.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=2),
    )

    dataset = merged

    dataset.sort_values("date", inplace=True)
    dataset["inventory_mmbbl"] = dataset["inventory_mmbbl"].ffill()
    dataset["inventory_change"] = dataset["inventory_change"].ffill()
    dataset["kalshi_prob"] = dataset["kalshi_prob"].ffill().clip(0, 1)
    dataset["rbob_settle"] = dataset["rbob_settle"].interpolate()

    dataset["rbob_7d_change"] = dataset["rbob_settle"].diff(7)
    dataset["price_7d_change"] = dataset["regular_gas_price"].diff(7)
    dataset["inventory_2w_change"] = dataset["inventory_mmbbl"].diff(2)
    dataset["lag_1"] = dataset["regular_gas_price"].shift(1)
    dataset["lag_7"] = dataset["regular_gas_price"].shift(7)
    dataset["lag_14"] = dataset["regular_gas_price"].shift(14)
    dataset["target_future_price"] = dataset["regular_gas_price"].shift(
        -config.model.horizon_days
    )
    dataset.dropna(inplace=True)
    dataset = dataset.reset_index(drop=True)

    meta_dir = Path("data_proc") / "meta"
    latest_date = dataset["date"].max() if not dataset.empty else None
    as_of = None
    if latest_date is not None and not pd.isna(latest_date):
        as_of = pd.Timestamp(latest_date).normalize().date().isoformat()
    meta_payload = {
        "source": "dataset",
        "mode": "assembled",
        "as_of": as_of,
        "records": int(len(dataset)),
        "columns": list(dataset.columns),
        "inputs": [
            str(aaa_path),
            str(rbob_path),
            str(eia_path),
            str(kalshi_path),
        ],
    }
    write_meta(meta_dir / "dataset.json", meta_payload)

    return dataset
