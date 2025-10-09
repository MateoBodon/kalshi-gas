"""Combine processed datasets into a modeling table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kalshi_gas.config import PipelineConfig


def load_processed_frame(path: Path, date_col: str = "date") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset missing: {path}")
    return pd.read_csv(path, parse_dates=[date_col])


def assemble_dataset(config: PipelineConfig) -> pd.DataFrame:
    """Return merged time-series for modeling."""
    aaa_path = config.data.processed_dir / "aaa_daily.csv"
    eia_path = config.data.processed_dir / "eia_weekly.csv"
    rbob_path = config.data.processed_dir / "rbob_prices.csv"
    kalshi_path = config.data.processed_dir / "kalshi_markets.csv"

    aaa = load_processed_frame(aaa_path)
    rbob = load_processed_frame(rbob_path)
    eia = load_processed_frame(eia_path)
    kalshi = load_processed_frame(kalshi_path)

    rbob.rename(columns={"rbob_price": "rbob_settle"}, inplace=True)

    kalshi = (
        kalshi.groupby("date")
        .agg({"prob_yes": "mean"})
        .rename(columns={"prob_yes": "kalshi_prob"})
        .reset_index()
    )

    dataset = (
        aaa.merge(rbob, on="date", how="left")
        .merge(eia[["date", "inventory_mmbbl", "inventory_change"]], on="date", how="left")
        .merge(kalshi, on="date", how="left")
    )

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
    dataset["target_future_price"] = dataset["regular_gas_price"].shift(-config.model.horizon_days)
    dataset.dropna(inplace=True)

    return dataset.reset_index(drop=True)
