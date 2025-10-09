"""Weighted ensemble combining model families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
import numpy as np

from kalshi_gas.models.market_prior import MarketPriorModel
from kalshi_gas.models.nowcast import NowcastModel
from kalshi_gas.models.pass_through import PassThroughModel


@dataclass
class EnsembleModel:
    weights: Dict[str, float]

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if total == 0:
            raise ValueError("Ensemble weights sum to zero")
        self.normalized_weights = {k: v / total for k, v in self.weights.items()}
        self.nowcast = NowcastModel()
        self.pass_through = PassThroughModel()
        self.market_prior = MarketPriorModel()

    def fit(self, dataset: pd.DataFrame) -> None:
        price_series = dataset["regular_gas_price"]
        self.nowcast.fit(price_series)
        self.pass_through.fit(dataset)
        self.market_prior.fit(dataset)
        preds = self.predict(dataset)
        residuals = dataset["target_future_price"] - preds["ensemble"]
        self.residual_std = float(np.nanstd(residuals))
        if not np.isfinite(self.residual_std) or self.residual_std == 0.0:
            self.residual_std = float(np.nanstd(dataset["target_future_price"]))

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        nowcast_sim = self.nowcast.predict(dataset["regular_gas_price"])
        nowcast_pred = pd.Series(
            nowcast_sim.mean,
            index=dataset.index,
            name="nowcast",
        )
        pass_pred = self.pass_through.predict(dataset)
        market_pred = self.market_prior.predict(dataset)

        preds = pd.concat(
            [nowcast_pred, pass_pred, market_pred],
            axis=1,
        )
        ensemble = sum(
            preds[column] * weight for column, weight in self.normalized_weights.items()
        )
        preds["ensemble"] = ensemble
        return preds
