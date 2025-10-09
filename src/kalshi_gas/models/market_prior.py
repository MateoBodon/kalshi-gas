"""Market prior model translating Kalshi probabilities."""

from __future__ import annotations

import pandas as pd

from kalshi_gas.models.regression import LinearModel


class MarketPriorModel:
    def __init__(self) -> None:
        self.regressor = LinearModel(
            feature_cols=["kalshi_prob", "lag_1"],
        )

    def fit(self, dataset: pd.DataFrame) -> None:
        self.regressor.fit(dataset)

    def predict(self, dataset: pd.DataFrame) -> pd.Series:
        preds = self.regressor.predict(dataset)
        preds.name = "market_prior"
        return preds
