"""Nowcast model based on lagged AAA prices."""

from __future__ import annotations

import pandas as pd

from kalshi_gas.models.regression import LinearModel


class NowcastModel:
    def __init__(self) -> None:
        self.regressor = LinearModel(
            feature_cols=["lag_1", "lag_7", "lag_14", "price_7d_change"],
        )

    def fit(self, dataset: pd.DataFrame) -> None:
        self.regressor.fit(dataset)

    def predict(self, dataset: pd.DataFrame) -> pd.Series:
        preds = self.regressor.predict(dataset)
        preds.name = "nowcast"
        return preds
