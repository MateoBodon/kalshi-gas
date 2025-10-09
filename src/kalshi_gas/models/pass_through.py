"""Pass-through model linking RBOB futures to pump prices."""

from __future__ import annotations

import pandas as pd

from kalshi_gas.models.regression import LinearModel


class PassThroughModel:
    def __init__(self) -> None:
        self.regressor = LinearModel(
            feature_cols=["rbob_settle", "rbob_7d_change", "inventory_change"],
        )

    def fit(self, dataset: pd.DataFrame) -> None:
        self.regressor.fit(dataset)

    def predict(self, dataset: pd.DataFrame) -> pd.Series:
        preds = self.regressor.predict(dataset)
        preds.name = "pass_through"
        return preds
