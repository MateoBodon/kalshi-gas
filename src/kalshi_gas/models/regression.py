"""Regression helpers for deterministic forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class LinearModel:
    feature_cols: Sequence[str]
    target_col: str = "target_future_price"
    model: LinearRegression = field(default_factory=LinearRegression)

    def fit(self, frame: pd.DataFrame) -> None:
        train = frame.dropna(subset=list(self.feature_cols) + [self.target_col])
        if train.empty:
            raise ValueError("No rows available for training")
        X = train[list(self.feature_cols)].values
        y = train[self.target_col].values
        self.model.fit(X, y)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        data = frame.copy()
        data = data.ffill().bfill()
        X = data[list(self.feature_cols)].values
        preds = self.model.predict(X)
        return pd.Series(preds, index=frame.index, name=f"{self.target_col}_pred")

    def residuals(self, frame: pd.DataFrame) -> pd.Series:
        preds = self.predict(frame)
        actuals = frame[self.target_col]
        return actuals - preds
