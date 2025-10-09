"""Nowcast model leveraging state-space predictive simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents


def _bounded_mean(mean: float, lower: float, upper: float) -> float:
    return float(np.clip(mean, lower, upper))


@dataclass
class NowcastSimulation:
    date: pd.Timestamp
    mean: float
    variance: float
    samples: np.ndarray


class NowcastModel:
    def __init__(
        self,
        horizon: int = 7,
        simulations: int = 500,
        drift_bounds: tuple[float, float] = (-0.1, 0.1),
        residual_window: int = 28,
    ) -> None:
        self.horizon = horizon
        self.simulations = simulations
        self.drift_bounds = drift_bounds
        self.residual_window = residual_window
        self.model: UnobservedComponents | None = None
        self.fit_results = None
        self.residuals: np.ndarray | None = None

    def fit(self, series: pd.Series) -> None:
        series = series.dropna()
        if series.empty:
            raise ValueError("Nowcast requires non-empty series")

        self.model = UnobservedComponents(
            series.astype(float),
            level="local linear trend",
        )
        self.fit_results = self.model.fit(disp=False)
        if self.fit_results is None:
            raise RuntimeError("Nowcast fit failed to produce results")

        resid = self.fit_results.resid
        if isinstance(resid, pd.Series):
            resid = resid.to_numpy()
        resid = np.asarray(resid, dtype=float)
        if resid.size == 0:
            resid = np.array([0.0])
        self.residuals = resid[-self.residual_window :]
        if self.residuals.size == 0:
            self.residuals = resid

    def simulate(self, steps: int | None = None) -> NowcastSimulation:
        if self.model is None or self.fit_results is None:
            raise RuntimeError("Nowcast model must be fit before simulation")

        steps = steps or self.horizon
        forecast = self.fit_results.get_forecast(steps=steps)
        mean_forecast = forecast.predicted_mean.iloc[-1]
        var_forecast = forecast.var_pred_mean.iloc[-1]

        lower, upper = self.drift_bounds
        bounded_mean = _bounded_mean(mean_forecast, lower, upper)

        residuals = self.residuals if self.residuals is not None else np.array([0.0])
        bootstrap_indices = np.random.randint(0, len(residuals), size=self.simulations)
        bootstrap_resid = residuals[bootstrap_indices]

        noise = np.random.normal(
            0, np.sqrt(max(var_forecast, 1e-8)), size=self.simulations
        )
        samples = bounded_mean + noise + bootstrap_resid

        target_date = forecast.row_labels[-1]
        if not isinstance(target_date, pd.Timestamp):
            target_date = pd.Timestamp(target_date)

        return NowcastSimulation(
            date=target_date,
            mean=float(bounded_mean),
            variance=float(var_forecast),
            samples=samples,
        )

    def predict(self, series: pd.Series) -> NowcastSimulation:
        self.fit(series)
        return self.simulate(self.horizon)
