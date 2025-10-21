"""Nowcast model leveraging state-space predictive simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents


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
        self.last_observation: float | None = None

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
        self.last_observation = float(series.iloc[-1])

    def simulate(self, steps: int | None = None) -> NowcastSimulation:
        if self.model is None or self.fit_results is None:
            raise RuntimeError("Nowcast model must be fit before simulation")
        if self.last_observation is None:
            raise RuntimeError("Nowcast model missing last observation")

        steps = steps or self.horizon
        forecast = self.fit_results.get_forecast(steps=steps)
        mean_path = forecast.predicted_mean.to_numpy(dtype=float)
        var_path = forecast.var_pred_mean.to_numpy(dtype=float)

        lower, upper = self.drift_bounds
        prev_levels = np.concatenate(([self.last_observation], mean_path[:-1]))
        raw_drifts = mean_path - prev_levels
        drift_means = np.clip(raw_drifts, lower, upper)
        drift_range = max(upper - lower, 0.0)
        drift_std = drift_range / 6 if drift_range > 0 else 0.0

        step_variances = np.diff(np.concatenate(([0.0], var_path)))
        step_variances = np.maximum(step_variances, 0.0)

        residuals = self.residuals if self.residuals is not None else np.array([0.0])
        levels = np.full(self.simulations, self.last_observation, dtype=float)

        for idx in range(steps):
            if drift_std > 0:
                drift_samples = np.random.normal(
                    loc=drift_means[idx], scale=drift_std, size=self.simulations
                )
                drift_samples = np.clip(drift_samples, lower, upper)
            else:
                drift_samples = np.full(self.simulations, drift_means[idx], dtype=float)

            gaussian_noise = np.random.normal(
                0.0, np.sqrt(max(step_variances[idx], 1e-8)), size=self.simulations
            )
            bootstrap_noise = residuals[
                np.random.randint(0, len(residuals), size=self.simulations)
            ]
            levels = levels + drift_samples + gaussian_noise + bootstrap_noise

        samples = levels
        mean_forecast = float(np.mean(samples))
        var_forecast = float(np.var(samples, ddof=0))

        target_date = forecast.row_labels[-1]
        if not isinstance(target_date, pd.Timestamp):
            target_date = pd.Timestamp(target_date)

        return NowcastSimulation(
            date=target_date,
            mean=mean_forecast,
            variance=var_forecast,
            samples=samples,
        )

    def predict(self, series: pd.Series) -> NowcastSimulation:
        self.fit(series)
        return self.simulate(self.horizon)
