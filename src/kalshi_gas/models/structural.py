"""Structural pass-through modelling with lag selection and asymmetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

LagCandidates = Sequence[int]


@dataclass
class PassThroughResult:
    alpha: float
    beta: float
    beta_up: float | None
    beta_dn: float | None
    lag: int
    r2: float
    criterion: float

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "beta_up": self.beta_up,
            "beta_dn": self.beta_dn,
            "lag": self.lag,
            "r2": self.r2,
        }


def _prepare_frame(
    price: pd.Series,
    rbob: pd.Series,
    lag: int,
    asymmetry: bool,
) -> pd.DataFrame:
    delta_price = price.astype(float).diff()
    delta_rbob = rbob.astype(float).diff()

    frame = pd.DataFrame(
        {
            "delta_price": delta_price,
            "rbob_change": delta_rbob.shift(lag),
        }
    )

    if asymmetry:
        up = delta_rbob.clip(lower=0.0).shift(lag)
        dn = delta_rbob.clip(upper=0.0).shift(lag)
        frame["rbob_up"] = up
        frame["rbob_dn"] = dn

    frame = frame.dropna()
    return frame


def _fit_ols(
    frame: pd.DataFrame, columns: Iterable[str]
) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = sm.add_constant(frame[list(columns)], has_constant="add")
    y = frame["delta_price"]
    model = sm.OLS(y, X)
    return model.fit()


def fit_structural_pass_through(
    data: pd.DataFrame,
    price_col: str = "regular_gas_price",
    rbob_col: str = "rbob_settle",
    lags: LagCandidates = (7, 8, 9, 10),
    asymmetry: bool = False,
    criterion: str = "aic",
) -> dict[str, float | int | None]:
    """
    Fit structural pass-through using lag selection and optional asymmetry.

    We minimise AIC over the grid of candidate lags (default 7–10 days).
    """

    if criterion.lower() not in {"aic", "bic"}:
        raise ValueError("criterion must be 'aic' or 'bic'")

    if len(lags) == 0:
        raise ValueError("At least one lag must be provided")

    price = data[price_col]
    rbob = data[rbob_col]

    best: PassThroughResult | None = None

    for lag in lags:
        frame = _prepare_frame(price, rbob, lag, asymmetry)
        if frame.empty:
            continue

        columns = ["rbob_change"]
        if asymmetry:
            columns = ["rbob_up", "rbob_dn"]

        results = _fit_ols(frame, columns)

        score = results.aic if criterion.lower() == "aic" else results.bic
        alpha = float(results.params.get("const", 0.0))

        if asymmetry:
            beta_up = float(results.params.get("rbob_up", np.nan))
            beta_dn = float(results.params.get("rbob_dn", np.nan))
            beta = float(np.nanmean([beta_up, beta_dn]))
        else:
            beta = float(results.params.get("rbob_change", np.nan))
            beta_up = None
            beta_dn = None

        current = PassThroughResult(
            alpha=alpha,
            beta=beta,
            beta_up=beta_up,
            beta_dn=beta_dn,
            lag=int(lag),
            r2=float(results.rsquared),
            criterion=float(score),
        )

        if best is None or current.criterion < best.criterion:
            best = current

    if best is None:
        raise RuntimeError("Unable to fit pass-through model for supplied data")

    return best.as_dict()


def rolling_alpha_path(
    data: pd.DataFrame,
    lag: int,
    window: int = 26,
    price_col: str = "regular_gas_price",
    rbob_col: str = "rbob_settle",
) -> pd.Series:
    """Estimate rolling intercept α_t for level mapping Retail_t ≈ α_t + β·F_{t-L}.

    Uses ordinary least squares on a rolling window. Returns a Series indexed to
    the input with NaNs where the window is insufficient.
    """
    df = data[[price_col, rbob_col]].copy()
    df["rbob_lag"] = df[rbob_col].astype(float).shift(int(lag))
    alphas = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        lo = max(0, i - int(window) + 1)
        sub = df.iloc[lo : i + 1].dropna()
        if len(sub) < max(10, int(window) // 2):
            continue
        X = np.vstack([np.ones(len(sub)), sub["rbob_lag"].to_numpy(dtype=float)]).T
        y = sub[price_col].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        alphas[i] = float(coef[0])
    return pd.Series(alphas, index=data.index, name="alpha_t")
