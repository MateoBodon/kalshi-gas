"""Pass-through model linking RBOB futures to pump prices."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from kalshi_gas.models.regression import LinearModel
from kalshi_gas.models.structural import fit_structural_pass_through


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

    @staticmethod
    def bootstrap_asymmetry_ci(
        dataset: pd.DataFrame,
        structural_result: Dict[str, float | int | None],
        *,
        price_col: str = "regular_gas_price",
        rbob_col: str = "rbob_settle",
        samples: int = 500,
        alpha: float = 0.05,
        seed: int | None = None,
    ) -> tuple[float, float, float]:
        """Bootstrap percentile CI for beta_up - beta_dn asymmetry."""

        beta_up = structural_result.get("beta_up")
        beta_dn = structural_result.get("beta_dn")
        lag = structural_result.get("lag")

        if beta_up is None or beta_dn is None:
            raise ValueError("Structural result lacks asymmetry coefficients")
        if lag is None:
            raise ValueError("Structural result missing lag")
        if samples <= 0:
            raise ValueError("samples must be positive")

        base_delta = float(beta_up) - float(beta_dn)

        rng = np.random.default_rng(seed)
        draws: list[float] = []

        for _ in range(samples):
            sample = dataset.sample(
                frac=1.0,
                replace=True,
                random_state=rng.integers(0, 1 << 30),
            ).sort_index()
            try:
                fitted = fit_structural_pass_through(
                    sample,
                    price_col=price_col,
                    rbob_col=rbob_col,
                    lags=(int(lag),),
                    asymmetry=True,
                )
                up = fitted.get("beta_up")
                dn = fitted.get("beta_dn")
                if up is None or dn is None:
                    continue
                draws.append(float(up) - float(dn))
            except RuntimeError:
                continue

        if not draws:
            return base_delta, base_delta, base_delta

        lower = float(np.quantile(draws, alpha / 2))
        upper = float(np.quantile(draws, 1 - alpha / 2))
        return base_delta, lower, upper
