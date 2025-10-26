# Methodology

## Nowcast Module

We estimate the daily retail price nowcast with a local linear trend state‐space model fitted to the AAA national average series.

\[
\begin{aligned}
 y_t &= \mu_t + \varepsilon_t, & \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2) \\
 \mu_t &= \mu_{t-1} + \beta_{t-1} + \eta_t, & \eta_t \sim \mathcal{N}(0, \sigma_\eta^2) \\
 \beta_t &= \beta_{t-1} + \zeta_t, & \zeta_t \sim \mathcal{N}(0, \sigma_\zeta^2)
\end{aligned}
\]

We simulate the predictive distribution for the settlement horizon by combining the Gaussian forecast with a residual bootstrap. Under risk stress, the drift prior bounds are widened upward by \(+\Delta_{\text{drift}}\) as described in the pipeline.

## Data Alignment (Nearest-week joins)

Live data can be sparse on any given day. To avoid empty merges, the assembler aligns weekly EIA and RBOB series to daily AAA using nearest‑week as‑of joins (backward direction) with sensible tolerances (10 days for EIA/RBOB; 2 days for Kalshi). This preserves continuity without leaking future information.

## Structural Pass-through

RBOB pass-through is estimated on lagged futures changes. The asymmetric mapping is

\[
\Delta P_t = \alpha + \beta^{\uparrow} \max(\Delta F_{t-L}, 0) + \beta^{\downarrow} \min(\Delta F_{t-L}, 0) + \epsilon_t,
\]

where \(L \in \{7,8,9,10\}\) is chosen by minimising the Akaike Information Criterion. Under tight WPSR conditions we up-weight \( \beta^{\uparrow}\) relative to \( \beta^{\downarrow} \) and apply an \(\alpha\) shift to reflect refinery outages.

## Posterior Blending

The posterior is a mixture of the empirical nowcast samples \(\hat{F}\) and the Kalshi-implied prior CDF \(F_0\):

\[
F(z) = (1-w)\,\hat{F}(z) + w\,F_0(z), \quad w = \texttt{prior\_weight} \in [0,1].
\]

Sensitivity tables perturb RBOB deltas and structural intercepts before evaluating the probability of exceeding price thresholds.

## Scores

- **Brier score**: \(\text{BS} = \tfrac{1}{N} \sum (p_i - o_i)^2\).
- **CRPS (sample form)**: \(\text{CRPS} = \mathbb{E}|X - y| - \tfrac{1}{2}\mathbb{E}|X - X'|\), with \(X,X'\) independent draws from the posterior samples.

These metrics are stored in `data_proc/backtest_metrics.json` for inspection.
