# Gas Market Forecast Report

Generated on 2025-10-30T22:04:32+00:00
Provenance sidecars: [data_proc/meta/aaa.json](data_proc/meta/aaa.json), [data_proc/meta/eia.json](data_proc/meta/eia.json), [data_proc/meta/kalshi.json](data_proc/meta/kalshi.json), [data_proc/meta/rbob.json](data_proc/meta/rbob.json), [data_proc/meta/dataset.json](data_proc/meta/dataset.json)
## Headline Call
**Final Probability:** 4.0% that AAA national regular > $3.10 on 2025-10-31
_As of 2025-10-30._
> As of 2025-10-30, we assign 4.0% to AAA National Average Regular > $3.10 on Oct 31, 2025. The posterior mean price is $3.03, blending a prior weight of 0.10. Key adjustments: WPSR tightness: boosted upside pass-through and +3¢ alpha lift.

## Data Sources & Provenance
| Source | Mode | As Of | Fresh | Records | Path |
| --- | --- | --- | --- | --- | --- |
| aaa | last_good | 2025-10-30 | yes | 1 | /Users/mateobodon/Documents/Programming/Projects/kalshi-gas/data/raw/last_good.aaa.csv |
| eia | last_good | 2025-10-24 | yes | 1869 | /Users/mateobodon/Documents/Programming/Projects/kalshi-gas/data/raw/last_good.eia.csv |
| kalshi | last_good | 2025-10-30 | yes | 666 | /Users/mateobodon/Documents/Programming/Projects/kalshi-gas/data/raw/last_good.kalshi.csv |
| rbob | last_good | 2025-10-24 | yes | 2056 | /Users/mateobodon/Documents/Programming/Projects/kalshi-gas/data/raw/last_good.rbob.csv |

> Market reference: series n/a, event n/a (Kalshi resolves off AAA).

## Risk Dashboard
- NHC alert: **OFF** (0 active storms vs threshold 1)
- WPSR alert: **ON** (latest change -5.94 vs -3.00)
- Tightness flag: **ON**, draw 5.94 mmbbl, refinery util 89.6%
- NHC overrides: operational OFF, analyst OFF
- Tail adjustments: WPSR tightness: boosted upside pass-through and +3¢ alpha lift

### Risk Metrics
| Metric | Value | Trigger |
| --- | --- | --- |
| WPSR draw (mmbbl) | 5.94 | > 3.00 |
| Refinery utilisation (%) | 89.6 | < 90.0 |
| NHC active storms | 0 | > 1 |
| Analyst override | OFF | Manual |
| Event date | 2025-10-31 | — |
| Event threshold (USD/gal) | 3.10 | — |
| Days to event | 1 | — |
| β_eff (Δpump / ΔRBOB) | 0.0000 | — |
| Alpha lift applied (USD/gal) | 0.0000 | — |
| Tail adjustments | WPSR tightness: boosted upside pass-through and +3¢ alpha lift | — |
| Dataset as_of | 2025-10-30 | — |

![Risk Box](../figures/risk_box.png)
_As of 2025-10-30 • Sources: AAA daily (USD/gal); EIA WPSR weekly (mmbbl); CME RB settle ($/gal)_

![Fundamentals Dashboard](../figures/wpsr_dashboard.png)
_As of 2025-10-30 • Sources: AAA daily (USD/gal); EIA WPSR weekly (mmbbl); CME RB settle ($/gal)_

## Forecast Performance
### Ensemble Metrics
| Metric | Value |
| --- | --- |
| brier_score | 0.0051 |
| crps | 0.0083 |
| rmse | 0.0147 |
| brier_score_se | 0.0032 |
| brier_carry | 0.0050 |
| carry_rmse | 0.0081 |
| crps_carry | 0.0064 |
| brier_rbob | 0.4562 |
| crps_rbob | 0.0816 |
| brier_prior | 0.3401 |
| crps_prior | 0.1279 |
| posterior_brier | 0.0051 |
| prior_weight_calibrated | 0.0000 |
| posterior_brier_se | 0.0032 |
| crps_posterior | 0.0083 |

### Benchmarks
| Model | Brier | CRPS |
| --- | --- | --- |
| Posterior | 0.0051 | 0.0083 |
| Ensemble | 0.0051 | 0.0083 |
| RBOB Only | 0.4562 | 0.0816 |
| Carry Forward | 0.0050 | 0.0064 |
| Kalshi Prior | 0.3401 | 0.1279 |

### Freeze-date Metrics (central threshold)
| Model | Brier | CRPS |
| --- | --- | --- |
| Posterior | 0.0000 | 0.0548 |
| Carry Forward | 0.0000 | 0.0050 |
| RBOB Only | 0.0000 | 0.0071 |
| Kalshi Prior | 0.3138 | 0.2252 |

#### Freeze Snapshot
| Series | Date | Value (USD/gal) |
| --- | --- | --- |
| AAA today | 2025-10-30 | 3.0380 |
| AAA week-ago | 2025-10-23 | 3.0680 |
| AAA month-ago | 2025-09-30 | 3.1600 |
| RBOB settle | 2025-10-30 | 1.8840 |

#### Freeze-date Parameters
| Parameter | Value |
| --- | --- |
| Event threshold | 3.10 |
| Threshold gap | 0.0620 |
| α lift | 0.0000 |
| β_eff | 0.0000 |
| Prior weight | 0.10 |
| Residual σ | 0.0084 |
| Point forecast | 3.0298 |
| Tail probability | 3.98% |
| WTI proxy | n/a |

### Calibration Table
| Forecast Bin | Observed Frequency | Count |
| --- | --- | --- |
| 0.01 | 0.00 | 17 |
| 0.16 | 0.00 | 2 |
| 0.22 | 0.00 | 1 |
| 0.51 | 0.00 | 1 |
| 0.78 | 0.50 | 2 |
| 1.00 | 1.00 | 178 |

![Forecast vs Actual](../figures/nowcast.png)
_As of 2025-10-30 • Sources: AAA daily (USD/gal); EIA WPSR weekly (mmbbl); CME RB settle ($/gal)_

![Calibration Curve](../figures/calibration.png)
_As of 2025-10-30 • Sources: AAA daily (USD/gal); EIA WPSR weekly (mmbbl); CME RB settle ($/gal)_

## Model Outputs
![Sensitivity Bars](../figures/sensitivity_bars.png)
_As of 2025-10-30 • Sources: AAA daily (USD/gal); EIA WPSR weekly (mmbbl); CME RB settle ($/gal)_

![AAA Daily Δ Histogram](../figures/aaa_delta_hist.png)
_As of 2025-10-30 • Sources: AAA daily (USD/gal); EIA WPSR weekly (mmbbl); CME RB settle ($/gal)_

_Horizon-aware T+1 adjustments:_

| Scenario | β_eff | Δα | ΔP(> 3.10) |
| --- | --- | --- | --- |
| RBOB +$0.05 | 0.0000 | +0.0¢ | +0.00pp |
| RBOB -$0.05 | 0.0000 | +0.0¢ | +0.00pp |
| α +$0.02 | 0.0000 | +2.0¢ | +0.00pp |
| α -$0.02 | 0.0000 | -2.0¢ | +0.00pp |

### Posterior Snapshot
- Mean: 3.0298
- Variance: 0.0001
- 90% CI: [3.0158, 3.0437] (0.0140 down / 0.0139 up)
- 80% CI: [3.0188, 3.0408]
- Prior weight: 0.10 (file)
- Asymmetry: Δβ = -0.1237 (95% CI [-0.0641, 0.0290])
- P(X > 2.90) = 0.9397
- P(X > 2.95) = 0.9396
- P(X > 3.00) = 0.9395
- P(X > 3.05) = 0.0458
- P(X > 3.10) = 0.0398
- Dataset digest: dda40006e07ef3dbd3b2dafbeb9e51ef127e00dd806a95889faaa315ee9eab86


_α_t note_: latest α_t=2.33; 26w mean α_t=2.42

![Pass-through Fit](../figures/pass_through.png)
_As of 2025-10-30 • Sources: AAA daily (USD/gal); EIA WPSR weekly (mmbbl); CME RB settle ($/gal)_

![Market-implied Prior CDF](../figures/prior_cdf.png)
_As of 2025-10-30 • Sources: AAA daily (USD/gal); EIA WPSR weekly (mmbbl); CME RB settle ($/gal)_

![Posterior Density](../figures/posterior.png)
_As of 2025-10-30 • Sources: AAA daily (USD/gal); EIA WPSR weekly (mmbbl); CME RB settle ($/gal)_

### Sensitivity Grid (Detail)
| Threshold | ΔRBOB | ΔAlpha | Prob > Threshold |
| --- | --- | --- | --- |
| 2.90 | -0.050 | -0.050 | 0.9396 |
| 2.90 | -0.050 | 0.000 | 0.9397 |
| 2.90 | -0.050 | 0.050 | 0.9397 |
| 2.90 | 0.000 | -0.050 | 0.9396 |
| 2.90 | 0.000 | 0.000 | 0.9397 |
| 2.90 | 0.000 | 0.050 | 0.9397 |
| 2.90 | 0.050 | -0.050 | 0.9396 |
| 2.90 | 0.050 | 0.000 | 0.9397 |
| 2.90 | 0.050 | 0.050 | 0.9397 |
| 2.95 | -0.050 | -0.050 | 0.9395 |
| 2.95 | -0.050 | 0.000 | 0.9396 |
| 2.95 | -0.050 | 0.050 | 0.9397 |
| 2.95 | 0.000 | -0.050 | 0.9395 |
| 2.95 | 0.000 | 0.000 | 0.9396 |
| 2.95 | 0.000 | 0.050 | 0.9397 |
| 2.95 | 0.050 | -0.050 | 0.9395 |
| 2.95 | 0.050 | 0.000 | 0.9396 |
| 2.95 | 0.050 | 0.050 | 0.9397 |
| 3.00 | -0.050 | -0.050 | 0.0458 |
| 3.00 | -0.050 | 0.000 | 0.9395 |
| 3.00 | -0.050 | 0.050 | 0.9396 |
| 3.00 | 0.000 | -0.050 | 0.0458 |
| 3.00 | 0.000 | 0.000 | 0.9395 |
| 3.00 | 0.000 | 0.050 | 0.9396 |
| 3.00 | 0.050 | -0.050 | 0.0458 |
| 3.00 | 0.050 | 0.000 | 0.9395 |
| 3.00 | 0.050 | 0.050 | 0.9396 |
| 3.05 | -0.050 | -0.050 | 0.0398 |
| 3.05 | -0.050 | 0.000 | 0.0458 |
| 3.05 | -0.050 | 0.050 | 0.9395 |
| 3.05 | 0.000 | -0.050 | 0.0398 |
| 3.05 | 0.000 | 0.000 | 0.0458 |
| 3.05 | 0.000 | 0.050 | 0.9395 |
| 3.05 | 0.050 | -0.050 | 0.0398 |
| 3.05 | 0.050 | 0.000 | 0.0458 |
| 3.05 | 0.050 | 0.050 | 0.9395 |
| 3.10 | -0.050 | -0.050 | 0.0398 |
| 3.10 | -0.050 | 0.000 | 0.0398 |
| 3.10 | -0.050 | 0.050 | 0.0458 |
| 3.10 | 0.000 | -0.050 | 0.0398 |
| 3.10 | 0.000 | 0.000 | 0.0398 |
| 3.10 | 0.000 | 0.050 | 0.0458 |
| 3.10 | 0.050 | -0.050 | 0.0398 |
| 3.10 | 0.050 | 0.000 | 0.0398 |
| 3.10 | 0.050 | 0.050 | 0.0458 |

## Appendix: Meta Files
- data_proc/meta/aaa.json
- data_proc/meta/eia.json
- data_proc/meta/kalshi.json
- data_proc/meta/rbob.json
- data_proc/meta/dataset.json
- Submission SHA: 95726ccf103dedbbf309dc224fbbdf24c7497ae4
