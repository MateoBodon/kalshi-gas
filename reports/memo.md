# Gas Market Forecast Report

Generated on 2025-10-30T20:06:45+00:00
Provenance sidecars: [data_proc/meta/aaa.json](data_proc/meta/aaa.json), [data_proc/meta/eia.json](data_proc/meta/eia.json), [data_proc/meta/kalshi.json](data_proc/meta/kalshi.json), [data_proc/meta/rbob.json](data_proc/meta/rbob.json), [data_proc/meta/dataset.json](data_proc/meta/dataset.json)
## Headline Call
**Final Probability:** 8.0% that AAA national regular > $3.10 on 2025-10-31
_As of 2025-10-30._
As of 2025-10-30, we assign 8.0% to AAA National Average Regular > $3.10 on Oct 31, 2025. The posterior mean price is $3.07, blending a prior weight of 0.20. Key adjustments: WPSR tightness: boosted upside pass-through and +3¢ alpha lift.

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
| β_eff (Δpump / ΔRBOB) | 0.0004 | — |
| Alpha lift applied (USD/gal) | 0.0043 | — |
| Tail adjustments | WPSR tightness: boosted upside pass-through and +3¢ alpha lift | — |
| Dataset as_of | 2025-10-30 | — |

![Risk Box](../figures/risk_box.png)
_As of 2025-10-30 • Sources: AAA (daily), EIA WPSR (Wed 10:30 ET)_

![Fundamentals Dashboard](../figures/wpsr_dashboard.png)
_As of 2025-10-30 • Sources: AAA (daily), EIA WPSR (Wed 10:30 ET)_

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
| Posterior | 0.0000 | 0.0561 |
| Carry Forward | 0.0000 | 0.0050 |
| RBOB Only | 0.0000 | 0.0071 |
| Kalshi Prior | 0.3138 | 0.2252 |

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
_As of 2025-10-30 • Sources: AAA (daily), EIA WPSR (Wed 10:30 ET)_

![Calibration Curve](../figures/calibration.png)
_As of 2025-10-30 • Sources: AAA (daily), EIA WPSR (Wed 10:30 ET)_

## Model Outputs
![Sensitivity Bars](../figures/sensitivity_bars.png)
_As of 2025-10-30 • Sources: AAA (daily), EIA WPSR (Wed 10:30 ET)_

### Posterior Snapshot
- Mean: 3.0713
- Variance: 0.0001
- 90% CI: [3.0564, 3.0862] (0.0149 down / 0.0149 up)
- 80% CI: [3.0598, 3.0828]
- Prior weight: 0.20 (file)
- Asymmetry: Δβ = -0.1237 (95% CI [-0.0615, 0.0312])
- P(X > 2.90) = 0.8789
- P(X > 2.95) = 0.8786
- P(X > 3.00) = 0.8783
- P(X > 3.05) = 0.8654
- P(X > 3.10) = 0.0796
- Dataset digest: dda40006e07ef3dbd3b2dafbeb9e51ef127e00dd806a95889faaa315ee9eab86


_α_t note_: latest α_t=2.33; 26w mean α_t=2.42

![Pass-through Fit](../figures/pass_through.png)
_As of 2025-10-30 • Sources: AAA (daily), EIA WPSR (Wed 10:30 ET)_

![Market-implied Prior CDF](../figures/prior_cdf.png)
_As of 2025-10-30 • Sources: AAA (daily), EIA WPSR (Wed 10:30 ET)_

![Posterior Density](../figures/posterior.png)
_As of 2025-10-30 • Sources: AAA (daily), EIA WPSR (Wed 10:30 ET)_

### Sensitivity Grid (Detail)
| Threshold | ΔRBOB | ΔAlpha | Prob > Threshold |
| --- | --- | --- | --- |
| 2.90 | -0.050 | -0.050 | 0.8786 |
| 2.90 | -0.050 | 0.000 | 0.8789 |
| 2.90 | -0.050 | 0.050 | 0.8792 |
| 2.90 | 0.000 | -0.050 | 0.8786 |
| 2.90 | 0.000 | 0.000 | 0.8789 |
| 2.90 | 0.000 | 0.050 | 0.8793 |
| 2.90 | 0.050 | -0.050 | 0.8786 |
| 2.90 | 0.050 | 0.000 | 0.8789 |
| 2.90 | 0.050 | 0.050 | 0.8792 |
| 2.95 | -0.050 | -0.050 | 0.8783 |
| 2.95 | -0.050 | 0.000 | 0.8786 |
| 2.95 | -0.050 | 0.050 | 0.8789 |
| 2.95 | 0.000 | -0.050 | 0.8783 |
| 2.95 | 0.000 | 0.000 | 0.8786 |
| 2.95 | 0.000 | 0.050 | 0.8789 |
| 2.95 | 0.050 | -0.050 | 0.8783 |
| 2.95 | 0.050 | 0.000 | 0.8786 |
| 2.95 | 0.050 | 0.050 | 0.8789 |
| 3.00 | -0.050 | -0.050 | 0.8643 |
| 3.00 | -0.050 | 0.000 | 0.8783 |
| 3.00 | -0.050 | 0.050 | 0.8786 |
| 3.00 | 0.000 | -0.050 | 0.8654 |
| 3.00 | 0.000 | 0.000 | 0.8783 |
| 3.00 | 0.000 | 0.050 | 0.8786 |
| 3.00 | 0.050 | -0.050 | 0.8641 |
| 3.00 | 0.050 | 0.000 | 0.8783 |
| 3.00 | 0.050 | 0.050 | 0.8786 |
| 3.05 | -0.050 | -0.050 | 0.0796 |
| 3.05 | -0.050 | 0.000 | 0.8643 |
| 3.05 | -0.050 | 0.050 | 0.8783 |
| 3.05 | 0.000 | -0.050 | 0.0796 |
| 3.05 | 0.000 | 0.000 | 0.8654 |
| 3.05 | 0.000 | 0.050 | 0.8783 |
| 3.05 | 0.050 | -0.050 | 0.0796 |
| 3.05 | 0.050 | 0.000 | 0.8641 |
| 3.05 | 0.050 | 0.050 | 0.8783 |
| 3.10 | -0.050 | -0.050 | 0.0796 |
| 3.10 | -0.050 | 0.000 | 0.0796 |
| 3.10 | -0.050 | 0.050 | 0.8643 |
| 3.10 | 0.000 | -0.050 | 0.0796 |
| 3.10 | 0.000 | 0.000 | 0.0796 |
| 3.10 | 0.000 | 0.050 | 0.8654 |
| 3.10 | 0.050 | -0.050 | 0.0796 |
| 3.10 | 0.050 | 0.000 | 0.0796 |
| 3.10 | 0.050 | 0.050 | 0.8641 |

## Appendix: Meta Files
- data_proc/meta/aaa.json
- data_proc/meta/eia.json
- data_proc/meta/kalshi.json
- data_proc/meta/rbob.json
- data_proc/meta/dataset.json
- Submission SHA: a9ad1d1c531632a7b2fd7f73c4c2eb456336e7cf
