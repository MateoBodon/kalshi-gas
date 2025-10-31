## [Unreleased]
- No unreleased changes.

## [2025-10-30] Mateo–Ly Kalshi Final Submission
### Added
- Live Kalshi AAA v2 integration with RSA-PSS signing (pinned `KXAAAGASM-25OCT31`).
- `scripts/update_kalshi_bins.py` supports API-key auth, series/event filters, and bid/ask mid mapping to CDF.
- Posterior clipping to reduce CRPS explosions; added pass-through fit, prior CDF, posterior plots.
- Freeze-date metrics surfaced in memo; new Make targets (`calibrate`, `freeze-backtest`, `sweep-ensemble`).

### Changed
- Nowcast horizon set dynamically to month-end based on latest dataset date.
- Deck headline uses central (event) threshold probability.
- AAA ETL adds an HTML fallback with user-agent when the JSON endpoint forbids access.

### Docs
- README covers live integration (Kalshi RSA-PSS, EIA series), acceptance workflow, and reproduction commands.
- Submission artefacts documented (`Mateo_Ly_Kalshi_Submission.pdf`, `SUBMISSION.md`, `reports/memo.md`).
