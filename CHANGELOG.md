## [Unreleased]
### Added
- Live Kalshi AAA v2 integration with RSA-PSS signing (pinned `KXAAAGASM-25OCT31`).
- `scripts/update_kalshi_bins.py` supports API-key auth, series/event filters, and bid/ask mid mapping to CDF.
- Posterior clipping to reduce CRPS explosions; added pass-through fit, prior CDF, posterior plots.
- Freeze-date metrics surfaced in memo; new Make targets (`calibrate`, `freeze-backtest`, `sweep-ensemble`).

### Changed
- Nowcast horizon set dynamically to month-end based on latest dataset date.
- Deck headline uses central (event) threshold probability.
- AAA ETL: added HTML fallback with user-agent when JSON endpoint forbids.

### Docs
- README: Live integration section (Kalshi RSA-PSS, EIA series), acceptance & repro commands.
