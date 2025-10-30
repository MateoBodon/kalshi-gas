# Daily Ops Runbook

## 1. Manual Label Logging (AAA / EIA / GasBuddy)

1. **08:45 ET** – visit [AAA Fuel Prices](https://gasprices.aaa.com/) and record the national *Regular* average.
2. Pull the same value from [EIA Today in Energy – Daily Petroleum Prices](https://www.eia.gov/todayinenergy/prices.php) (AAA row) and note whether it matches.
3. Optionally capture the [GasBuddy Fuel Insights](https://www.gasbuddy.com/charts) national price for direction checks.
4. Append the observation using either the interactive helper:
   ```bash
   python3 scripts/log_prompt.py
   ```
   or the direct CLI:
   ```bash
   python3 scripts/log_manual_label.py \
     --operator MB \
     --aaa 3.112 \
     --eia 3.115 \
     --gasbuddy 3.09 \
     --notes "08:45 ET snapshot"
   ```
   (Add `--time 2025-10-26T08:45:00` for backfills.)
5. **~16:30 ET** – repeat steps 1–4 if AAA has changed. Use separate entries with updated notes such as “PM check”.

All entries land in `data_raw/manual_logs.csv` with ISO timestamps for auditability. Each team member should initial the `--operator` field.

## 2. RBOB Settlements (front & second month)

- Source delayed settlement data (e.g., CME RB contracts) after the 14:28–14:30 ET settlement window.
- Prepare a CSV with columns `date,settle,second_month,spread,settle_change,second_change,source` (omit columns if values are unavailable).
- Ingest using:
  ```bash
  python3 scripts/update_rbob_csv.py --from-csv path/to/rbob_settlements.csv
  ```
  or append ad‑hoc values with `--settle`/`--second-month` flags.
- The processed ETL converts the front-month settle into `rbob_price` while preserving additional features.

## 3. Weekly EIA Data

- **Retail anchor (Monday 17:00 ET):** download the U.S. Regular series from the [Gasoline & Diesel Fuel Update](https://www.eia.gov/petroleum/gasdiesel/). Save the CSV/XLS in `data_raw/eia_weekly_retail.csv` and keep the original file as provenance. API v2 fallback: `https://api.eia.gov/v2/seriesid/PET.EMM_EPMRR_PTE_NUS_DPG?api_key=...`.
- **Fundamentals (Wednesday 10:30 ET):** pull WPSR tables (stocks, product supplied) or call API v2 (`PET.WGTSTUS1.W`, etc.). Store in `data_raw/wpsr_state.yml` via `scripts/update_wpsr_state.py` if manual adjustments are needed.

After each refresh, run the pipeline to regenerate processed files and reports:
```bash
python3 -m kalshi_gas.cli report
python3 -m kalshi_gas.backtest.calibrate_prior
python3 scripts/cli.py init-raw-templates
```

## 4. QA & Cross-Checks

- Use MacroMicro or other licensed dashboards **only** to validate logged values; do not ingest third-party data into the repository unless licensing permits redistribution.
- `data_proc/meta/` contains auto-generated provenance JSON. Do not edit manually.

## 5. Ownership

| Task | Primary | Backup |
| ---- | ------- | ------ |
| AAA/EIA manual logging | TBD | TBD |
| RBOB settlements ingest | TBD | TBD |
| Weekly EIA updates | TBD | TBD |

Document any anomalies in the `notes` field (e.g., “AAA flat, no EIA update today”).

### Suggested Reminders

Set local reminders so the morning (08:45 ET) and afternoon (16:30 ET) checks never slip. Example `cron` entry (
adjust for your local timezone):

```
45 13 * * 1-5 cd /path/to/kalshi-gas && /usr/bin/python3 scripts/log_prompt.py
30 20 * * 1-5 cd /path/to/kalshi-gas && /usr/bin/python3 scripts/log_prompt.py
```

The helper will prompt for prices and write them to `data_raw/manual_logs.csv` with a full audit trail.

## Final Submission Checklist (Oct 28 – Oct 30, 2025)

**Oct 28 (Tue) — Data refresh & calibration**
- Pull latest AAA national (AM/PM), append to `data_raw/last_good.aaa.csv`, rerun `make report`.
- Update Kalshi bins via `python scripts/update_kalshi_bins.py --manual-survival ... --as-of 2025-10-28` if the live API is unavailable; verify `data_raw/kalshi_bins.yml.meta.json` stamps today.
- Run `make calibrate`, `make freeze-backtest`, and `make sweep-ensemble`; archive `data_proc/backtest_metrics.json` and `data_proc/ensemble_weight_sweep.csv`.

**Oct 29 (Wed) — WPSR drop & scenario sweep**
- 10:30 ET: ingest WPSR (`scripts/freeze_wpsr_html.py` or live API), update `data_raw/wpsr_state.yml`, rerun `make report`.
- Rebuild figures/deck; note any scenario shifts (ΔP) in memo “Risks” section.
- If Kalshi markets move materially, refresh bins and capture the new meta stamp.

**Oct 30 (Thu) — Final memo/deck freeze**
- From a clean clone: `make report` then `make deck`; confirm build artifacts in `build/`.
- Record final git commit SHA via `git rev-parse HEAD` and paste into `reports/memo.md` appendix (cite as “Submission SHA”).
- Double-check `data_proc/meta/*.json` freshness; zip `build/` + memo for submission.

> Tip: include the SHA, run timestamp, and data sources on the cover slide so judges can audit reproducibility in minutes.
