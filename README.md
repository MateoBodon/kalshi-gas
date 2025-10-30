# Kalshi Gas Analytics

Forecast and reporting toolkit for the YUHA × Kalshi Market Research & Modeling Competition. The repo combines authoritative public data (AAA resolver, EIA fundamentals, CME/Nasdaq RBOB futures, Kalshi market priors), builds an ensemble forecast, scores it, and auto-generates submission-ready memo/deck artefacts.

---

## Current Snapshot *(run: 2025‑10‑30 16:22 UTC)*

- **Target:** AAA US national regular gasoline price, Oct 31 2025 resolution threshold **$3.10/gal**.
- **Posterior probability:** **72.3 %** that AAA > $3.10 (posterior mean **$3.26**, 90 % CI [3.12, 3.39]).
- **Key drivers:** 5.94 mmbbl gasoline stock draw (week ending Oct 24), refinery utilisation 93 %, Kalshi prior tilted bullish (45 %/20 %/8 % for ≥ $3.05/3.10/3.15 as of Oct 30).
- **Risk gates:** Storm override off; utilisation above 90 % so no tightness flag, but inventory draw trigger is on.
- **Backtests (Jan 2023–Sep 2025 freeze schedule):**
  - Posterior Brier 0.146 (vs carry 0.048) — market prior still carries most signal.
  - Posterior CRPS 0.060 (carry 0.032). Ensemble sweep keeps prior dominant (best log-score weight 100 % prior; best CRPS weight 90 % prior / 10 % nowcast).

### Live data status

| Feed | Source | Latest as-of | Notes |
| --- | --- | --- | --- |
| AAA national average | https://gasprices.aaa.com (HTML fallback) | **2025‑10‑30** | Resolver; twice-daily capture recommended. |
| WPSR fundamentals | https://www.eia.gov/petroleum/supply/weekly/ | **2025‑10‑24** | Next release expected 2025‑10‑31 10:30 ET. |
| RBOB settle (front / 2nd month) | CME settle (manual entry) | **2025‑10‑29** | Oct 30 settle pending – update via `scripts/update_rbob_csv.py`. |
| Kalshi bins | Trade API v2 (RSA-PSS) | **2025‑10‑30** | `scripts/update_kalshi_bins.py` pulls ≥ $3.05/$3.10/$3.15 midpoints. |
| NHC flag | data_raw/nhc_flag.yml | OFF | Manual override for storm risk. |

Latest memo/deck/report artefacts are zipped in `build/submission_oct30.zip`.

---

## Repository highlights

- **ETL** (`src/kalshi_gas/etl`): resilient fallbacks (live → last_good → sample), provenance metadata, HTML parsers for AAA/WPSR.
- **Models** (`src/kalshi_gas/models`): local-trend nowcast, asymmetric pass-through, isotonic Kalshi prior, ensemble/posterior with scenario sensitivities.
- **Pipeline** (`src/kalshi_gas/pipeline/run_all.py`): one-command rebuild of ETL → dataset → risk gating → posterior → memo/deck.
- **Backtesting** (`src/kalshi_gas/pipeline/backtest.py`): freeze-date harness, proper scoring (Brier/CRPS), calibration plots.
- **Reporting** (`src/kalshi_gas/reporting`): Markdown memo & deck templated with figure footers, “Why this matters” call-outs, SHA stamping.
- **Automation** (`Makefile`): reproducible targets for report, calibrations, backtests, ensemble sweep, lint/tests.

---

## Reproducing the forecast

1. **Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```
2. **Credentials (optional but recommended)**
   ```bash
   export KALSHI_GAS_USE_LIVE=1
   export EIA_API_KEY="..."
   export KALSHI_API_KEY_ID="..."
   export KALSHI_PRIVATE_KEY_PATH="$HOME/.kalshi/kalshi_private_key.pem"
   # Optional: export KALSHI_PY_PRIVATE_KEY_PEM="..."
   ```
3. **Update live snapshots (if permitted)**
   ```bash
   python scripts/update_kalshi_bins.py          # Kalshi API (RSA-PSS signed)
   python scripts/update_rbob_csv.py --date YYYY-MM-DD --settle <value> --source "CME settle ..."
   python scripts/freeze_wpsr_html.py            # pulls latest WPSR summary HTML
   ```
   *(For AAA the HTML fallback runs inside the ETL; capture a screenshot for audit.)*
4. **Full rebuild**
   ```bash
   make report            # ETL → posterior → memo/deck
   make calibrate         # prior weight sweep
   make freeze-backtest   # historical freeze-date evaluation
   make sweep-ensemble    # ensemble weight grid search
   ```
5. **Outputs**
   - `build/memo/report.md` & `build/deck/deck.md`
   - `build/figures/*.png`
   - `data_proc/backtest_metrics.json`, `data_proc/ensemble_weight_sweep.{csv,json}`
   - `build/submission_oct30.zip` (memo, deck, figures, provenance)

Run `make test` for the 36 unit/integration tests; `make lint` for formatting/static checks.

---

## Data ingest & manual updates

- **AAA national average:** live JSON endpoint is blocked (403). ETL scrapes the homepage with a browser user agent; save screenshots or copy values into `data_raw/last_good.aaa.csv` if you capture them manually.
- **RBOB settles:** Nasdaq Data Link CHRIS endpoints are firewall-protected; until whitelisted, pull CME official settles manually and append via `scripts/update_rbob_csv.py`. CSV bulk import accepts `date,settle,second_month,spread,...`.
- **WPSR fundamentals:** rerun `scripts/freeze_wpsr_html.py` after each Wednesday release (or download HTML/PDF and replace `data_raw/wpsr_summary.html`). Risk gates read refinery utilisation and stock draws from that snapshot.
- **Kalshi bins:** `scripts/update_kalshi_bins.py` signs requests with RSA-PSS (API key + private key). Falls back to `data_raw/kalshi_bins.yml` if credentials are absent.
- **Storm / analyst overrides:** toggle `data_raw/nhc_flag.yml` and `data_raw/wpsr_state.yml` as described in `docs/RUNBOOK.md`.

Provenance sidecars in `data_proc/meta/*.json` capture mode, as-of, record counts, and fallback chains for auditability.

---

## Submission package checklist

`build/submission_oct30.zip` contains:
- `memo/report.md` – competition memo with timestamp/SHA, probability call, sensitivities.
- `deck/deck.md` – slide deck outline with “Why this matters” notes.
- `figures/*.png` – nowcast cone, pass-through fit, prior CDF, posterior density, calibration, risk dashboard, sensitivities.
- `metadata/data_provenance.json` + `data_proc/meta/*.json` – data lineage.

Before final submission:
1. Capture latest AAA/RBOB/Kalshi snapshots (document sources).
2. Pull the new WPSR release (if available) and rerun `make report`.
3. Rebuild memo/deck/zip and confirm SHA printed in the memo appendix (`git rev-parse HEAD`).

---

## Future improvements

- Automate CME/Nasdaq settle ingestion (requires paid Data Link plan or alternative licensed feed).
- Integrate EIA API retries/backoff when v2 endpoints return 500 immediately after release.
- Expand Kalshi prior to include additional bins (≥ $3.00/3.05/3.10/3.15) for finer interpolation.
- Add sensitivity dashboards for alternative thresholds (e.g., $3.00, $3.20) and regional AAA dispersion.
- Stand up daily CI job to rebuild memo when new data arrive (GitHub Actions + scheduled cron).

---

## Support & questions

- Daily ops checklist, manual logging, and reminder scripts are documented in `docs/RUNBOOK.md`.
- For API credential rotation or data snapshots, update `config` files and rerun `make report`.
- Questions / future work ideas: open an issue or annotate the README with new data links.

Good luck in the YUHA × Kalshi competition—this repository is ready to generate and defend an evidence-backed forecast, with clear instructions for refreshing data as new information arrives. ***
