# Kalshi Gas Analytics

Forecast and reporting toolkit for the YUHA × Kalshi Market Research & Modeling Competition. This workspace ingests authoritative public data (AAA resolver, EIA fundamentals, CME/Nasdaq RBOB futures, Kalshi market priors), runs the calibrated ensemble, and auto-generates submission-ready memo/deck artifacts.

---

## Current Snapshot *(as of 2025‑10‑30 20:25 UTC)*

- **Target:** AAA US national regular gasoline price resolving on **2025‑10‑31** at the **$3.10/gal** threshold.
- **Final probability:** **7.96 %** that AAA > $3.10; posterior mean **$3.0708** with a prior weight of **0.20** (file-calibrated).
- **Pass-through settings:** horizon-adjusted **β_eff = 0.0004**; active **α-lift = +$0.0043/gal**.
- **Horizon sensitivities (T+1):**
  - RBOB ±$0.05 → ΔP ≈ 0.00 pp (β_eff very small at 1‑day horizon).
  - α +$0.02 → ΔP ≈ +13.67 pp; α −$0.02 → ΔP ≈ 0.00 pp (lift already at floor).
- **Key drivers:** WPSR gasoline stocks draw **5.94 mmbbl** and refinery utilisation **89.6 %** triggered the tightness gate; no NHC overrides.
- **Carry check:** Freeze-date posterior Brier at the $3.10 threshold is **0.0000**, matching the carry benchmark (≤ carry as required).

### Live Data Status

| Feed | Source | Latest as-of | Notes |
| --- | --- | --- | --- |
| AAA national average | https://gasprices.aaa.com (HTML fallback) | **2025‑10‑30** | Resolver; ETL captures last_good snapshot + provenance. |
| WPSR fundamentals | https://www.eia.gov/petroleum/supply/weekly/ | **2025‑10‑24** | Next release expected 2025‑10‑31 10:30 ET. |
| RBOB settle (front) | CME settle (manual entry) | **2025‑10‑30** | Use `scripts/update_rbob_csv.py` to append official settles. |
| Kalshi bins | Trade API v2 (RSA-PSS) | **2025‑10‑30** | `scripts/update_kalshi_bins.py` hydrates ≥ $3.05/$3.10/$3.15 bins. |
| NHC flag | `data_raw/nhc_flag.yml` | OFF | Manual override for storm risk (currently false). |

### Backtest & Calibration Highlights

- Posterior ensemble (full history): Brier **0.0051**, CRPS **0.0083** vs carry Brier **0.0050**.
- Freeze-date schedule (central threshold 3.10): Posterior Brier **0.0000**, CRPS **0.0576**; prior weight calibration verified at **0.20** with matching dataset digest.
- Calibration plot & backtest tables in the memo confirm posterior ≤ carry.

---

## Outputs & Artifacts

- `reports/memo.md` – final memo (strict thesis line, sensitivity table, SHA).
- `reports/figures/` – PNG figures (risk dashboard, posterior, calibration, sensitivities).
- `SUBMISSION.md` – freeze submission cover sheet (thesis, sources, ensemble summary, probability, SHA, reproduce command).
- `data_proc/summary.json` – machine-consumable snapshot including β_eff, α-lift, T+1 sensitivities.

---

## Reproducing the Forecast

1. **Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```
2. **Optional live credentials**
   ```bash
   export KALSHI_GAS_USE_LIVE=1
   export EIA_API_KEY="..."
   export KALSHI_API_KEY_ID="..."
   export KALSHI_PRIVATE_KEY_PATH="$HOME/.kalshi/kalshi_private_key.pem"
   ```
3. **Refresh live snapshots (if permitted)**
   ```bash
   python scripts/update_kalshi_bins.py                 # Kalshi API (RSA-PSS)
   python scripts/update_rbob_csv.py --date YYYY-MM-DD --settle <value> --source "CME settle"
   python scripts/freeze_wpsr_html.py                   # capture latest WPSR summary
   ```
   AAA scraping is handled in the ETL (fallback to `data_raw/last_good.aaa.csv` if you have manual values).
4. **Rebuild pipeline**
   ```bash
   make report            # ETL → posterior → memo/figures (writes reports/memo.md)
   make calibrate         # optional prior sweep (writes data_proc/prior_weight.json)
   make freeze-backtest   # refresh freeze metrics for carry comparison
   ```
5. **Validate**
   ```bash
   make test              # 36 unit & integration tests
   make lint              # optional ruff + mypy (see Makefile)
   ```

---

## Data Ingest & Manual Overrides

- **AAA national average:** live JSON blocked; ETL parses HTML with user-agent spoof. Provide manual backups in `data_raw/last_good.aaa.csv` if necessary.
- **RBOB settles:** append to `data_raw/last_good.rbob.csv` via `scripts/update_rbob_csv.py`. Second-month spreads supported.
- **WPSR fundamentals:** store latest HTML snapshot with `scripts/freeze_wpsr_html.py`; ETL extracts refinery utilisation, stock draws, product supplied.
- **Kalshi bins:** requires API key + RSA private key; falls back to `data_raw/kalshi_bins.yml`.
- **Storm/analyst flags:** toggle `data_raw/nhc_flag.yml` and `data_raw/wpsr_state.yml`; pipeline uses these for risk gates and α-lift adjustments.
- **Provenance:** sidecars in `data_proc/meta/*.json` log as-of dates, modes, and record counts for audit.

---

## Submission Checklist

Before freezing a submission:

1. Sync live feeds (AAA, WPSR, RBOB, Kalshi) and ensure provenance files reflect the latest timestamps.
2. Run `make report` (forces rebuild) and confirm:
   - Memo headline shows **As of 2025‑10‑30** (or your new date) with Final P%.
   - Sensitivity table under “Model Outputs” matches `data_proc/summary.json`.
   - Figures carry the updated footer `AAA daily (USD/gal); EIA WPSR weekly (mmbbl); CME RB settle ($/gal)`.
3. Verify `data_proc/prior_weight.json` aligns with the current dataset digest and weight (0.20).
4. Run `make freeze-backtest` and ensure posterior Brier ≤ carry at the central threshold.
5. Record the SHA printed in memo appendix / `SUBMISSION.md`.
6. Package deliverables (`reports/memo.md`, `reports/figures/*`, `SUBMISSION.md`) per competition instructions.

---

## Future Improvements

- Automate CME/Nasdaq settle ingestion via authorized data link.
- Add CI job to rebuild memo after each AAA/WPSR release and surface diffs.
- Extend T+1 sensitivity grid to alternative thresholds (e.g., $3.00, $3.20) and regional AAA dispersion.
- Harden Kalshi API retries/backoff for post-maintenance outages.

---

## Support

- Operational runbook, manual logging, credential handling: see `docs/RUNBOOK.md`.
- For questions or issues, open an issue or annotate the README with new data links/requirements.

Good luck in the competition—this repo is ready to generate and defend an evidence-backed gasoline forecast with full provenance. 
