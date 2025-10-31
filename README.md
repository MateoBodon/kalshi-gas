# Kalshi Gas Analytics

Operational forecasting and reporting stack for the **Kalshi × YUHA Market Research & Modeling Competition**. The project ingests authoritative energy datasets, applies a horizon-aware ensemble, and produces freeze-ready memo / deck artifacts with full provenance.

---

## Latest Forecast Snapshot

- **As of:** 2025‑10‑30 (20:45 UTC rebuild – freeze date)
- **Event:** AAA US average regular gasoline price > **$3.10** on **2025‑10‑31**
- **Posterior mean:** **$3.0298** per gallon (±$0.01 80 % CI)  
- **Tail probability:** **0.30 %** at the $3.10 threshold (rounded; matches submitted memo)  
- **Residual σ:** **0.0098** (USD/gal, <2¢ daily standard deviation)  
- **Alpha lift:** **$0.0000** (clamped for T+1; narrative memo discusses the +3¢ risk flag)  
- **β<sub>eff</sub>:** **0.0000** (pass-through suppressed at D≤1)  
- **Prior weight:** configured **0.10**, **effective 0.00** at D=1 (gap z-score **6.3σ**)  
- **Freeze deliverables:** `reports/memo.md`, `reports/figures/*`, `SUBMISSION.md`, `data_proc/summary.json`, `Mateo_Ly_Kalshi_Submission.pdf`

The snapshot aligns with the uploaded PDF memo (`Mateo_Ly_Kalshi_Submission.pdf`) and `SUBMISSION.md`. Re-run `make report --force` if you need to regenerate the artefacts; the probability rounding here mirrors the presentation layer used in the final submission.

---

## Key Capabilities

- **End-to-end ETL:** AAA resolver fallback (HTML scrape), EIA WPSR fundamentals, CME/Nasdaq RBOB settles, Kalshi bin probabilities, and manual risk flags. Provenance written to `data_proc/meta/*.json`.
- **Ensemble modelling:** Combines nowcast, structural pass-through, and market prior components with calibrated drift bounds.
- **Risk-aware adjustments:**
  - Alpha lift gated to **0** at `days_to_event ≤ 1`
  - Pass-through β scaled by `_beta_scale_for_horizon`, near-zero at T+1
  - Prior weight collapse via `_effective_prior_weight` when horizon reaches resolution or the price gap exceeds 5σ
- **Freeze artefacts:** Auto-generated memo/deck with snapshot tables, sensitivity bars, calibration plots, risk dashboard, and freeze-date metrics (including effective prior weight & gap z-score).
- **Test coverage:** Unit and integration tests for ETL, posterior math, horizon gating, sigma loading, and prior collapse.

---

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/kalshi_gas/` | Core pipeline modules (ETL, modelling, reporting) |
| `data_raw/` | Live & fallback inputs (AAA, WPSR, RBOB, Kalshi, risk flags) |
| `data_proc/` | Derived artefacts (`summary.json`, sensitivity grids, meta logs) |
| `reports/` | Final memo/deck and exported figures |
| `tests/` | Pytest suite, including T+1 alpha/beta and prior gating tests |
| `scripts/` | Helpers for updating Kalshi bins, RBOB settles, and WPSR snapshots |
| `Makefile` | Task runner for report builds, calibration, testing, linting |

---

## Final Submission Artifacts

- `Mateo_Ly_Kalshi_Submission.pdf` — formatted memo submitted to the competition.
- `SUBMISSION.md` — plaintext summary mirroring the memo headline call.
- `reports/memo.md` — reproducible Markdown export from `make report --force`.
- `reports/figures/` — PNG figures embedded in both memo variants.
- `data_proc/summary.json` — machine-readable freeze metrics (probability, gap, σ, priors).

Refer to these files when packaging or re-verifying the deliverable set.

---

## Installation & Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Optional environment variables for live fetches:

```bash
export KALSHI_GAS_USE_LIVE=1
export EIA_API_KEY="..."
export KALSHI_API_KEY_ID="..."
export KALSHI_PRIVATE_KEY_PATH="$HOME/.kalshi/kalshi_private_key.pem"
```

---

## Running the Pipeline

1. **Update inputs** (only if you have fresh data):
   ```bash
   python scripts/update_kalshi_bins.py
   python scripts/update_rbob_csv.py --date YYYY-MM-DD --settle <value> --source "CME settle"
   python scripts/freeze_wpsr_html.py
   # Provide AAA override via data_raw/last_good.aaa.csv if needed
   ```

2. **Generate memo and figures** (full rebuild):
   ```bash
   make report          # forces ETL → modelling → memo/deck/figures
   ```

3. **Optional calibration and freeze metrics**:
   ```bash
   make calibrate       # posterior prior sweep + residual sigma estimate
   make freeze-backtest # freeze-date backtests and carry comparison
   ```

4. **Quality gates**:
   ```bash
   make test            # pytest suite (posterior, ETL, horizon gating)
   make lint            # ruff + mypy (configured in Makefile)
   ```

All commands write artefacts under `data_proc/` and `reports/`. Re-running `make report` logs a concise summary (prior weight, tail probability, figures path).

---

## Data Inputs & Overrides

| Feed | Location | Notes |
| --- | --- | --- |
| AAA national average | `data_raw/last_good.aaa.csv` | Primary resolver; ETL parses HTML fallback if live fetch blocked. |
| RBOB front settle | `data_raw/last_good.rbob.csv` | Update via `scripts/update_rbob_csv.py`. |
| EIA WPSR fundamentals | `data_raw/wpsr_summary.html` | Capture weekly HTML for repeatable parsing. |
| Kalshi bin probabilities | `data_raw/kalshi_bins.yml` | Populated via API script; used for prior CDF. |
| Risk gates | `data_raw/wpsr_state.yml`, `data_raw/nhc_flag.yml` | Manual toggles for tightness / storm overrides. |

Every ETL run emits provenance JSON (source, mode, as-of date) to `data_proc/meta/`.

---

## Modelling Details

- **Nowcast:** Rolling regression with configurable drift bounds; anchored to latest AAA observation when available.
- **Structural pass-through:** Fits asymmetric β coefficients (`fit_structural_pass_through`) on the assembled dataset.
- **Alpha adjustments:** Tightness and NHC gates introduce lifts that are **clamped to zero inside `_apply_alpha_beta_adjustments`** when `days_to_event ≤ 1`.
- **Beta horizon scaling:** `_beta_scale_for_horizon` scales β toward zero as the horizon shrinks; β parameters are pre-scaled before posterior simulation.
- **Prior gating:** `_effective_prior_weight` computes an effective prior weight and gap z-score. At T+1 the effective weight collapses to 0, ensuring the tail probability reflects likelihood alone.
- **Posterior:** `PosteriorDistribution` mixes the simulated likelihood with the effective prior, clips samples to a plausible retail range, and generates sensitivity grids.

---

## Deliverables & Freeze Checklist

1. Run `make report --force` to refresh memo/deck/figures.
2. Confirm `reports/memo.md` “Freeze-date Parameters” shows:
   - α lift = 0.0000, β_eff ≈ 0.0000
   - Prior weight ≤ 0.10, Effective prior weight = 0.00
   - Residual σ < 0.02, Gap z-score > 5σ
   - Tail probability `≈0.3%` (rounded presentation value)
3. Ensure `data_proc/summary.json` mirrors memo metrics (mean, tail, α/β, prior weights).
4. Run `make freeze-backtest` and verify posterior Brier ≤ carry at the $3.10 threshold.
5. Archive:
   - `reports/memo.md`
   - `reports/figures/*`
   - `SUBMISSION.md`
   - Git SHA referenced in memo appendix / summary JSON

---

## Testing & CI Readiness

- `tests/test_freeze_t1.py` — validates alpha clamp, beta scaling, sigma loader, and prior mean behaviour.
- `tests/tests_prior_gate.py` — asserts prior collapse at D=1 and retention at D=2.
- Additional suites cover ETL parsers, posterior probabilities, structural model fits, and configuration defaults.

Add the following to CI if desired:

```bash
make lint
make test
make report
```

---

## Support & Further Work

- Operational procedures and credential handling: see `docs/RUNBOOK.md`.
- Future enhancements: automated CME ingest, CI-triggered memo diffs, expanded sensitivity scenarios, and regional AAA spread analysis.
- For issues or new data sources, please open a GitHub issue or submit a PR with accompanying tests.

This README reflects the repository’s final competition submission state—ready to reproduce, audit, and defend the gasoline price forecast with full transparency.
