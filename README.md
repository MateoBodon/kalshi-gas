# Kalshi Gas Analytics

End-to-end analytics pipeline for U.S. gasoline markets combining AAA retail prices, EIA fundamentals, RBOB futures, and Kalshi market priors. The workflow builds an ensemble forecast, scores it with proper scoring rules, applies risk gates, and assembles a memo-ready report — all reproducible via `make report`.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
make report
```

The command pulls (or falls back to bundled samples for offline mode) each data source, trains the ensemble, runs a backtest, and writes artefacts to `build/`:

- `build/figures/*.png`: visuals for memo
- `build/memo/forecast_results.csv`: scored test set
- `build/memo/report.md`: rendered memo

## One-command Rebuild

Use `make report` at any time to rerun ETL, modelling, posterior generation, risk gating, and report compilation with the current data snapshots. The command is idempotent and will refresh figures and memo outputs in `build/`.

## How to Reproduce

1. Create or activate the virtual environment and install dependencies (see Quickstart).
2. (Optional) Update `data_raw/nhc_flag.yml` or `data_raw/wpsr_state.yml` with the latest analyst inputs if live data are gated.
3. Ensure any “as-of” timestamp is recorded in the memo by setting the ISO string you used for data pulls inside `data_raw/wpsr_state.yml` (add an `as_of:` field if desired). Judges can confirm the run time via the memo header’s `Generated on …` entry.
4. Run `make report` for a full rebuild or `make figures` if you only need refreshed PNGs.

The repo is fully self-contained; live API credentials are optional. All sample data are bundled, so the commands above will always succeed offline.

## Live Data

Set `KALSHI_GAS_USE_LIVE=1` to enable live HTTP pulls. Additional credentials:

| Source | Variables |
| --- | --- |
| EIA | `EIA_API_KEY` |
| Kalshi | `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`, `KALSHI_API_BASE` (default `https://api.elections.kalshi.com`), optional `KALSHI_SERIES_TICKER` and `KALSHI_EVENT_TICKER` |

Example session (RSA-PSS per Kalshi docs):

```bash
source .venv/bin/activate
export KALSHI_GAS_USE_LIVE=1
export EIA_API_KEY="<your_eia_key>"
export KALSHI_API_BASE="https://api.elections.kalshi.com"
export KALSHI_API_KEY_ID="<your_key_id>"
export KALSHI_PRIVATE_KEY_PATH="$HOME/.kalshi/kalshi_private_key.pem"
# AAA gas series/event (update monthly to last day):
export KALSHI_SERIES_TICKER="KXAAAGASM"
export KALSHI_EVENT_TICKER="KXAAAGASM-25OCT31"
make update-kalshi-bins && make report
```

EIA/RBOB default API series (with HTML fallbacks):
- RBOB weekly: `PET.RBRTWD.W`
- Gasoline stocks weekly: `PET.WGTSTUS1.W`
- Retail weekly US regular (optional): `PET.EMM_EPMRR_PTE_NUS_DPG.W`

## Fallback & Freshness

Each ETL component follows a deterministic fallback chain: live API → `data_raw/last_good.*` snapshot → bundled samples in `data/sample/`. Provenance sidecars under `data_proc/meta/` capture the current mode and as-of timestamps. Run `make check-fresh` to assert that the latest last-good snapshots meet freshness guardrails; sample mode automatically skips the check for offline runs.

> ℹ️ The EIA Weekly Petroleum Status Report generally posts Wednesdays at 10:30 ET; release times shift when U.S. federal holidays fall early in the week. Update your `last_good` snapshot accordingly if live pulls are delayed.

## Project Layout

- `src/kalshi_gas/etl`: Extract/transform/load tasks
- `src/kalshi_gas/data`: Dataset assembly logic
- `src/kalshi_gas/models`: Nowcast, pass-through, market-prior, and ensemble models
- `src/kalshi_gas/backtest`: Scoring rules and evaluation harness
- `src/kalshi_gas/reporting`: Visuals and memo builder
- `src/kalshi_gas/risk`: NHC/WPSR risk gate checks

## Development

```bash
make lint
make test
```

CI runs linting and tests on push (see `.github/workflows/ci.yml`).

## Acceptance & Repro

- CRPS-optimal (production) weights:
```bash
cat > config.crps.yaml <<'YML'
data:
  raw_dir: data/raw
  interim_dir: data/interim
  processed_dir: data/processed
  external_dir: data/external
  build_dir: build
model:
  ensemble_weights:
    nowcast: 0.0
    pass_through: 1.0
    market_prior: 0.0
  calibration_bins: 10
  horizon_days: 7
  prior_weight: 0.35
risk_gates:
  nhc_active_threshold: 1
  wpsr_inventory_cutoff: -3.0
YML
python -m kalshi_gas.cli report --config config.crps.yaml
```

- Freeze-date backtest and metrics at 3.10:
```bash
make freeze-backtest
python - <<'PY'
import json, pathlib
s=json.loads(pathlib.Path('data_proc/backtest_metrics.json').read_text())
print(json.dumps(s.get('per_threshold',{}).get('3.10',{}), indent=2))
PY
```
