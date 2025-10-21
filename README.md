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
| Kalshi | `KALSHI_EMAIL`, `KALSHI_PASSWORD` |

Without these, the pipeline defaults to deterministic sample data.

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
