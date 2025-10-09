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

## Live Data

Set `KALSHI_GAS_USE_LIVE=1` to enable live HTTP pulls. Additional credentials:

| Source | Variables |
| --- | --- |
| EIA | `EIA_API_KEY` |
| Kalshi | `KALSHI_EMAIL`, `KALSHI_PASSWORD` |

Without these, the pipeline defaults to deterministic sample data.

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
