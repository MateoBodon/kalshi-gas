# Usage Guide

## Configuration

Pipeline settings live in `kalshi_gas.config.DEFAULT_CONFIG`. Override by supplying a YAML file and passing `--config path/to/config.yml` to the CLI.

```yaml
data:
  build_dir: build
model:
  ensemble_weights:
    nowcast: 0.5
    pass_through: 0.3
    market_prior: 0.2
  calibration_bins: 12
  horizon_days: 7
risk_gates:
  nhc_active_threshold: 2
  wpsr_inventory_cutoff: -2.0
event:
  name: AAA > $3.10 on Oct 31, 2025
  resolution_date: 2025-10-31
  threshold: 3.10
```

## Workflow Targets

| Command | Description |
| --- | --- |
| `python -m kalshi_gas.cli report` | Run the full ETL → model → memo pipeline |
| `make report` | Convenience wrapper around the CLI |
| `make test` | Execute the pytest suite |
| `make lint` | Run Ruff (style) and mypy (typing) checks |
| `make live-check` | Diagnose live sources: status, probes, and env hints |
| `python scripts/log_prompt.py` | Interactive prompt for manual AAA/EIA logging |
| `python scripts/log_manual_label.py --operator MB --aaa 3.112 --eia 3.115` | Append a manual AAA/EIA observation |
| `python scripts/bootstrap_last_good.py` | Promote offline CSVs to `last_good.*` snapshots |
| `python scripts/update_wpsr_state.py --latest-change -3.8 --refinery-util 88.5 --product-supplied 8.4` | Refresh risk-gate inventory context |
| `python scripts/update_nhc_flag.py --flag --note "Disturbance in Gulf"` | Toggle the NHC analyst override |
| `python scripts/simulate_market_data.py` | Generate a full synthetic dataset for offline modelling |

## Extending Data Sources

Add an ETL by implementing `Extractor`, `Transformer`, and `Loader`, then wiring it into `run_all_etl`. Store long-lived credentials in environment variables; the harness only promotes live requests when `KALSHI_GAS_USE_LIVE=1`.

### Kalshi RSA (Trade API v2)

The ETL and the `update_kalshi_bins` script support signed requests with:

- `KALSHI_API_KEY_ID`
- `KALSHI_PRIVATE_KEY_PATH` (PEM) or `KALSHI_PY_PRIVATE_KEY_PEM` (inline PEM)
- Optional: `KALSHI_PRIVATE_KEY_PASSPHRASE`
- `KALSHI_API_BASE` (default `https://api.elections.kalshi.com`)

If v2 fails, the ETL falls back to v1 (session login) or sample mode.

### EIA API

Set `EIA_API_KEY` and the ETL will pull the weekly series. If the API returns 403, the pipeline falls back to last_good or bundled samples; you can snapshot HTML using `make pull-eia-html` and wire an HTML fallback if desired.

## Reproducible Reports

Reports are rendered with Jinja templates at `src/kalshi_gas/reporting/templates`. Customise visuals by editing `reporting/visuals.py` or use additional figures by returning new paths in the CLI and referencing them in the template.

For daily operations and manual logging, see `docs/RUNBOOK.md`.

### Live-mode robustness

When live data are sparse (e.g., one live AAA day, EIA blocked), the pipeline trains the model on the stable sample frame, anchors the nowcast to the live AAA level, and uses the live Kalshi bins prior. Figures and memo still generate without crashing.
