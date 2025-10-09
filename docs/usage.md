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
```

## Workflow Targets

| Command | Description |
| --- | --- |
| `python -m kalshi_gas.cli report` | Run the full ETL → model → memo pipeline |
| `make report` | Convenience wrapper around the CLI |
| `make test` | Execute the pytest suite |
| `make lint` | Run Ruff (style) and mypy (typing) checks |

## Extending Data Sources

Add an ETL by implementing `Extractor`, `Transformer`, and `Loader`, then wiring it into `run_all_etl`. Store long-lived credentials in environment variables; the harness only promotes live requests when `KALSHI_GAS_USE_LIVE=1`.

## Reproducible Reports

Reports are rendered with Jinja templates at `src/kalshi_gas/reporting/templates`. Customise visuals by editing `reporting/visuals.py` or use additional figures by returning new paths in the CLI and referencing them in the template.
