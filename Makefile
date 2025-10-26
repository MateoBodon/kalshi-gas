PYTHON ?= python3

.PHONY: install lint test report deck models figures clean check-fresh calibrate freeze-backtest sweep-ensemble update-kalshi-bins report-live pull-eia-html pull-wpsr-html live-check
.PHONY: log-prompt log-entry daily-run

install:
	$(PYTHON) -m pip install -e .[dev]

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

report: .mplcache
	MPLBACKEND=Agg MPLCONFIGDIR=$(CURDIR)/.mplcache XDG_CACHE_HOME=$(CURDIR)/.mplcache $(PYTHON) -m kalshi_gas.cli report

deck:
	$(PYTHON) -m kalshi_gas.reporting.deck

check-fresh:
	$(PYTHON) scripts/check_freshness.py

models:
	$(PYTHON) -m pytest tests/test_prior_isotonic.py tests/test_structural_map.py tests/test_nowcast_sim.py tests/test_posterior_prob.py

figures:
	$(PYTHON) -m kalshi_gas.cli report

clean:
	rm -rf build data/raw/* data/interim/* data/processed/*

calibrate: .mplcache
	MPLBACKEND=Agg MPLCONFIGDIR=$(CURDIR)/.mplcache XDG_CACHE_HOME=$(CURDIR)/.mplcache $(PYTHON) -m kalshi_gas.backtest.calibrate_prior

freeze-backtest:
	$(PYTHON) -m kalshi_gas.pipeline.backtest

sweep-ensemble:
	$(PYTHON) -m kalshi_gas.backtest.sweep_ensemble

update-kalshi-bins:
	$(PYTHON) scripts/update_kalshi_bins.py

# Convenience live run (requires env: KALSHI_GAS_USE_LIVE=1, EIA_API_KEY, and Kalshi creds if used)
report-live:
	KALSHI_GAS_USE_LIVE=1 $(PYTHON) -m kalshi_gas.cli report

# Snapshot helpers for WPSR/EIA HTML (for parser tests or delayed API days)
pull-wpsr-html:
	$(PYTHON) scripts/freeze_wpsr_html.py

pull-eia-html:
	$(PYTHON) scripts/freeze_eia_html.py

live-check:
	$(PYTHON) scripts/live_check.py

log-prompt:
	$(PYTHON) scripts/log_prompt.py

log-entry:
	$(PYTHON) scripts/log_manual_label.py --operator $(OP) --aaa $(AAA) --eia $(EIA) --gasbuddy $(GB) --notes $(NOTES)

daily-run: log-prompt report calibrate

.mplcache:
	mkdir -p .mplcache
