PYTHON ?= python3

.PHONY: install lint test report models figures clean check-fresh

install:
	$(PYTHON) -m pip install -e .[dev]

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

report:
	$(PYTHON) -m kalshi_gas.cli report

check-fresh:
	$(PYTHON) scripts/check_freshness.py

models:
	$(PYTHON) -m pytest tests/test_prior_isotonic.py tests/test_structural_map.py tests/test_nowcast_sim.py tests/test_posterior_prob.py

figures:
	$(PYTHON) -m kalshi_gas.cli report

clean:
	rm -rf build data/raw/* data/interim/* data/processed/*
