PYTHON ?= python3

.PHONY: install lint test report clean

install:
	$(PYTHON) -m pip install -e .[dev]

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

report:
	$(PYTHON) -m kalshi_gas.cli report

clean:
	rm -rf build data/raw/* data/interim/* data/processed/*
