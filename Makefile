.PHONY: install test run simulate

install:
	pip install -e .[dev]

test:
	pytest -q

run:
	uvicorn sort_drift_sentinel.api.main:app --reload --port 8010

simulate:
	python scripts/run_simulation.py
