.PHONY: install test audit clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest

audit:
	evaltrust audit examples/clean_win.json --plain

clean:
	rm -rf .pytest_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
