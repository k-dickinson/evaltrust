PYTHON ?= python3

.PHONY: install test audit sbom

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

audit:
	$(PYTHON) -m evaltrust.cli audit examples/clean_win.json

sbom:
	$(PYTHON) -m pip install -q "cyclonedx-bom>=4.0"
	$(PYTHON) -m cyclonedx_py environment -o sbom.json
	@echo "Wrote sbom.json (CycloneDX)."
