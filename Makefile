PYTHON ?= python

.PHONY: test validate validate-quick check

test:
	$(PYTHON) -m unittest discover -s tests -v

validate:
	$(PYTHON) scripts/validate_results.py

validate-quick:
	$(PYTHON) scripts/validate_results.py --quick

check: test validate
