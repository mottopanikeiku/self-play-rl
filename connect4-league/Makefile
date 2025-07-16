.PHONY: all clean test report
PYTHON := python3.11
VENV   := .venv

all: $(VENV)/bin/activate
	$(VENV)/bin/python scripts/train_agents.py
	$(VENV)/bin/python scripts/tournament.py
	$(VENV)/bin/python scripts/report.py

$(VENV)/bin/activate: requirements.txt pyproject.toml
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	touch $(VENV)/bin/activate

test: $(VENV)/bin/activate
	$(VENV)/bin/pytest tests/

clean:
	rm -rf $(VENV) data/*.pt data/*.json data/*.gif

.DEFAULT_GOAL := all 