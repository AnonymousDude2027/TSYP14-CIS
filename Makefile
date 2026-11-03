# TSFDI - reproducibility entry points.
#
#   make all           setup + data verify + gonogo + calibrate + experiments + report
#   make setup         create a virtual environment and install pinned requirements
#   make setup-all     setup + the optional dev/app/report requirements
#   make report-pdf    typeset report, appendix and the documentation bundle (XeLaTeX)
#   make data-verify   check the committed bundle against the signed manifest
#   make gonogo        run the go/no-go experiment first (it gates the claims)
#   make calibrate     fit and calibrate every model (writes results/cache)
#   make experiments   standard profile, grid, baselines, ablations, evasion, diagnosis
#   make report        tables, figures and docs/report.md from results/raw
#   make verify        compare results/ with expected_results.json (LAST step of the chain:
#                      it is only a reproduction check after experiments/report have run)
#   make dashboard     launch the Streamlit dashboard
#   make static        write results/static_report.html (no Streamlit needed)
#   make test          unit tests (no network)
#   make bench         provenance overhead benchmark
#   make clean         remove generated results (keeps the bundle)

PY ?= python3
VENV ?= .venv
export TSFDI_ROOT := $(CURDIR)
export PYTHONPATH := $(CURDIR)/src

.PHONY: all setup setup-all data-verify gonogo calibrate experiments report report-pdf verify dashboard static test bench clean sites

all: data-verify gonogo calibrate experiments report verify static
	@echo "== all done (add `make report-pdf` for the PDFs in docs/) =="

setup:
	$(PY) -m venv $(VENV) && . $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "activate with: source $(VENV)/bin/activate"

setup-all: setup
	. $(VENV)/bin/activate && pip install -r requirements-dev.txt -r requirements-app.txt -r requirements-report.txt

data-verify:
	$(PY) -m tsfdi data-verify

gonogo:
	$(PY) -m tsfdi gonogo

calibrate:
	$(PY) -m tsfdi calibrate

experiments:
	$(PY) -m tsfdi experiments

report:
	$(PY) -m tsfdi report-figures
	$(PY) -m tsfdi report-text

report-pdf:
	$(PY) -m tsfdi report-pdf

verify:
	$(PY) -m tsfdi verify

bench:
	$(PY) -m tsfdi bench

sites:
	$(PY) -m tsfdi sites

test:
	$(PY) -m pytest -q tests

dashboard:
	streamlit run src/tsfdi/app/dashboard.py

static:
	$(PY) -m tsfdi report-static

clean:
	rm -rf results/cache results/raw results/tables results/figures results/*.csv results/*.json results/*.md results/static_report.html
