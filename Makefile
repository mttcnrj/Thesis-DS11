PY ?= python

.PHONY: test test-data lint nb01 phase1 recover

test:            ## synthetic tests only (< 1 min)
	$(PY) -m pytest -m fast -q

test-data:       ## tests on the real processed files
	$(PY) -m pytest -m data -q

lint:
	$(PY) -m ruff check pasig scripts tests

phase1:          ## Notebook 01 as a script: parse, QC, coverage, eval hours, gates G1 G2 G3 G5 G6
	$(PY) scripts/run_phase.py --phase 1

nb01:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_load_qc.ipynb

recover:         ## cloud-container fallback: rebuild data/raw from Drive connector downloads
	$(PY) scripts/drive/recover_downloads.py
