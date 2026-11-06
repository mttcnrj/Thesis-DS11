PY ?= python

.PHONY: test test-data lint nb01 nb02 phase1 phase2 recover

test:            ## synthetic tests only (< 1 min)
	$(PY) -m pytest -m fast -q

test-data:       ## tests on the real processed files
	$(PY) -m pytest -m data -q

lint:
	$(PY) -m ruff check pasig scripts tests

phase1:          ## Notebook 01 as a script: parse, QC, coverage, eval hours, gates G1 G2 G3 G5 G6
	$(PY) scripts/run_phase.py --phase 1

phase2:          ## Notebook 02 as a script: G4, G20, link analysis, H1 (G7), alert levels (G8)
	$(PY) scripts/run_phase.py --phase 2

nb02:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_link_analysis.ipynb

nb01:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_load_qc.ipynb

recover:         ## cloud-container fallback: rebuild data/raw from Drive connector downloads
	$(PY) scripts/drive/recover_downloads.py
