PY ?= python

.PHONY: test test-data lint nb01 nb02 nb03 nb04 phase1 phase2 phase3 phase4 recover smoke

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

phase3:          ## Notebook 03 as a script: tide (G9), features, train/val windows, scalers
	$(PY) scripts/run_phase.py --phase 3

phase4:          ## Notebook 04 as a script: Granger selection (G10, H2)
	$(PY) scripts/run_phase.py --phase 4

nb04:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_granger.ipynb

nb03:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_features_split.ipynb

smoke:           ## Section 12 smoke test on synthetic data (needs TensorFlow)
	$(PY) scripts/smoke/smoke_pipeline.py --device cpu

nb02:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_link_analysis.ipynb

nb01:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_load_qc.ipynb

recover:         ## cloud-container fallback: rebuild data/raw from Drive connector downloads
	$(PY) scripts/drive/recover_downloads.py
