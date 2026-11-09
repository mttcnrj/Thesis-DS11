PY ?= python

.PHONY: test test-data lint nb01 nb02 nb03 nb04 nb05 phase1 phase2 phase3 phase4 phase5 rehearsal5 phase6 rehearsal6 nb06 phase7 rehearsal7 nb07 data-check recover smoke

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

phase5:          ## Notebook 05 on the chosen laptop (long; resumable per stage): G11-G17, G19; tags settings-frozen
	$(PY) scripts/run_phase.py --phase 5 --tag

rehearsal5:      ## Notebook 05 dry run with tiny settings, outputs under runs/rehearsal/ (never the real gates)
	PASIG_PROFILE=rehearsal $(PY) scripts/run_phase.py --phase 5

phase6:          ## Notebook 06: G18 opens the 2024 test season ONCE (after G17 + tag settings-frozen)
	$(PY) scripts/run_phase.py --phase 6

rehearsal6:      ## Notebook 06 dry run on the 2023 stand-in (2024 never read), outputs under runs/rehearsal/
	PASIG_PROFILE=rehearsal $(PY) scripts/run_phase.py --phase 6

phase7:          ## Notebook 07 after G18: walk-forward, ablations (D20), case study, Aviles 2024 (resumable per stage)
	$(PY) scripts/run_phase.py --phase 7

rehearsal7:      ## Notebook 07 dry run (2023 stands in for 2024), outputs under runs/rehearsal/
	PASIG_PROFILE=rehearsal $(PY) scripts/run_phase.py --phase 7

nb07:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/07_ablations_walkforward.ipynb

nb06:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/06_evaluate.ipynb

nb05:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/05_train.ipynb

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

data-check:      ## raw data in data/raw match the manifest (SHA-256); rebuild missing Phase 1/3 inputs
	$(PY) -c "from pasig.local import prepare; prepare(7)"

recover:         ## OBSOLETE since the raw data are committed; kept for the record (cloud-container Drive fallback)
	$(PY) scripts/drive/recover_downloads.py
