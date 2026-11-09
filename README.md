# pasig-forecast — Pasig River (EFCOS Pandacan) 1–6 h water-level forecasting

Thesis code for *Pasig River (EFCOS Pandacan) 1–6 h Water-Level Forecasting as a Flood-Risk Indicator for
Sampaloc* (Baltazar, Cañarejo, Serrano; adviser Gernale).

A hybrid LSTM–GRU network forecasts the hourly river level at the MMDA EFCOS Pandacan gauge 1–6 h ahead
(wet season, June–November) from the gauge's own history, the predicted tide, the clock and PAGASA Science
Garden weather. One forecast drives two alerts: **high water ≥ 11.69 m** and **flood risk ≥ 12.00 m**, both
derived from training data only. It is compared with simple baselines, a standalone LSTM and alert
classifiers on the 2024 season, which is opened once.

* **Specification:** `IMPLEMENTATION_PLAN.md`. Decision IDs (D1–D20), gate IDs (G0–G20) and contradiction IDs
  (C1–C6) from it are used throughout the code, configs and `gates/`.
* **Handover summary of the build session:** `docs/session_context.md` — start here in a new chat.
* **Data are in the repository:** the raw MMDA and PAGASA Science Garden files (31 MB, 465 files) are committed
  in `data/raw/`, so a clone runs end to end without Google Drive (decision R1). Every notebook and
  `run_phase.py` first checks them against the SHA-256 manifest and, in a fresh clone, rebuilds the generated
  inputs (Phases 1 and 3, about 1–2 minutes, once).

## Status

| Phase | Notebook / command | State |
|---|---|---|
| 1 Load + QC | `notebooks/01_load_qc.ipynb` / `make phase1` | **done** — G1, G2, G3, G5, G6 passed |
| 2 Link analysis, alert levels | `notebooks/02_link_analysis.ipynb` / `make phase2` | **done** — G4, G20, G7, G8 |
| 3 Tide, features, windows | `notebooks/03_features_split.ipynb` / `make phase3` | **done** — G9, train/val windows, scalers |
| 4 Granger selection | `notebooks/04_granger.ipynb` / `make phase4` | **done** — G10 |
| 5 Training | `notebooks/05_train.ipynb` / `make phase5` | **built, dry-run passes**; real run on the laptop (G11–G17, G19) |
| 6 Test evaluation (2024) | `notebooks/06_evaluate.ipynb` / `make phase6` | **built, dry-run passes** on a 2023 stand-in; **2024 not opened** (G18) |
| 7 Robustness | `notebooks/07_ablations_walkforward.ipynb` / `make phase7` | **built, dry-run passes**: walk-forward, 6 ablations incl. D20, case study, Aviles 2024 |

Phases 1–4 were run on the real data in a cloud container; they are deterministic, so re-running them on the
laptop must reproduce the committed gates and tables. All final results (Notebooks 05–07) must come from the
chosen laptop (D16). Tests: `pytest` → 102 passed, 4 skipped (the skips need laptop outputs).

### Results so far (real data)

| Item | Result | Where |
|---|---|---|
| Rain timing (G3, D5) | 6-hourly stamps mark the period end (99.7% of rain days); rain usable 1 h after | `gates/G03_*` |
| Upstream stations (G5, D2) | none eligible (Sto. Niño 78.1% in 2024; Fort Santiago 0% in 2018) | `reports/tables/eligibility.csv` |
| Status legend (G4, D3) | gate and pump codes used (amended checks, P2-a); sky codes dropped | `gates/G04_*` |
| **H1** (G7) | **partly supported**: OR 1.207 per 0.1 m after rain (CI 1.143–1.275); 2023 AUC gain 0.004 (CI −0.022 to 0.028) | `gates/G07_*` |
| Alert levels (G8) | **11.69 m** (P95 of training) and **12.00 m** (≥ 50% stress hours) | `configs/alert_levels.yaml` |
| Tide (G9, D7) | P1, K2, Q1 added; residual peaks remain (logged) | `artifacts/tide_*.json` |
| **H2** (G10) | **supported**: rain (p = 6.1e-48) and pressure (p = 2.2e-6) selected | `configs/selected_features.yaml` |
| Table III-6 | recomputed (train 16,937 cases at lead 1; val 4,091) | `reports/tables/table_III-6_recomputed.csv` |
| H3, H4 | not yet — decided in Notebook 06 on the laptop by the rules in `configs/eval.yaml` | — |

Every data-vs-plan finding is in `reports/transparency_log.md` (F1–F30); every decision and implementation
choice in `docs/decisions.md` (D1–D20, P1–P7).

## Ground rules (Plan Section 0.3)

* Splits: train 2018–2022, validation 2023, test 2024; chronological, wet season only.
* Data: MMDA files and PAGASA **Science Garden** only (no Port Area, no Kaggle, no other data).
* Everything learned from data comes from training seasons (2023 for tuning only). **The 2024 test season is
  read once, in Notebook 06, after the `settings-frozen` tag** — enforced by gate G18 and by tests.
* No SMOTE or synthetic training data. Results are reported whichever way they fall.

## Setup (chosen laptop: Ryzen 7 7735HS, 16 GB, RTX 4060 Laptop GPU 8 GB → WSL2 Ubuntu)

Full step-by-step instructions: **`docs/laptop_runbook.md`**.

```bash
# inside WSL2 Ubuntu; keep the repo in ~/, not /mnt/c
git clone https://github.com/mttcnrj/Thesis-DS11.git ~/pasig-forecast && cd ~/pasig-forecast
git checkout feat/pasig-forecast-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements-dev.txt "tensorflow[and-cuda]" joblib
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"   # must not be []
make data-check                   # raw data match the manifest; builds data/interim + data/processed once
pip freeze > requirements.txt     # commit; never upgrade mid-thesis
```

## Run order

Long jobs run as scripts (inside `tmux`), not notebook cells. Every phase can also be run from its notebook
(VS Code or Jupyter, opened from the repository), which calls the same functions and ends with that phase's
tests. Notebooks can be opened in any order once their gates exist: missing generated inputs are rebuilt
automatically from `data/raw/`.

```bash
make test                                          # pytest -m fast (synthetic, < 1 min)
make smoke                                         # Section 12 smoke test (7 PASS); also --device gpu
python scripts/run_phase.py --phase 1 --tag        # G1 G2 G3 G5 G6; tags phase1-frozen
python scripts/run_phase.py --phase 2              # G4 G20 G7 G8 -> configs/alert_levels.yaml
python scripts/run_phase.py --phase 3              # G9, features, train/val windows, scalers
python scripts/run_phase.py --phase 4              # G10 -> configs/selected_features.yaml
make test-data                                     # tests on the real processed files
python scripts/run_phase.py --phase 5 --stage compute      # G14 -> G13 -> G15 (benchmark)
python scripts/run_phase.py --phase 5 --stage variants     # G11, G12
python scripts/run_phase.py --phase 5 --stage grid         # 60 hybrid + 18 LSTM configs (long)
python scripts/run_phase.py --phase 5 --stage finals       # 5 seeds each, determinism on
python scripts/run_phase.py --phase 5 --stage baselines
python scripts/run_phase.py --phase 5 --stage thresholds
python scripts/run_phase.py --phase 5 --stage checks       # G16 — REVIEW before freezing
python scripts/run_phase.py --phase 5 --stage freeze --tag # G17, tag settings-frozen
python scripts/run_phase.py --phase 6              # G18: opens 2024 ONCE (--stage unseal|predict|evaluate)
python scripts/run_phase.py --phase 7              # --stage walkforward|ablations|casestudy|aviles
```

* Stages are resumable: finished configs/seeds are skipped on a re-run.
* Re-running `predict`/`evaluate` of Phase 6 after a crash does not re-open the test; re-running `unseal`
  needs `--rerun-reason "..."`, which is logged and must be reported in the transparency log.

### Dry runs (rehearsal profile)

```bash
make rehearsal5    # PASIG_PROFILE=rehearsal: 2 epochs, 1,500 cases, 2 configs per grid, 2 seeds
make rehearsal6    # 2023 validation windows stand in for 2024 (2024 is never read)
make rehearsal7
```

Rehearsal outputs (gates, generated configs, checkpoints, tables, figures) go to `runs/rehearsal/`
(git-ignored) and never touch the real gates. **Rehearsal numbers are not results.**

## Gates

Each gate writes `gates/G##_<name>.json` (rule, evidence, decision, timestamp, git SHA, profile);
`require_gates()` refuses to start a phase whose prerequisites are missing or failed.

| Gate | Phase | Gate | Phase |
|---|---|---|---|
| G0 machine | before anything | G11 tidal input variant, G12 output form | NB05 |
| G1 raw files, G2 parsing, G3 rain timing | NB01 | G13 device, G14 determinism, G15 seed budget | NB05 |
| G5 eligibility, G6 phase-1 frozen | NB01 | G16 persistence sanity + bug checklist | NB05 |
| G4 status legend, G20 Quiapo 2021 | NB02 | G17 settings frozen (`frozen_manifest.json`) | NB05 |
| G7 H1, G8 alert levels | NB02 | G18 test unsealed once | NB06 |
| G9 tide fit | NB03 | G19 hyperparameter edge | NB05 |
| G10 Granger outcome | NB04 | | |

## Layout

```
configs/    hand-written: paths, splits (incl. walk-forward folds), stations, qc, tide, features, granger,
            link, model_hybrid, baselines, eval (H3/H4 rules, NB06/NB07 settings)
            GENERATED by gates: pagasa, eligible_inputs, alert_levels, selected_features,
            (laptop) model_inputs, compute, thresholds, frozen_manifest.json
gates/      G##_<name>.json (committed)
artifacts/  tide_<station>.json, scalers.json, feature_names.json; frozen/ (checkpoints, git-ignored, hashed)
pasig/      config.py gates.py phase1.py … phase7.py      (one orchestrator per notebook)
  io/       manifest (G1), mmda_waterlevel (EFCOS, G2), mmda_pumping (Aviles/Quiapo, D3), pagasa (D4, D5, G3)
  qc/       rules (range, spike, flat runs), gaps (<= 3 h fill), coverage (G5)
  link/     stats (Wilson, events, bootstraps, clustered logit), legend (G4), quiapo (G20)
  features/ tide (D7, G9), windows (Section 9.1, D8), scaling (D9)
  selection/ stationarity, granger, correlation (Section 9.2)
  models/   hybrid (Figure III-4, D9, D20 loss), baselines (Table III-4), offsets (D11)
  train/    data, loop (run folders, determinism D15), grid (D12), compute (G13-G15)
  eval/     metrics, alerts (D19 equal recall), events, dm_test, bootstrap (Section 10.1)
  local.py  prepare(): raw-data check against the manifest + rebuild of missing local inputs
notebooks/  01–07 (thin: call pasig, show results, run the phase's tests)
scripts/    run_phase.py, record_g0.py, smoke/ (from smoke_tests.zip); sync_drive.py and drive/ only if the
            Drive source files ever change (not needed: the raw data are committed)
tests/      19 files: raw_data, parsers, time_alignment, qc, splits, gapfill, gates, link, windows, leakage, granger,
            reproducibility, offsets, same_hours, metrics, events, dm, unseal, walkforward
data/       raw/ (committed raw files, byte-for-byte), manifest/ (committed); interim/ processed/ (generated, git-ignored)
runs/       training run folders <stamp>_<tag>_seed<k>/ (git-ignored); runs/rehearsal/ for dry runs
reports/    tables/ figures/ logs/ compute/ transparency_log.md
docs/       decisions.md, data_dictionary.md, laptop_runbook.md, session_context.md
```

## Outputs by phase

| Phase | Main outputs |
|---|---|
| 1 | `data/interim/{waterlevel_long,pumping_long,pagasa_6h,pagasa_daily}.parquet`, `data/processed/{hourly_master,eval_hours}.parquet`, `reports/tables/table_III-6_recomputed.csv`, coverage tables, `reports/figures/fig_III-3_coverage_wet_season.png` |
| 2 | `configs/alert_levels.yaml`, link-analysis tables, `alert_sensitivity.csv`, `phase2_summary.json` |
| 3 | `artifacts/tide_*.json`, `artifacts/scalers.json`, `artifacts/feature_names.json`, `data/processed/windows_{train,val}.npz` |
| 4 | `configs/selected_features.yaml`, `reports/tables/granger_results.csv` |
| 5 | `configs/{compute,model_inputs,thresholds}.yaml`, `configs/frozen_manifest.json`, `reports/compute/<laptop>/`, validation tables, frozen checkpoints |
| 6 | `gates/TEST_UNSEALED.json`, `reports/tables/test_*` (level metrics, skill, alerts, equal recall, events, DM + Holm, `test_results.json` with H3/H4), `reports/figures/test_*` |
| 7 | `reports/tables/{walkforward,ablation,case_study,aviles}_*`, `reports/figures/case_event_*.png` |

## Still to do

* **On the laptop:** smoke test and benchmark, re-run Phases 1–4, Notebook 05 (review G16 before freezing),
  Notebook 06 once, Notebook 07; commit gates, generated configs and `reports/`.
* **Code (small, not yet done):** a Chapter IV report assembler (`pasig/eval/report.py`),
  `scripts/setup_wsl.sh`, a thresholds case in `tests/test_leakage.py`, and a statsmodels HAC cross-check in
  `tests/test_dm.py`.
* **Writing:** Plan Section 13 thesis-text edits plus the additions listed in `reports/transparency_log.md`.
* Delete the old remote branch `feat/affectionate-mccarthy-w2uqzk` on GitHub.
