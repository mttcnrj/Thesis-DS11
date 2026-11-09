# Session context — building the Pasig forecasting pipeline (6–7 Nov 2026)

A handover document for Claude chat (claude.ai Project) and for the group. It records what was built in
the Claude Code session that implemented `IMPLEMENTATION_PLAN.md`, every decision taken, the results so far,
and what is left. The authoritative details stay in the repository; this file points to them.

* **Thesis:** *Pasig River (EFCOS Pandacan) 1–6 h Water-Level Forecasting as a Flood-Risk Indicator for
  Sampaloc* (Baltazar, Cañarejo, Serrano; adviser Gernale).
* **Repository:** `mttcnrj/Thesis-DS11`, branch **`feat/pasig-forecast-pipeline`** (no pull request opened).
* **Specification:** `IMPLEMENTATION_PLAN.md` (repo root). Decision IDs D1–D20, gate IDs G0–G20, contradiction
  IDs C1–C6 come from it.
* **Detailed records (read these for exact numbers):** `docs/decisions.md` (D1–D20 outcomes + implementation
  choices P1-…P7-), `reports/transparency_log.md` (findings F1–F30), `gates/*.json` (every gate's rule,
  evidence, decision, git SHA), `docs/laptop_runbook.md` (how to run everything on the laptop), `README.md`.

---

## 1. Status in one paragraph

**All seven notebooks are implemented.** Phases 1–4 (Notebooks 01–04) were **run on the real data** in a
cloud container and their gates, configs and tables are committed. Notebooks 05–07 (training, the 2024 test,
robustness) are **built and pass full CPU dry runs** ("rehearsal profile"), but their real runs must happen on
the chosen laptop (D16). **The 2024 test season has not been opened.** Tests: 99 passed, 4 skipped (skips =
outputs that only exist after the laptop run). Four small code gaps against the plan remain (Section 9).

## 2. Ground rules followed throughout (Plan Section 0.3 + user instructions)

* Target = hourly EFCOS **Pandacan** river level, 1–6 h ahead, wet season (June–November) only.
* Splits: train 2018–2022, validation 2023, test 2024; chronological, never shuffled.
* Data: only MMDA files and PAGASA **Science Garden**. No Port Area (D6), no Kaggle, no other data.
* Everything learned from data comes from training seasons (or 2023 for tuning only). **2024 is read once,
  in Notebook 06, after the `settings-frozen` tag (gate G18).**
* No SMOTE or synthetic/resampled training data. Rare alerts: LR class weights, tuned thresholds, trigger
  offsets, rare-event metrics, and the D20 weighted-loss ablation.
* Report every result whichever way it falls; if a model fails to beat persistence, run the bug checklist
  (G16) and then report as is.
* The `build-/evaluate-/iterate-/audit-ml-pipeline` skills were not used (plan rule).
* User instruction: "when a decision is needed, always pick the most optimal and best fit" — decisions were
  taken that way and recorded; two were put to the user (G4 amendment, Quiapo 2023/2024 sheets).

## 3. Timeline of the session (user requests → what was done)

| # | User request | Result | Commit |
|---|---|---|---|
| 1 | Implement Phase 1; data in the Drive folder "Thesis II" | Raw data retrieved (465 files), Phase 1 run, gates G1–G3, G5, G6 | `5f141f5` |
| 2 | "Anything to decide before Phase 2?" | Raised F1 (real Quiapo 2023/2024 sheets) and the G4 risk | — |
| 3 | "Proceed with (a), start Phase 2" | Phase 2 run; G4 checks failed literally → user chose **"Amend checks"** (P2-a); G4, G7, G8, G20 | `2db720a` |
| 4 | Laptop specs image; check `smoke_tests.zip`; rename branch; Phase 3 | G0 recorded; zip found and installed; branch renamed; Phase 3 run (G9) | `b791216`, `152d444`, `aa84e97` |
| 5 | "Is the smoke test needed before Phase 4? Why is phase3.py missing?" | Not needed (only before Notebook 05 on the laptop); `phase3.py` existed, README fixed | — |
| 6 | Phase 4, "pick the most optimal" | Phase 4 run (G10, H2 supported) | `af7cd28` |
| 7 | "How much is left?" / "Implement everything first, then run at once — bad idea?" | Advised: build NB05–07 + CPU dry runs here, run smoke/benchmark on the laptop early, run phase by phase there with review stops at G16/G17 and before NB06 | — |
| 8 | Build Notebook 05 | Built + rehearsal (all 8 stages pass) | `6e789b8` |
| 9 | Build Notebook 06 | Built + rehearsal on a 2023 stand-in | `b64eacc` |
| 10 | `nvidia-smi` image; "implement the remaining tasks" | G0 updated with GPU; Notebook 07 built + rehearsal | `af9320b` |
| 11 | "Is everything in the plan done?" | Audit: done except 4 small code gaps + laptop-only items (Section 9) | — |
| 12 | Create a session summary for Claude chat | `docs/session_context.md` | `09a2943` |
| 13 | Update the README | README rewritten (status, results, run order, gates, layout, to-do) | `2d78881` |
| 14 | Update all markdown files | decisions (D8–D20 outcomes), transparency log (F31), data dictionary (later phases), data/README, runbook (troubleshooting, final step), smoke README note; `IMPLEMENTATION_PLAN.md` left unedited (it is the pre-registered specification) | `4b8a280` |
| 15 | Put the datasets in the repository; make every notebook work locally without Google Drive | raw data committed (R1); `pasig.local.prepare` in every notebook and `run_phase.py`; fresh-clone test passed (F32) | see git log |

## 4. Environment and how the data were obtained

* **Since 9 Nov 2026 the raw data are committed in `data/raw/` (R1); no Google Drive step is needed.**
* Work ran in a Claude Code cloud container (Python 3.13, pandas 3.0, TensorFlow 2.21 CPU). Direct
  `drive.google.com` access was blocked, so the raw files were fetched through the Google Drive connector and
  rebuilt locally (`scripts/drive/recover_downloads.py`, `scripts/drive/drive_inventory.json`). The manifest
  `data/manifest/raw_manifest.csv` stores a SHA-256 for every file (465 files: 458 `use`, 7 `exclude`).
* `odfpy` (for `QUIAPO 2018.ods`) needed a manual install in the container; on the laptop `pip` works normally.
* **Chosen laptop (G0, D16):** AMD Ryzen 7 7735HS (8 cores / 16 threads), 16 GB RAM, NVIDIA GeForce RTX 4060
  Laptop GPU **8188 MiB**, driver **595.97**, CUDA 13.2 (driver side), Python in **WSL2 Ubuntu**. Only this
  laptop's specs were provided (chosen with `--choose`). `nvidia-smi` was run in PowerShell; it must also
  work inside WSL2 before the benchmark.
* Phases 1–4 are deterministic: re-running them on the laptop must reproduce the committed gates and tables.

## 5. Gates recorded so far

| Gate | Decision / outcome |
|---|---|
| G0 machine | chosen-laptop (RTX 4060 Laptop 8 GB → WSL2) |
| G1 raw files | passed (Taguig files excluded; QUIAPO 2023 = 2024 identical → one copy kept, dated by sheet headers) |
| G2 parsing | passed (09:00 rollover, impossible dates logged, HIGHEST/LOWEST check formula-aware — P1-a) |
| G3 rain timing | **period_end** (99.7% of 1,172 rain days vs 53.7%) → rain usable 1 h after the period ends |
| G4 status legend | gate + pump fields used (after amendment P2-a); **sky codes dropped** (21.1% > 20%) |
| G5 eligibility | **no eligible upstream station** (Sto. Niño min 78.1% in 2024; Fort Santiago 0% in 2018) |
| G6 phase 1 frozen | passed; tag `phase1-frozen` |
| G7 H1 | **partly supported** |
| G8 alert levels | **high water 11.69 m, flood risk 12.00 m** (reproduce the preliminary levels) |
| G9 tide fit | P1, K2, Q1 added at every station; residual peaks remain (logged) |
| G10 Granger | **rain and pressure selected; H2 supported** |
| G20 Quiapo 2021 | passed → Quiapo check reported with and without 2021 |
| G11–G19 | implemented; run on the laptop (rehearsal versions exist only under `runs/rehearsal/`) |

## 6. Results by phase (real data, Phases 1–4)

### Phase 1 — load and QC (Notebook 01)
* Recomputed Table III-6 (`reports/tables/table_III-6_recomputed.csv`):

| split | observed unflagged h | cases lead 1 / 6 | hours ≥ 11.69 m | events ≥ 11.69 | hours ≥ 12.0 m | events ≥ 12.0 |
|---|---|---|---|---|---|---|
| train 2018–22 | 18,995 | 16,937 / 16,699 | 964 (5.1%) | 255 | 80 (0.42%) | 22 |
| val 2023 | 4,138 | 4,091 / 4,081 | 172 (4.2%) | 50 | 13 | 5 |
| test 2024 | 4,009 | 3,729 / 3,692 | 438 (10.9%) | 103 | 54 (1.35%) | 12 |

  (The 2024 row counts hours from Phase 1 QC metadata as the plan requires; no model has seen 2024.)
* Pandacan coverage before gap fill 77–95% per season (the reference "91–99%" was counted after the ≤ 3 h
  fill) — F2; Figure III-3 redrawn (`reports/figures/fig_III-3_coverage_wet_season.png`).
* Pandacan QC removed 119 training hours (98 flat 11.89 m runs, 18 spikes, 3 typos); all 13 spikes > 11.69 m
  hand-checked; P95 stays 11.69 m (F4). 2023–2024 QC applied unchanged; **Carina's peak survives (12.98 m,
  24 Jul 2024 15:00)** (F5).
* Bugs caught by tests before any result was used: blank 6-hourly rain read as trace (F9); a lint-induced
  empty-sheet parser regression (F17, fixed + Phase 1 re-run identical).

### Phase 2 — link analysis and alert levels (Notebook 02)
* **G4 amendment (P2-a, approved by the user):** literal checks for gate `O` (100% fail: open gate equalises
  both sides) and `nP` (42% fail: pumping draws the sump down) failed for physical reasons; amended checks
  fail 1.1% and 0.3%. Literal results kept in the gate evidence.
* **Quiapo 2023/2024 sheets (P2-b, user chose option (a)):** outside the H1 verdict; 2023 reported
  descriptively; 2024 sheet sealed until after NB06.
* **H1 partly supported (G7):** OR per 0.1 m of Pandacan level after rain 1.207 (95% CI 1.143–1.275,
  day-clustered; 17,132 h, 840 days, 448 stress hours); level × rain interaction significant (OR 1.53);
  2023 ROC-AUC 0.824 → 0.829, gain 0.004 (CI −0.022 to +0.028) does not exclude 0.
* G20 passed (Quiapo 2021 r with Pandacan 0.915 vs 0.885–0.917 confirmed years); Quiapo check OR 1.067
  (without 2021) / 1.066 (with) — same direction as Aviles, weaker (F16).
* Alert levels **11.69 m / 12.00 m** (stress share at ≥ 12.0 m: 55%, n = 80); sensitivity table in
  `reports/tables/alert_sensitivity.csv`; 2023 check recorded, not used (F15).

### Phase 3 — tide, features, windows (Notebook 03)
* Tide fit (D7) on all months of 2018–2022 above floor + 0.05 m (Pandacan floor 10.65 m, P3-a). G9 base
  constituents failed → P1, K2, Q1 added (max 3); residual peaks beside M2/K1/O1 remain (seasonally
  modulated tides), frozen and logged (F18).
* Features: sequence = Pandacan level, Δlevel, latest SG 6 h rain (24 steps); tabular = 28 calendar + 12 tide
  + 3 rain + previous-day pressure (46 columns available); constituent-pair variant = 120 columns (D8).
* Windows: train 18,386 cases, val 4,288; only train/val built here — 2024 windows are built in NB06 only.
* Scalers min-max on training cases; target scaler = Pandacan channel scaler (D9 add-back exact).

### Phase 4 — Granger selection (Notebook 04)
* Rain F = 59.0, p = 6.1e-48; pressure F = 9.9, p = 2.2e-6; RH p = 0.74; mean temperature p = 0.29;
  Bonferroni 0.05/4 = 0.0125 → **rain + pressure selected, H2 supported** (F22).
* Ridge validation RMSE gains are tiny (≤ 0.3 mm) — significance ≠ linear skill (F23).
* Upstream stations (descriptive only, D2): both "Granger-cause" Pandacan at 23 h, but adding them worsens the
  ridge RMSE (F24). Correlation screen picks rain + RH (used by the correlation-selection ablation, F25).

## 7. Notebooks 05–07 — design (built, dry-run, real run pending)

All run as resumable scripts: `python scripts/run_phase.py --phase N [--stage S]`. The **rehearsal profile**
(`PASIG_PROFILE=rehearsal`: 2 epochs, 1,500 training cases, 2 configs per grid, 2 seeds) writes everything to
`runs/rehearsal/` (git-ignored) and never touches the real gates. **No rehearsal number is a result.**

### Notebook 05 — training (`pasig/phase5.py`, stages compute → variants → grid → finals → baselines → thresholds → checks → freeze)
* compute: G14 determinism → G13 device → G15 seed budget via `scripts/smoke/bench_compute.py` (twice per
  device) → `configs/compute.yaml`.
* variants: G11 (constituent pairs vs predicted tide only) and G12 (add-back vs direct level), 3 seeds each,
  switch only if better by > 1 SD of the default's seeds (P5-c) → `configs/model_inputs.yaml`.
* grid: D12 two-stage hybrid grid (24 + 36 configs), G19 edge check, standalone LSTM grid (18).
* finals: hybrid and LSTM × 5 seeds, determinism on → `artifacts/frozen/` (git-ignored; hashed in the manifest).
* baselines (Table III-4): persistence, tide-only, tide + persisted residual, ridge, KNN regressor;
  LR (class weights + threshold) and KNN classifiers (threshold only, C4), two input sets, per lead × level.
* thresholds: D11 trigger offsets (one per model per level, shared by leads) + classifier thresholds →
  `configs/thresholds.yaml`.
* checks: **G16** persistence sanity + automated bug checklist (Section 9.7) — **review before freezing**.
* freeze: **G17** → `configs/frozen_manifest.json` (SHA-256 of every frozen file, now including
  `configs/eval.yaml`, so the H3/H4 rules are pre-registered) and tag `settings-frozen`.
* Keras 3 detail: the D9 add-back is a registered `LevelAtT` layer (not a Lambda) so checkpoints reload.

### Notebook 06 — the one look at 2024 (`pasig/phase6.py`, stages unseal → predict → evaluate)
* **G18 (blocking):** `settings-frozen` must be an ancestor of HEAD, every frozen hash must match, and
  `gates/TEST_UNSEALED.json` must not exist; opening writes it. A second opening needs `--rerun-reason`
  (logged; must go in the transparency log). Re-running `predict`/`evaluate` after a crash does not re-open.
* 2024 windows built only here with the frozen builder, tide, medians and scalers (a test forbids any other
  module from building test windows). Inference on CPU.
* Outputs: level metrics (all/rainy; neural mean ± SD over 5 seeds), skill vs persistence and vs the
  strongest simple baseline (chosen by 2023 RMSE), hourly alerts with frozen offsets/thresholds, equal-recall
  false alarms at Recall 0.90 (+0.80/0.95 sensitivity, D19), PR-AUC, 24 h moving-block bootstrap CIs
  (2,000 reps), events (±2 h detection, warning time, event bootstrap), Diebold–Mariano (HLN, Newey–West)
  with Holm over 72 tests, and the **H3/H4 verdicts by the rules written in `configs/eval.yaml`**:
  * H3: supported if the hybrid's squared error is significantly lower (Holm p < 0.05) than all six baselines
    at all six leads on all hours; partly if ≥ 1 pair; not if none.
  * H4 (high water only): per lead and classifier family (LR, KNN; input set with the higher 2023 F1), the
    hybrid wins if FP at Recall ≥ 0.90 is lower with CI < 0 **and** PR-AUC is higher with CI > 0; supported
    if it wins every lead against both families, partly if some, not if none. Flood-risk results descriptive.
* Rehearsal proof: on the 2023 stand-in, the reloaded checkpoints, frozen baselines and refitted KNNs
  reproduced Notebook 05's saved predictions exactly (difference 0.0), and the rebuilt windows equal Phase 3's
  bit for bit.

### Notebook 07 — robustness (`pasig/phase7.py`, stages walkforward → ablations → casestudy → aviles)
* **Walk-forward** (2018–20 → 2021, 2018–21 → 2022, 2018–22 → 2023): per fold the tide, alert levels, Granger
  selection, medians and scalers are refitted on the fold's training years; the network trains on all but the
  last training year, which is the inner validation (early stopping, ridge alpha, offsets) — interpretation
  recorded as **P7-a**. Models: hybrid + persistence, tide-only, tide + residual, ridge. DM + Holm (72 tests).
  The 2023 fold is flagged not independent. Never reads 2024. Data tests: +5 m on a fold's evaluation year
  changes none of the fold's fits; the 2018–22 fold reproduces the frozen pipeline exactly (11.69/12.00 m,
  rain + pressure, Phase 3 windows and scalers bit for bit).
* **Ablations (Table III-5):** no Granger, correlation selection (rain + RH), no rainfall, no tabular path,
  second LSTM instead of GRU, **weighted loss (D20, w ∈ {2, 5, 10} chosen on 2023 by high-water F1)**. Own
  trigger offsets on 2023; scored on the NB06 hours; compared with the main hybrid on the same seeds
  (seed-noise rule, DM + Holm over 36, descriptive); D20 report: overall RMSE, RMSE at ≥ 11.69 m, error at
  flood-risk peaks, recall at both levels.
* **Case study:** every 2024 flood-risk event (Carina = the event covering 24 Jul 2024): detection, warning
  time, first alert, forecast/error at the peak per model × lead, one figure per event.
* **Aviles 2024 check:** the `post_eval_only` Aviles blocks are parsed only here (after G18), 1 Jun–15 Jul,
  compared descriptively with the alert levels and the training seasons.

## 8. Code map

```
pasig/config.py gates.py local.py phase1.py … phase7.py   orchestrators; local.py = raw check + local rebuild
  io/        manifest (G1), EFCOS parser (G2), pumping parser (D3), PAGASA (D4, D5, G3)
  qc/        rules (range, spike, flat runs), gaps (≤ 3 h fill), coverage (G5)
  link/      stats (Wilson, events, bootstraps, clustered logit), legend (G4), quiapo (G20)
  features/  tide (D7, G9), windows (Section 9.1), scaling (D9)
  selection/ stationarity, granger, correlation
  models/    hybrid (Figure III-4, D9, D20 loss), baselines (Table III-4), offsets (D11)
  train/     data, loop (run folders, determinism), grid (D12), compute (G13–G15)
  eval/      metrics, alerts (D19), events, dm_test, bootstrap
scripts/     run_phase.py, record_g0.py, smoke/ (from smoke_tests.zip); sync_drive.py, drive/ (obsolete)
tests/       19 files (all Section 11 tests + gates, offsets, unseal, walk-forward, same hours, raw data)
notebooks/   01–07, thin: call pasig, show results, run that phase's tests
configs/     hand-written + GENERATED by gates (pagasa, eligible_inputs, alert_levels, selected_features,
             later model_inputs, compute, thresholds, frozen_manifest.json)
```

Note: Phase 4 was refactored in Notebook 07 (`select_candidates`, `model_columns`) so a fold reuses the G10
rule; re-running Phase 4 reproduced the committed G10 result exactly (F31).

Column-level definitions of every data file, artifact, prediction table, gate file and run folder:
`docs/data_dictionary.md`.

## 9. What is NOT done

### Code gaps found by the plan audit (can be done in the cloud; not yet done — user was asked)
1. `pasig/eval/report.py`: one assembler that builds the Chapter IV tables/figures from the NB06–07 outputs
   (the outputs exist, as ~20 separate files).
2. `scripts/setup_wsl.sh` (the steps exist only in `docs/laptop_runbook.md`).
3. `tests/test_leakage.py`: add "+5 m on 2024 leaves the thresholds identical" (scalers, tide, alert levels,
   Granger are already covered).
4. `tests/test_dm.py`: cross-check the Newey–West variance against statsmodels' HAC (statsmodels has no DM test).

### Only possible on the laptop (by design)
* `pip freeze > requirements.txt`, `make smoke` (7 PASS on CPU and GPU), benchmark → G13–G15.
* Notebook 05 real run (G11–G17, G19; long, in tmux), review G16, freeze and tag `settings-frozen`.
* Notebook 06 (opens 2024 once; G18; H3/H4 verdicts), Notebook 07.
* Plan Section 15 items: real Chapter IV numbers, hardware and measured run time for Chapter III from
  `reports/compute/`, results reported whichever way they fall.

### For the group (writing)
* Plan Section 13: 31 thesis-text edits; extra ones under "Thesis text to add" in
  `reports/transparency_log.md` (e.g. before-fill coverage, amended G4 checks, G9 additions, 120 constituent
  columns, Table III-6 replacement, walk-forward design, H3/H4 rules).
* Item 5 of Plan Section 14 (tense, working title) stays open.
* Delete the old remote branch `feat/affectionate-mccarthy-w2uqzk` on GitHub (the session could not).

## 10. How to run on the laptop (summary of `docs/laptop_runbook.md`)

1. WSL2 Ubuntu, `nvidia-smi` inside WSL, clone into `~/pasig-forecast`, checkout `feat/pasig-forecast-pipeline`,
   venv, `pip install -e . -r requirements-dev.txt "tensorflow[and-cuda]" joblib`, GPU visible to TF.
2. `make smoke` and `python scripts/smoke/smoke_pipeline.py --device gpu` (7 PASS each).
3. `make data-check` (the raw data come with the clone, R1), then Phases 1–4; `git diff` must show the same
   decisions; `pip freeze > requirements.txt`.
4. Notebook 05 in tmux, stage by stage; **stop at G16 and review**; then `--stage freeze --tag`.
5. Notebook 06 deliberately, once (`python scripts/run_phase.py --phase 6`).
6. Notebook 07 (`--phase 7`, resumable).
7. Commit gates, generated configs, `reports/` (not `artifacts/frozen/`, which is hashed in the manifest).

## 11. Points worth knowing in a new chat

* "Partly supported" H1 and "supported" H2 are real results; H3/H4 do not exist yet (2024 unopened).
* The rehearsal verdicts printed during development (e.g. "H3 partly / H4 not supported") are **not
  results** — 2-epoch models, thresholds tuned on the same 2023 hours.
* Two choices were made with the user's explicit approval: the G4 amendment (P2-a) and Quiapo option (a)
  (P2-b). The walk-forward inner-validation design (P7-a) is an interpretation of the plan text; change it
  before the laptop run if the group reads the plan differently.
* Interpretations to watch: the plan says the hybrid's output "Lambda" layer — implemented as a registered
  layer (same computation); KNN models are refitted from frozen hyperparameters in NB06 rather than stored.
* Every gate JSON carries the git SHA of the run; every training run folder carries config, git SHA and
  environment (`runs/<stamp>_<tag>_seed<k>/`).
