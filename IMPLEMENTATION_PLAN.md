# Implementation Plan for Claude Code
## Pasig River (EFCOS Pandacan) 1–6 h Water-Level Forecasting as a Flood-Risk Indicator for Sampaloc

Version: 6 Nov 2026. Team: Baltazar, Cañarejo, Serrano (BS CS, UST). Adviser: Gernale.

This file tells Claude Code what to build, in what order, and which decisions and gates
the code must enforce. Read it fully before writing code. Every decision has an ID
(`D1`–`D20`) and every automatic check has an ID (`G0`–`G20`); use these IDs in code
comments, config keys, gate files and commit messages so the thesis text can be traced
back to the code.

---

## 0. How to use this file

### 0.1 Sources, in order of authority

1. **The data files** (MMDA and PAGASA Science Garden). They are ground truth. If code finds
   that the data contradict this plan or the thesis, stop and report the mismatch before
   continuing.
2. **This plan's decision register (Section 4)**, which records decisions made on 6 Nov 2026.
   Where a decision amends the thesis text, the thesis will be edited to match (Section 13).
3. **The revised Chapters I–III** (`11_2_Revised_Chapters_1-3.pdf`; project copy
   `REVISED THESIS.pdf`) and `claude/project_context.md` in the claude.ai Project.
4. `claude/class_imbalance_and_safeguards.md` (no SMOTE; class weights only for Logistic Regression).
5. `Thesis Template.pdf` for formatting of any text output.

`THESIS 511.pdf` and the earlier runs (`outputs/run_20261016_1540`, `outputs/run_20261019_0749`)
are reference only. Never copy their numbers into results.

### 0.2 Contradictions already found (resolved as stated)

| ID | Contradiction | Resolution |
|---|---|---|
| C1 | Project description says "seven Colab notebooks"; decision D16 runs everything locally in VS Code on one Windows laptop | Run locally (D16). Keep code portable: no Colab-only calls, paths from `configs/paths.yaml`, so Colab stays a fallback |
| C2 | Project Instructions field says "log how each of the 335 conflicting duplicate hours is resolved"; `claude/project_context.md` (31 Oct) says that figure came from the old pipeline and is not used | Follow project_context: EFCOS rows for impossible dates (e.g. 31 June from 31-row templates) are set aside and logged; real dates have no repeated hours. Readings of exactly 0.0 m are missing |
| C3 | project_context Part D: tide fitted "on the training seasons"; D7 fits on all 12 months of 2018–2022 | D7 wins (6 Nov). Dry months of training years are not validation/test data |
| C4 | Chapter III says "class weights" for both LR and KNN classifiers; scikit-learn's `KNeighborsClassifier` has no class weights | LR: class weights + tuned threshold. KNN classifier: tuned decision threshold only |
| C5 | `QUIAPO 2023.xlsx` and `QUIAPO 2024.xlsx` are identical (677,095 bytes) and both start on a sheet headed "January 1–15, 2021"; Notebook 1 (3 Nov) found a Quiapo 2021 block inside `QUIAPO 2022.xlsx` that "stays excluded pending the adviser" | D1: use one copy only, dated by its sheet headers. D18 (6 Nov): any Quiapo 2021 data (the block in `QUIAPO 2022.xlsx` and/or that copy) is used for the secondary Quiapo check only if it passes the identity gate G20, and that check is reported with and without it |
| C6 | Chapter III p. 53 labels the hyperparameter table "Table III-2" (duplicate) | It is Table III-3; fix in the chapter (Section 13) |

### 0.3 Non-negotiable rules (apply everywhere)

- **Scope:** target is the hourly Pasig River level at MMDA EFCOS **Pandacan** (the river gauge, not
  the MMDA Pandacan pumping station), forecast 1–6 h ahead, **wet season only (June–November)**.
  "Flood" means flood risk for the Aviles–Sampaloc drainage area. Never claim street flood depth.
  Never call the Pasig the sole cause of flooding.
- **One forecast, two alerts:** high water = forecast ≥ 11.69 m; flood risk = forecast ≥ 12.0 m.
  Both levels are derived from training data only (Phase 2) and then frozen.
- **Splits:** train 2018–2022, validation 2023, test 2024. Chronological, never shuffled.
- **Data:** only MMDA files and PAGASA **Science Garden** files. **No Port Area (D6), no Kaggle file,
  no SRTM, no other data.** Science Garden is "a rain gauge representing rainfall over upland Quezon
  City" — never say it measures runoff.
- **Leakage:** every quantity learned from data (QC thresholds, scalers, tide fits, alert levels,
  Granger results, hyperparameters, thresholds, trigger offsets) comes from training seasons, or from
  2023 for tuning only. The 2024 test season is read **once**, in Notebook 6, after `settings-frozen`.
- **Honesty:** report every analysis whichever way it falls. If the hybrid loses, say so. If a model
  fails to beat persistence, hunt for a bug first (G16); if none, report it as is.
- **No SMOTE or any synthetic/resampled training data.** Rare alerts are handled by LR class weights,
  tuned thresholds, trigger offsets and rare-event metrics.
- **Skills:** do not use the `build-`, `evaluate-`, `iterate-` or `audit-ml-pipeline` skills (they tie the
  project to skrub/skore, which do not fit a two-input Keras model). Useful: explore-data,
  statistical-analysis, testing-strategy, validate-data, debug, code-review, dataviz, documentation, learn.

---

## 1. Study summary (what the code must answer)

| SOP | Question | Notebook | Hypothesis and pre-set decision rule |
|---|---|---|---|
| 1 | Is the Pandacan level associated with drainage stress (≥ 2 Aviles pumps) after accounting for Science Garden rain? At which levels do the alerts sit? | 02 | **H1** supported if the adjusted odds ratio's 95% CI (per 0.1 m) excludes 1 **and** the 2023 ROC-AUC gain from adding the level to rain has a day-block bootstrap 95% CI excluding 0; partly supported if one holds; not supported if neither |
| 2 | Which candidates Granger-cause the de-tided Pandacan level? | 04 | **H2**: at least one candidate significant after Bonferroni |
| 3 | How is the hybrid designed and trained? | 05 | — |
| 4 | Does the hybrid beat the baselines in error and alerts? | 06, 07 | **H3**: lower error than KNN regressor, ridge, standalone LSTM, tide models and persistence at every lead (DM test, Holm). **H4** (high-water level only): fewer false alarms than LR and KNN classifiers at Recall = 0.90, and higher PR-AUC (moving-block bootstrap CIs). Recall 0.80 and 0.95 are reported as a sensitivity check but do not change the verdict (D19). Flood-risk results are descriptive |

The study is complete when every planned analysis has run and been reported.

---

## 2. Environment (decision D16)

### 2.1 Machine selection — gate G0

All runs happen locally in **VS Code** on **one Windows laptop**: whichever of the group's two Windows
laptops has the stronger GPU. Run on both:

```powershell
Get-CimInstance Win32_VideoController | Select-Object Name
```

Rules:
- Only **NVIDIA** GPUs count (TensorFlow needs CUDA). Intel/AMD entries, including integrated GPUs, are ignored.
- Both NVIDIA → higher-tier card wins. Confirm with `nvidia-smi` (model, VRAM). Do not read memory from
  `Win32_VideoController.AdapterRAM` (capped at 4 GB).
- One NVIDIA → that laptop. Neither → laptop with more CPU cores:
  `Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors`.
- **G0 output:** `gates/G00_machine.json` with the command output, chosen laptop, reason; copy into
  `docs/decisions.md` under D16.

### 2.2 Setup on the chosen laptop

| Component | NVIDIA GPU present | No NVIDIA GPU |
|---|---|---|
| Python runs in | WSL2 Ubuntu (TensorFlow ≥ 2.11 has no native-Windows GPU support) | Native Windows |
| TensorFlow | `pip install "tensorflow[and-cuda]"` | `pip install tensorflow` |
| VS Code extensions | Python, Jupyter, WSL ("WSL: Connect to WSL") | Python, Jupyter |
| Repo location | `~/pasig-forecast` in the Linux filesystem (not `/mnt/c`) | Local folder outside OneDrive |

Other packages: pandas, NumPy, scipy, scikit-learn, statsmodels ≥ 0.14, pyarrow, openpyxl, odfpy
(`QUIAPO 2018.ods`), utide (optional), matplotlib, psutil, pytest, pyyaml, nbstripout, ruff.

Steps: install NVIDIA driver + `wsl --install -d Ubuntu` (GPU case) → clone repo → `python3 -m venv .venv`
→ `pip install -e . -r requirements-dev.txt` → check `tf.config.list_physical_devices('GPU')` (must be
non-empty on an NVIDIA laptop) → data via Google Drive for desktop (non-owners add a shortcut to
`Thesis II` in My Drive; in WSL mount with `sudo mount -t drvfs G: /mnt/g` if needed) →
`scripts/sync_drive.py` copies raw files once into `data/raw/` → `pip freeze > requirements.txt`, commit,
never upgrade mid-thesis.

### 2.3 Long runs

- Run long jobs as scripts (`python scripts/run_phase.py --phase 5`), not notebook cells. In WSL use
  `tmux`; on native Windows a separate PowerShell window.
- Plugged in, power mode "Best performance", sleep "Never", Windows Update paused during Phases 4–5.
- One heavy job at a time. The grid runner is resumable (skips configs whose result row exists).
- Every run writes `env.txt`: Python, TF, CUDA/cuDNN, GPU name, CPU model, physical cores, RAM.
- All final results (final seeds, ablations, walk-forward, 2024 test) come from the chosen laptop only.
  Other members develop code and run `pytest -m fast` on their machines.

---

## 3. Data inventory and known file problems

### 3.1 Files

| Source | Files | Layout | Use |
|---|---|---|---|
| MMDA EFCOS water level 2018–2024 | `Waterlevel Data 2018-2024/<Station> 2018-2024/<year>/<m>. <Month>.xlsx` for Pandacan, Sto. Niño, Fort Santiago, Rosario JS, Rosario LS (2018–2020), Montalban | "FORM 2-1 HOURLY WATERLEVEL CHART": one row per day; 24 hourly columns **09:00 → 08:00**; then AVE., DATE, HIGHEST, LOWEST | Pandacan = target; Sto. Niño, Fort Santiago = candidates (if eligible); Montalban = QC helper |
| MMDA pumping stations | `AVILES 2018–2024.xlsx`, `QUIAPO 2018.ods`, `QUIAPO 2019–2024.xlsx` | Half-month sheets; day columns; **two rows per hour** (inside/sump with a status code, then outside/river) | Phase 2 only; never model inputs |
| MMDA WOCR | `WOCR PS - <year>.pdf` | Monthly pump operating hours | Context for Phase 2 |
| PAGASA 6-hourly | `6-Hourly Data.csv` | Two header rows; `Date(UTC)` as `M/D/YYYY H:MM`; Port Area and Science Garden side by side | **Science Garden columns only** |
| PAGASA daily | `Science Garden Daily Data.csv` | `YEAR,MONTH,DAY,RAINFALL,TMAX,TMIN,TMEAN,RH,WIND_SPEED,WIND_DIRECTION,ATM PRESSURE` | Daily weather candidates |
| Not used | `Port Area Daily Data.csv`, `Kaggle Dataset/Flood_Prediction_NCR_Philippines.csv` | — | Excluded (D6; Kaggle by group decision) |

### 3.2 Parsing rules forced by the files

- **EFCOS day starts at 09:00.** For the row of day *d*: column *i* (0-based, 9:00 first) → timestamp
  `Timestamp(year, month, d, 9) + i hours`. Columns 0:00–8:00 belong to day *d+1*.
- Month from the file name; year from the `YEAR :` cell, asserted equal to the folder year. Header
  text is unreliable (`2024/4. April.xlsx` says "February 1-31"; many say "1-31").
- **Impossible-date rows** (day 31 in a 30-day month, day 29–31 in February when invalid) are set aside
  and logged to `reports/logs/impossible_dates.csv`; they are not data.
- **Missing tokens:** `**`, `****`, `**.**`, `*`, `-`, blanks, any non-numeric cell → NaN; keep the raw
  token. **A reading of exactly 0.0 m is missing.**
- Cross-check each day: parsed max/min vs the sheet's HIGHEST/LOWEST when no hour is missing; log
  mismatches (`reports/logs/highest_lowest_mismatch.csv`). A day-level mismatch rate above 1% for a file
  means a column shift: fix the parser, do not continue.
- **Aviles values are centimetres** (`1135` → 11.35 m); **Quiapo values are metres**. Assert 9 ≤ m ≤ 14.
- Pumping hour labels: Aviles `12MN`, `1AM`…; Quiapo `12:00:00 AM`… Map both to 0–23.
- **Status codes (D3):** regex `^(?P<gate>[OC])?(?P<sky>[A-Z]+)$` for plain codes and
  `(?P<pumps>\d+)\s*P\s*(?P<sky>[A-Z]+)` when pumps run. Store `pumps_running` (0 if no pump token),
  `gate_open` (True/False/NaN), `sky`, `raw_status`. Meanings are assumed and checked (gate G4).
- Skip sheets whose readings are all blank (empty templates, e.g. `AVILES 2018.xlsx` Sheet20–24).
- **PAGASA:** trace `T` (6-hourly) and `-1` (daily) → **0.05 mm (D4)**; `-999` → NaN. 6-hourly UTC → PHT
  (+8 h). Availability per D5/G3. Daily values usable from **09:00 PHT the day after their label**
  (a daily value covers 08:00 PHT on its date to 08:00 PHT the next day).

### 3.3 Known file problems (log each in `reports/transparency_log.md`)

| File | Problem | Handling |
|---|---|---|
| `QUIAPO 2020.xlsx`, `QUIAPO 2021.xlsx` | Contain **Taguig** Pumping Station data (labelled 2019) | Exclude |
| `QUIAPO 2023.xlsx`, `QUIAPO 2024.xlsx` | Identical files; first sheet headed 2021 | **D1**: use one copy; date by sheet headers (G1). If its sheets are 2021 data → **D18**/G20 |
| `QUIAPO 2022.xlsx` | Contains a Quiapo 2021 block that behaves like a Pasig gauge (r = 0.897 with Pandacan) | **D18**: used for the secondary Quiapo check only if it passes G20; reported with and without it |
| `AVILES 2024.xlsx` | Mislabeled sheets; no data after 15 Jul 2024 | Date by sheet content; 2024 Aviles used only for the post-evaluation descriptive check |
| Rosario JS 2024 | January only | Station fails coverage; not a candidate |
| `WOCR PS - 2022.pdf` / `(1)` / `2022 1.pdf`, `WOCR PS -2020 1.pdf` | Duplicates | Keep one after hash check |
| `Pandacan 2024/4. April.xlsx` | 137 KB vs ~34 KB; wrong header text | Parse the April sheet only |

### 3.4 Reference numbers for sanity checks (recompute; do not quote as results)

From `claude/project_context.md` and Notebook 1 (3 Nov 2026, `outputs/run_20261103_2038`):

| Quantity | Value |
|---|---|
| Pandacan wet-season coverage | 91–99% every year |
| Sto. Niño passes 80% | only 2018 and 2022 → **fails** the every-season rule |
| Fort Santiago coverage 2018–2024 | 0, 2, 86, 98, 66, 87, 96% → **fails** |
| Science Garden rain coverage | 97.5–100% every year |
| Usable Pandacan hours train / val / test | 20,867 / 4,336 / 4,283 |
| High-water hours (≥ 11.69 m) | 972 (4.7%) / 174 / 440 (10.3%) |
| Flood-risk hours (≥ 12.0 m) | 82 (0.39%) / 13 / 55; events 24 / 5 / 13 |
| Forecast cases lead 1 h / 6 h (Pandacan + SG inputs) | train 16,158 / 15,953; val 4,113 / 4,103; test 3,752 / 3,716 |
| Pandacan training hours removed in QC (3 Nov) | 103 (5 one-hour spikes; flat runs at exactly 11.89 m after Ulysses, 25–26 Jul 2021 and during Paeng) |
| P95 of training Pandacan | 11.69 m |
| Max Pandacan (Carina, 24 Jul 2024) | 12.98 m |

If a rebuilt number differs materially, investigate before continuing; report the new number, not the old one.

---

## 4. Decision register

Status as of 6 Nov 2026: all decided. D2, D18, D19 and D20 settle the items earlier reserved for the adviser (Section 14); the group decided them. "Gate" means the code decides automatically at run time
using a pre-set rule; record the outcome in the gate file.

| ID | Decision | Final choice | Enforced by |
|---|---|---|---|
| D1 | Quiapo 2023/2024 duplicate files | Use **one** copy; date its readings by the sheet headers, not the file name. If the sheets are 2021 data, D18 applies | G1 |
| D2 | Upstream stations failing the 80% every-season rule (Section 14, item 1) | **Keep the rule.** Sto. Niño and Fort Santiago are not model inputs if they fail G5. They still go through the full Granger procedure on their available training hours, reported as **descriptive** results, never used as inputs. **No missing-indicator variant.** Why: Fort Santiago has 0% and 2% coverage in 2018–2019, so those seasons cannot be tested; Sto. Niño passes only 2018 and 2022; the reference run's no-upstream ablation changed 6 h RMSE by 0.0004 m against ~0.004 m seed noise; and the coverage rule is already in Chapter III, so keeping it needs no rewrite | G5 |
| D3 | Aviles/Quiapo status codes (no legend available; MMDA cannot be contacted) | Use the **assumed legend** (Section 8.1) and validate each part on 2018–2022 data before use | G4 |
| D4 | Trace rainfall value | **0.05 mm** (half the 0.1 mm limit) | Parser + test |
| D5 | When a 6-hourly total is usable | Test settles whether stamps are period start or end; the total becomes usable **1 h after its period ends** | G3 + test |
| D6 | Port Area | **Excluded** (also has no 2024 data and gaps in the 2020 and 2023 wet seasons) | Parser drops columns |
| D7 | Tide fit | Fit on **all 12 months of 2018–2022**, using only hours **> floor + 0.05 m** (Pandacan floor ≈ 10.63 m); check residual spectrum | G9 |
| D8 | Tidal inputs in tabular path | **Predicted tide at t+1…t+6 and its change from t** (12 cols). Constituent sin/cos pairs (84 cols) tried once on validation; kept only if better | G11 |
| D9 | Output parameterization | Head predicts **change from current level**, a fixed layer adds Lₜ back → output is still the level. Compared once with direct-level output on validation | G12 |
| D10 | Model-selection criterion | **Mean validation RMSE over the six leads** (2023) | Grid code |
| D11 | Trigger offsets at 12.0 m | **One offset per model per alert level, shared by all six leads** | Offset code |
| D12 | Hyperparameter search | **Two-stage grid, 60 configs** (Section 9.3) | Grid code |
| D13 | Seeds for walk-forward and ablations | **5** if the chosen-device projection of the plan is ≤ 10 h; otherwise **3** for walk-forward and ablations (final configs always 5) | G15 |
| D14 | Hyperparameter ranges | **Chapter III ranges exactly.** If the winner is at the low edge, report it; never widen after seeing results | G19 |
| D15 | GPU determinism | Grid may run with determinism off; final seeds, ablations, walk-forward and test **always** with determinism on. If the GPU cannot reproduce a run, final runs fall back to CPU | G14 |
| D16 | Platform | Local VS Code on the Windows laptop with the stronger NVIDIA GPU (else more CPU cores) | G0 |
| D17 | Train on GPU or CPU | GPU only if TF sees an NVIDIA GPU **and** it passes G14 **and** its benchmark projection beats the CPU's; else CPU | G13 |
| D18 | Quiapo 2021 data (block inside `QUIAPO 2022.xlsx`, and the single 2023/2024 copy if its sheets are 2021) (Section 14, item 2) | **Use it, conditionally**, in the secondary Quiapo check only (never a model input, never used for the alert levels, which come from Aviles). It must pass the identity gate G20. Report the Quiapo check **with and without** 2021. Why: 2021 is a training year, and the only file named 2021 holds Taguig data, so this block is the only way to cover 2021; adding a confirmed year strengthens the independent check at no risk to the main analysis | G20 |
| D19 | Comparison point for H4 (Section 14, item 3) | **Keep Recall = 0.90** as the pre-registered point for the H4 verdict. Also report false alarms at Recall **0.80 and 0.95** and the full precision–recall curves as a descriptive sensitivity check. If a model cannot reach exactly 0.90 (e.g. KNN probabilities move in steps of 1/k), use the first threshold giving Recall ≥ 0.90 and report the Recall achieved. Why: Recall is the priority, and 0.90 means at most 1 in 10 high-water hours missed; it was fixed before any model ran, so changing it now would move the goalposts; the 2024 test year has about 440 high-water hours, enough to estimate it; 0.95 at 6 h forces many false alarms for every model, and 0.80 misses too much for a warning | Eval code |
| D20 | Weighted-loss ablation (Section 14, item 4) | **Add it** as a sixth ablation variant, not as the main model. Weighted masked MSE: each valid target ≥ the high-water level (from `alert_levels.yaml`) gets weight w, others 1. Choose w from {2, 5, 10} on 2023 with the final hybrid hyperparameters, 1 seed each, by the highest high-water F1 after the variant's own trigger-offset tuning (same rule as D11). Then train with the G15 seed count. Report overall RMSE, RMSE on hours ≥ 11.69 m, mean error at flood-risk event peaks, and Recall at both levels. If it beats the main model, report that honestly and recommend it in Chapter V; the main model stays as pre-registered. Why: the reference run under-forecast storm peaks by 0.1–0.3 m (flood-risk Recall 0.46 at 6 h); this answers the training-side imbalance question without synthetic data, for about 8 extra runs | Ablation code |

Each decision gets an entry in `docs/decisions.md`: date, choice, reason, gate file path, and the
chapter rows it changes (Section 13).

---

## 5. Gates

A gate is a function that evaluates a pre-set rule and writes `gates/G##_<name>.json`:

```json
{"gate": "G13", "name": "device", "passed": true, "decision": "gpu",
 "rule": "GPU iff visible NVIDIA GPU and G14 pass and gpu_plan_hours < cpu_plan_hours",
 "evidence": {"gpu": "NVIDIA GeForce RTX 4060 Laptop GPU", "gpu_plan_hours": 4.1, "cpu_plan_hours": 9.8},
 "timestamp": "2026-11-10T21:03:00+08:00", "git_sha": "abc1234"}
```

Implement in `pasig/gates.py`:

```python
def record_gate(gate_id: str, name: str, passed: bool, rule: str, evidence: dict,
                decision: str | None = None) -> None: ...
def require_gates(*gate_ids: str) -> None:
    """Raise RuntimeError naming every gate that is missing or failed. Called in the first cell of
    every notebook and at the start of scripts/run_phase.py."""
```

A failed **blocking** gate stops the pipeline. A failed **switching** gate selects the fallback branch
and is recorded as passed with the decision taken.

| Gate | Type | Where | Rule | Pass → | Fail → |
|---|---|---|---|---|---|
| G0 machine | record | before anything | Section 2.1 rules | `configs/compute.yaml: machine` | — |
| G1 raw files | blocking | NB01 step 1.1 | Every raw file has a SHA-256 and a status (`use`/`exclude`/`pending_G20`); every identical-hash group has exactly one non-excluded copy; Taguig files excluded; the single Quiapo copy is dated by sheet headers; Quiapo 2021 data (the block in `QUIAPO 2022.xlsx` and/or that copy) marked `pending_G20` | continue | stop; list offending files |
| G2 parsing | blocking | NB01 step 1.2–1.4 | 09:00 rollover test passes; impossible-date rows logged; HIGHEST/LOWEST mismatch ≤ 1% of complete days per file; Aviles values in 9–14 m after ÷100 | continue | stop; fix parser |
| G3 rain timing (D5) | switching | NB01 step 1.4 | For rain days, compare daily total with (a) sum of 00/06/12/18 UTC reports of that date and (b) 06/12/18 UTC + 00 UTC next date. Convention = option matching ≥ 95% of rain days (±0.2 mm). Then usable time = period end + 1 h | write `configs/pagasa.yaml` convention | neither ≥ 95% → stop and report |
| G4 status legend (D3) | switching | NB02 start | For each code part in Section 8.1, the check fails on ≤ 20% of applicable training hours | use field | drop that field from Phase 2 and log it |
| G5 eligibility (D2) | record | NB01 step 1.8 | Station eligible iff ≥ 80% usable hours in every season 2018–2024 | `configs/eligible_inputs.yaml` | station excluded (expected for Sto. Niño, Fort Santiago) |
| G6 phase 1 frozen | blocking | end NB01 | G1–G3, G5 written; tests 1.x pass; `eval_hours.parquet` written | tag `phase1-frozen` | stop |
| G7 H1 result | record | end NB02 | H1 rule (Section 1) applied to computed stats | record verdict | — |
| G8 alert levels | record | NB02 | High water = P95 of training Pandacan (observed, unfilled, unflagged), rounded 0.01 m; flood risk = lowest 0.05 m grid level where ≥ 50% of training hours at/above are stress hours | `configs/alert_levels.yaml` | — |
| G9 tide fit (D7) | switching | NB03 | After fitting M2 S2 N2 K1 O1 M4 MS4, no residual-spectrum peak within ±3% of 12.0, 12.42, 23.93, 24.0, 25.82 h exceeds 3× the median spectral power nearby | freeze tide coefs | add next constituent (P1, K2, Q1, M3, then 2N2), refit, recheck (max 3 additions), log each |
| G10 Granger outcome | record | NB04 | Pre-stated: none pass → model uses Pandacan history + tabular only; all pass → call it screening and let the "no Granger" ablation decide | `configs/selected_features.yaml` | — |
| G11 tidal input variant (D8) | switching | NB05 before grid | Reference config (L1×64, GRU 32, dense 1×16, dropout 0.2, lr 1e-3, batch 128), 3 seeds each: predicted-tide-only vs + constituent pairs. Keep pairs only if mean val RMSE (mean over leads) is lower by more than 1 SD of the seed spread | `configs/model_inputs.yaml` | keep predicted tide only |
| G12 output form (D9) | switching | NB05 before grid | Same reference config, 3 seeds: add-back (default) vs direct level. Switch to direct only if it is lower by more than 1 SD of seed spread | `configs/model_inputs.yaml` | keep add-back |
| G13 device (D17) | switching | NB05 start | GPU iff TF lists an NVIDIA GPU **and** G14 passes on GPU **and** `bench_compute` GPU plan hours < CPU plan hours (use the second of two back-to-back runs) | `configs/compute.yaml: device: gpu` | `device: cpu` |
| G14 determinism (D15) | switching | NB05 start | With `enable_op_determinism()`, a 1-epoch run repeated twice on the chosen device gives identical predictions (atol 1e-6) | final runs on that device, determinism on | final runs on CPU with determinism on; grid may stay on GPU without determinism |
| G15 seed budget (D13) | switching | NB05 after benchmark | Recommended-plan projection on the chosen device ≤ 10 h | `seeds_wf_ablation: 5` | `seeds_wf_ablation: 3` (final configs stay 5); flag Chapter rows 4 and 26 |
| G16 persistence sanity | blocking | NB05 after each model family; NB06 | Each model's validation RMSE < persistence at lead ≥ 2 (lead 1 may tie). If not → run the bug checklist (Section 9.7). Pass if checklist finds and fixes a bug, or documents none | continue | if no bug found, record "beats persistence: no" and continue (report as is) |
| G17 settings frozen | blocking | end NB05 | `alert_levels.yaml`, `selected_features.yaml`, `model_inputs.yaml`, `thresholds.yaml`, `compute.yaml`, scalers, tide coefs and final checkpoints exist; their SHA-256 written to `configs/frozen_manifest.json` | tag `settings-frozen` | stop |
| G18 test unsealed once | blocking | NB06 first cell | `settings-frozen` is an ancestor of HEAD; frozen hashes match; no `TEST_UNSEALED` exists (or a rerun reason is logged) | write `TEST_UNSEALED` | stop |
| G19 hyperparameter edge (D14) | record | NB05 after grid | Winner at a range boundary (LSTM 64 & 1 layer, GRU 32, dense 16, dropout 0.2/0.5, lr bound, batch bound) | note for Chapter IV limitation | — |
| G20 Quiapo 2021 identity (D18) | switching | NB02 start, after G4 | All four hold: (a) layout and units match confirmed Quiapo files (metres, `O`/`C` codes, two rows per hour); (b) sheets dated 2021; (c) the correlation of its outside elevation with Pandacan is no more than 0.05 below the lowest such correlation among confirmed Quiapo years (2018, 2019, 2022); (d) its values are not a copy of Aviles or of the Taguig files over the same hours (correlation < 0.99 and not equal). If the 2022-file block and the 2023/2024 copy overlap in dates, their values must agree; identical → one series; different → fail | include 2021 in the Quiapo check (with/without both reported) | exclude 2021, log the failed condition |

---

## 6. Repository layout

```text
pasig-forecast/
├── IMPLEMENTATION_PLAN.md          # this file
├── README.md                       # setup, run order, how to reproduce Chapter IV
├── pyproject.toml                  # package `pasig`; pip install -e .
├── requirements.txt                # frozen on the chosen laptop
├── requirements-dev.txt            # pytest, ruff, nbstripout, pre-commit
├── Makefile                        # make smoke | test | nb01 … nb07
├── .gitignore                      # data/raw, data/interim, data/processed, runs/
├── .gitattributes                  # nbstripout filter
├── .pre-commit-config.yaml         # ruff, nbstripout, pytest -m fast
├── configs/
│   ├── paths.yaml                  # data root on the chosen laptop
│   ├── splits.yaml                 # wet months 6–11; train/val/test years; walk-forward folds
│   ├── stations.yaml               # station ids, globs, units (m/cm), parser
│   ├── qc.yaml                     # ranges, spike size, flat-run length, exemptions (Section 7.5)
│   ├── tide.yaml                   # constituent list (G9 may extend)
│   ├── features.yaml               # window 24, leads 1–6, candidates, rain windows
│   ├── model_hybrid.yaml           # D12 two-stage grid; D14 ranges
│   ├── baselines.yaml              # ridge, KNN, LR grids
│   ├── eval.yaml                   # bootstrap 2000 reps, 24 h blocks, recall 0.90, event merge 6 h, ±2 h
│   ├── pagasa.yaml                 # GENERATED by G3
│   ├── eligible_inputs.yaml        # GENERATED by G5
│   ├── alert_levels.yaml           # GENERATED by G8
│   ├── selected_features.yaml      # GENERATED by G10
│   ├── model_inputs.yaml           # GENERATED by G11, G12
│   ├── compute.yaml                # GENERATED by G0, G13, G14, G15
│   ├── thresholds.yaml             # GENERATED in NB05 (classifier thresholds, trigger offsets)
│   └── frozen_manifest.json        # GENERATED by G17
├── gates/                          # G##_<name>.json, committed
├── artifacts/                      # fitted objects: tide_<station>.json, scalers, frozen checkpoints
├── data/                           # git-ignored except README and manifest
│   ├── manifest/raw_manifest.csv   # committed: path, sha256, bytes, station, year, status, reason
│   ├── raw/ interim/ processed/ synthetic/
├── pasig/
│   ├── config.py  gates.py
│   ├── io/        manifest.py  mmda_waterlevel.py  mmda_pumping.py  pagasa.py
│   ├── qc/        rules.py  gaps.py  coverage.py
│   ├── link/      stress.py  legend.py  logit.py  auc_gain.py  events.py  alert_levels.py
│   ├── features/  tide.py  calendar.py  rainfall.py  weather.py  windows.py  scaling.py
│   ├── selection/ stationarity.py  granger.py  correlation.py
│   ├── models/    hybrid.py  lstm.py  baselines.py  offsets.py
│   ├── train/     loop.py  grid.py  seeds.py
│   ├── eval/      metrics.py  alerts.py  events.py  dm_test.py  bootstrap.py  report.py
│   └── utils/     timing.py  hardware.py  runs.py
├── notebooks/                      # thin: import pasig, call functions, show results
│   ├── 01_load_qc.ipynb
│   ├── 02_link_analysis.ipynb
│   ├── 03_features_split.ipynb
│   ├── 04_granger.ipynb
│   ├── 05_train.ipynb
│   ├── 06_evaluate.ipynb           # ONLY place 2024 targets are read
│   └── 07_ablations_walkforward.ipynb
├── scripts/
│   ├── sync_drive.py  run_phase.py  setup_wsl.sh
│   └── smoke/  common.py  smoke_pipeline.py  bench_compute.py  README.md   # from smoke_tests.zip
├── tests/   (Section 11)
├── runs/                           # git-ignored: <yyyymmdd_hhmm>_<tag>_seed<k>/
├── reports/ figures/ tables/ logs/ compute/ transparency_log.md
└── docs/    decisions.md  data_dictionary.md
```

Notebooks hold no logic: they import `pasig`, call functions, display outputs and end by running that
phase's tests (`pytest tests/test_xxx.py -q`), stopping if any fail.

---

## 7. Notebook 01 — Load and quality-check (all objectives)

**Outputs:** `data/interim/waterlevel_long.parquet` (all months), `pumping_long.parquet`,
`pagasa_6h.parquet`, `pagasa_daily.parquet`, `data/processed/hourly_master.parquet` (wet season),
`eval_hours.parquet`, coverage tables, logs. Gates G1, G2, G3, G5, G6.

### 7.1 Manifest (G1)
`scripts/sync_drive.py`: copy `MMDA Dataset/` and `PAGASA Dataset/` into `data/raw/`; compute SHA-256;
write `raw_manifest.csv`; group identical hashes. Apply Section 3.3 statuses. Inspect the first sheet
header of each Quiapo file and record the year it actually holds.

### 7.2 EFCOS water-level parser (G2)
Implement Section 3.2. Output long table `time_pht, station, value_m, raw_token, source_file, sheet,
flag_impossible_date`. Keep **all months** (the tide fit needs dry months, D7).

### 7.3 Pumping parser
Half-month sheets → `time_pht, station, inside_m, outside_m, pumps_running, gate_open, sky, raw_status`.

### 7.4 PAGASA parser (G3)
Science Garden only. Run the G3 convention test, then set `available_from = period_end_pht + 1 h`.
Rolling features (24 h, 72 h) use only reports available at t.

### 7.5 QC rules (set on 2018–2022 only, then applied automatically to 2023–2024)
| Rule | Detail |
|---|---|
| Valid range | per station from `qc.yaml` (e.g. Pandacan 9–15 m); outside → NaN + flag |
| Spike | \|xₜ−xₜ₋₁\| > s and \|xₜ−xₜ₊₁\| > s with opposite signs; s per station (prototype 0.5 m) |
| Flat run | ≥ L identical values; L per station. **Pandacan:** remove flat runs at exactly 11.89 m of the lengths found on 3 Nov (after Ulysses, 25–26 Jul 2021, Paeng); keep shorter 11.89 m runs. Low-tide floor runs (≈ 10.63–10.66 m) are exempt. **Sto. Niño:** flat runs inside 11.8–12.5 m are exempt unless Montalban's 3-h median rises ≥ 0.5 m while Sto. Niño stays flat 12 h later; also report coverage without the exemption |
| Hand check | Every flagged **training** Pandacan value > 11.69 m checked against neighbouring stations, the Aviles river-side gauge and rainfall; rules adjusted until real peaks survive |
| 2023–2024 | Rules applied unchanged; flagged values become NaN; nothing edited by hand; every removal logged; Pandacan values > 11.69 m listed separately, including whether Carina's peak survives |

Expected: ~103 Pandacan training hours removed; P95 stays 11.69 m.

### 7.6 Gap fill
Gaps ≤ 3 h forward-filled with `<col>_filled = True`; longer gaps stay NaN. Filled hours are never targets.

### 7.7 Hourly master
One hourly PHT index for June–November 2018–2024; one column per variable + `_filled` + `_qcflag`;
column `season` = year.

### 7.8 Coverage and eligibility (G5)
Coverage per station per season (share of unflagged readings, before fill) and per calendar year.
Redraw Figure III-3 from this table (label: wet season). Write `eligible_inputs.yaml`.

### 7.9 Evaluation hours
Hour t is a scored case at lead h iff the 24 Pandacan readings t−23…t exist (filled allowed), the
Pandacan reading at t+h is observed and not filled, and the Science Garden report needed at t exists
(and every eligible input's window is complete). Save `eval_hours.parquet` (one boolean per lead).
Recompute Table III-6 (hours, cases at leads 1 and 6, alert hours, events per split).

### 7.10 Tests → G6
`test_parsers.py`, `test_time_alignment.py`, `test_qc.py`, `test_splits.py`. Then tag `phase1-frozen`.

---

## 8. Notebook 02 — Link analysis and alert levels (SOP 1)

Training seasons for fitting; 2023 for checking; 2024 Aviles only after Notebook 6.

### 8.1 Status legend check (G4, D3)

| Code part | Assumed meaning | Check on 2018–2022 | Used for |
|---|---|---|---|
| Leading `O`/`C` | Outfall gate open/closed | `O` hours: outside < inside in most hours; `C` hours: outside ≥ inside | open-gate share |
| `nP` (`1P`, `2P`…) | n pumps running | pump hours cluster at high sump and high Pandacan levels; digit ≤ max seen at that station | stress label (≥ 2 pumps) |
| `nP` with no `O`/`C` | gate closed while pumping | outside ≥ inside in most such hours | open-gate share |
| `F`, `C` (sky) | fair, cloudy | Science Garden rain in the matching 6 h near zero more often than for rain codes | not used in model |
| `R`, `LR`, `LRS`, `S` | rain, light rain, light rain shower; `S` = shower if its rainfall matches R/LR, else sunny | same rainfall comparison | not used in model |

Reference: project_context found ≥ 1 pump in 14% of "no rain" hours vs 85% of "light rain" hours.

### 8.2 First assumption
Pandacan vs Aviles outside elevation: Pearson, Spearman, mean difference, correlation of hourly changes,
over all training hours and hours ≥ 11.69 m.

### 8.3 Grid tables
For L on a 0.05 m grid: share of stress hours (Wilson 95% CI), share with open gate, share of dry hours
(SG 24 h rain < 5 mm), also by event (runs ≥ L, gaps < 6 h merged; event-bootstrap CIs).

### 8.4 Logistic regression
`statsmodels.Logit`, SE clustered by calendar day: M0 = ln(1+R₂₄); M1 = M0 + L; M2 = M1 + L·ln(1+R₂₄).
Report OR per 0.1 m with 95% CI; LR test M0 vs M1 and M1 vs M2 (unclustered fits, stated); 2023 ROC-AUC
of M0 and M1 and the gain with 2,000 day-block bootstrap reps. Quiapo gate records as a second check
for the confirmed Quiapo years (2018, 2019, 2022) plus 2021 if G20 passes, reported **with and without**
2021 (D18). No real Quiapo 2023 data are expected, so the Quiapo check covers training years only.
Apply the H1 rule → G7.
Reference: OR ×1.19 (1.13–1.26) after rain; AUC gain 0.006 (CI −0.019 to +0.030) → "partly supported".

### 8.5 Alert levels (G8)
Rules in G8. Sensitivity grid: stress ≥ 1/2/3 pumps × cut-off 40/50/60% × levels 11.90–12.10 m →
`reports/tables/alert_sensitivity.csv`. Check both levels on 2023 (exceedance rate, dry-alert share) and
record without changing them. Write `alert_levels.yaml` (levels, rule, git SHA, date).

---

## 9. Notebooks 03–05 — Features, Granger, training

### 9.1 Notebook 03 — tide, features, windows, split
- **Tide (D7, G9):** least-squares harmonic fit per water-level series on all months of 2018–2022 from
  `waterlevel_long.parquet`, hours above floor + 0.05 m only. Save amplitudes/phases/SEs to
  `artifacts/tide_<station>.json`. `predict_tide(station, times)` works for any hour.
- **Feature table:**

| Path | Feature | Shape | Known at t because |
|---|---|---|---|
| Sequence | Pandacan level and Δlevel, t−23…t | 24×2 | observed |
| Sequence | Each **eligible and selected** river station, t−23…t | 24×k | observed |
| Sequence | Latest Science Garden 6 h total, step-held | 24×1 | available ≥ 1 h after period end (D5) |
| Tabular | Hour-of-day and day-of-year sin/cos at t and t+1…t+6 | 28 | clock |
| Tabular | Predicted tide at t+1…t+6 and change from t (D8) | 12 | tide model |
| Tabular | SG rain 24 h and 72 h totals, hours since last report | 3 | available reports only |
| Tabular | Previous-day SG RH, pressure, mean temperature (if selected) | ≤3 | daily values from 09:00 next day |

- **Windows:** built inside each season (no window crosses the dry-season gap). Arrays per split:
  `X_seq (N,24,C)`, `X_tab (N,F)`, `y (N,6)`, `valid (N,6)`, `t_index (N,)`.
- **Scaling:** min-max fitted on training cases only. **Target scaler = the Pandacan sequence channel's
  scaler (D9)**, so the add-back is exact. Invalid targets encoded as −1 after scaling (masked loss).

### 9.2 Notebook 04 — Granger selection (SOP 2)
Candidates: eligible river stations (Sto. Niño, Fort Santiago if they pass G5), Science Garden rainfall
(6-hourly steps), Science Garden RH, pressure, mean temperature (daily, against the daily max of the
de-tided Pandacan level). Steps:
1. De-tide every water-level series with its own tide fit.
2. ADF per training season (`autolag="BIC"`); difference once if p ≥ 0.05; same transform for all seasons.
3. Lag matrices built **within** each season; drop NaN rows; stack seasons; add season dummies.
4. p = 1…Pmax (24 hourly, 4 six-hourly, 3 daily); choose p by BIC of the unrestricted OLS.
5. F-test restricted (own lags) vs unrestricted at that p; select if p-value < 0.05 / m (m = number tested).
6. Beside each p-value, report the validation RMSE gain of a ridge model on the 24 h window with vs
   without the candidate.
7. Correlation screen for the ablation: same count of candidates ranked by max |lagged cross-correlation|
   (lags 1–24 h) with de-tided Pandacan on training data.
Write `selected_features.yaml` with all statistics → G10. Report every result.

**Ineligible stations (D2):** Sto. Niño and Fort Santiago, if they fail G5, still go through steps 1–6
on the training hours they have (seasons with too little data are skipped and named). Their results go
into a separate "descriptive only" table and are never used as model inputs, and they do not count
toward m for the Bonferroni level of the eligible candidates.

### 9.3 Notebook 05 — models and training (SOP 3)

**Start-of-notebook gates:** run `scripts/smoke/bench_compute.py` on CPU (and GPU if NVIDIA) twice;
evaluate **G14 → G13 → G15**; write `compute.yaml`. Then **G11** and **G12** on the reference config.

**Hybrid (Figure III-4), reference code in `scripts/smoke/common.py::build_hybrid`:**

```python
seq = Input((24, C)); x = seq
for _ in range(cfg.lstm_layers):                      # 1–2 layers, 64 or 128 units
    x = LSTM(cfg.lstm_units, return_sequences=True)(x)
    x = Dropout(cfg.dropout)(x)                       # dropout BETWEEN layers; never recurrent_dropout
x = Dropout(cfg.dropout)(GRU(cfg.gru_units)(x))       # 32 or 64 units
tab = Input((F,)); z = tab
for _ in range(cfg.dense_layers):                     # 1–2 layers, 16 or 32 units, ReLU
    z = Dense(cfg.dense_units, activation="relu")(z)
h = Dense(cfg.head_units, activation="relu")(Concatenate()([x, z]))
delta = Dense(6, activation="linear", name="change")(h)          # D9
level_t = Lambda(lambda s: s[:, -1, 0:1])(seq)                    # scaled Pandacan at t
out = Add(name="levels")([delta, Concatenate()([level_t] * 6)])   # level at t+1 … t+6
```

- Loss: **masked MSE** over valid targets only (`y_true > -0.5`).
- Adam; at most 150 epochs; `EarlyStopping(val_loss, patience=10, restore_best_weights=True)`.
- **Standalone LSTM:** same inputs, tabular vector repeated at every time step, same add-back (D9).
- Keep the cuDNN path: default activations, `recurrent_dropout=0`, `unroll=False`, no masking layers. float32.

**Grid (D12, D14 ranges, criterion D10):**
- Stage 1: 24 configs = LSTM layers {1,2} × LSTM units {64,128} × GRU {32,64} × batch {32,64,128};
  fixed dropout 0.2, lr 1e-3, dense 1×16. One seed each.
- Stage 2: 36 configs on the stage-1 winner = dropout {0.2,0.35,0.5} × lr {1e-3,5e-4,1e-4} ×
  dense {(1,16),(1,32),(2,16),(2,32)}. One seed each.
- Resumable: one CSV row per config (`runs/grid_stage{1,2}.csv`); skip existing rows.
- Then G19. Final hybrid config retrained with seeds 0–4 (determinism on, device per G13/G14).
- Standalone LSTM grid: units {64,128} × dropout {0.2,0.35,0.5} × lr {1e-3,5e-4,1e-4} (18), then 5 seeds.

**Baselines (Table III-4):**

| Model | Task | Inputs | Tuned on 2023 |
|---|---|---|---|
| Persistence | level | last observed level for all leads | — |
| Tide-only | level | predicted tide + seasonal terms (training years) | — |
| Tide + persisted residual | level | tide(t+h) + (Lₜ − tideₜ) | — |
| Ridge | level | flattened 24 h window + tabular | alpha 0.01…100 |
| KNN regressor | level | same | k {5,11,21,41,81,161}, uniform/distance (extend until best k is interior) |
| Logistic Regression | alert | (a) current-hour inputs, (b) 24 h window + tabular | C, **class weights**, decision threshold |
| KNN classifier | alert | same two input sets | k, weighting, **decision threshold only** (C4) |

Classifiers: one per lead × alert level × input set (24 each).

**Thresholds and trigger offsets (D11):** for each level-forecasting model and each alert level, search
offsets −0.30…+0.30 m (0.01 m) on 2023, maximizing F1; **one offset shared by all six leads**. Alert iff
forecast ≥ level − offset. Classifier thresholds tuned on 2023 by F1. Write `thresholds.yaml`.

### 9.4 Seeds and determinism
`keras.utils.set_random_seed(seed)`; `tf.config.experimental.enable_op_determinism()` for every final,
ablation, walk-forward and test run (D15). Seeds 0–4 (or 0–2 per G15 for walk-forward/ablations).

### 9.5 Run folders
`runs/<yyyymmdd_hhmm>_<tag>_seed<k>/`: `config_snapshot.yaml`, `git_sha.txt`, `env.txt`, `model.keras`,
`history.json`, `timing.json`, `preds_val.parquet`.

### 9.6 End of Notebook 05 → G16, G17
Persistence sanity (G16) for every model; then freeze (G17), tag `settings-frozen`.

### 9.7 Bug checklist when a model does not beat persistence (G16)
1. Target alignment: `y[:, h-1]` equals level at t+h (test_windows).
2. Scaler: target scaler equals the Pandacan channel scaler; inverse transform applied once.
3. Masked loss: invalid targets really −1 and excluded.
4. Leakage in reverse: inputs shifted by one hour (rain availability, daily values).
5. Learning: training loss falls; validation loss not flat from epoch 1; lr not too high.
6. Evaluation: only observed, unfilled targets scored; same hours for all models.

---

## 10. Notebooks 06–07 — Evaluation, robustness (SOP 4)

### 10.1 Notebook 06 (G18 first)
Predict 2024 with every model (all seeds for neural models). Save
`preds_test.parquet`: `t, lead, model, seed, y_true, y_pred, valid, rainy`.

| Group | Measures | Granularity |
|---|---|---|
| Level error | RMSE, MAE, R², skill = 1 − RMSE_model/RMSE_ref vs persistence **and** vs strongest simple baseline | per lead; all hours and rainy hours |
| Hourly alerts | Precision, Recall, F1, PR-AUC (beside the alert-hour share), FN (missed), FP (false alarms) | per lead × level; rainy and dry |
| Equal recall (H4, D19) | FP at Recall = 0.90 for the verdict; also at 0.80 and 0.95 (descriptive). Threshold moved on test for this comparison only; if exactly 0.90 is unreachable, use the first threshold with Recall ≥ 0.90 and report the Recall achieved | high-water level, per lead |
| Events | detected if an alert falls from 2 h before to 2 h after the first exceedance; warning lead time | per lead × level |
| Uncertainty | moving-block bootstrap (24 h blocks, 2,000 reps) for differences; event bootstrap for events; seeds as mean ± SD | all comparisons |

- Rainy hour = ≥ 5 mm SG rain reported in the previous 24 h (from reports available at t).
- **Diebold–Mariano** on squared errors, Newey–West variance with lag max(h−1, ⌊4(n/100)^(2/9)⌋), HLN
  small-sample correction, Holm over the whole family; all hours and rainy hours; neural models use the
  seed with median validation RMSE. Reference implementation: `scripts/smoke/common.py::diebold_mariano`.
- Accuracy reported for reference only.
- Apply the H3 and H4 rules; record verdicts.

### 10.2 Notebook 07
- **Walk-forward:** 2018–20 → 2021, 2018–21 → 2022, 2018–22 → 2023; per fold refit scaler, tide,
  alert levels, Granger selection, thresholds/offsets from that fold's training years (QC rules stay as
  frozen). Report the 2023 fold separately (not independent).
- **Ablations (Table III-5), six variants:** no Granger selection; correlation selection; no rainfall;
  no tabular path; GRU replaced by a second LSTM; **weighted loss (D20)**. Seeds per G15; same evaluation
  hours. Implement the weighted loss as the masked MSE times a per-target weight (w for valid targets ≥ the
  high-water level, 1 otherwise; the weight uses the true future level, which is allowed in a training loss
  but never as an input). No missing-indicator variant (D2).
- **Case study:** every 2024 flood-risk event (~12–13), including 24 Jul 2024 (Carina): observed level,
  each lead's forecast, alert times, hours of warning before first exceedance.
- **Deferred check:** compare alert levels descriptively with Aviles June–15 Jul 2024 records.

---

## 11. Tests (pytest)

`pytest -m fast` (synthetic fixtures, < 1 min) and `pytest -m data` (real processed files).

| File | Asserts | Required test # |
|---|---|---|
| `test_parsers.py` | 09:00 rollover (day-31 0:00 → 1st of next month); impossible dates set aside; missing tokens and 0.0 m → NaN; Aviles cm → m; month from file name | — |
| `test_time_alignment.py` | 6 h total invisible until 1 h after its period end; daily values at 09:00 PHT next day; G3 convention still holds | 3 |
| `test_qc.py` | planted spikes/flat runs flagged; floor runs kept; 11.89 m rule reproduces | — |
| `test_splits.py` | no train timestamp ≥ any val/test timestamp; wet months only; no window crosses seasons | 1 |
| `test_gapfill.py` | only gaps ≤ 3 h filled and flagged; filled hours never valid targets | 5 |
| `test_leakage.py` | +5 m on every 2024 value leaves scalers, tide coefs, alert levels, Granger results, thresholds identical | 2 |
| `test_windows.py` | `y[:, h-1]` = level at t+h; tabular t+h terms come from the clock | 4 |
| `test_same_hours.py` | every model and ablation, incl. persistence, scored on identical hours | 6 |
| `test_reproducibility.py` | same seed + config → identical predictions (atol 1e-6) | 7 |
| `test_granger.py` | finds a planted 3 h lead; rejects noise at α = 0.01 | — |
| `test_metrics.py`, `test_events.py`, `test_dm.py` | hand-computed values; DM matches statsmodels where available | — |
| `test_gates.py` | `require_gates` raises on missing/failed gates | — |

Required test 8 is gate G16.

---

## 12. Smoke tests and compute benchmark (already written)

`scripts/smoke/` from `smoke_tests.zip`:
- `smoke_pipeline.py`: all phases on synthetic data with the real quirks; prints 7 checks (rain usable only
  1 h after period end; Granger finds planted lead; Granger rejects noise; scaler ignores test; target =
  t+1; loss falls; same seed → same predictions). Measured: 31 s on 2 vCPUs, 10 epochs, 3,000 cases.
- `bench_compute.py`: seconds per epoch for all 24 architecture × batch combinations + LSTM + sklearn;
  writes `projection.csv` and `summary.json`. Reference (2 vCPU cloud, TF 2.21, determinism on, 60 epochs/run):
  two-stage plan ≈ 16 h; full factorial (864 configs) ≈ 123 h. Batch 128 ≈ 1.9× faster than batch 32 on CPU.
  The D20 weighted-loss ablation adds about 8 runs (≈ 1 h on that reference machine); include it when
  comparing the projection with the 10 h limit of G15.
- `common.py` already implements D5 (rain lag), D7 (tide fit on all training-year months above floor),
  D9 (`add_back_level=True`, `level_scaler`). Move these into `pasig/` unchanged.

Run on the chosen laptop:
```bash
python scripts/smoke/smoke_pipeline.py --device cpu            # must print 7 PASS
python scripts/smoke/bench_compute.py  --device cpu --measure-epochs 5   # run twice; use the second
# NVIDIA only:
python scripts/smoke/smoke_pipeline.py --device gpu
python scripts/smoke/bench_compute.py  --device gpu --measure-epochs 5
python scripts/smoke/bench_compute.py  --device gpu --measure-epochs 5 --no-determinism
```
Commit `summary.json` and `projection.csv` to `reports/compute/<laptop>/`; they feed G13 and G15.

---

## 13. Thesis text that must change (do not edit the PDF; the team edits the source)

Pages are PDF pages of `11_2_Revised_Chapters_1-3.pdf`.

| # | Where | Change | Decision |
|---|---|---|---|
| 1 | Ch I §C p. 8 | "the tidal phase" → "the predicted tide" | D8 |
| 2 | Ch I Fig. I-2 p. 11 | "predicted tide and its harmonics" → "predicted tide" (redraw) | D8 |
| 3 | Ch I §G p. 13 | add: status codes had no legend; meanings inferred and checked | D3 |
| 4 | Ch II §F p. 31 | only if G15 → 3 seeds: five seeds for final networks, three for walk-forward/ablations | D13 |
| 5 | Ch III Table III-1 p. 43 | Quiapo note: 2020–21 other station; 2023/24 identical, one copy used, dated by sheets | D1 |
| 6 | Ch III Ph1 p. 44 | rainfall "assigned to the hour its period ended" → "made available one hour after its period ended" | D5 |
| 7 | Ch III Ph1 p. 44 | trace → 0.05 mm; missing −999 | D4 |
| 8 | Ch III Ph1 p. 45 | add the 2023/2024 Quiapo duplicate | D1 |
| 9 | Ch III Fig. III-3 p. 46 | redraw from rebuilt coverage | D1 |
| 10 | Ch III Ph2 p. 46 | add status-code reading and checks | D3 |
| 11 | Ch III Ph2 p. 47 | Quiapo check for the confirmed Quiapo years (2018, 2019, 2022) and, if its identity was confirmed, 2021 from a block filed under other years; results shown with and without 2021 | D1, D18 |
| 12 | Ch III Table III-2 p. 49 | rainfall known "≥ 1 h after period end" | D5 |
| 13 | Ch III Table III-2 p. 50 | tide row → predicted tide t+1…t+6 and its change | D8 |
| 14 | Ch III Ph3 p. 50 | tide fit on all months of 2018–2022, hours > floor + 0.05 m | D7 |
| 15 | Ch III Ph4 p. 52 | head predicts change; current level added back; direct form compared once | D9 |
| 16 | Ch III Fig. III-4 p. 53 | "tidal harmonics" → "predicted tide" | D8 |
| 17 | Ch III p. 53 | table title → "Table III-3 Hyperparameter Settings and Search Ranges" | C6 |
| 18 | Ch III Table III-3 p. 54 | head row: change + current level | D9 |
| 19 | Ch III Table III-3 p. 54 | loss: MSE over observed target hours only | plan |
| 20 | Ch III Ph4 p. 54 | two-stage grid (Section 9.3); criterion = mean val RMSE over six leads | D12, D10 |
| 21 | Ch III Table III-4 p. 55 | tide-only "fitted on the training years" | D7 |
| 22 | Ch III Ph4 p. 57 | trigger offset "shared by all six lead times" | D11 |
| 23 | Ch III §C p. 60 | rainfall total appears "one hour after" its period ends | D5 |
| 24 | Ch III §D p. 60 | replace Colab sentence with local VS Code on the chosen Windows laptop (+ WSL2 GPU sentence if G13 = gpu); delete Bisong (2019) reference | D16, D17 |
| 25 | Ch III §E p. 63 | file problems: other station, identical files, mislabeled sheets | D1 |
| 26 | Ch III §F.9 p. 67 | only if G15 → 3 seeds | D13 |
| 27 | Ch III Ph3 p. 52 | after "A candidate was kept only if it passed the coverage rule and the corrected Granger test": add that stations failing the coverage rule were still tested on their available training hours and reported, but not used as inputs | D2 |
| 28 | Ch III §F.7 p. 66 | after the Recall 0.90 comparison: add that it was repeated at Recall 0.80 and 0.95 as a sensitivity check, that the H4 decision used 0.90 only, and that where 0.90 could not be reached exactly the first threshold reaching at least 0.90 was used and the achieved Recall reported | D19 |
| 29 | Ch III Table III-5 p. 59 | add row: **Weighted loss** — "Training targets at or above the high-water level given weight w (2, 5 or 10, chosen on 2023) in the mean-squared-error loss" — "Does emphasizing high-water hours in training reduce the under-forecasting of peaks?" | D20 |
| 30 | Ch III Ph5 p. 58 | "The ablation study in Table III-5 removed one component at a time" → "changed one component at a time" (the weighted-loss variant adds a component) | D20 |
| 31 | Ch III Ph4, no-SMOTE paragraph (from `class_imbalance_and_safeguards.md` §7B) | add a closing sentence: "A weighted-loss variant, which gave hours at or above the high-water level more weight during training, was tested as an ablation (Table III-5)." | D20 |

Also from `class_imbalance_and_safeguards.md`: reword "class weights" to "class weights for Logistic
Regression and a decision threshold tuned on 2023 for both classifiers" (C4), and add the no-SMOTE paragraph.

---

## 14. Items formerly reserved for the adviser

The adviser left these to the group. Items 1–4 were decided on 6 Nov 2026; item 5 stays open and does
not affect the code.

| # | Item | Status | Decision |
|---|---|---|---|
| 1 | Keep Sto. Niño and Fort Santiago if they fail the every-season coverage rule? | Decided | **D2**: rule kept; tested descriptively only; no missing-indicator variant |
| 2 | Use the Quiapo 2021 block (inside `QUIAPO 2022.xlsx`, and possibly the single 2023/2024 copy)? | Decided | **D18**: used in the secondary Quiapo check if it passes G20; reported with and without it |
| 3 | Is Recall 0.90 the right comparison point for H4? | Decided | **D19**: 0.90 for the verdict; 0.80 and 0.95 as sensitivity |
| 4 | Weighted-loss ablation (training-side imbalance treatment)? | Decided | **D20**: added as a sixth ablation; main model unchanged |
| 5 | Proposal vs final-manuscript tense; working title | **Open** | Writing only; no code depends on it |

---

## 15. Definition of done

- [ ] Every raw file in the manifest with a status; G1 passed
- [ ] All gates G0–G20 have written their files; `docs/decisions.md` has D1–D20
- [ ] `pytest -m fast` and `pytest -m data` pass, including `test_leakage.py`
- [ ] Every Chapter IV table and figure regenerates from one run of Notebooks 06–07
- [ ] Hardware and measured run time stated in Chapter III from `reports/compute/`
- [ ] Preliminary numbers (11.69 m, 12.0 m, Table III-6, Figure III-3) replaced by recomputed ones
- [ ] Results reported whichever way they fall, including any case where the hybrid does not win
