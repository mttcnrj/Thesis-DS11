# Decision register (D1–D20) and implementation choices (P1–P7)

Source: IMPLEMENTATION_PLAN.md Section 4 (decided 6 Nov 2026). Each entry: date, choice, reason, gate file,
Chapter rows it changes (Plan Section 13). "Outcome" records what the gate decided when it has run;
"decided on the laptop" = code done and dry-run, the real gate runs on the chosen laptop (D16).

Status (7 Nov 2026): Phases 1–4 run on real data; Notebooks 05–07 built and dry-run; 2024 not opened.

| ID | Date | Choice | Reason (short) | Gate file | Chapter rows | Outcome |
|---|---|---|---|---|---|---|
| D1 | 2026-11-06 | One copy of the identical QUIAPO 2023/2024 files; dated by sheet headers | identical SHA-256; file names unreliable | `gates/G01_raw_files.json` | 5, 8, 9, 11, 25 | G1 passed. Kept `QUIAPO 2023.xlsx`; its sheets are headed **2021, 2023 and 2024** (F1). 2023 sheet: descriptive only; 2024 sheet: sealed until after NB06 (decision P2-b) |
| D2 | 2026-11-06 | Keep the 80% every-season rule; failing stations tested descriptively only; no missing-indicator variant | Fort Santiago 0–2% in 2018–19; tiny ablation effect | `gates/G05_eligibility.json`, `gates/G10_granger_outcome.json` | 27 | G5: **no eligible upstream input** (Sto. Niño min 78.1% in 2024; Fort Santiago 0% in 2018). Descriptive Granger in NB04: both significant, both worsen the ridge validation RMSE |
| D3 | 2026-11-06 | Assumed status-code legend, validated on 2018–2022 | no MMDA legend | `gates/G04_status_legend.json` | 3, 10 | G4: gate and pump fields **used** after the post-hoc amendment P2-a (literal checks failed for physical reasons, reported); **sky field dropped** (F/C dry check fails 21.1% > 20%); `S` reads as shower at Aviles |
| D4 | 2026-11-06 | Trace rainfall = 0.05 mm | half the 0.1 mm limit | parser + `tests/test_time_alignment.py` | 7 | implemented (6-hourly `T`, daily `-1`) |
| D5 | 2026-11-06 | 6-hourly total usable 1 h after its period ends | no look-ahead | `gates/G03_rain_timing.json` | 6, 12, 23 | G3: stamps mark the **period end** (99.7% of 1,172 rain days match vs 53.7%) |
| D6 | 2026-11-06 | Port Area excluded | no 2024 data, gaps | `gates/G01_raw_files.json` | — | Port Area file excluded; Port Area columns dropped from the 6-hourly file |
| D7 | 2026-11-06 | Tide fit on all 12 months of 2018–2022, hours > floor + 0.05 m | more tidal cycles | `gates/G09_tide_fit.json` | 14, 21 | fitted for Pandacan (floor 10.65 m), Sto. Niño (11.32 m), Fort Santiago (10.59 m); `artifacts/tide_<station>.json` |
| D8 | 2026-11-06 | Predicted tide t+1…t+6 and change (12 cols); constituent pairs tried once | | G11 (NB05) | 1, 2, 13, 16 | 12 tide columns in `X_tab`; constituent pairs prepared as `X_tab_cons` (**120** columns = 10 frozen constituents × sin/cos × 6 leads, not 84, because G9 added 3 constituents); G11 implemented (NB05 `variants`; switch rule P5-c); decided on the laptop |
| D9 | 2026-11-06 | Head predicts change; add-back of L_t | | G12 (NB05) | 15, 18 | implemented (registered `LevelAtT` layer, P5-m; target scaler = Pandacan channel); G12 decided on the laptop |
| D10 | 2026-11-06 | Selection by mean validation RMSE over six leads | | grid code | 20 | implemented (`pasig/train/grid.py`) |
| D11 | 2026-11-06 | One trigger offset per model per alert level, shared by leads | | offset code | 22 | implemented (`pasig/models/offsets.py`, P5-j; also per ablation and per walk-forward fold) |
| D12 | 2026-11-06 | Two-stage grid, 60 configs | | grid code | 20 | implemented (24 + 36, resumable; + 18 standalone LSTM) |
| D13 | 2026-11-06 | 5 seeds if projection ≤ 10 h else 3 for walk-forward/ablations | | G15 | 4, 26 | implemented (projection includes the D20 runs and G11/G12, P5-d); decided on the laptop |
| D14 | 2026-11-06 | Chapter III ranges exactly | | G19 | — | implemented (`model_hybrid.yaml` ranges; G19 records edge winners); decided on the laptop |
| D15 | 2026-11-06 | Determinism on for final runs; CPU fallback | | G14 | — | implemented (finals, ablations, walk-forward always deterministic; NB06 inference on CPU); G14 decided on the laptop |
| D16 | 2026-11-06 | Local VS Code on the Windows laptop with the stronger NVIDIA GPU (else more cores) | | `gates/G00_machine.json` | 24 | **G0 recorded**: the group's chosen laptop (AMD Ryzen 7 7735HS, 8 cores/16 threads, 16 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU) → Python in WSL2 Ubuntu. Only this laptop's output was provided. `nvidia-smi` (6 Nov): **8188 MiB**, driver 595.97, CUDA 13.2 (F30). Phases 1–3 here ran in a cloud container (deterministic); re-run them on the laptop |
| D17 | 2026-11-06 | GPU only if visible, G14 passes, projection faster | | G13 | 24 | implemented (NB05 `compute`, second of two benchmark runs); decided on the laptop |
| D18 | 2026-11-06 | Quiapo 2021 used conditionally (G20) in the secondary check; with/without reported | only route to 2021 | `gates/G20_quiapo_2021_identity.json` | 11 | G20 **passed** (a–d all hold; copies identical → one series): 2021 included; Quiapo check reported with and without it |
| D19 | 2026-11-06 | Recall 0.90 for the H4 verdict; 0.80/0.95 sensitivity | | eval code | 28 | implemented (`pasig/eval/alerts.py`, P6-g; rule in `configs/eval.yaml`, frozen by G17) |
| D20 | 2026-11-06 | Weighted-loss ablation (w ∈ {2,5,10}) | | ablation code | 29, 30, 31 | implemented (`weighted_masked_mse`; weight search and report in NB07, P7-f); run on the laptop |

## Gate outcomes recorded in Phase 2

| Gate | Outcome |
|---|---|
| G4 | all gate/pump fields used (amended checks P2-a); sky dropped |
| G20 | passed: Quiapo 2021 is Quiapo (r with Pandacan 0.915 vs confirmed 0.885–0.917; not a copy of Aviles r = 0.955 or Taguig r ≤ 0.19) |
| G7 | **H1 partly supported**: OR per 0.1 m (M1, day-clustered) 1.207 (1.143–1.275) excludes 1; 2023 ROC-AUC gain 0.004 (−0.022 to +0.028) does not exclude 0 |
| G8 | high water **11.69 m** (P95), flood risk **12.00 m** (first 0.05 m level ≥ 11.69 with ≥ 50% stress hours: 55%, n = 80) → `configs/alert_levels.yaml` (frozen) |

## Decisions taken during Phase 2 (6 Nov 2026)

| Ref | Decision | Why | Chapter text |
|---|---|---|---|
| P2-a | **Post-hoc amendment of two G4 checks**, approved by the group after the literal checks failed. Gate `O` fails if the river side is *above* the sump (was: river ≥ sump); `nP` fails if the river side is *not above* the sump (was: neither sump nor Pandacan above its no-pump median). Literal results stay in the G4 evidence | An open gate equalises both sides (99% of `O` hours read exactly equal, so the literal check failed 100%); pumping draws the sump down (median 10.20 vs 10.60 m), so a "high sump" test fails by design (42%). Amended checks fail 1.1% and 0.3%. Independent support: pump count rises with rain (median R24 1.8 → 6 → 28 → 76 mm for 0–3 pumps) and with Pandacan level (11.15 → 11.68 m) | Ch III Ph2 (row 10): describe the checks as amended and why, with both results; Ch I §G (row 3) unchanged |
| P2-b | **Quiapo 2023/2024 sheets (F1): option (a)**. Outside the H1 verdict; the 2023 sheet is reported as a descriptive 2023 check of the alert levels; the 2024 sheet stays sealed until after NB06 | keeps the pre-registered design; uses the found data openly | Ch III Ph2 (row 11) and Table III-1 note (row 5) |
| P2-c | Operational definitions fixed in `configs/link.yaml` before the statistics: dry hour = SG 24 h rain < 5 mm; events = runs ≥ L with gaps < 6 h merged; event-level stress = event contains ≥ 1 stress hour; L and ln(1+R24) centred at training means (M2 main effects at mean rain); grid = multiples of 0.05 m from 10.80 to 12.60 m; flood-risk scan starts at the high-water level | the plan leaves these open | Ch III Ph2 |

## Gate outcomes and choices in Phase 3 (6 Nov 2026)

| Ref | Item | Detail |
|---|---|---|
| G9 | tide fit | rule applied as written: base 7 constituents failed the spectrum check at every station, so P1, K2, Q1 were added (max 3). Residual peaks remain beside M2/K1/O1 (consistent with seasonally modulated tides that a constant-amplitude fit cannot remove); frozen with the 3 additions and logged. Pandacan fit RMSE 0.190 m; tide-only RMSE on 2023 wet hours 0.158 m (recorded, not used) |
| P3-a | floor | floor = 1st percentile of the station's QC'd training-year readings, rounded 0.01 m (Pandacan 10.65 m; plan ≈ 10.63 m) — fixed in `configs/tide.yaml` before fitting |
| P3-b | G9 implementation | periodogram = \|FFT\|² of the fit residuals on the hourly grid (mean removed, unfitted hours 0, 4× zero padding); peak window ±3%, nearby band 3–20% (a Lomb-Scargle version was too slow for the 5-year grid) |
| P3-c | test windows | NB03 builds **train and validation windows only**; the 2024 windows are built by the same frozen function in NB06 after G18 (Section 0.3: 2024 read once, in NB06) |
| P3-d | missing inputs | a missing 6 h rain report counts as 0 mm in rain inputs (38 blank reports 2018–2024; the report needed at t must still exist, Phase 1 rule); the first hour of a season has Δlevel = 0; missing previous-day weather → training median (0 cases needed it) |
| P3-e | case set | a case is any hour scored at ≥ 1 lead in `eval_hours.parquet`; `valid` = its lead flags, so all models share the same scored hours (train 18,386 cases; 16,937 at lead 1, 16,699 at lead 6; val 4,288; 4,091 / 4,081) |
| P3-f | branch | work branch renamed to `feat/pasig-forecast-pipeline` (the old `feat/affectionate-mccarthy-w2uqzk` could not be deleted from this session; delete it on GitHub) |

## Gate outcomes and choices in Phase 4 (6 Nov 2026)

| Ref | Item | Detail |
|---|---|---|
| G10 | Granger outcome | **some pass**: Science Garden rainfall (p = 6.1e-48, lag 4 six-hour steps) and daily pressure (p = 2.2e-6, lag 3 days) selected at p < 0.05/4 = 0.0125; RH (p = 0.74) and mean temperature (p = 0.29) not selected. **H2 supported.** Model inputs → `configs/selected_features.yaml`: sequence = Pandacan level, Δlevel, latest SG 6 h rain; tabular = calendar (28), tide (12), SG rain 24 h / 72 h / hours since report, previous-day SG pressure |
| P4-a | candidates and time steps | eligible candidates only (G5 left no river station): rain at its 6-hourly step against the **mean** de-tided Pandacan over each report period (≥ 4 of 6 h present); daily RH/pressure/tmean against the **max** de-tided Pandacan over the PAGASA day 09:00 D–08:00 D+1 (≥ 18 of 24 h), the window the daily value is usable for |
| P4-b | stationarity | ADF (BIC) per training season; differenced once for all seasons if any season has p ≥ 0.05 (Pandacan series were differenced at both steps; rain and tmean kept in levels; RH and pressure differenced) |
| P4-c | lag choice | BIC of the unrestricted OLS compared on the common sample valid at the maximum lag (the smoke version compared different samples); F-test on the rows valid at the chosen lag, restricted model on exactly those rows |
| P4-d | ridge gain (step 6) | ridge per lead on the Phase 3 windows, alpha by leave-one-season-out CV on training (grid 0.01–100; 0.01 chosen, the grid edge, reported not widened), scored on 2023 valid targets (mean RMSE over leads). Gains are tiny: rain +0.21 mm, pressure +0.26 mm, RH −0.01 mm, tmean +0.005 mm |
| P4-e | correlation screen | max \|lagged cross-correlation\| (1–24 h) of each candidate as available at t, transformed like its Granger series, with de-tided Pandacan (differenced): rain 0.036, RH 0.019, pressure 0.018, tmean 0.004 → the correlation-selection ablation uses **rain + RH** (same count as Granger: 2) |
| P4-f | descriptive stations (D2) | Sto. Niño (all 5 seasons; lag 23 h, p = 1.6e-166) and Fort Santiago (2020–2022 only; 2018–2019 skipped for < 50% coverage; lag 23 h, p = 4.1e-43) would pass any threshold, but adding either 24 h window **worsens** the ridge validation RMSE (−0.27 mm and −2.0 mm on the cases where their windows are complete); never inputs, not in m |

## Notebook 05: implementation choices (built and dry-run, 6 Nov 2026; real run pending on the laptop)

| Ref | Item | Choice | Why |
|---|---|---|---|
| P5-a | stages | eight resumable stages (compute → variants → grid → finals → baselines → thresholds → checks → freeze), each in its own process | the TensorFlow device must be fixed before TF starts (G13/G14 may put the grid and the finals on different devices, D15); a crash or reboot loses at most one config |
| P5-b | rehearsal profile | `PASIG_PROFILE=rehearsal` runs everything with 2 epochs, 1,500 training cases and 2 configs per grid; gates, generated configs, checkpoints and tables go to `runs/rehearsal/` (git-ignored) | full dry run without touching the real gates; real Phase 1–4 outputs are still read |
| P5-c | G11 / G12 switch rule | switch only if mean RMSE of the alternative < mean of the default − 1 SD (ddof = 1) of the default's 3 seeds; G12 is evaluated after G11 with G11's choice; the shared default runs are reused | Section 5 says "1 SD of the seed spread"; the default's spread is the reference |
| P5-d | G15 projection | `bench_compute` recommended plan + 6th ablation variant + D20 weight choice (3 runs) + G11/G12 (12 runs) at the median run cost | Section 12 asks to include D20; the bench predates the 6th variant and the G11/G12 runs |
| P5-e | grid device | finals always with determinism on the G13 device; the grid may run on a GPU without determinism if G14 fails there but the GPU is faster (D15) | as D15 |
| P5-f | early stopping | validation loss on 2023 (masked MSE), patience 10, max 150 epochs, best weights restored | Section 9.3 |
| P5-g | LSTM grid batch | the stage-1 winner's batch size | the plan fixes units × dropout × lr only |
| P5-h | baselines | tide-only = OLS of observed training wet-season Pandacan on the predicted tide + 2 annual harmonics; ridge and KNN one model per lead (valid targets), single alpha / (k, weights) chosen by mean 2023 RMSE over leads; KNN k extended (×2) while the best k is the largest | Table III-4 |
| P5-i | classifiers | per lead × level × input set: (a) last time step of the sequence inputs + tabular, (b) flattened 24 h window + tabular; hyperparameters and threshold by the best 2023 F1 (LR: C × class weight; KNN: k × weighting, threshold only, C4) | Section 9.3 |
| P5-j | trigger offsets | one offset per model per level shared by all leads (D11); for neural models the offset maximises the mean F1 over the final seeds; ties → offset closest to 0 | D11 |
| P5-k | G16 checklist | automated: target alignment, target scaler = Pandacan channel, invalid targets = −1, rain availability lag = 1 h, training loss falls, identical scored hours across models | Section 9.7 items 1–6 |
| P5-l | frozen files | Keras checkpoints and fitted baselines in `artifacts/frozen/` (git-ignored, produced on the laptop) with SHA-256 in `configs/frozen_manifest.json` (committed); KNN models are refitted in NB06 from the frozen hyperparameters (deterministic; avoids storing the training set 30 times) | G17, G18 |
| P5-m | Keras 3 | the D9 add-back uses a registered `LevelAtT` layer instead of a Python `Lambda`, and the losses are registered, so checkpoints reload safely in NB06 | serialisation only; same computation |

## Notebook 06: implementation choices (built and dry-run, 6 Nov 2026; the real test is opened once on the laptop)

Fixed in `configs/eval.yaml` before the test season is opened.

| ID | Topic | Choice | Why |
|---|---|---|---|
| P6-a | G18 | blocking: tag `settings-frozen` is an ancestor of HEAD; every file in `frozen_manifest.json` (incl. `configs/eval.yaml`, so the H3/H4 rules are pre-registered) has its SHA-256; `gates/TEST_UNSEALED.json` absent. Opening writes the marker (time, commit). A second opening needs `--rerun-reason`, which is appended to the marker and must go in the transparency log | Section 0.3: the test is read once |
| P6-b | test windows | built only in NB06 with the Phase 3 `build_windows`, the frozen tide coefficients, training medians and scalers; the layout must equal `feature_names.json`. The rehearsal rebuilds the 2023 windows through the same path and asserts they equal Phase 3's saved arrays bit for bit | the 2024 inputs are made exactly as the training inputs were |
| P6-c | inference device | CPU (`CUDA_VISIBLE_DEVICES=-1`) | scoring is cheap and CPU results are reproducible on any machine |
| P6-d | neural seeds | level metrics and alerts: mean ± SD over the 5 seeds; DM tests and H4: the seed with the median 2023 RMSE; events: every seed reported | `eval.yaml` (neural_seed_for_tests) |
| P6-e | rainy hours | ≥ 5 mm Science Garden rain in the previous 24 h from reports available at t (`sg_rain_24h_in`); dry = the rest | uses only information available at issue time |
| P6-f | Diebold–Mariano | squared error, Newey–West lags max(h − 1, ⌊4(n/100)^(2/9)⌋), HLN correction, t(n − 1); each pair on the hours both models score; Holm over 72 tests (6 baselines × 6 leads × all/rainy); "hybrid better" = Holm p < 0.05 and negative mean loss difference | Section 10.1; H3 rule in `eval.yaml` |
| P6-g | equal recall (D19) | hybrid score = forecast level (the offset does not change the ranking); classifiers = probability; same hours; ties admitted together; first threshold with Recall ≥ target, achieved Recall reported; CIs: 24 h moving-block bootstrap, 2,000 reps, percentile (rehearsal: 200) | D19 |
| P6-h | classifier input set for H4 | per family, the set with the higher mean 2023 F1 at high water (read from `thresholds.yaml`) | fixed before the test |
| P6-i | strongest simple baseline | lowest mean 2023 RMSE among persistence, tide-only, tide + persisted residual | fixed before the test |
| P6-j | events | observed levels = targets of the valid (hour, lead) pairs; runs ≥ level with gaps < 6 h merged; detected if an alert targets first exceedance ± 2 h; warning time from the earliest such alert's issue hour; event-bootstrap 95% CI | `eval.yaml` |

## Notebook 07: implementation choices (built and dry-run, 6 Nov 2026)

Fixed in `configs/eval.yaml` (`nb07`) before the test season is opened.

| ID | Topic | Choice | Why |
|---|---|---|---|
| P7-a | walk-forward fits | per fold, from the fold's training years only: tide (D7 + G9 rule), alert levels (G8 rule on Aviles stress), Granger selection (G10 rule, same m and Bonferroni), training medians, scalers. The network trains on all but the last training year; that last year is the inner validation (early stopping, ridge alpha, trigger offsets). QC rules, G5 eligibility, G11/G12 choices and the final hyperparameters stay frozen | Section 10.2 ("from that fold's training years"); the evaluation year is never touched. Same layout as the main run (train → validation → test) |
| P7-b | walk-forward models | hybrid (seeds per G15) + persistence, tide-only, tide + persisted residual, ridge | the strongest simple baselines and the linear model; the standalone LSTM and KNN would double the run time without changing what the walk-forward tests (stability of the hybrid's advantage) |
| P7-c | walk-forward statistics | RMSE / skill per fold and lead, alerts with the fold's own levels and offsets, DM (hybrid median-inner-validation seed vs each baseline), Holm over 3 folds × 4 baselines × 6 leads; descriptive. The 2023 fold is flagged `independent = False` | 2023 chose the hyperparameters |
| P7-d | ablation training | final hybrid hyperparameters, frozen windows/scalers, 2018–22 training, early stopping on 2023, seeds 0…n−1 per G15, determinism on; each variant gets its own trigger offsets on 2023 (D11 rule) | Table III-5; one component changed at a time |
| P7-e | ablation inputs | no Granger = all eligible candidates (`no_granger_ablation`); correlation selection = the same number of candidates ranked by lagged cross-correlation; no rainfall = the `sg_rain` group removed (sequence + tabular); no tabular = sequence path only; second LSTM = `LSTM(gru_units)` in place of the GRU | `selected_features.yaml`, `granger.yaml` |
| P7-f | D20 weight | w ∈ {2, 5, 10}, seed 0, by high-water F1 on 2023 after the variant's own offset tuning; ties → smaller w; the seed-0 run is reused for the final seeds | D20 |
| P7-g | ablation comparison | against the main hybrid on the same seeds (NB06 predictions): RMSE difference per lead, "beyond seed noise" if \|difference\| > SD (ddof 1) of the main hybrid; DM (each variant's median-validation seed vs the main hybrid's NB06 test seed), Holm over 36; descriptive | the G11/G12 seed-noise rule, applied the same way |
| P7-h | case study | flood-risk events of the test season (NB06 definition), per model × lead: detected, warning time, first alert issue hour, forecast and error at the observed peak; one figure per event with the hybrid's leads 1, 3, 6; the Carina event = the one covering 24 Jul 2024 | Section 10.2 |
| P7-i | Aviles 2024 | parsed only here, after G18, from the `post_eval_only` blocks; 1 Jun – 15 Jul 2024; per level: hours at/above, stress share at/above and below, gate-open share, dry share; beside the training-season values. Descriptive; levels unchanged | Section 10.2 deferred check |

## Repository decision (9 Nov 2026)

| ID | Choice | Why | Plan text affected |
|---|---|---|---|
| R1 | **The raw data are committed in `data/raw/`** (31 MB, 465 files, byte-for-byte; `.gitattributes` marks them binary so no line-ending conversion can change a SHA-256). Generated data (`data/interim`, `data/processed`) stay git-ignored and are rebuilt locally by `pasig.local.prepare`, which every notebook and `run_phase.py` call first: it checks every raw file against `raw_manifest.csv`, rebuilds missing Phase 1/3 inputs once per clone, and restores committed files that come out identical apart from run stamps | requested by the group: the data are small, and a clone should run without waiting for Google Drive; committing the exact bytes also makes G1 verifiable by anyone | Plan Section 6 says `data/` is git-ignored except the README and manifest, and Section 2.2 copies the data from Drive with `sync_drive.py`; both superseded for the raw files only. `sync_drive.py` remains for the case where the Drive files change |

## Implementation choices made in Phase 1 (recorded for the thesis text)

| Ref | Choice | Why |
|---|---|---|
| P1-a | G2 HIGHEST/LOWEST check is formula-aware: when a file's mismatch rate is > 1%, each mismatching day's formula is read; days whose `=MAX(Bk:Yk)` points to another row and whose cached value matches that row are counted as source-sheet errors, not parser errors | 4 Fort Santiago files (Aug/Nov 2020, Aug/Nov 2021) have formulas for days 7–9 that reference rotated rows; all 8 mismatching days are explained and the readings are continuous across day boundaries. A real column shift would leave unexplained mismatches and still fails G2 |
| P1-b | Flat-run rules per station (configs/qc.yaml): Pandacan only the 11.89 m rule (runs ≥ 5 h); Fort Santiago ≥ 6 h outside the 10.60–10.70 m floor; Rosario ≥ 12 h; Sto. Niño ≥ 12 h outside 11.8–12.5 m plus the Montalban override inside; Montalban none | non-tidal gauges (Sto. Niño pool, Montalban) legitimately hold for days; tidal gauges do not. Lengths read from 2018–2022 only |
| P1-c | Valid ranges per station: Pasig/Fort Santiago 9–15 m, Rosario 9–16 m, Sto. Niño 10.5–22 m, Montalban 19–30 m | Montalban uses a ~21 m datum; Sto. Niño peaks reach ~21 m in 2018 |
| P1-d | Non-numeric typo-like tokens (`11..39`, `11.09+`, `10.6+8`) are NaN like every other non-numeric cell (raw token kept in `waterlevel_long.parquet`) | Section 3.2 rule; 54 such cells across all stations (8 at Pandacan), listed in `reports/logs/efcos_missing_tokens.csv` |
| P1-e | Pumping readings outside 9–14 m after unit conversion are set to NaN and logged (227 Aviles, 71 Quiapo) | typos such as 105 (cm) or 1.3 (m) |
| P1-f | Rolling rain totals (24 h, 72 h) = sum of the last 4 / 12 reports available at t; NaN if any is missing | only reports available at t (D5) |
