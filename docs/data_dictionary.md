# Data dictionary

Phase 1 outputs first, then the arrays, artifacts and prediction tables of later phases.
All times are Philippine time (PHT, UTC+8), hourly, naive timestamps.

# Raw data (committed, decision R1)

`data/raw/` holds the source files exactly as in the `Thesis II` Drive folder: `MMDA Dataset/` (EFCOS FORM 2-1
workbooks per station and month; Aviles and Quiapo pumping workbooks; WOCR PDFs) and `PAGASA Dataset/` (Science
Garden 6-hourly and daily CSVs; the Port Area columns are dropped, D6), plus the excluded Kaggle file. Their
SHA-256, size, station, year and G1 status are in `data/manifest/raw_manifest.csv`. Parsing rules: Plan Section 3.2.

# Phase 1

## data/interim/waterlevel_long.parquet (all months, all EFCOS stations)
| column | meaning |
|---|---|
| time_pht | hour of the reading (day *d* row, column *i* → d 09:00 + i h) |
| station | pandacan, sto_nino, fort_santiago, montalban, rosario_js, rosario_ls |
| value_m | reading in m; NaN for missing tokens and exact 0.0 |
| raw_token | the original non-numeric cell (e.g. `**`, `#REF!`, `11..39`) or `0`; null when numeric |
| day | sheet day label of the row |
| source_file, sheet | provenance |
| flag_impossible_date | always False here (impossible rows are in `reports/logs/impossible_dates.csv`) |
| flag_duplicate_time | True if two files produced the same station-hour (none) |

## data/interim/pumping_long.parquet (Aviles, Quiapo; Phase 2 only, never model inputs)
time_pht, station, inside_m (sump), outside_m (river side), pumps_running (0 if no pump token), gate_open
(1 open / 0 closed / NaN), sky, raw_status, status_parsed, inside/outside_token, *_range_flag (outside 9–14 m →
NaN), flag_impossible_date, flag_duplicate_time (the two identical Quiapo 2021 copies), block_status
(`use` / `pending_G20`), block_year, source_file, sheet, block_header. 2024 Aviles and Quiapo 2023/2024 blocks are
not included (see `data/manifest/pumping_blocks.csv`).

## data/interim/pagasa_6h.parquet, pagasa_daily.parquet (Science Garden only)
6-hourly: stamp_utc, stamp_pht, rain_6h_mm (T → 0.05), rain_trace, tmax/tmin_6h_c, period_start/end_pht,
available_from_pht (= period end + 1 h, D5). Daily: date, rain_day_mm (−1 → 0.05; −999 → NaN), tmax, tmin, tmean,
rh, wind_speed, wind_direction, atm_pressure, period_start/end_pht (08:00 → 08:00), available_from_pht (09:00 the
next day).

## data/processed/hourly_master.parquet (June–November 2018–2024)
| column | meaning |
|---|---|
| season, split | year; train (2018–22) / val (2023) / test (2024) |
| `<station>` | QC'd value, gaps ≤ 3 h forward-filled within the season |
| `<station>_filled` | True where the value was filled (never a target) |
| `<station>_qcflag` | '' clean, `range`, `spike`, `flat`, `flat_11.89`, `flat_upstream_rise`, `missing` |
| sg_rain_6h_latest | latest Science Garden 6 h total available at t (step-held) |
| sg_rain_24h, sg_rain_72h | sum of the last 4 / 12 available reports (NaN if any missing) |
| sg_hours_since_report | hours since the latest report became available |
| sg_report_ok | the report needed at t exists (non-missing, < 6 h old) |
| sg_daily_rh, sg_daily_atm_pressure, sg_daily_tmean, sg_daily_rain_day_mm | previous-day values usable from 09:00 PHT |

## data/processed/eval_hours.parquet
Index time_pht (same as the master); season, split; lead_1 … lead_6: hour t is a scored case at lead h iff
the 24 Pandacan readings t−23…t exist inside the season (filled allowed), Pandacan at t+h is observed and not
filled (same season), the Science Garden report needed at t exists, and every eligible input's window is
complete (no eligible inputs after G5).

## data/manifest/
`raw_manifest.csv` (path, sha256, bytes, source, station, year, status, reason) and `pumping_blocks.csv` (one row
per half-month block: file, sheet, header text, year, month, days, date_source, status, reason, dup_dating).

---

# Later phases

## data/processed/windows_{train,val}.npz (Phase 3)
| key | shape | meaning |
|---|---|---|
| X_seq | (N, 24, 3) | t−23…t: Pandacan level, Δlevel, latest SG 6 h rain (raw units; scaled at load) |
| X_tab | (N, 46) | 28 calendar (hour/day-of-year sin/cos at t…t+6), 12 tide (predicted tide t+1…t+6 and change from t), SG rain 24 h / 72 h / hours since report, previous-day SG RH, pressure, mean temperature. Models use only the columns in `selected_features.yaml` |
| X_tab_cons | (N, 120) | D8 variant: 10 tide constituents × sin/cos × 6 leads (used only if G11 switches) |
| y | (N, 6) | Pandacan level at t+1…t+6 (NaN where not a valid target) |
| valid | (N, 6) | lead flags from `eval_hours.parquet` (all models are scored on these) |
| t_index | (N,) | issue hour t (int64 ns) |

Train 18,386 cases (16,937 valid at lead 1), validation 4,288 (4,091). Test windows are never saved.

## artifacts/ (Phases 3–5)
* `tide_<station>.json` — harmonic tide fit (D7, G9): constituents, periods, amplitudes, phases, standard errors,
  floor, fit years.
* `scalers.json` — min-max scalers fitted on training cases: `seq`, `tab`, `tab_cons`, and `y` (= the Pandacan
  sequence channel's scaler, D9), with column names.
* `feature_names.json` — column names of `X_seq`, `X_tab`, `X_tab_cons`; training medians for missing daily values;
  eligible stations.
* `frozen/` (laptop, git-ignored, SHA-256 in `configs/frozen_manifest.json`) — `hybrid_seed{k}.keras`,
  `lstm_seed{k}.keras`, `baselines.joblib` (ridge models per lead, logistic classifiers, tide-only coefficients).

## Prediction tables (reports/tables/, Notebooks 05–07)
Long format, one row per issue hour × lead × model × seed (seed −1 for deterministic baselines).

| file | columns | contents |
|---|---|---|
| `preds_val_neural.parquet`, `preds_val_baselines.parquet` | t, lead, model, seed, y_true, y_pred, valid | 2023 predictions (NB05) |
| `probs_val_classifiers.parquet` | t, lead, model (`logistic_a_current_hour`, …), level, prob, truth | 2023 classifier probabilities |
| `preds_test.parquet` | + rainy | 2024 predictions of every model, all seeds (NB06) |
| `probs_test.parquet` | t, lead, model, level, prob, alert, truth, rainy | 2024 classifier outputs with frozen thresholds |
| `preds_ablation_{val,test}.parquet` | as `preds_test` | the six ablation variants (`abl_<name>`) |
| `preds_walkforward.parquet` | + fold | walk-forward predictions on each fold's evaluation year |

`y_true` is the observed, unfilled level at t+lead; only rows with `valid = True` are scored. `rainy` = at least
5 mm of Science Garden rain in the previous 24 h from reports available at t.

## Chapter IV report (reports/chapter4/, `make report`)
Built by `pasig/eval/report.py` from the files above; never edit by hand.

| file | contents |
|---|---|
| `chapter4.md` | every Chapter IV table formatted for the manuscript (Tables IV-1 … IV-28), H3/H4 verdicts, figure list, hardware and measured training time (Chapter III) |
| `table_IV-NN_<name>.csv` | the numbers behind Table IV-NN at full precision (wide: model × lead, or the source table's columns) |
| `compute_runtime.csv` | run group, runs, hours, mean epochs — summed from `runs/*/timing.json` |
| `fig_walkforward_rmse.png`, `fig_ablation_rmse_diff.png` | walk-forward RMSE by fold; ablation RMSE difference vs the main hybrid with the seed-SD band |

## Gate files (gates/G##_<name>.json)
gate, name, passed, decision, rule, evidence, timestamp (PHT), git_sha, profile (`full` or `rehearsal`).
`gates/TEST_UNSEALED.json` (written by G18) lists every opening of the test season: time, git SHA, reason, profile.

## Run folders (runs/<yyyymmdd_hhmm>_<tag>_seed<k>/, git-ignored)
`config_snapshot.yaml`, `git_sha.txt`, `env.txt` (Python, TensorFlow, CUDA/cuDNN, GPU, CPU, cores, RAM,
determinism), `model.keras`, `history.json`, `timing.json`, `preds_val.parquet`. Resumable run logs:
`runs/grid_stage{1,2}.csv`, `runs/finals.csv`, `runs/walkforward.csv`, `runs/ablations.csv`.
Dry runs write the same structure under `runs/rehearsal/`.
