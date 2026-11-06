# Data dictionary (Phase 1 outputs)

All times are Philippine time (PHT, UTC+8), hourly, naive timestamps.

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
