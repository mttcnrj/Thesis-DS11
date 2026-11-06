# Decision register (D1–D20)

Source: IMPLEMENTATION_PLAN.md Section 4 (decided 6 Nov 2026). Each entry: date, choice, reason, gate file,
Chapter rows it changes (Plan Section 13). "Outcome" records what the gate decided when it has run.

| ID | Date | Choice | Reason (short) | Gate file | Chapter rows | Outcome |
|---|---|---|---|---|---|---|
| D1 | 2026-11-06 | One copy of the identical QUIAPO 2023/2024 files; dated by sheet headers | identical SHA-256; file names unreliable | `gates/G01_raw_files.json` | 5, 8, 9, 11, 25 | G1 passed. Kept `QUIAPO 2023.xlsx`; its sheets are headed **2021, 2023 and 2024** (see transparency log F1) |
| D2 | 2026-11-06 | Keep the 80% every-season rule; failing stations tested descriptively only; no missing-indicator variant | Fort Santiago 0–2% in 2018–19; tiny ablation effect | `gates/G05_eligibility.json` | 27 | G5: **no eligible upstream input** (Sto. Niño min 78.1% in 2024; Fort Santiago 0% in 2018) |
| D3 | 2026-11-06 | Assumed status-code legend, validated on 2018–2022 | no MMDA legend | G4 (NB02) | 3, 10 | parser stores `pumps_running`, `gate_open`, `sky`, `raw_status`; G4 pending (Phase 2) |
| D4 | 2026-11-06 | Trace rainfall = 0.05 mm | half the 0.1 mm limit | parser + `tests/test_time_alignment.py` | 7 | implemented (6-hourly `T`, daily `-1`) |
| D5 | 2026-11-06 | 6-hourly total usable 1 h after its period ends | no look-ahead | `gates/G03_rain_timing.json` | 6, 12, 23 | G3: stamps mark the **period end** (99.7% of 1,172 rain days match vs 53.7%) |
| D6 | 2026-11-06 | Port Area excluded | no 2024 data, gaps | `gates/G01_raw_files.json` | — | Port Area file excluded; Port Area columns dropped from the 6-hourly file |
| D7 | 2026-11-06 | Tide fit on all 12 months of 2018–2022, hours > floor + 0.05 m | more tidal cycles | G9 (NB03) | 14, 21 | `waterlevel_long.parquet` keeps all months; pending Phase 3 |
| D8 | 2026-11-06 | Predicted tide t+1…t+6 and change (12 cols); constituent pairs tried once | | G11 (NB05) | 1, 2, 13, 16 | pending |
| D9 | 2026-11-06 | Head predicts change; add-back of L_t | | G12 (NB05) | 15, 18 | pending |
| D10 | 2026-11-06 | Selection by mean validation RMSE over six leads | | grid code | 20 | pending |
| D11 | 2026-11-06 | One trigger offset per model per alert level, shared by leads | | offset code | 22 | pending |
| D12 | 2026-11-06 | Two-stage grid, 60 configs | | grid code | 20 | pending |
| D13 | 2026-11-06 | 5 seeds if projection ≤ 10 h else 3 for walk-forward/ablations | | G15 | 4, 26 | pending |
| D14 | 2026-11-06 | Chapter III ranges exactly | | G19 | — | pending |
| D15 | 2026-11-06 | Determinism on for final runs; CPU fallback | | G14 | — | pending |
| D16 | 2026-11-06 | Local VS Code on the Windows laptop with the stronger NVIDIA GPU (else more cores) | | G0 (`scripts/record_g0.py`) | 24 | **G0 not yet recorded** (needs both laptops' PowerShell output). Phase 1 here ran in a cloud container; it is deterministic and must be re-run on the chosen laptop with `make phase1` |
| D17 | 2026-11-06 | GPU only if visible, G14 passes, projection faster | | G13 | 24 | pending |
| D18 | 2026-11-06 | Quiapo 2021 used conditionally (G20) in the secondary check; with/without reported | only route to 2021 | G20 (NB02) | 11 | G1 marks both 2021 copies `pending_G20`; the two copies are value-identical (one series) |
| D19 | 2026-11-06 | Recall 0.90 for the H4 verdict; 0.80/0.95 sensitivity | | eval code | 28 | pending |
| D20 | 2026-11-06 | Weighted-loss ablation (w ∈ {2,5,10}) | | ablation code | 29, 30, 31 | pending |

## Implementation choices made in Phase 1 (recorded for the thesis text)

| Ref | Choice | Why |
|---|---|---|
| P1-a | G2 HIGHEST/LOWEST check is formula-aware: when a file's mismatch rate is > 1%, each mismatching day's formula is read; days whose `=MAX(Bk:Yk)` points to another row and whose cached value matches that row are counted as source-sheet errors, not parser errors | 4 Fort Santiago files (Aug/Nov 2020, Aug/Nov 2021) have formulas for days 7–9 that reference rotated rows; all 8 mismatching days are explained and the readings are continuous across day boundaries. A real column shift would leave unexplained mismatches and still fails G2 |
| P1-b | Flat-run rules per station (configs/qc.yaml): Pandacan only the 11.89 m rule (runs ≥ 5 h); Fort Santiago ≥ 6 h outside the 10.60–10.70 m floor; Rosario ≥ 12 h; Sto. Niño ≥ 12 h outside 11.8–12.5 m plus the Montalban override inside; Montalban none | non-tidal gauges (Sto. Niño pool, Montalban) legitimately hold for days; tidal gauges do not. Lengths read from 2018–2022 only |
| P1-c | Valid ranges per station: Pasig/Fort Santiago 9–15 m, Rosario 9–16 m, Sto. Niño 10.5–22 m, Montalban 19–30 m | Montalban uses a ~21 m datum; Sto. Niño peaks reach ~21 m in 2018 |
| P1-d | Non-numeric typo-like tokens (`11..39`, `11.09+`, `10.6+8`) are NaN like every other non-numeric cell (raw token kept in `waterlevel_long.parquet`) | Section 3.2 rule; 54 such cells across all stations (8 at Pandacan), listed in `reports/logs/efcos_missing_tokens.csv` |
| P1-e | Pumping readings outside 9–14 m after unit conversion are set to NaN and logged (227 Aviles, 71 Quiapo) | typos such as 105 (cm) or 1.3 (m) |
| P1-f | Rolling rain totals (24 h, 72 h) = sum of the last 4 / 12 reports available at t; NaN if any is missing | only reports available at t (D5) |
