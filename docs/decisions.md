# Decision register (D1–D20)

Source: IMPLEMENTATION_PLAN.md Section 4 (decided 6 Nov 2026). Each entry: date, choice, reason, gate file,
Chapter rows it changes (Plan Section 13). "Outcome" records what the gate decided when it has run.

| ID | Date | Choice | Reason (short) | Gate file | Chapter rows | Outcome |
|---|---|---|---|---|---|---|
| D1 | 2026-11-06 | One copy of the identical QUIAPO 2023/2024 files; dated by sheet headers | identical SHA-256; file names unreliable | `gates/G01_raw_files.json` | 5, 8, 9, 11, 25 | G1 passed. Kept `QUIAPO 2023.xlsx`; its sheets are headed **2021, 2023 and 2024** (F1). 2023 sheet: descriptive only; 2024 sheet: sealed until after NB06 (decision P2-b) |
| D2 | 2026-11-06 | Keep the 80% every-season rule; failing stations tested descriptively only; no missing-indicator variant | Fort Santiago 0–2% in 2018–19; tiny ablation effect | `gates/G05_eligibility.json`, `gates/G10_granger_outcome.json` | 27 | G5: **no eligible upstream input** (Sto. Niño min 78.1% in 2024; Fort Santiago 0% in 2018). Descriptive Granger in NB04: both significant, both worsen the ridge validation RMSE |
| D3 | 2026-11-06 | Assumed status-code legend, validated on 2018–2022 | no MMDA legend | `gates/G04_status_legend.json` | 3, 10 | G4: gate and pump fields **used** after the post-hoc amendment P2-a (literal checks failed for physical reasons, reported); **sky field dropped** (F/C dry check fails 21.1% > 20%); `S` reads as shower at Aviles |
| D4 | 2026-11-06 | Trace rainfall = 0.05 mm | half the 0.1 mm limit | parser + `tests/test_time_alignment.py` | 7 | implemented (6-hourly `T`, daily `-1`) |
| D5 | 2026-11-06 | 6-hourly total usable 1 h after its period ends | no look-ahead | `gates/G03_rain_timing.json` | 6, 12, 23 | G3: stamps mark the **period end** (99.7% of 1,172 rain days match vs 53.7%) |
| D6 | 2026-11-06 | Port Area excluded | no 2024 data, gaps | `gates/G01_raw_files.json` | — | Port Area file excluded; Port Area columns dropped from the 6-hourly file |
| D7 | 2026-11-06 | Tide fit on all 12 months of 2018–2022, hours > floor + 0.05 m | more tidal cycles | `gates/G09_tide_fit.json` | 14, 21 | fitted for Pandacan (floor 10.65 m), Sto. Niño (11.32 m), Fort Santiago (10.59 m); `artifacts/tide_<station>.json` |
| D8 | 2026-11-06 | Predicted tide t+1…t+6 and change (12 cols); constituent pairs tried once | | G11 (NB05) | 1, 2, 13, 16 | 12 tide columns in `X_tab`; constituent pairs prepared as `X_tab_cons` (**120** columns = 10 frozen constituents × sin/cos × 6 leads, not 84, because G9 added 3 constituents); G11 pending |
| D9 | 2026-11-06 | Head predicts change; add-back of L_t | | G12 (NB05) | 15, 18 | pending |
| D10 | 2026-11-06 | Selection by mean validation RMSE over six leads | | grid code | 20 | pending |
| D11 | 2026-11-06 | One trigger offset per model per alert level, shared by leads | | offset code | 22 | pending |
| D12 | 2026-11-06 | Two-stage grid, 60 configs | | grid code | 20 | pending |
| D13 | 2026-11-06 | 5 seeds if projection ≤ 10 h else 3 for walk-forward/ablations | | G15 | 4, 26 | pending |
| D14 | 2026-11-06 | Chapter III ranges exactly | | G19 | — | pending |
| D15 | 2026-11-06 | Determinism on for final runs; CPU fallback | | G14 | — | pending |
| D16 | 2026-11-06 | Local VS Code on the Windows laptop with the stronger NVIDIA GPU (else more cores) | | `gates/G00_machine.json` | 24 | **G0 recorded**: the group's chosen laptop (AMD Ryzen 7 7735HS, 8 cores/16 threads, 16 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU) → Python in WSL2 Ubuntu. Only this laptop's output was provided; VRAM to confirm with `nvidia-smi`. Phases 1–3 here ran in a cloud container (deterministic); re-run them on the laptop |
| D17 | 2026-11-06 | GPU only if visible, G14 passes, projection faster | | G13 | 24 | pending |
| D18 | 2026-11-06 | Quiapo 2021 used conditionally (G20) in the secondary check; with/without reported | only route to 2021 | `gates/G20_quiapo_2021_identity.json` | 11 | G20 **passed** (a–d all hold; copies identical → one series): 2021 included; Quiapo check reported with and without it |
| D19 | 2026-11-06 | Recall 0.90 for the H4 verdict; 0.80/0.95 sensitivity | | eval code | 28 | pending |
| D20 | 2026-11-06 | Weighted-loss ablation (w ∈ {2,5,10}) | | ablation code | 29, 30, 31 | pending |

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

## Implementation choices made in Phase 1 (recorded for the thesis text)

| Ref | Choice | Why |
|---|---|---|
| P1-a | G2 HIGHEST/LOWEST check is formula-aware: when a file's mismatch rate is > 1%, each mismatching day's formula is read; days whose `=MAX(Bk:Yk)` points to another row and whose cached value matches that row are counted as source-sheet errors, not parser errors | 4 Fort Santiago files (Aug/Nov 2020, Aug/Nov 2021) have formulas for days 7–9 that reference rotated rows; all 8 mismatching days are explained and the readings are continuous across day boundaries. A real column shift would leave unexplained mismatches and still fails G2 |
| P1-b | Flat-run rules per station (configs/qc.yaml): Pandacan only the 11.89 m rule (runs ≥ 5 h); Fort Santiago ≥ 6 h outside the 10.60–10.70 m floor; Rosario ≥ 12 h; Sto. Niño ≥ 12 h outside 11.8–12.5 m plus the Montalban override inside; Montalban none | non-tidal gauges (Sto. Niño pool, Montalban) legitimately hold for days; tidal gauges do not. Lengths read from 2018–2022 only |
| P1-c | Valid ranges per station: Pasig/Fort Santiago 9–15 m, Rosario 9–16 m, Sto. Niño 10.5–22 m, Montalban 19–30 m | Montalban uses a ~21 m datum; Sto. Niño peaks reach ~21 m in 2018 |
| P1-d | Non-numeric typo-like tokens (`11..39`, `11.09+`, `10.6+8`) are NaN like every other non-numeric cell (raw token kept in `waterlevel_long.parquet`) | Section 3.2 rule; 54 such cells across all stations (8 at Pandacan), listed in `reports/logs/efcos_missing_tokens.csv` |
| P1-e | Pumping readings outside 9–14 m after unit conversion are set to NaN and logged (227 Aviles, 71 Quiapo) | typos such as 105 (cm) or 1.3 (m) |
| P1-f | Rolling rain totals (24 h, 72 h) = sum of the last 4 / 12 reports available at t; NaN if any is missing | only reports available at t (D5) |
