# Transparency log

Every known file problem (Plan Section 3.3) and every data-vs-plan finding from Phase 1. Numbers are
from the run recorded in `gates/` (git SHA in each gate file).

## Known file problems (Section 3.3) — handling as implemented

| File | Problem | Handling | Evidence |
|---|---|---|---|
| `QUIAPO 2020.xlsx`, `QUIAPO 2021.xlsx` | Taguig Pumping Station data (sheet "2019", LAND/LAKE columns) | excluded (content check: "TAGUIG" in the sheet header) | `data/manifest/raw_manifest.csv` |
| `QUIAPO 2023.xlsx` = `QUIAPO 2024.xlsx` | identical SHA-256 | one copy kept (`QUIAPO 2023.xlsx`), dated by sheet headers | G1 |
| `QUIAPO 2022.xlsx` | sheet `Sheet4` holds Quiapo 2021, `Sheet1` holds 2022 | 2021 block `pending_G20`; 2022 `use` | `data/manifest/pumping_blocks.csv` |
| `AVILES 2024.xlsx` | sheets "AUGUST 16-31", "OCT 1-15", "NOV 16-30" carry the headers of Jul 16-31, Sep 1-15 and Oct 16-31 | 2024 Aviles blocks are `post_eval_only` (not parsed in Phase 1); duplicate datings flagged in `pumping_blocks.csv` (`dup_dating`) for the NB07 check | G1 |
| `AVILES 2023.xlsx`, `AVILES 2024.xlsx` "APRIL 16-30" | no header text | dated from sheet name + file year (`date_source = sheet_name+file_year`) | |
| `AVILES 2018.xlsx` Sheet20–Sheet50 | empty templates | no block header → not parsed | |
| Rosario JS 2024 | `1. January 2024.xlsx` holds daily sheets in another layout plus a **December 2023** multi-station FORM 2-1 workbook (YEAR cell 2023) | set aside (YEAR cell ≠ folder year); Rosario JS 2024 coverage = 0% | G2, `reports/logs/efcos_files.csv` |
| WOCR PDFs | `WOCR PS - 2022.pdf` = `WOCR PS - 2022 (1).pdf`; `WOCR PS -2020.pdf` = `WOCR PS -2020 (1).pdf` (Drive names use "(1)", the plan says " 1") | one copy of each kept after the hash check | G1 |
| `Pandacan 2024/4. April.xlsx` | wrong header text | single FORM 2-1 sheet; month from the file name | |
| `Fort Santiago 2020/11. November.xlsx.xlsx` | double extension | parsed normally | |

## Data-vs-plan findings (Plan 0.1: data are ground truth — reported, not hidden)

**F1. Real-looking Quiapo 2023 and 2024 sheets exist.** *Resolved 6 Nov 2026 by option (a) (decision P2-b):
outside the H1 verdict; 2023 reported descriptively; 2024 sealed until after NB06.* The single QUIAPO 2023/2024 copy has three sheets, headed
2021, 2023 and 2024. The 2023 and 2024 sheets are not copies of 2021 or 2022 (outside-elevation correlation
r = 0.007–0.031 and ≤ 5% equal values against both). The plan (C5, Section 8.4) expected no real Quiapo 2023
data. Per Section 8.4 the Quiapo check covers training years only, so these blocks are `exclude` in G1 (reason
recorded). **Team decision needed before NB02** if they should be used to check the alert levels on 2023.

**F2. Pandacan coverage and "usable hours" in the reference were counted after the ≤ 3 h gap fill.** Rebuilt
before-fill coverage (Section 7.8 definition, used for G5): 94.7, 88.3, 84.8, **77.0**, 87.7, 94.2, 91.3% (2018–2024).
After fill: 99.3, 96.9, 92.6, 90.4, 96.0, 98.7, 97.7% — this reproduces the reference "91–99%". Usable Pandacan
hours observed-or-filled 20,876 / 4,336 / 4,292 vs reference 20,867 / 4,336 / 4,283; strictly observed and
unflagged (the only hours that can be targets) 18,995 / 4,138 / 4,009. The missing hours are genuine `*`/`**`
tokens (2,990 `**`, 1,813 `*`) and 2,466 zero readings. One whole day (31 Oct 2018) has no row in its sheet.
Pandacan is the target, not a candidate input, so G5 does not apply to it; the 2021 season is thinner than
the reference suggested.

**F3. Fort Santiago 2023 coverage is 60.1%, not 87%.** The raw 2023 wet-season sheets have 1,676 `**` cells
(61.4% raw coverage), so the gap is in the source files. Fort Santiago fails G5 either way (0% in 2018).

**F4. Pandacan QC removed 119 training wet-season hours (reference ≈ 103):** 98 flat-run hours at exactly
11.89 m (identical to the reference), 18 one-hour spikes (reference 5), 3 out-of-range typos (e.g. 1.07 m).
All 13 spikes above 11.69 m were hand-checked (`reports/tables/qc_handcheck_pandacan_train.csv`): each is a
single-hour jump (smaller side 0.64–1.01 m) that neither the neighbouring hours, Fort Santiago nor the Aviles river-side
gauge shows; real peaks (multi-hour) survive. P95 of training Pandacan stays **11.69 m**.

**F5. 2023–2024 QC (rules unchanged, nothing edited by hand).** Pandacan values > 11.69 m removed: 2023-07-17
02:00 (11.80), 2023-09-17 19:00 (11.73), 2024-06-10 04:00 (11.88), 2024-07-14 23:00 (11.94), 2024-09-27 00:00
(12.45), 2024-11-16 10:00 (12.25), all one-hour spikes. **Carina's peak survives: 12.98 m on 24 Jul 2024 15:00.**

**F6. G3: 6-hourly stamps mark the period end** (option b matches 99.7% of 1,172 rain days; option a 53.7%).

**F7. HIGHEST/LOWEST template errors** in 4 Fort Santiago sheets (see decisions P1-a); 8 days, all explained.

**F8. Sto. Niño coverage depends strongly on the 11.8–12.5 m flat-run exemption:** with it 85.6, 93.1, 91.3, 94.5,
95.9, 97.4, 78.1%; without it 72.9, 47.7, 38.2, 51.0, 94.9, 56.4, 63.4%. It fails the every-season rule either way
(2024 with the exemption; most seasons without), matching the expected outcome.

**F9. A blank-cell bug was caught by tests before any results were used:** blank 6-hourly rain cells were
being read as trace; fixed (blanks are missing, `T` is trace).

## Phase 2 findings (Notebook 02)

**F10. The literal G4 checks for gate `O` and pump `nP` failed for physical, not legend, reasons** and were
amended with the group's approval (decision P2-a). Literal: gate `O` 100% fail at both stations (river = sump in
99% of `O` hours); pumps 42.3% (Aviles) / 35.4% (Quiapo) fail. Amended: 1.1% / 1.0% and 0.3% / 0.3%.
`gate_C` 1.0% / 0.2% fail; `nP` with no O/C → gate closed 0.2% / 0.1% fail; pump digits ≤ 4 at both stations.

**F11. Sky codes dropped (G4):** fair/cloudy hours had ≥ 1 mm Science Garden rain in the matching 6 h in 21.1%
of hours (limit 20%); rain codes failed 4.2%. Not used in any model. `S` behaves like rain at Aviles (95.8%
rainy vs 95.8% for rain codes): read as "shower".

**F12. G20 passed.** Quiapo 2021 (one series; the two copies are identical): metres (99.9% in 9–14 m), O/C/nP
codes (99.97%), outside row present (99.8%); r(outside, Pandacan) = 0.915 vs 0.885 / 0.894 / 0.917 in 2018 /
2019 / 2022; not a copy of Aviles (r = 0.955, 1.2% equal values) or Taguig 2021 (r ≤ 0.19).

**F13. First assumption:** Pandacan vs Aviles river side, training hours: Pearson 0.934, Spearman 0.940,
hourly changes 0.706; Aviles reads 0.218 m higher on average (SD 0.119 m), so the two gauges may not share a
datum. At Pandacan ≥ 11.69 m: Pearson 0.598, changes 0.672.

**F14. H1 partly supported (G7).** Training: 17,132 hours, 840 days, 448 stress hours. OR per 0.1 m after
rain (M1) 1.207 (95% CI 1.143–1.275, day-clustered); LR M0 vs M1 = 162.6 (p < 1e-36). The level × rain
interaction is significant (M2 interaction OR 1.53, 1.07–2.21; LR 15.4, p = 8.7e-5): the level matters more
when it has rained. 2023: AUC M0 0.824, M1 0.829, gain 0.004 (−0.022 to +0.028; 2,000 day-block reps, 182
days, 227 stress hours). Reference: OR 1.19 (1.13–1.26), gain 0.006 (−0.019 to +0.030) → same verdict.

**F15. Alert levels (G8) reproduce the preliminary ones: 11.69 m and 12.00 m.** Stress share at/above:
11.90 m 34.9%, 11.95 m 42.3%, 12.00 m 55.0% (n = 80). Sensitivity: with ≥ 1 pump the rule gives 11.85 m
(cut-off 50%); with ≥ 3 pumps 12.20 m; with ≥ 2 pumps and cut-offs 40/60% 11.95/12.05 m. 2023 check (recorded,
not used): 4.2% of hours ≥ 11.69 m (29% of them dry), 0.31% ≥ 12.0 m (13 h, none dry); stress share at/above
23.8% and 46.2%.

**F16. Quiapo secondary check:** OR per 0.1 m (≥ 2 Quiapo pumps, after rain) 1.067 (1.046–1.088) without 2021
and 1.066 (1.046–1.086) with 2021; same direction as Aviles, weaker. Quiapo 2023 (descriptive): stress share
25.6% at/above 11.69 m and 53.8% at/above 12.0 m (13 h).

**F17. A parser regression was caught before Phase 2:** an empty-sheet case (`zip(strict=True)`) introduced
by a lint fix after the Phase 1 run; fixed with a test, and Phase 1 re-run (identical results).

## Thesis text to add beyond Plan Section 13
* Ch III §7.8 / Figure III-3: coverage counted after QC and **before** gap fill (Pandacan 77–95%).
* Table III-6: replace with `reports/tables/table_III-6_recomputed.csv`.
* Ch III Ph2: legend checks as amended (P2-a) with both results; sky codes dropped; Quiapo 2023 descriptive.
* Table III-1 / Ph1: Quiapo workbook also holds 2023 and 2024 sheets (F1).

## Notes on the environment of this run
Phase 1 outputs in this commit were produced in a Claude Code cloud container (Python 3.13, pandas 3.0)
from the Drive files listed in `data/manifest/raw_manifest.csv` (SHA-256 per file). The plan requires all
final results to come from the chosen laptop (D16); Phase 1 is deterministic, so re-running `make phase1`
there must reproduce the same manifest hashes, gates and tables.
