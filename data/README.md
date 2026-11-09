# data/

`raw/` and `manifest/` are committed; `interim/` and `processed/` are generated locally and git-ignored.
Column definitions: `docs/data_dictionary.md`.

| Folder | Contents | Written by |
|---|---|---|
| `raw/` | **committed** (31 MB, decision R1): `MMDA Dataset/`, `PAGASA Dataset/` and `Kaggle Dataset/` exactly as in the `Thesis II` Drive folder, stored byte-for-byte (`.gitattributes`: binary). The Kaggle file is kept only so G1's manifest stays complete; it is **excluded** (Section 0.3) and never read | copied once from Drive (6 Nov 2026); checked by `pasig.local.check_raw` |
| `manifest/` | `raw_manifest.csv` (SHA-256, source, station, year and G1 status of all 465 raw files: 458 `use`, 7 `exclude`) and `pumping_blocks.csv` (one row per half-month pumping block with its status: `use`, `pending_G20`, `descriptive_only`, `post_eval_only`, `exclude`) — **committed** | Phase 1 (G1) |
| `interim/` | long tables: water levels (all months), pumping, PAGASA 6-hourly and daily, QC'd water levels | Phase 1 (rebuilt automatically in a fresh clone) |
| `processed/` | `hourly_master.parquet` (wet seasons 2018–2024), `eval_hours.parquet` (scored cases per lead), `windows_{train,val}.npz` | Phases 1 and 3 (rebuilt automatically) |
| `synthetic/` | reserved for synthetic test fixtures (the smoke tests generate their own data) | — |

Rules:

* `raw/` is never edited. Every file's SHA-256 is in the manifest; every notebook and `run_phase.py` check it
  first (`pasig.local.prepare`), and `tests/test_raw_data.py` checks it too.
* A fresh clone has no `interim/` or `processed/`: the first notebook or phase that needs them runs Phase 1
  (and Phase 3 for the windows) once, about 1–2 minutes. Committed gates, configs and tables that come out
  identical apart from run stamps are restored byte for byte; a real difference is reported (`make data-check`
  does the same on demand).
* The 2024 test-season **windows** are not stored here: Notebook 06 builds them in memory after gate G18.
  The 2024 Aviles pumping blocks (`post_eval_only`) are parsed only in Notebook 07, after G18.
* Google Drive is no longer needed. Only if the source files on Drive change: `scripts/sync_drive.py` copies
  them again, then re-run Phase 1 and commit the new files with the new manifest. `scripts/drive/` records how
  the files were first retrieved in the cloud container (obsolete).
