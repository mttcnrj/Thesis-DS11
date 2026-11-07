# data/

Git-ignored except this file and `manifest/`. Column definitions: `docs/data_dictionary.md`.

| Folder | Contents | Written by |
|---|---|---|
| `raw/` | copy of `MMDA Dataset/`, `PAGASA Dataset/` and `Kaggle Dataset/` from the `Thesis II` Drive folder. The Kaggle file is copied only so that G1 can record its hash; it is **excluded** (Section 0.3) and never read | `python scripts/sync_drive.py --src "<mounted Thesis II>"` |
| `manifest/` | `raw_manifest.csv` (SHA-256, source, station, year and G1 status of all 465 raw files: 458 `use`, 7 `exclude`) and `pumping_blocks.csv` (one row per half-month pumping block with its status: `use`, `pending_G20`, `descriptive_only`, `post_eval_only`, `exclude`) — **committed** | Phase 1 (G1) |
| `interim/` | long tables: water levels (all months), pumping, PAGASA 6-hourly and daily, QC'd water levels | Phase 1 |
| `processed/` | `hourly_master.parquet` (wet seasons 2018–2024), `eval_hours.parquet` (scored cases per lead), `windows_{train,val}.npz` | Phases 1 and 3 |
| `synthetic/` | reserved for synthetic test fixtures (the smoke tests generate their own data) | — |

Rules:

* `raw/` is never edited. Every file's SHA-256 is in the manifest; re-running Phase 1 on another machine must
  reproduce the same hashes.
* The 2024 test-season **windows** are not stored here: Notebook 06 builds them in memory after gate G18.
  The 2024 Aviles pumping blocks (`post_eval_only`) are parsed only in Notebook 07, after G18.
* In a cloud container without the Drive desktop client, `scripts/drive/` rebuilds `raw/` from Google Drive
  connector downloads (`drive_inventory.json` lists file ids, paths and byte counts; `recover_downloads.py`
  decodes and checks sizes; `make recover`).
