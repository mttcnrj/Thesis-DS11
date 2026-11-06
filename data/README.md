# data/

Git-ignored except this file and `manifest/`.

* `raw/` – copy of `MMDA Dataset/`, `PAGASA Dataset/`, `Kaggle Dataset/` from the `Thesis II` Drive folder
  (`python scripts/sync_drive.py --src "<mounted Thesis II>"`). In a cloud container without the Drive client,
  `scripts/drive/` rebuilds it from Google Drive connector downloads (`drive_inventory.json` lists file ids,
  paths and byte counts; `recover_downloads.py` decodes and checks sizes).
* `interim/`, `processed/` – written by Phase 1 (`make phase1`). See `docs/data_dictionary.md`.
* `manifest/raw_manifest.csv` – SHA-256 and G1 status of every raw file (committed).
