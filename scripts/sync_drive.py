"""Copy the raw data once from the mounted Google Drive folder into data/raw/ and write the manifest
(Section 2.2, 7.1). Run on the chosen laptop after Google Drive for desktop has synced `Thesis II`.

    python scripts/sync_drive.py --src "/mnt/g/My Drive/Thesis II"      # WSL
    python scripts/sync_drive.py                                        # uses configs/paths.yaml: drive_root

Only `MMDA Dataset/`, `PAGASA Dataset/` and `Kaggle Dataset/` are copied (the Kaggle file is excluded by G1 but
hashed so the manifest is complete). Existing identical files are left alone. G1 itself is evaluated in NB01.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pasig.config import load, path  # noqa: E402
from pasig.io import manifest as mf  # noqa: E402

FOLDERS = ["MMDA Dataset", "PAGASA Dataset", "Kaggle Dataset"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=load("paths").get("drive_root"))
    a = ap.parse_args()
    if not a.src:
        raise SystemExit("set --src or configs/paths.yaml: drive_root")
    src, dst = Path(a.src), path("raw")
    n = 0
    for folder in FOLDERS:
        for p in sorted((src / folder).rglob("*")):
            if not p.is_file() or p.name.lower() == "desktop.ini" or p.name.startswith("~$"):
                continue
            out = dst / p.relative_to(src)
            if out.exists() and out.stat().st_size == p.stat().st_size and mf.sha256(out) == mf.sha256(p):
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            n += 1
    print(f"copied {n} files into {dst}")
    man, bl = mf.build_manifest(dst)
    mdir = path("manifest")
    mdir.mkdir(parents=True, exist_ok=True)
    man.drop(columns=["dup_group"]).to_csv(mdir / "raw_manifest.csv", index=False)
    bl.to_csv(mdir / "pumping_blocks.csv", index=False)
    print(man.status.value_counts().to_string())


if __name__ == "__main__":
    main()
