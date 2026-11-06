"""Rebuild data/raw/ from Google Drive connector downloads (cloud-container fallback for D16/C1).

Why this exists: in a Claude Code cloud container the Drive desktop client is unavailable and
drive.google.com is not reachable, so raw files are fetched with the Google Drive connector
(`download_file_content`), whose results are base64 JSON stored either in a session transcript
(*.jsonl) or a spilled tool-result file (*.txt). This script scans those logs, decodes every
download whose file id is in `drive_inventory.json`, checks the byte count against Drive's
metadata, and writes the file to `data/raw/<drive path>`.

On the chosen laptop use `scripts/sync_drive.py` (copies from the mounted Drive) instead.

Usage: python scripts/drive/recover_downloads.py [--logs ~/.claude/projects] [--out data/raw]
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Matches the connector payload both raw ({"content":"..","id":"..") and JSON-escaped inside a
# transcript line ({\"content\":\"..\",\"id\":\"..).
PAT = re.compile(r'\\?"content\\?"\s*:\s*\\?"([A-Za-z0-9+/=]{16,})\\?"\s*,\s*\\?"id\\?"\s*:\s*\\?"([A-Za-z0-9_-]{20,})\\?"')


def scan(log_root: Path) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for p in list(log_root.rglob("*.jsonl")) + list(log_root.rglob("*.txt")):
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if '"content' not in text:
            continue
        for b64, fid in PAT.findall(text):
            if fid in found:
                continue
            try:
                found[fid] = base64.b64decode(b64, validate=True)
            except Exception:  # truncated/garbled payload: skip, another copy may exist
                continue
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(Path.home() / ".claude" / "projects"))
    ap.add_argument("--out", default=str(HERE.parents[1] / "data" / "raw"))
    ap.add_argument("--inventory", default=str(HERE / "drive_inventory.json"))
    args = ap.parse_args()

    inv = json.loads(Path(args.inventory).read_text())
    found = scan(Path(args.logs))
    out = Path(args.out)
    ok, bad, missing = 0, [], []
    for item in inv:
        data = found.get(item["id"])
        if data is None:
            dest = out / item["path"]
            if dest.exists() and dest.stat().st_size == item["bytes"]:
                ok += 1
            else:
                missing.append(item)
            continue
        if len(data) != item["bytes"]:
            bad.append((item["path"], len(data), item["bytes"]))
            continue
        dest = out / item["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.read_bytes() != data:
            dest.write_bytes(data)
        ok += 1
    print(f"inventory={len(inv)} written_or_present={ok} size_mismatch={len(bad)} missing={len(missing)}")
    for b in bad:
        print("SIZE MISMATCH", b)
    Path(args.out).parent.joinpath("manifest").mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.joinpath("manifest", "drive_missing.json").write_text(json.dumps(missing, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
