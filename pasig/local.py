"""Run everything from a local clone: the raw data are committed in data/raw/ (decision R1), so no notebook
waits for Google Drive.

`prepare(n)` is called in the first cell of every notebook and by `scripts/run_phase.py` before phase n:

1. checks that every raw file listed in data/manifest/raw_manifest.csv exists with its recorded SHA-256;
2. rebuilds any generated input that phase n needs but a fresh clone does not have (data/interim and
   data/processed are git-ignored): Phase 1 outputs, and the Phase 3 train/validation windows;
3. after a rebuild, compares every committed gate, generated config, artifact and table with what was there
   before. Files that differ only in run stamps (timestamp, git SHA, profile, date) are restored byte for byte,
   so a clean clone stays clean; any real difference is kept and reported loudly (the data or code changed).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from pasig.config import path, root

PHASE1_OUTPUTS = [("interim", "waterlevel_long.parquet"), ("interim", "waterlevel_qc.parquet"),
                  ("interim", "pumping_long.parquet"), ("interim", "pagasa_6h.parquet"),
                  ("interim", "pagasa_daily.parquet"), ("processed", "hourly_master.parquet"),
                  ("processed", "eval_hours.parquet")]
PHASE3_OUTPUTS = [("processed", "windows_train.npz"), ("processed", "windows_val.npz")]
# which generated inputs each phase reads
NEEDS = {1: [], 2: [1], 3: [1], 4: [1, 3], 5: [1, 3], 6: [1, 3], 7: [1, 3]}
WATCHED = ["gates", "configs", "artifacts", "reports", "data/manifest"]
VOLATILE = {"timestamp", "git_sha", "profile", "date"}


class RawDataError(RuntimeError):
    pass


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check_raw(verbose: bool = True) -> dict:
    """Every manifest file present in data/raw with its recorded SHA-256 (G1 evidence stays valid)."""
    man = pd.read_csv(path("manifest") / "raw_manifest.csv")
    raw = path("raw")
    missing = [r.path for r in man.itertuples() if not (raw / r.path).exists()]
    changed = [r.path for r in man.itertuples() if (raw / r.path).exists() and _sha256(raw / r.path) != r.sha256]
    if missing or changed:
        raise RawDataError(
            f"data/raw does not match data/manifest/raw_manifest.csv: {len(missing)} missing, {len(changed)} changed "
            f"(first: {(missing + changed)[:3]}). The raw data are committed in the repository: run "
            "`git checkout -- data/raw` (or `git lfs`/`git pull` if the clone is partial). Only if the source files "
            "on Drive really changed, refresh them with scripts/sync_drive.py and re-run Phase 1.")
    if verbose:
        print(f"raw data OK: {len(man)} files match the manifest ({int(man['status'].eq('use').sum())} used)")
    return {"files": len(man), "missing": missing, "changed": changed}


def _missing(outputs) -> list[str]:
    return [f"{d}/{f}" for d, f in outputs if not (path(d) / f).exists()]


def _snapshot() -> dict[str, bytes]:
    snap = {}
    for d in WATCHED:
        base = root() / d
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file() and "rehearsal" not in p.parts and p.suffix != ".pyc":
                    snap[str(p.relative_to(root()))] = p.read_bytes()
    return snap


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def _same_content(rel: str, a: bytes, b: bytes) -> bool:
    if a == b:
        return True
    try:
        if rel.endswith(".json"):
            return _strip(json.loads(a)) == _strip(json.loads(b))
        if rel.endswith((".yaml", ".yml")):
            return _strip(yaml.safe_load(a)) == _strip(yaml.safe_load(b))
    except Exception:
        return False
    return False


def _rebuild(phase: int) -> dict:
    """Run one phase as a script, then restore committed files whose content did not change."""
    before = _snapshot()
    print(f"rebuilding generated inputs: python scripts/run_phase.py --phase {phase} (once per clone)", flush=True)
    env = {k: v for k, v in os.environ.items() if k != "PASIG_PROFILE"}   # Phases 1-4 have no rehearsal variant
    code = subprocess.call([sys.executable, "scripts/run_phase.py", "--phase", str(phase), "--no-prepare"],
                           cwd=root(), env=env)
    if code != 0:
        raise RuntimeError(f"Phase {phase} failed while rebuilding local inputs (exit {code})")
    after = _snapshot()
    restored, differ = [], []
    for rel, old in before.items():
        new = after.get(rel)
        if new is None or new == old:
            continue
        if _same_content(rel, old, new):
            (root() / rel).write_bytes(old)
            restored.append(rel)
        else:
            differ.append(rel)
    new_files = [rel for rel in after if rel not in before]
    if differ:
        print("WARNING: the rebuild changed committed results (not just run stamps). Compare with `git diff` "
              "before going on:\n  " + "\n  ".join(differ[:20]), flush=True)
    return {"phase": phase, "restored_stamps": len(restored), "changed": differ, "new": new_files}


def prepare(phase: int, build: bool = True) -> dict:
    """Make phase `phase` runnable from a local clone (see module docstring)."""
    out = {"raw": check_raw()}
    for need in NEEDS.get(phase, []):
        miss = _missing(PHASE1_OUTPUTS if need == 1 else PHASE3_OUTPUTS)
        if not miss:
            continue
        if not build:
            raise RuntimeError(f"missing {miss}: run `python scripts/run_phase.py --phase {need}` first")
        if need == 3 and _missing(PHASE1_OUTPUTS):
            out["rebuild_1"] = _rebuild(1)
        out[f"rebuild_{need}"] = _rebuild(need)
    print(f"local inputs for phase {phase} ready")
    return out
