"""Gates (Section 5): pre-set rules evaluated at run time, each writing gates/G##_<name>.json.

A failed *blocking* gate stops the pipeline (record it with passed=False, then raise).
A failed *switching* gate selects the fallback branch and is recorded as passed=True with the
decision taken. *Record* gates always pass and just store the outcome.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from pasig.config import out_dir, path, profile
from pasig.utils.gitinfo import git_sha

PHT = timezone(timedelta(hours=8))


def _gate_dir() -> Path:
    """Where gates are WRITTEN (runs/rehearsal/gates in the rehearsal profile)."""
    return out_dir("gates")


def _read_dirs() -> list[Path]:
    dirs = [_gate_dir()]
    real = path("gates")
    if profile() == "rehearsal" and real not in dirs:
        dirs.append(real)          # Phases 1-4 gates are read from gates/
    return dirs


def _norm(gate_id: str) -> str:
    n = int(gate_id.upper().lstrip("G"))
    return f"G{n:02d}"


def gate_file(gate_id: str) -> Path | None:
    for d in _read_dirs():
        hits = sorted(d.glob(f"{_norm(gate_id)}_*.json"))
        if hits:
            return hits[0]
    return None


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def record_gate(gate_id: str, name: str, passed: bool, rule: str, evidence: dict,
                decision: str | None = None) -> Path:
    gid = _norm(gate_id)
    for old in _gate_dir().glob(f"{gid}_*.json"):
        old.unlink()
    rec = {
        "gate": f"G{int(gid[1:])}",
        "name": name,
        "passed": bool(passed),
        "decision": decision,
        "rule": rule,
        "evidence": _jsonable(evidence),
        "timestamp": datetime.now(PHT).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "profile": profile(),
    }
    out = _gate_dir() / f"{gid}_{name}.json"
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def read_gate(gate_id: str) -> dict | None:
    f = gate_file(gate_id)
    return json.loads(f.read_text(encoding="utf-8")) if f else None


def require_gates(*gate_ids: str) -> None:
    """Raise RuntimeError naming every gate that is missing or failed. Called in the first cell of
    every notebook and at the start of scripts/run_phase.py."""
    problems = []
    for g in gate_ids:
        rec = read_gate(g)
        if rec is None:
            problems.append(f"{_norm(g)} missing")
        elif not rec.get("passed", False):
            problems.append(f"{_norm(g)} failed ({rec.get('name')})")
    if problems:
        raise RuntimeError("Required gates not satisfied: " + "; ".join(problems))


class GateFailure(RuntimeError):
    """Raised after a blocking gate is recorded as failed."""
