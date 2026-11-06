"""Config loading. Every path comes from configs/paths.yaml (C1: portable, no Colab-only calls)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("PASIG_ROOT", Path(__file__).resolve().parents[1]))

# Rehearsal profile (PASIG_PROFILE=rehearsal): Notebooks 05-07 run end to end with tiny settings and write their
# gates, GENERATED configs, checkpoints and tables under runs/rehearsal/ (git-ignored). Inputs from Phases 1-4
# are still read from their normal places. Real runs use the default profile ("full").
PHASE5_GENERATED = ("compute", "model_inputs", "thresholds")


def profile() -> str:
    return os.environ.get("PASIG_PROFILE", "full")


def rehearsal_dir() -> Path:
    return ROOT / "runs" / "rehearsal"


def out_dir(kind: str) -> Path:
    """Output folder for Phase 5+ products: gates, tables, figures, logs, runs, frozen (checkpoints)."""
    base = rehearsal_dir() if profile() == "rehearsal" else ROOT
    sub = {"gates": "gates", "tables": "reports/tables", "figures": "reports/figures", "logs": "reports/logs",
           "runs": "runs", "frozen": "artifacts/frozen", "compute": "reports/compute"}[kind]
    p = base / sub if not (profile() == "rehearsal" and kind == "runs") else base / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def root() -> Path:
    return ROOT


def _config_file(name: str) -> Path:
    if profile() == "rehearsal" and name in PHASE5_GENERATED:
        return rehearsal_dir() / "configs" / f"{name}.yaml"
    return ROOT / "configs" / f"{name}.yaml"


@lru_cache(maxsize=None)
def _load(name: str, prof: str) -> dict:
    p = _config_file(name)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load(name: str) -> dict:
    """Load configs/<name>.yaml (Phase 5 GENERATED configs come from runs/rehearsal/configs in rehearsal)."""
    return _load(name, profile())


load.cache_clear = _load.cache_clear  # type: ignore[attr-defined]


def path(key: str) -> Path:
    """Resolve a key of configs/paths.yaml to an absolute path."""
    v = load("paths")[key]
    if v is None:
        raise KeyError(f"configs/paths.yaml: '{key}' is not set")
    p = Path(v)
    return p if p.is_absolute() else ROOT / p


def write_yaml(name: str, data: dict, header: str = "") -> Path:
    """Write a GENERATED config (configs/<name>.yaml) and clear the cache."""
    p = _config_file(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    p.write_text((header.rstrip() + "\n" if header else "") + text, encoding="utf-8")
    load.cache_clear()
    return p


def wet_months() -> list[int]:
    return list(load("splits")["wet_months"])


def split_of_year(year: int) -> str:
    s = load("splits")
    if year in s["train_years"]:
        return "train"
    if year in s["val_years"]:
        return "val"
    if year in s["test_years"]:
        return "test"
    return "none"
