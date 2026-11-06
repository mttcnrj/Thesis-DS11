"""Config loading. Every path comes from configs/paths.yaml (C1: portable, no Colab-only calls)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("PASIG_ROOT", Path(__file__).resolve().parents[1]))


def root() -> Path:
    return ROOT


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    """Load configs/<name>.yaml (cached; call `load.cache_clear()` after writing a generated config)."""
    p = ROOT / "configs" / f"{name}.yaml"
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def path(key: str) -> Path:
    """Resolve a key of configs/paths.yaml to an absolute path."""
    v = load("paths")[key]
    if v is None:
        raise KeyError(f"configs/paths.yaml: '{key}' is not set")
    p = Path(v)
    return p if p.is_absolute() else ROOT / p


def write_yaml(name: str, data: dict, header: str = "") -> Path:
    """Write a GENERATED config (configs/<name>.yaml) and clear the cache."""
    p = ROOT / "configs" / f"{name}.yaml"
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
