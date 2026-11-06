"""Two-stage grid (D12) with D14 ranges and the D10 criterion; resumable (one CSV row per config)."""
from __future__ import annotations

import itertools

import pandas as pd

from pasig.config import load, out_dir, profile
from pasig.models.hybrid import HybridConfig
from pasig.train.loop import train_run


def stage1_configs() -> list[HybridConfig]:
    c = load("model_hybrid")
    s, f = c["stage1"], c["stage1"]["fixed"]
    out = [HybridConfig(lstm_layers=ll, lstm_units=lu, gru_units=gu, batch=b, head_units=c["head_units"], **f)
           for ll, lu, gu, b in itertools.product(s["lstm_layers"], s["lstm_units"], s["gru_units"], s["batch"])]
    return _limit(out, "stage1_limit")


def stage2_configs(winner: HybridConfig) -> list[HybridConfig]:
    c = load("model_hybrid")["stage2"]
    out = []
    for do, lr, (dl, du) in itertools.product(c["dropout"], c["lr"], c["dense"]):
        d = winner.as_dict() | {"dropout": do, "lr": lr, "dense_layers": dl, "dense_units": du}
        out.append(HybridConfig(**d))
    return _limit(out, "stage2_limit")


def lstm_configs(batch: int) -> list[HybridConfig]:
    c = load("model_hybrid")["lstm"]
    out = [HybridConfig(lstm_units=u, dropout=do, lr=lr, batch=batch)
           for u, do, lr in itertools.product(c["units"], c["dropout"], c["lr"])]
    return _limit(out, "lstm_limit")


def _limit(cfgs: list, key: str) -> list:
    if profile() == "rehearsal":
        return cfgs[: load("model_hybrid")["rehearsal"][key]]
    return cfgs


def run_grid(name: str, kind: str, cfgs: list[HybridConfig], inp, tr, va, seed: int, deterministic: bool,
             log=print) -> pd.DataFrame:
    """Train every config not yet in runs/<name>.csv (resumable); returns the full table."""
    path = out_dir("runs") / f"{name}.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    have = set(done["tag"]) if len(done) else set()
    for i, cfg in enumerate(cfgs, 1):
        tag = f"{name}_{cfg.tag()}"
        if tag in have:
            continue
        r = train_run(kind, cfg, inp, tr, va, seed, deterministic, tag)["row"]
        done = pd.concat([done, pd.DataFrame([r])], ignore_index=True)
        done.to_csv(path, index=False)
        log(f"  [{i}/{len(cfgs)}] {tag}: val RMSE {r['val_rmse_mean']:.4f} m ({r['epochs']} epochs, {r['seconds']:.0f} s)")
    return done[done["tag"].isin([f"{name}_{c.tag()}" for c in cfgs])].reset_index(drop=True)


def best(table: pd.DataFrame) -> HybridConfig:
    """D10: lowest mean validation RMSE over the six leads."""
    r = table.sort_values("val_rmse_mean").iloc[0]
    keys = HybridConfig().as_dict().keys()
    return HybridConfig(**{k: (int(r[k]) if isinstance(HybridConfig().as_dict()[k], int) else float(r[k])) for k in keys})
