"""One training run (Section 9.3-9.5): seeds, determinism (D15), early stopping, run folder.

Run folder runs/<yyyymmdd_hhmm>_<tag>_seed<k>/: config_snapshot.yaml, git_sha.txt, env.txt, model.keras,
history.json, timing.json, preds_val.parquet.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pasig.config import load, out_dir, profile
from pasig.models.hybrid import HybridConfig, build_hybrid, build_lstm, masked_mse
from pasig.train.data import Inputs, repeat_tab
from pasig.utils.gitinfo import git_sha
from pasig.utils.hardware import env_text

_DET = {"on": False}


def hp() -> dict:
    """Training hyperparameters with the rehearsal overrides applied in the rehearsal profile."""
    c = dict(load("model_hybrid"))
    if profile() == "rehearsal":
        c.update({k: v for k, v in c["rehearsal"].items() if k in ("max_epochs", "patience", "gate_seeds", "final_seeds")})
    return c


def set_seed(seed: int, deterministic: bool) -> None:
    import keras
    import tensorflow as tf
    keras.utils.set_random_seed(seed)
    if deterministic and not _DET["on"]:
        tf.config.experimental.enable_op_determinism()      # process-wide; cannot be switched off again
        _DET["on"] = True


def rmse_by_lead(y_pred: np.ndarray, y_true: np.ndarray, valid: np.ndarray) -> list[float]:
    return [float(np.sqrt(np.mean((y_pred[valid[:, h], h] - y_true[valid[:, h], h]) ** 2))) for h in range(y_true.shape[1])]


def predict_levels(model, X, y_scaler, kind: str) -> np.ndarray:
    Xin = repeat_tab(X) if kind == "lstm" else X
    p = model.predict(Xin, batch_size=1024, verbose=0)
    return p * y_scaler.span + y_scaler.lo


def build(kind: str, cfg: HybridConfig, inp: Inputs, data: dict, loss=masked_mse, **variant):
    n_seq = data["X"]["seq"].shape[2]
    n_tab = data["X"]["tab"].shape[1] if "tab" in data["X"] else 0      # no-tabular ablation passes seq only
    if kind == "hybrid":
        return build_hybrid(cfg, (data["X"]["seq"].shape[1], n_seq), n_tab, add_back=inp.add_back, loss=loss, **variant)
    if kind == "lstm":
        return build_lstm(cfg.lstm_units, cfg.dropout, cfg.lr, (data["X"]["seq"].shape[1], n_seq + n_tab),
                          add_back=inp.add_back, loss=loss)
    raise ValueError(kind)


def train_run(kind: str, cfg: HybridConfig, inp: Inputs, tr: dict, va: dict, seed: int, deterministic: bool,
              tag: str, save: bool = True, loss=masked_mse, max_epochs: int | None = None, **variant) -> dict:
    import keras
    h = hp()
    set_seed(seed, deterministic)
    model = build(kind, cfg, inp, tr, loss=loss, **variant)
    Xtr = repeat_tab(tr["X"]) if kind == "lstm" else tr["X"]
    Xva = repeat_tab(va["X"]) if kind == "lstm" else va["X"]
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=h["patience"], restore_best_weights=True)]
    t0 = time.perf_counter()
    hist = model.fit(Xtr, tr["Y"], validation_data=(Xva, va["Y"]), epochs=max_epochs or h["max_epochs"],
                     batch_size=cfg.batch, verbose=0, callbacks=cb, shuffle=True)
    secs = time.perf_counter() - t0
    pred = predict_levels(model, va["X"], va["y_scaler"], kind)
    rm = rmse_by_lead(pred, va["y_raw"], va["valid"])
    ep = len(hist.history["loss"])
    row = {"kind": kind, "tag": tag, "seed": seed, "deterministic": deterministic, **cfg.as_dict(),
           "inputs": inp.tag, "use_constituents": inp.use_constituents, "add_back": inp.add_back,
           **{f"val_rmse_lead{i + 1}": v for i, v in enumerate(rm)}, "val_rmse_mean": float(np.mean(rm)),
           "epochs": ep, "best_epoch": int(np.argmin(hist.history["val_loss"]) + 1), "seconds": round(secs, 2),
           "seconds_per_epoch": round(secs / max(ep, 1), 3), "train_loss_first": float(hist.history["loss"][0]),
           "train_loss_last": float(hist.history["loss"][-1]), "val_loss_first": float(hist.history["val_loss"][0]),
           "val_loss_best": float(np.min(hist.history["val_loss"])), **{f"variant_{k}": v for k, v in variant.items()}}
    if save:
        row["run_dir"] = str(_save_run(model, hist, row, pred, va, kind, cfg, inp, tag, seed))
    return {"row": row, "model": model, "pred_val": pred}


def _save_run(model, hist, row, pred, va, kind, cfg, inp, tag, seed) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    d = out_dir("runs") / f"{stamp}_{tag}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config_snapshot.yaml").write_text(yaml.safe_dump({"kind": kind, "config": cfg.as_dict(), "inputs": inp.__dict__,
                                                            "profile": profile()}, sort_keys=False))
    (d / "git_sha.txt").write_text(git_sha() + "\n")
    (d / "env.txt").write_text(env_text({"deterministic": row["deterministic"]}))
    model.save(d / "model.keras")
    (d / "history.json").write_text(json.dumps({k: [float(x) for x in v] for k, v in hist.history.items()}))
    (d / "timing.json").write_text(json.dumps({"seconds": row["seconds"], "epochs": row["epochs"],
                                               "seconds_per_epoch": row["seconds_per_epoch"]}))
    preds_frame(va, pred, f"{kind}:{tag}", seed).to_parquet(d / "preds_val.parquet", index=False)
    return d


def preds_frame(data: dict, pred: np.ndarray, model: str, seed: int | None = None) -> pd.DataFrame:
    n, L = pred.shape
    return pd.DataFrame({"t": np.repeat(data["t_index"].to_numpy(), L), "lead": np.tile(np.arange(1, L + 1), n),
                         "model": model, "seed": -1 if seed is None else seed,
                         "y_true": data["y_raw"].reshape(-1), "y_pred": pred.reshape(-1), "valid": data["valid"].reshape(-1)})
