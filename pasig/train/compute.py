"""Compute gates for Notebook 05 (Section 9.3 start): G14 determinism, G13 device, G15 seed budget.

Run as a module in a subprocess so the device is fixed before TensorFlow starts:
    python -m pasig.train.compute --probe
    python -m pasig.train.compute --determinism --device gpu
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from pasig.config import root

CPU_ENV = {"CUDA_VISIBLE_DEVICES": "-1"}


def sub_env(device: str) -> dict:
    env = dict(os.environ)
    if device == "cpu":
        env.update(CPU_ENV)
    return env


def call_module(args: list[str], device: str = "cpu") -> dict:
    out = subprocess.run([sys.executable, "-m", "pasig.train.compute", *args], cwd=root(), env=sub_env(device),
                         capture_output=True, text=True)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    if out.returncode != 0 or not lines:
        raise RuntimeError(f"compute subprocess failed: {out.stderr[-2000:]}")
    return json.loads(lines[-1])


def probe() -> dict:
    from pasig.utils.hardware import hardware_info, visible_nvidia_gpu
    return {"nvidia_gpu": visible_nvidia_gpu(), "hardware": hardware_info()}


def determinism(device: str, n_cases: int = 512) -> dict:
    """G14: with enable_op_determinism, a 1-epoch run repeated twice gives identical predictions (atol 1e-6)."""
    import numpy as np

    from pasig.models.hybrid import HybridConfig
    from pasig.train.data import Inputs, load_split
    from pasig.train.loop import predict_levels, set_seed
    from pasig.utils.hardware import configure_device
    configure_device(device)
    inp = Inputs.from_config()
    tr = load_split("train", inp, max_cases=n_cases, seed=1)
    va = load_split("val", inp, max_cases=256, seed=2)
    from pasig.train.loop import build
    preds = []
    for _ in range(2):
        set_seed(123, deterministic=True)
        cfg = HybridConfig(lstm_units=64, gru_units=32, batch=128)
        m = build("hybrid", cfg, inp, tr)
        m.fit(tr["X"], tr["Y"], epochs=1, batch_size=cfg.batch, verbose=0)
        preds.append(predict_levels(m, va["X"], va["y_scaler"], "hybrid"))
    diff = float(np.max(np.abs(preds[0] - preds[1])))
    return {"device": device, "max_abs_diff": diff, "identical": bool(diff <= 1e-6)}


def run_bench(device: str, out: Path, target_cases: int, quick: bool, train_cases: int | None,
              measure_epochs: int, no_determinism: bool = False) -> dict:
    """scripts/smoke/bench_compute.py (Section 12); returns its summary.json and projection.csv."""
    cmd = [sys.executable, "scripts/smoke/bench_compute.py", "--device", device, "--target-cases", str(target_cases),
           "--measure-epochs", str(measure_epochs), "--out", str(out)]
    if quick:
        cmd.append("--quick")
    if train_cases:
        cmd += ["--train-cases", str(train_cases)]
    if no_determinism:
        cmd.append("--no-determinism")
    r = subprocess.run(cmd, cwd=root(), env=sub_env(device), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bench_compute failed on {device}: {r.stderr[-2000:]}")
    d = out / (f"bench_{device}" + ("_nodet" if no_determinism else ""))
    return {"summary": json.loads((d / "summary.json").read_text()), "projection": pd.read_csv(d / "projection.csv"),
            "dir": str(d)}


def plan_hours(summary: dict, projection: pd.DataFrame, seeds: int = 5) -> dict:
    """Recommended-plan projection extended for this plan: six ablation variants (not five), the D20 weight
    choice (3 runs), and the G11/G12 reference runs (4 variants x 3 seeds)."""
    p = projection.set_index("item")["hours"]
    median_run_h = summary["median_epoch_s"] * summary["expected_epochs"] / 3600
    base = float(summary["recommended_total_hours"])
    abl = float(p.get("Ablations (5 variants x 5 seeds)", 0.0))
    extra = abl * (6 / 5 - 1) + 3 * median_run_h + 12 * median_run_h
    total5 = base + extra
    # walk-forward + ablations scale with the seed count
    wf = float(p.get("Walk-forward (3 folds x hybrid + LSTM x 5 seeds)", 0.0))
    seed_part = abl * 6 / 5 + wf
    total3 = total5 - seed_part * (1 - 3 / 5)
    return {"bench_recommended_hours": base, "extra_hours": round(extra, 2), "plan_hours_5_seeds": round(total5, 2),
            "plan_hours_3_seeds_wf_ablation": round(total3, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--determinism", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.probe:
        print(json.dumps(probe(), default=str))
    elif a.determinism:
        print(json.dumps(determinism(a.device)))


if __name__ == "__main__":
    main()
