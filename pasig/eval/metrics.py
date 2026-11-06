"""Level-error metrics (Section 10.1)."""
from __future__ import annotations

import numpy as np


def level_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    e = pred - truth
    ss_res = float(np.sum(e**2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return {"n": int(len(e)), "rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e))),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "bias": float(np.mean(e))}


def skill(rmse_model: float, rmse_ref: float) -> float:
    return 1 - rmse_model / rmse_ref if rmse_ref > 0 else float("nan")
