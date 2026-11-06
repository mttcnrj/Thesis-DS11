"""Diebold-Mariano test on squared errors with Newey-West variance and the HLN small-sample correction.
Moved unchanged from scripts/smoke/common.py::diebold_mariano (Section 10.1)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano(e_a: np.ndarray, e_b: np.ndarray, h: int) -> dict:
    """d = e_a^2 - e_b^2; negative mean d means model a has the smaller squared error."""
    d = e_a**2 - e_b**2
    d = d[~np.isnan(d)]
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    lags = max(h - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    v = np.dot(dc, dc) / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * np.dot(dc[k:], dc[:-k]) / n
    dm = dbar / np.sqrt(v / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return {"dm": float(dm), "dm_hln": float(hln), "p_value": float(p), "n": int(n), "mean_loss_diff": float(dbar),
            "nw_lags": int(lags)}
