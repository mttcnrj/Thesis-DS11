"""Pooled within-season Granger F-test with BIC lag choice (Section 9.2 steps 3-5).

Adapted from scripts/smoke/common.py::pooled_granger: lag matrices are built inside each season (no lag
crosses the dry-season gap), rows with any NaN are dropped, seasons are stacked with season dummies (no
separate intercept). Change from the smoke version: BIC is compared on the common sample valid at the
maximum lag, so every candidate lag order is scored on the same rows.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _design(ys: list[np.ndarray], xs: list[np.ndarray], p: int, with_x: bool, pmax: int | None = None):
    rows_y, rows_X = [], []
    start = pmax if pmax is not None else p
    for si, (y, x) in enumerate(zip(ys, xs, strict=True)):
        n = len(y)
        if n <= start + 1:
            continue
        idx = np.arange(start, n)
        cols = [y[idx - i] for i in range(1, p + 1)]
        if with_x:
            cols += [x[idx - i] for i in range(1, p + 1)]
        d = np.zeros((len(idx), len(ys)))
        d[:, si] = 1.0
        Xm = np.column_stack([d] + cols)
        yy = y[idx]
        ok = ~np.isnan(Xm).any(axis=1) & ~np.isnan(yy)
        if pmax is not None:                      # common sample: rows valid up to the maximum lag
            full = [y[idx - i] for i in range(1, pmax + 1)] + [x[idx - i] for i in range(1, pmax + 1)]
            ok &= ~np.isnan(np.column_stack(full)).any(axis=1)
        rows_y.append(yy[ok])
        rows_X.append(Xm[ok])
    X = np.vstack(rows_X)
    keep = X.any(axis=0) | (np.arange(X.shape[1]) >= len(ys))   # drop dummies of seasons with no rows
    return np.concatenate(rows_y), X[:, keep]


def _rss(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r)


def pooled_granger(ys: list[np.ndarray], xs: list[np.ndarray], max_lag: int) -> dict:
    bic = {}
    for p in range(1, max_lag + 1):
        yy, XX = _design(ys, xs, p, True, pmax=max_lag)
        n, k = XX.shape
        bic[p] = n * np.log(_rss(yy, XX) / n) + k * np.log(n)
    p = min(bic, key=bic.get)
    yu, Xu = _design(ys, xs, p, True)
    # restricted model (own lags + season dummies) on exactly the unrestricted rows: drop the p x-lag columns
    Xr = Xu[:, : Xu.shape[1] - p]
    rss_u, rss_r = _rss(yu, Xu), _rss(yu, Xr)
    df1, df2 = p, len(yu) - Xu.shape[1]
    F = ((rss_r - rss_u) / df1) / (rss_u / df2)
    return {"lag": int(p), "F": float(F), "df1": int(df1), "df2": int(df2), "p_value": float(stats.f.sf(F, df1, df2)),
            "n": int(len(yu)), "bic_by_lag": {int(k): float(v) for k, v in bic.items()}}
