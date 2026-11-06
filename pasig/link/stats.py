"""Statistics for the link analysis (Section 8): Wilson intervals, events, bootstraps, clustered logit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return p, c - h, c + h


def event_ids(t: pd.DatetimeIndex, merge_gap_h: int = 6) -> np.ndarray:
    """Group sorted timestamps (hours at/above a level) into events; gaps < merge_gap_h hours are merged."""
    if len(t) == 0:
        return np.array([], dtype=int)
    gaps = np.diff(t.to_numpy()) / np.timedelta64(1, "h")
    return np.concatenate([[0], np.cumsum(gaps >= merge_gap_h)])


def bootstrap_share(flags: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile 95% CI of a share by resampling units (e.g. events) with replacement."""
    flags = np.asarray(flags, dtype=float)
    if len(flags) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(flags), size=(reps, len(flags)))
    m = flags[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fit_logit(df: pd.DataFrame, y: str, xs: list[str], cluster: str | None):
    X = sm.add_constant(df[xs].astype(float), has_constant="add")
    mod = sm.Logit(df[y].astype(float), X)
    if cluster is None:
        return mod.fit(disp=0, maxiter=200)
    groups = pd.factorize(df[cluster])[0]
    return mod.fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})


def lr_test(small, big) -> dict:
    stat = 2 * (big.llf - small.llf)
    df = int(big.df_model - small.df_model)
    return {"lr_stat": float(stat), "df": df, "p_value": float(stats.chi2.sf(stat, df))}


def odds_ratio(res, var: str, per: float) -> dict:
    b, se = res.params[var], res.bse[var]
    z = 1.959964
    return {"beta": float(b), "se": float(se), "or": float(np.exp(b * per)),
            "ci_low": float(np.exp((b - z * se) * per)), "ci_high": float(np.exp((b + z * se) * per)),
            "p_value": float(res.pvalues[var])}


def predict(res, df: pd.DataFrame, xs: list[str]) -> np.ndarray:
    X = sm.add_constant(df[xs].astype(float), has_constant="add")
    return np.asarray(res.predict(X))


def auc_gain_day_bootstrap(y: np.ndarray, p0: np.ndarray, p1: np.ndarray, days: np.ndarray,
                           reps: int, seed: int) -> dict:
    """ROC-AUC of two fixed models on the same hours, and the gain AUC1 - AUC0 with a calendar-day block
    bootstrap (days resampled with replacement; the models are not refitted)."""
    auc0, auc1 = roc_auc_score(y, p0), roc_auc_score(y, p1)
    rng = np.random.default_rng(seed)
    ud, inv = np.unique(days, return_inverse=True)
    rows_by_day = [np.flatnonzero(inv == k) for k in range(len(ud))]
    gains, a0s, a1s = [], [], []
    for _ in range(reps):
        pick = rng.integers(0, len(ud), size=len(ud))
        idx = np.concatenate([rows_by_day[k] for k in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        a0, a1 = roc_auc_score(yy, p0[idx]), roc_auc_score(yy, p1[idx])
        a0s.append(a0)
        a1s.append(a1)
        gains.append(a1 - a0)
    g = np.array(gains)
    return {"auc_m0": float(auc0), "auc_m1": float(auc1), "gain": float(auc1 - auc0),
            "gain_ci_low": float(np.percentile(g, 2.5)), "gain_ci_high": float(np.percentile(g, 97.5)),
            "auc_m0_ci": [float(np.percentile(a0s, 2.5)), float(np.percentile(a0s, 97.5))],
            "auc_m1_ci": [float(np.percentile(a1s, 2.5)), float(np.percentile(a1s, 97.5))],
            "reps_used": int(len(g)), "n_hours": int(len(y)), "n_days": int(len(ud)), "positives": int(y.sum())}
