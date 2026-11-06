"""ADF per training season (Section 9.2 step 2)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_pvalue(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 30 or np.nanstd(x) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(adfuller(x, autolag="BIC")[1])


def choose_transform(by_season: dict[int, pd.Series], p_thr: float = 0.05) -> tuple[str, dict]:
    """'level' if every season rejects a unit root (p < p_thr); else 'diff1' for ALL seasons."""
    ps = {int(k): adf_pvalue(v) for k, v in by_season.items()}
    nonstat = [k for k, p in ps.items() if not (p < p_thr)]
    return ("diff1" if nonstat else "level"), {"adf_p_by_season": ps, "seasons_p_ge_threshold": nonstat}


def apply_transform(s: pd.Series, how: str) -> pd.Series:
    return s.diff() if how == "diff1" else s
