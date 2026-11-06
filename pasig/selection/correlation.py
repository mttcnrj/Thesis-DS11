"""Correlation screen for the 'correlation selection' ablation (Section 9.2 step 7)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def max_lagged_xcorr(y: pd.Series, x: pd.Series, lags: range) -> tuple[float, int]:
    """max over lags k of |corr(y_t, x_{t-k})|, computed within each season (index carries the season)."""
    best, lag = 0.0, None
    for k in lags:
        xs = x.groupby(x.index.year).shift(k)
        j = pd.concat([y, xs], axis=1).dropna()
        if len(j) < 30:
            continue
        r = abs(float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]))
        if r > best:
            best, lag = r, k
    return best, lag
