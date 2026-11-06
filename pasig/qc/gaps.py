"""Gap fill (Section 7.6): gaps of <= 3 h are forward-filled and flagged `<col>_filled`; longer gaps stay
NaN. Filled hours are never forecast targets. Fill never crosses a season boundary (callers pass one
season at a time or a series on a continuous hourly grid)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fill_short_gaps(s: pd.Series, max_gap: int = 3) -> tuple[pd.Series, pd.Series]:
    isna = s.isna().to_numpy()
    gid = np.cumsum(~isna)                       # gap hours share the id of the last observed hour
    run = pd.Series(isna).groupby(gid).transform("sum").to_numpy()
    started = gid > 0                            # a gap at the very start has nothing to carry forward
    # a gap is fillable only if it is closed (an observation follows) and short
    nxt_obs = pd.Series(np.where(~isna, np.arange(len(s)), np.nan)).bfill().notna().to_numpy()
    fill_mask = isna & started & (run <= max_gap) & nxt_obs
    filled = s.ffill().where(fill_mask, s)
    return filled, pd.Series(fill_mask, index=s.index)
