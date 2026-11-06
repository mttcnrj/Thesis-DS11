"""Event detection and warning time (Section 10.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def events(level_obs: pd.Series, level: float, merge_gap_h: int) -> pd.DataFrame:
    """Observed events: hours >= level grouped into runs, gaps < merge_gap_h merged. One row per event."""
    t = level_obs.index[(level_obs >= level).to_numpy()]
    if len(t) == 0:
        return pd.DataFrame(columns=["event", "first", "last", "peak_m", "hours"])
    gaps = np.diff(t.to_numpy()) / np.timedelta64(1, "h")
    eid = np.concatenate([[0], np.cumsum(gaps >= merge_gap_h)])
    df = pd.DataFrame({"t": t, "eid": eid, "v": level_obs.loc[t].to_numpy()})
    return df.groupby("eid").agg(first=("t", "min"), last=("t", "max"), peak_m=("v", "max"), hours=("t", "size")) \
             .reset_index().rename(columns={"eid": "event"})


def detect(ev: pd.DataFrame, alert_target_times: pd.DatetimeIndex, lead: int, window_h: int) -> pd.DataFrame:
    """An event is detected at lead h if an alert's target hour (issue hour t + h) falls in
    [first - window, first + window]; warning time = first exceedance - issue hour of the earliest such alert."""
    a = pd.DatetimeIndex(alert_target_times).sort_values()
    out = []
    for _, e in ev.iterrows():
        lo, hi = e["first"] - pd.Timedelta(hours=window_h), e["first"] + pd.Timedelta(hours=window_h)
        hit = a[(a >= lo) & (a <= hi)]
        if len(hit):
            issue = hit.min() - pd.Timedelta(hours=lead)
            out.append({"event": e["event"], "detected": True, "warning_h": (e["first"] - issue) / pd.Timedelta(hours=1)})
        else:
            out.append({"event": e["event"], "detected": False, "warning_h": np.nan})
    return pd.DataFrame(out)


def event_bootstrap_ci(flags: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    flags = np.asarray(flags, dtype=float)
    if len(flags) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = flags[rng.integers(0, len(flags), size=(reps, len(flags)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
