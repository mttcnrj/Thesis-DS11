"""Feature table and windows (Section 9.1; D8, D9).

Sequence path (24 x C): Pandacan level, its hourly change, latest Science Garden 6 h total (step-held, usable
1 h after its period ends), plus every eligible-and-selected river station (none after G5).
Tabular path: hour-of-day and day-of-year sin/cos at t and t+1..t+6 (28); predicted tide at t+1..t+6 and its
change from t (12, D8); SG rain 24 h / 72 h and hours since the last report (3); previous-day SG RH, pressure
and mean temperature (3, used only if Granger selects them in NB04).

Windows are built inside each season (never across the dry-season gap). A case is an hour t that is scored at
one or more leads in `eval_hours.parquet`; `valid[:, h-1]` is that hour's lead-h flag, so every model is scored
on the same hours. Missing 6 h reports count as 0 mm in the rain inputs (38 blank reports in 2018-2024);
missing previous-day weather is imputed with the training median.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEADS = 6
WINDOW = 24
SEQ_BASE = ["pandacan", "d_pandacan", "sg_rain_6h_latest"]
WEATHER_TAB = ["sg_rain_24h_in", "sg_rain_72h_in", "sg_hours_since_report"]
DAILY_TAB = ["sg_daily_rh", "sg_daily_atm_pressure", "sg_daily_tmean"]


def calendar_features(times: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for h in range(0, LEADS + 1):
        tt = times + pd.Timedelta(hours=h)
        doy = tt.dayofyear.to_numpy()
        hr = tt.hour.to_numpy()
        cols += [np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
                 np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
        names += [f"hour_sin_t{h}", f"hour_cos_t{h}", f"doy_sin_t{h}", f"doy_cos_t{h}"]
    return np.column_stack(cols), names


def tide_features(times: pd.DatetimeIndex, tide) -> tuple[np.ndarray, list[str]]:
    now = tide.predict(times)
    fut = np.column_stack([tide.predict(times + pd.Timedelta(hours=h)) for h in range(1, LEADS + 1)])
    names = [f"tide_t{h}" for h in range(1, LEADS + 1)] + [f"tide_change_t{h}" for h in range(1, LEADS + 1)]
    return np.column_stack([fut, fut - now[:, None]]), names


def constituent_features(times: pd.DatetimeIndex, tide) -> tuple[np.ndarray, list[str]]:
    """D8 variant (tried once by G11): sin/cos of every frozen constituent at t+1..t+6."""
    th0 = tide._hours(times)
    cols, names = [], []
    for h in range(1, LEADS + 1):
        th = th0 + h
        for k, p in tide.periods.items():
            w = 2 * np.pi / p
            cols += [np.cos(w * th), np.sin(w * th)]
            names += [f"{k}_cos_t{h}", f"{k}_sin_t{h}"]
    return np.column_stack(cols), names


def rain_inputs(six: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """24 h / 72 h totals of the last 4 / 12 reports available at t, missing reports as 0 mm."""
    rep = six.dropna(subset=["available_from_pht"]).sort_values("available_from_pht")
    avail = rep["available_from_pht"].to_numpy()
    rain = np.nan_to_num(rep["rain_6h_mm"].to_numpy())
    pos = np.searchsorted(avail, index.to_numpy(), side="right") - 1
    csum = np.concatenate([[0.0], np.cumsum(rain)])
    out = pd.DataFrame(index=index)
    for k, name in ((4, "sg_rain_24h_in"), (12, "sg_rain_72h_in")):
        lo = np.clip(pos - k + 1, 0, None)
        out[name] = np.where(pos >= 0, csum[pos + 1] - csum[lo], np.nan)
    latest = np.where(pos >= 0, rain[np.clip(pos, 0, len(rain) - 1)], np.nan)
    out["sg_rain_6h_latest"] = latest
    return out


def feature_frame(master: pd.DataFrame, six: pd.DataFrame, extra_stations: list[str]) -> pd.DataFrame:
    f = pd.DataFrame(index=master.index)
    f["season"] = master["season"]
    f["pandacan"] = master["pandacan"]
    d = master["pandacan"].groupby(master["season"]).diff()
    f["d_pandacan"] = d.fillna(0.0)            # first hour of a season has no previous reading: change = 0
    f = f.join(rain_inputs(six, master.index))
    f["sg_hours_since_report"] = master["sg_hours_since_report"]
    for c in DAILY_TAB:
        f[c] = master[c]
    for st in extra_stations:
        f[st] = master[st]
    return f


def build_windows(master: pd.DataFrame, ev: pd.DataFrame, six: pd.DataFrame, tide, split: str,
                  extra_stations: list[str] | None = None, daily_fill: dict | None = None) -> dict:
    """Arrays for one split: X_seq (N,24,C), X_tab (N,F), X_tab_cons (N,K), y (N,6), valid (N,6), t_index (N,)."""
    extra_stations = list(extra_stations or [])
    f = feature_frame(master, six, extra_stations)
    seq_cols = SEQ_BASE + extra_stations
    leads = [f"lead_{h}" for h in range(1, LEADS + 1)]
    case = ev["split"].eq(split) & ev[leads].any(axis=1)
    t_index = ev.index[case]
    pos = master.index.get_indexer(t_index)
    S = f[seq_cols].to_numpy(dtype=float)
    X_seq = np.stack([S[p - WINDOW + 1: p + 1] for p in pos]) if len(pos) else np.zeros((0, WINDOW, len(seq_cols)))
    if not (master["season"].to_numpy()[pos - WINDOW + 1] == master["season"].to_numpy()[pos]).all():
        raise AssertionError("a window crosses a season boundary")
    cal, cal_n = calendar_features(t_index)
    tid, tid_n = tide_features(t_index, tide)
    cons, cons_n = constituent_features(t_index, tide)
    wx = f.loc[t_index, WEATHER_TAB].to_numpy(dtype=float)
    dx = f.loc[t_index, DAILY_TAB].copy()
    n_daily_missing = int(dx.isna().sum().sum())
    if daily_fill is not None:
        dx = dx.fillna(daily_fill)
    X_tab = np.column_stack([cal, tid, wx, dx.to_numpy(dtype=float)])
    lvl = np.concatenate([master["pandacan"].where(~master["pandacan_filled"]).to_numpy(), np.full(LEADS, np.nan)])
    y = np.column_stack([lvl[pos + h] for h in range(1, LEADS + 1)]) if len(pos) else np.zeros((0, LEADS))
    valid = ev.loc[t_index, leads].to_numpy(dtype=bool)
    y = np.where(valid, y, np.nan)
    if np.isnan(X_seq).any() or np.isnan(X_tab[:, : len(cal_n) + len(tid_n) + len(WEATHER_TAB)]).any():
        raise AssertionError("NaN in window inputs")
    return {"X_seq": X_seq.astype(np.float32), "X_tab": X_tab.astype(np.float32), "X_tab_cons": cons.astype(np.float32),
            "y": y.astype(np.float32), "valid": valid, "t_index": t_index.to_numpy(),
            "seq_names": seq_cols, "tab_names": cal_n + tid_n + WEATHER_TAB + DAILY_TAB, "cons_names": cons_n,
            "n_daily_missing_filled": n_daily_missing}
