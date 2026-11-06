"""PAGASA Science Garden parsers (Section 3.2, 7.4). Port Area columns are dropped (D6).

Science Garden is a rain gauge representing rainfall over upland Quezon City; it does not measure runoff.

* 6-hourly file: two header rows, `Date(UTC)` as M/D/YYYY H:MM. Trace `T` -> 0.05 mm (D4); blank -> NaN.
* Daily file: YEAR, MONTH, DAY, RAINFALL, ... ; -1 (trace) -> 0.05 mm (D4); -999 -> NaN. A daily value
  covers 08:00 PHT on its date to 08:00 PHT the next day and is usable from 09:00 PHT the day after
  its label.
* G3 (D5) settles whether 6-hourly stamps mark the start or the end of their period; a total becomes
  usable 1 h after its period ends.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRACE_MM = 0.05          # D4: half the 0.1 mm reporting limit
UTC_TO_PHT = pd.Timedelta(hours=8)
AVAIL_LAG = pd.Timedelta(hours=1)   # D5
G3_TOL_MM = 0.2
G3_MIN_SHARE = 0.95

SG_6H_COLS = {4: "rain_6h_mm", 5: "tmax_6h_c", 6: "tmin_6h_c"}   # Science Garden block of the CSV


def _num(series: pd.Series, trace_token) -> tuple[pd.Series, pd.Series]:
    raw = series.astype("string").str.strip()
    if trace_token is not None:
        is_trace = raw.eq(str(trace_token)).fillna(False).astype(bool)   # blanks are missing, not trace
    else:
        is_trace = pd.Series(False, index=series.index)
    val = pd.to_numeric(raw, errors="coerce").astype(float)
    val = val.where(~is_trace, TRACE_MM)
    return val, is_trace


def read_6hourly(path: Path) -> pd.DataFrame:
    """Science Garden 6-hourly reports. Returns stamp_utc, stamp_pht, rain_6h_mm, rain_trace, tmax/tmin."""
    head = pd.read_csv(path, nrows=1, header=None, dtype=str, encoding="utf-8-sig").iloc[0].tolist()
    if "Science Garden" not in str(head[4]):
        raise ValueError(f"unexpected 6-hourly layout, column 4 header = {head[4]!r}")
    df = pd.read_csv(path, skiprows=2, header=None, dtype=str, encoding="utf-8-sig")
    out = pd.DataFrame({"stamp_utc": pd.to_datetime(df[0], format="%m/%d/%Y %H:%M")})
    out["rain_6h_mm"], out["rain_trace"] = _num(df[4], "T")
    out["tmax_6h_c"], _ = _num(df[5], None)
    out["tmin_6h_c"], _ = _num(df[6], None)
    out["rain_raw"] = df[4]
    out["stamp_pht"] = out["stamp_utc"] + UTC_TO_PHT
    if out["stamp_utc"].duplicated().any():
        raise ValueError("duplicated 6-hourly stamps")
    return out.sort_values("stamp_utc").reset_index(drop=True)


def read_daily(path: Path) -> pd.DataFrame:
    """Science Garden daily values with -999 -> NaN and rainfall -1 -> 0.05 mm (D4)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    for c in ["rainfall", "tmax", "tmin", "tmean", "rh", "wind_speed", "wind_direction", "atm_pressure"]:
        df[c] = df[c].astype(float).mask(df[c] == -999)
    df["rain_trace"] = df["rainfall"].eq(-1)
    df["rainfall"] = df["rainfall"].mask(df["rain_trace"], TRACE_MM)
    # covers 08:00 PHT on `date` to 08:00 PHT next day; usable from 09:00 PHT the day after its label
    df["period_start_pht"] = df["date"] + pd.Timedelta(hours=8)
    df["period_end_pht"] = df["date"] + pd.Timedelta(days=1, hours=8)
    df["available_from_pht"] = df["date"] + pd.Timedelta(days=1, hours=9)
    return df.rename(columns={"rainfall": "rain_day_mm"})


def g3_convention_test(six: pd.DataFrame, daily: pd.DataFrame, tol: float = G3_TOL_MM) -> dict:
    """D5/G3. For rain days (daily total > 0.1 mm), compare the daily total with
    (a) sum of 00/06/12/18 UTC reports of that date  -> stamps mark period START;
    (b) 06/12/18 UTC + 00 UTC of the next date       -> stamps mark period END.
    The daily value covers 00 UTC d to 00 UTC d+1 (08:00 PHT to 08:00 PHT)."""
    s = six.set_index("stamp_utc")["rain_6h_mm"]
    rows = []
    for _, r in daily.iterrows():
        tot = r["rain_day_mm"]
        if not (tot > 0.1):
            continue
        d = r["date"]
        a_idx = [d + pd.Timedelta(hours=h) for h in (0, 6, 12, 18)]
        b_idx = [d + pd.Timedelta(hours=h) for h in (6, 12, 18, 24)]
        a = s.reindex(a_idx)
        b = s.reindex(b_idx)
        rows.append({"date": d, "daily": tot,
                     "a_sum": a.sum() if a.notna().all() else np.nan,
                     "b_sum": b.sum() if b.notna().all() else np.nan})
    t = pd.DataFrame(rows)
    ev = {}
    for k in ("a", "b"):
        ok = t[f"{k}_sum"].notna()
        match = (t.loc[ok, f"{k}_sum"] - t.loc[ok, "daily"]).abs() <= tol
        ev[f"option_{k}_n"] = int(ok.sum())
        ev[f"option_{k}_match_share"] = float(match.mean()) if ok.any() else float("nan")
    ev["rain_days"] = int(len(t))
    winner = None
    if ev["option_b_match_share"] >= G3_MIN_SHARE and ev["option_b_match_share"] >= ev["option_a_match_share"]:
        winner = "period_end"
    elif ev["option_a_match_share"] >= G3_MIN_SHARE:
        winner = "period_start"
    ev["convention"] = winner
    ev["table"] = t
    return ev


def add_availability(six: pd.DataFrame, convention: str) -> pd.DataFrame:
    """D5: total usable 1 h after its period ends (PHT)."""
    out = six.copy()
    if convention == "period_end":
        out["period_end_pht"] = out["stamp_pht"]
    elif convention == "period_start":
        out["period_end_pht"] = out["stamp_pht"] + pd.Timedelta(hours=6)
    else:
        raise ValueError(f"unknown convention {convention!r}")
    out["period_start_pht"] = out["period_end_pht"] - pd.Timedelta(hours=6)
    out["available_from_pht"] = out["period_end_pht"] + AVAIL_LAG
    return out


def hourly_rain_features(six: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """For each hour t: latest available 6-h total (step-held), 24 h and 72 h totals over reports
    available at t, hours since the last available report, and whether the report needed at t exists.

    Rolling totals use the reports whose availability time is <= t and whose period ended within the
    last 24 h / 72 h measured at the availability time (i.e. the last 4 / 12 available reports)."""
    rep = six.dropna(subset=["available_from_pht"]).sort_values("available_from_pht")
    avail = rep["available_from_pht"].to_numpy()
    rain = rep["rain_6h_mm"].to_numpy()
    pos = np.searchsorted(avail, index.to_numpy(), side="right") - 1   # last report available at t
    n = len(rep)
    latest = np.where(pos >= 0, rain[np.clip(pos, 0, n - 1)], np.nan)
    hrs_since = np.where(pos >= 0, (index.to_numpy() - avail[np.clip(pos, 0, n - 1)]) / np.timedelta64(1, "h"), np.nan)

    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(rain))])
    cnan = np.concatenate([[0], np.cumsum(np.isnan(rain))])

    def window(k):
        lo = np.clip(pos - k + 1, 0, None)
        tot = csum[pos + 1] - csum[lo]
        bad = (cnan[pos + 1] - cnan[lo]) > 0
        short = (pos - k + 1) < 0
        return np.where((pos >= 0) & ~bad & ~short, tot, np.nan)

    out = pd.DataFrame(index=index)
    out["sg_rain_6h_latest"] = latest
    out["sg_rain_24h"] = window(4)
    out["sg_rain_72h"] = window(12)
    out["sg_hours_since_report"] = hrs_since
    # the report needed at t: the latest one available at t, which must be the expected one
    # (no more than 6 h + 1 h lag old) and non-missing
    out["sg_report_ok"] = (~np.isnan(latest)) & (hrs_since < 6)
    return out
