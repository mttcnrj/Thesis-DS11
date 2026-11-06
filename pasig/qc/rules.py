"""QC rules (Section 7.5): range, one-hour spike, flat runs (station-specific; see configs/qc.yaml).

`apply_qc(long)` returns the hourly long table with `qc_flag` ('' when clean, else the first rule that
fired) and `value_qc` (NaN where flagged). Flags never edit values in place; every removal is logged by
the caller. The rules are fixed before 2023-2024 are seen and applied unchanged to them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pasig.config import load


def _runs(s: pd.Series) -> pd.DataFrame:
    """Runs of identical consecutive non-NaN readings on an hourly grid."""
    v = s.to_numpy()
    new = np.ones(len(v), dtype=bool)
    new[1:] = ~((v[1:] == v[:-1]) & ~np.isnan(v[1:]))
    gid = np.cumsum(new)
    df = pd.DataFrame({"gid": gid, "v": v, "t": s.index})
    df = df[~np.isnan(v)]
    return df.groupby("gid").agg(value=("v", "first"), start=("t", "first"), end=("t", "last"), length=("v", "size"))


def range_flags(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.notna() & ~s.between(lo, hi)


def spike_flags(s: pd.Series, thr: float) -> pd.Series:
    d1 = s - s.shift(1)          # jump into t
    d2 = s - s.shift(-1)         # jump out of t (sign as seen from t)
    return (d1.abs() > thr) & (d2.abs() > thr) & (np.sign(d1) == np.sign(d2))


def flat_flags(s: pd.Series, station: str, cfg: dict, helper: pd.Series | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Flat-run flags for one station. Returns (flags, table of flagged runs)."""
    flags = pd.Series(False, index=s.index)
    R = _runs(s)
    R = R[R.length >= 2]
    out = []
    stuck = cfg.get("stuck_value_m")
    if stuck is not None:
        hit = R[np.isclose(R.value, stuck) & (R.length >= cfg["stuck_min_len"])]
        for _, r in hit.iterrows():
            out.append({**r.to_dict(), "rule": f"flat_{stuck:.2f}"})
    L = cfg.get("generic_min_len")
    if L:
        cand = R[R.length >= L]
        band = cfg.get("exempt_band_m")
        for _, r in cand.iterrows():
            in_band = band is not None and band[0] <= r.value <= band[1]
            if not in_band:
                out.append({**r.to_dict(), "rule": "flat"})
                continue
            if station == "sto_nino" and helper is not None:
                if _montalban_rise_while_flat(r, helper, cfg):
                    out.append({**r.to_dict(), "rule": "flat_upstream_rise"})
    tab = pd.DataFrame(out)
    for _, r in tab.iterrows():
        flags.loc[r["start"]:r["end"]] = True
    return flags, tab


def _montalban_rise_while_flat(run, montalban: pd.Series, cfg: dict) -> bool:
    """Sto. Niño exemption check: Montalban's 3-h median rises >= rise_m over `lag` hours at some t,
    while Sto. Niño is still flat at t + lag."""
    lag = pd.Timedelta(hours=cfg["montalban_lag_h"])
    med = montalban.rolling(3, min_periods=2).median()
    rise = med - med.shift(cfg["montalban_lag_h"])
    window = rise.loc[run["start"]: run["end"] - lag]
    return bool((window >= cfg["montalban_rise_m"]).any())


def apply_qc(long: pd.DataFrame, flat_exemption: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply all rules to every station. Returns (hourly table with qc columns, flagged-run table).
    `flat_exemption=False` drops the Sto. Niño band exemption (for the 'without exemption' coverage)."""
    q = load("qc")
    parts, runs = [], []
    hourly = {}
    for st, g in long.groupby("station"):
        s = g.set_index("time_pht")["value_m"].sort_index()
        s = s[~s.index.duplicated()]
        hourly[st] = s.asfreq("h")
    for st, s in hourly.items():
        lo, hi = q["valid_range_m"][st]
        f_range = range_flags(s, lo, hi)
        s1 = s.mask(f_range)
        f_spike = spike_flags(s1, q["spike_m"].get(st, q["spike_m"]["default"])) & ~f_range
        s2 = s1.mask(f_spike)
        cfg = dict(q["flat_run"].get(st, {}))
        if not flat_exemption and "exempt_band_m" in cfg and st == "sto_nino":
            cfg.pop("exempt_band_m")
        helper = None
        if st == "sto_nino" and "montalban" in hourly:
            m = hourly["montalban"]
            mlo, mhi = q["valid_range_m"]["montalban"]
            helper = m.mask(~m.between(mlo, mhi))
        f_flat, tab = flat_flags(s2, st, cfg, helper)
        f_flat &= ~(f_range | f_spike)
        flag = np.select([f_range, f_spike, f_flat], ["range", "spike", "flat"], default="").astype(object)
        if len(tab):
            tab.insert(0, "station", st)
            runs.append(tab)
            for _, r in tab.iterrows():   # name the specific flat rule
                sl = (s.index >= r["start"]) & (s.index <= r["end"]) & (flag == "flat")
                flag[sl] = r["rule"]
        df = pd.DataFrame({"time_pht": s.index, "station": st, "value_m": s.to_numpy(), "qc_flag": flag})
        df["value_qc"] = df["value_m"].where(df["qc_flag"].eq(""))
        parts.append(df)
    return pd.concat(parts, ignore_index=True), (pd.concat(runs, ignore_index=True) if runs else pd.DataFrame())
