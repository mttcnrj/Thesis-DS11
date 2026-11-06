"""Phase 4 / Notebook 04: Granger selection (Section 9.2, SOP 2). Gate G10; hypothesis H2.

Eligible candidates (after G5): Science Garden rainfall (6-hourly), RH, pressure, mean temperature (daily).
Ineligible river stations (Sto. Niño, Fort Santiago) are tested the same way on their available training
hours and reported as descriptive results only (D2): never inputs, not counted in m.
"""
from __future__ import annotations

import datetime as dt
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import Ridge

from pasig import config
from pasig.config import load, path
from pasig.features.scaling import MinMax, load_scalers
from pasig.features.tide import load_tide
from pasig.gates import record_gate, require_gates
from pasig.phase3 import load_windows
from pasig.selection.correlation import max_lagged_xcorr
from pasig.selection.granger import pooled_granger
from pasig.selection.stationarity import apply_transform, choose_transform
from pasig.utils.gitinfo import git_sha

DAILY = {"sg_rh": "rh", "sg_pressure": "atm_pressure", "sg_tmean": "tmean"}


def _p(key: str) -> Path:
    p = path(key)
    p.mkdir(parents=True, exist_ok=True)
    return p


def train_years() -> list[int]:
    return list(load("splits")["train_years"])


# ----------------------------------------------------------------------------- series

def detided_hourly(master: pd.DataFrame, station: str) -> pd.Series:
    s = master.loc[master.split.eq("train"), station]
    return s - load_tide(station).predict(s.index)


def six_hourly_pair(y_h: pd.Series, six: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """y: mean de-tided Pandacan over each 6 h report period (start, end]; x: the period's rain total."""
    k = load("granger")["min_hours_per_period"]["six_hourly"]
    s6 = six.dropna(subset=["period_end_pht"]).set_index("period_end_pht").sort_index()
    s6 = s6[s6.index.year.isin(train_years()) & s6.index.month.isin(config.wet_months())]
    # assign each hour t to the period whose end is the first stamp >= t (hours start+1 .. end)
    ends = s6.index.to_numpy()
    pos = np.searchsorted(ends, y_h.index.to_numpy(), side="left")
    ok = pos < len(ends)
    lab = pd.Series(pd.DatetimeIndex(ends)[np.clip(pos, 0, len(ends) - 1)], index=y_h.index).where(ok)
    g = y_h.groupby(lab)
    y = g.mean().where(g.count() >= k)
    y = y.reindex(s6.index)
    return y, s6["rain_6h_mm"].astype(float)


def daily_pair(y_h: pd.Series, daily: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """y: max de-tided Pandacan over the PAGASA day 09:00 D .. 08:00 D+1; x: the daily value labelled D."""
    k = load("granger")["min_hours_per_period"]["daily"]
    day = (y_h.index - pd.Timedelta(hours=9)).normalize()
    g = y_h.groupby(day)
    y = g.max().where(g.count() >= k)
    d = daily.set_index("date")
    d = d[d.index.year.isin(train_years()) & d.index.month.isin(config.wet_months())]
    return y.reindex(d.index), d[col].astype(float)


def by_season(s: pd.Series) -> dict[int, pd.Series]:
    return {int(y): g for y, g in s.groupby(s.index.year)}


# ----------------------------------------------------------------------------- one test

def granger_test(y: pd.Series, x: pd.Series, max_lag: int, seasons: list[int] | None = None) -> dict:
    cfg = load("granger")
    seasons = seasons or sorted(set(y.index.year))
    yb = {k: v for k, v in by_season(y).items() if k in seasons}
    xb = {k: v for k, v in by_season(x).items() if k in seasons}
    ty, ey = choose_transform(yb, cfg["adf_p_threshold"])
    tx, ex = choose_transform(xb, cfg["adf_p_threshold"])
    ys, xs = [], []
    for k in seasons:
        yy = apply_transform(yb[k], ty)
        xx = apply_transform(xb[k].reindex(yb[k].index), tx)
        ys.append(yy.to_numpy(dtype=float))
        xs.append(xx.to_numpy(dtype=float))
    res = pooled_granger(ys, xs, max_lag)
    res.update({"transform_y": ty, "transform_x": tx, "adf_y": ey, "adf_x": ex, "seasons": seasons})
    return res


# ----------------------------------------------------------------------------- step 6: ridge RMSE gain

def _flat(a: dict, s: dict, seq_idx: list[int], tab_idx: list[int]) -> np.ndarray:
    xs = s["seq"].transform(a["X_seq"])[:, :, seq_idx].reshape(len(a["y"]), -1)
    xt = s["tab"].transform(a["X_tab"])[:, tab_idx]
    return np.hstack([xs, xt])


def ridge_rmse(Ftr, Ytr, Vtr, seasons_tr, Fva, Yva, Vva) -> dict:
    """Alpha by leave-one-season-out CV on training (mean RMSE over leads), then fit on all training and
    score 2023 (valid targets only). One ridge per lead."""
    alphas = load("granger")["ridge_gain"]["alphas"]
    warnings.filterwarnings("ignore", category=LinAlgWarning)   # collinear calendar sin/cos at tiny alpha
    cv = {}
    for a in alphas:
        errs = []
        for s in np.unique(seasons_tr):
            tr, te = seasons_tr != s, seasons_tr == s
            rm = []
            for h in range(Ytr.shape[1]):
                m1, m2 = tr & Vtr[:, h], te & Vtr[:, h]
                if m1.sum() < 50 or m2.sum() == 0:      # season without usable cases (e.g. Fort Santiago 2018): skip fold
                    continue
                p = Ridge(alpha=a).fit(Ftr[m1], Ytr[m1, h]).predict(Ftr[m2])
                rm.append(np.sqrt(np.mean((p - Ytr[m2, h]) ** 2)))
            if rm:
                errs.append(np.mean(rm))
        cv[a] = float(np.mean(errs))
    best = min(cv, key=cv.get)
    per = []
    for h in range(Ytr.shape[1]):
        m1, m2 = Vtr[:, h], Vva[:, h]
        p = Ridge(alpha=best).fit(Ftr[m1], Ytr[m1, h]).predict(Fva[m2])
        per.append(float(np.sqrt(np.mean((p - Yva[m2, h]) ** 2))))
    return {"alpha": best, "cv_rmse_by_alpha": cv, "val_rmse_by_lead": per, "val_rmse_mean": float(np.mean(per))}


def ridge_gains(groups: dict, extra_station_windows: dict | None = None) -> dict:
    """Validation RMSE (mean over leads) with vs without each candidate's feature group."""
    tr, va = load_windows("train"), load_windows("val")
    sc = load_scalers(path("artifacts") / "scalers.json")
    names = json.loads((path("artifacts") / "feature_names.json").read_text())
    seq_n, tab_n = names["seq"], names["tab"]
    seasons_tr = pd.DatetimeIndex(tr["t_index"]).year.to_numpy()
    daily_cols = [c for g in ("sg_rh", "sg_pressure", "sg_tmean") for c in groups[g]["tab"]]
    rain_seq, rain_tab = groups["sg_rain"]["seq"], groups["sg_rain"]["tab"]

    def idx(seq_drop=(), tab_drop=(), tab_keep_daily=()):
        si = [i for i, n in enumerate(seq_n) if n not in seq_drop]
        ti = [i for i, n in enumerate(tab_n) if n not in tab_drop and (n not in daily_cols or n in tab_keep_daily)]
        return si, ti

    def score(si, ti, extra=None, rows=None):
        Ftr, Fva = _flat(tr, sc, si, ti), _flat(va, sc, si, ti)
        Ytr, Yva, Vtr, Vva = tr["y"], va["y"], tr["valid"].copy(), va["valid"].copy()
        if extra is not None:
            Ftr, Fva = np.hstack([Ftr, extra["train"]]), np.hstack([Fva, extra["val"]])
        if rows is not None:
            Vtr &= rows["train"][:, None]
            Vva &= rows["val"][:, None]
            Ftr, Fva = np.nan_to_num(Ftr), np.nan_to_num(Fva)
        return ridge_rmse(Ftr, Ytr, Vtr, seasons_tr, Fva, Yva, Vva)

    out = {}
    base_si, base_ti = idx()                                   # base: everything except daily weather
    with_rain = score(base_si, base_ti)
    no_rain = score(*idx(seq_drop=rain_seq, tab_drop=rain_tab))
    out["sg_rain"] = {"with": with_rain, "without": no_rain, "gain_m": no_rain["val_rmse_mean"] - with_rain["val_rmse_mean"]}
    for g in ("sg_rh", "sg_pressure", "sg_tmean"):
        w = score(*idx(tab_keep_daily=groups[g]["tab"]))
        out[g] = {"with": w, "without": with_rain, "gain_m": with_rain["val_rmse_mean"] - w["val_rmse_mean"]}
    for st, ex in (extra_station_windows or {}).items():
        rows = {k: ~np.isnan(ex[k]).any(axis=1) for k in ("train", "val")}
        w = score(base_si, base_ti, extra=ex, rows=rows)
        wo = score(base_si, base_ti, rows=rows)
        out[st] = {"with": w, "without": wo, "gain_m": wo["val_rmse_mean"] - w["val_rmse_mean"],
                   "cases_with_complete_window": {k: int(v.sum()) for k, v in rows.items()}}
    return out


def station_windows(master: pd.DataFrame, station: str) -> dict:
    """24 h window of a river station for the train/val cases, min-max scaled on training cases; NaN kept."""
    out = {}
    s = master[station]
    pos_all = {}
    for sp in ("train", "val"):
        t = pd.DatetimeIndex(load_windows(sp)["t_index"])
        pos = master.index.get_indexer(t)
        v = s.to_numpy()
        out[sp] = np.stack([v[p - 23: p + 1] for p in pos])
        pos_all[sp] = pos
    sc = MinMax().fit(out["train"][~np.isnan(out["train"]).any(axis=1)][..., None])
    return {k: ((v - sc.lo[0]) / sc.span[0]).astype(float) for k, v in out.items()}


# ----------------------------------------------------------------------------- run

def run() -> dict:
    require_gates("G6", "G9")
    cfg = load("granger")
    proc, interim = path("processed"), path("interim")
    master = pd.read_parquet(proc / "hourly_master.parquet")
    master = master[master.split.ne("test")]
    six = pd.read_parquet(interim / "pagasa_6h.parquet")
    daily = pd.read_parquet(interim / "pagasa_daily.parquet")
    elig = load("eligible_inputs")
    y_h = detided_hourly(master, "pandacan")

    # ---- eligible candidates (count toward m)
    tests = {}
    y6, x6 = six_hourly_pair(y_h, six)
    tests["sg_rain"] = granger_test(y6, x6, cfg["max_lag"]["six_hourly"])
    tests["sg_rain"]["step"] = "6-hourly: mean de-tided Pandacan over the report period vs the period's rain total"
    for g, col in DAILY.items():
        yd, xd = daily_pair(y_h, daily, col)
        tests[g] = granger_test(yd, xd, cfg["max_lag"]["daily"])
        tests[g]["step"] = f"daily: max de-tided Pandacan over 09:00 D..08:00 D+1 vs {col} labelled D"
    for st in elig.get("eligible_stations") or []:
        tests[st] = granger_test(y_h, detided_hourly(master, st), cfg["max_lag"]["hourly"])
    m = len(tests)
    thr = cfg["alpha"] / m
    for r in tests.values():
        r["bonferroni_threshold"] = thr
        r["selected"] = bool(r["p_value"] < thr)

    # ---- ineligible stations: descriptive only (D2)
    desc = {}
    cov = pd.read_csv(path("tables") / "coverage_wet_season.csv", index_col=0)
    for st in elig.get("descriptive_only") or []:
        ok = [int(y) for y in train_years() if cov.loc[st, str(y)] >= cfg["min_season_coverage_descriptive"]]
        skipped = [int(y) for y in train_years() if int(y) not in ok]
        r = granger_test(y_h, detided_hourly(master, st), cfg["max_lag"]["hourly"], seasons=ok)
        r.update({"seasons_used": ok, "seasons_skipped": skipped, "nominal_p_lt_0.05": bool(r["p_value"] < 0.05),
                  "note": "descriptive only (D2): never an input, not counted in m"})
        desc[st] = r

    # ---- step 6: ridge validation RMSE gain beside each p-value
    groups = cfg["feature_groups"]
    stw = {st: station_windows(master, st) for st in desc}
    gains = ridge_gains(groups, stw)
    for k in tests:
        tests[k]["ridge_val_rmse_gain_m"] = gains[k]["gain_m"]
    for k in desc:
        desc[k]["ridge_val_rmse_gain_m"] = gains[k]["gain_m"]

    # ---- step 7: correlation screen (same count as Granger selects), eligible candidates
    hourly_x = _hourly_candidates(master, six, daily)
    ty = tests["sg_rain"]["transform_y"]
    yt = apply_transform(y_h, "level")
    xc = {}
    for k, x in hourly_x.items():
        how = tests[k]["transform_x"]
        xt = x.groupby(x.index.year).transform(lambda s, h=how: apply_transform(s, h))
        yy = yt.groupby(yt.index.year).transform(lambda s, h=ty: apply_transform(s, h))
        r, lag = max_lagged_xcorr(yy, xt, range(cfg["correlation_screen"]["lags_h"][0], cfg["correlation_screen"]["lags_h"][1] + 1))
        xc[k] = {"max_abs_xcorr": r, "lag_h": lag}
    n_sel = sum(r["selected"] for r in tests.values())
    ranked = sorted(xc, key=lambda k: xc[k]["max_abs_xcorr"], reverse=True)
    corr_sel = ranked[:n_sel]

    selected = [k for k, r in tests.items() if r["selected"]]
    h2 = "supported" if selected else "not supported"
    if not selected:
        outcome = "none pass: model uses Pandacan history + tabular (calendar, tide) only"
    elif len(selected) == m:
        outcome = "all pass: treated as screening; the 'no Granger selection' ablation decides"
    else:
        outcome = "some pass: selected candidates enter the model"
    seq_cols = ["pandacan", "d_pandacan"] + [c for g in selected for c in groups.get(g, {}).get("seq", [])]
    tab_base = [n for n in json.loads((path("artifacts") / "feature_names.json").read_text())["tab"]
                if n.startswith(("hour_", "doy_", "tide_"))]
    tab_cols = tab_base + [c for g in selected for c in groups.get(g, {}).get("tab", [])]
    corr_seq = ["pandacan", "d_pandacan"] + [c for g in corr_sel for c in groups.get(g, {}).get("seq", [])]
    corr_tab = tab_base + [c for g in corr_sel for c in groups.get(g, {}).get("tab", [])]
    config.write_yaml("selected_features", {
        "selected_candidates": selected, "bonferroni_threshold": thr, "m": m, "outcome": outcome, "h2": h2,
        "model_inputs": {"seq": seq_cols, "tab": tab_cols},
        "correlation_selection_ablation": {"candidates": corr_sel, "seq": corr_seq, "tab": corr_tab, "ranking": xc},
        "no_granger_ablation": {"description": "all eligible candidates enter regardless of the F-test",
                                "seq": ["pandacan", "d_pandacan"] + [c for g in tests for c in groups.get(g, {}).get("seq", [])],
                                "tab": tab_base + [c for g in tests for c in groups.get(g, {}).get("tab", [])]},
        "tests": {k: _slim(v) for k, v in tests.items()},
        "descriptive_only": {k: _slim(v) for k, v in desc.items()},
        "git_sha": git_sha(), "date": dt.date.today().isoformat(),
    }, header="# GENERATED by G10 (NB04). Frozen with settings-frozen (G17); never edit by hand.")
    rows = [{"candidate": k, "group": "eligible", **_row(v)} for k, v in tests.items()] + \
           [{"candidate": k, "group": "descriptive only (D2)", **_row(v)} for k, v in desc.items()]
    table = pd.DataFrame(rows)
    table.to_csv(_p("tables") / "granger_results.csv", index=False)
    record_gate("G10", "granger_outcome", True,
                "Pre-stated: none pass -> model uses Pandacan history + tabular only; all pass -> call it screening and "
                "let the 'no Granger' ablation decide. Select a candidate iff p < 0.05/m (m = eligible candidates tested)",
                {"tests": {k: _slim(v) for k, v in tests.items()}, "descriptive_only": {k: _slim(v) for k, v in desc.items()},
                 "correlation_screen": xc, "h2": h2}, decision=f"{outcome}; selected: {selected or 'none'}")
    return {"tests": tests, "descriptive": desc, "gains": gains, "xcorr": xc, "selected": selected, "h2": h2,
            "outcome": outcome, "table": table}


def _hourly_candidates(master: pd.DataFrame, six: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Candidates as available at each training hour t (for the correlation screen)."""
    idx = master.index[master.split.eq("train")]
    s6 = six.dropna(subset=["available_from_pht"]).sort_values("available_from_pht")
    pos = np.searchsorted(s6["available_from_pht"].to_numpy(), idx.to_numpy(), side="right") - 1
    rain = pd.Series(np.where(pos >= 0, s6["rain_6h_mm"].to_numpy()[np.clip(pos, 0, None)], np.nan), index=idx)
    out = {"sg_rain": rain}
    d = daily.sort_values("available_from_pht")
    m = pd.merge_asof(pd.DataFrame({"t": idx}), d[["available_from_pht", "rh", "atm_pressure", "tmean"]],
                      left_on="t", right_on="available_from_pht")
    for g, col in DAILY.items():
        out[g] = pd.Series(m[col].to_numpy(), index=idx)
    return out


def _slim(r: dict) -> dict:
    keep = ("lag", "F", "df1", "df2", "p_value", "n", "transform_y", "transform_x", "seasons", "seasons_used",
            "seasons_skipped", "selected", "bonferroni_threshold", "ridge_val_rmse_gain_m", "nominal_p_lt_0.05", "step", "note")
    out = {k: r[k] for k in keep if k in r}
    out["adf_p_y"] = r["adf_y"]["adf_p_by_season"]
    out["adf_p_x"] = r["adf_x"]["adf_p_by_season"]
    return out


def _row(r: dict) -> dict:
    return {"lag": r["lag"], "F": r["F"], "p_value": r["p_value"], "n": r["n"], "transform_y": r["transform_y"],
            "transform_x": r["transform_x"], "selected": r.get("selected"), "ridge_val_rmse_gain_m": r["ridge_val_rmse_gain_m"],
            "seasons": ",".join(map(str, r.get("seasons_used", r["seasons"])))}
