"""Phase 2 / Notebook 02: link analysis and alert levels (Section 8, SOP 1). Gates G4, G20, G7, G8.

Training seasons (2018-2022 wet) for fitting; 2023 for checking; 2024 pumping data are not read (sealed until
after Notebook 06). Operational rules are fixed in configs/link.yaml.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from pasig import config
from pasig.config import load, path
from pasig.gates import GateFailure, record_gate, require_gates
from pasig.io import mmda_pumping as mp
from pasig.link import legend as lg
from pasig.link import quiapo as qz
from pasig.link import stats as st
from pasig.utils.gitinfo import git_sha

PUMP_DIR = "MMDA Dataset/WOCR and Water elevation reading of Aviles and Quiapo Pumping Station"


def _p(key: str) -> Path:
    p = path(key)
    p.mkdir(parents=True, exist_ok=True)
    return p


def train_years() -> list[int]:
    return list(load("splits")["train_years"])


# ----------------------------------------------------------------------------- inputs

def load_inputs() -> dict:
    proc, interim = path("processed"), path("interim")
    master = pd.read_parquet(proc / "hourly_master.parquet")
    pump = pd.read_parquet(interim / "pumping_long.parquet")
    six = pd.read_parquet(interim / "pagasa_6h.parquet")
    blocks = pd.read_csv(path("manifest") / "pumping_blocks.csv")
    pump = pump[~pump.flag_impossible_date & pump.time_pht.notna()]
    if (pump.time_pht.dt.year >= min(load("splits")["test_years"])).any():
        raise RuntimeError("test-year pumping data must not be read before Notebook 06")
    # Pandacan observed, unfilled, unflagged
    pan = master["pandacan"].where(~master["pandacan_filled"])
    # Science Garden 6 h total of the period containing hour t (for the sky-code check only, not a model input)
    six = six.sort_values("period_start_pht")
    m = pd.merge_asof(pd.DataFrame({"t": master.index}), six[["period_start_pht", "period_end_pht", "rain_6h_mm"]],
                      left_on="t", right_on="period_start_pht", direction="backward")
    rain6 = pd.Series(np.where(m["t"] < m["period_end_pht"], m["rain_6h_mm"], np.nan), index=master.index)
    taguig = mp.parse_taguig(path("raw") / PUMP_DIR / "QUIAPO 2021.xlsx")
    return {"master": master, "pump": pump, "pandacan": pan, "rain6": rain6, "blocks": blocks, "taguig": taguig}


def station_frame(pump: pd.DataFrame, station: str, statuses=("use",)) -> pd.DataFrame:
    d = pump[pump.station.eq(station) & pump.block_status.isin(statuses)]
    d = d.sort_values(["time_pht", "source_file"]).drop_duplicates("time_pht").set_index("time_pht").sort_index()
    return d[d.index.month.isin(config.wet_months())]


def analysis_frame(inp: dict, station_df: pd.DataFrame, use_pumps: bool, use_gate: bool) -> pd.DataFrame:
    """Hourly frame on the master index: Pandacan (observed), ln(1+R24), stress label, gate, dry flag, season."""
    cfg = load("link")
    m = inp["master"]
    f = pd.DataFrame(index=m.index)
    f["season"] = m["season"]
    f["split"] = m["split"]
    f["L"] = inp["pandacan"]
    f["R24"] = m["sg_rain_24h"]
    f["lnR"] = np.log1p(f["R24"])
    f["dry"] = f["R24"] < cfg["dry_rain24_mm"]
    s = station_df.reindex(m.index)
    known = s["raw_status"].notna() & s["status_parsed"].fillna(False).astype(bool)
    f["pumps"] = s["pumps_running"].where(known)
    f["stress"] = (f["pumps"] >= cfg["stress_min_pumps"]).astype(float).where(known & use_pumps)
    f["gate_open"] = s["gate_open"].where(known & use_gate)
    f["inside"], f["outside"] = s["inside_m"], s["outside_m"]
    f["day"] = f.index.normalize()
    return f


# ----------------------------------------------------------------------------- G4

def step_legend(inp: dict) -> dict:
    tr = train_years()
    out = {}
    for stn in ("aviles", "quiapo"):
        d = station_frame(inp["pump"], stn)
        d = d[d.index.year.isin(tr)]
        out[stn] = lg.check_legend(d, inp["pandacan"], inp["rain6"])
        out[stn]["fields"] = lg.field_decisions(out[stn])
    # sky codes exist only at Aviles; the gate/pump fields decide per station
    dec = {"aviles": out["aviles"]["fields"], "quiapo": {k: v for k, v in out["quiapo"]["fields"].items() if k != "sky"}}
    dropped = [f"{s}.{k}" for s, f in dec.items() for k, v in f.items() if not v]
    record_gate("G4", "status_legend", True,
                "For each code part of the assumed legend (Section 8.1) the check fails on <= 20% of applicable "
                "2018-2022 hours -> field used; else the field is dropped from Phase 2 and logged (configs/link.yaml). "
                "Gate 'O' and pump 'nP' checks amended post hoc (decision P2-a); literal results kept in the evidence",
                out, decision=("all fields used" if not dropped else "dropped: " + ", ".join(dropped)))
    pd.DataFrame([{"station": s, "check": k, **v} for s, r in out.items() for k, v in r.items()
                  if isinstance(v, dict) and "fail_share" in v]).to_csv(_p("tables") / "g4_legend_checks.csv", index=False)
    return {"checks": out, "fields": dec}


# ----------------------------------------------------------------------------- G20

def step_g20(inp: dict) -> dict:
    q = inp["pump"][inp["pump"].station.eq("quiapo")]
    av = inp["pump"][inp["pump"].station.eq("aviles") & inp["pump"].block_status.eq("use")]
    pan_all = inp["pandacan"]
    ev = qz.g20(q, av, inp["taguig"], inp["blocks"], pan_all, config.wet_months())
    record_gate("G20", "quiapo_2021_identity", True,
                "(a) layout/units match confirmed Quiapo files; (b) sheets dated 2021; (c) corr(outside, Pandacan) "
                ">= lowest confirmed-year corr - 0.05; (d) not a copy of Aviles or the Taguig files over the same hours "
                "(corr < 0.99 and not equal); overlapping copies must agree (identical -> one series)",
                ev, decision="include 2021 in the Quiapo check (with/without reported)" if ev["passed"]
                else "exclude 2021: failed " + ", ".join(k for k, v in ev["conditions"].items() if not v))
    return ev


# ----------------------------------------------------------------------------- 8.2 first assumption

def step_first_assumption(f: pd.DataFrame) -> pd.DataFrame:
    tr = f[f.split.eq("train")]
    rows = []
    for name, sub in [("all training hours", tr), ("Pandacan >= 11.69 m", tr[tr.L >= 11.69])]:
        j = sub[["L", "outside"]].dropna()
        dl = (tr["L"] - tr["L"].shift(1)).where(tr.index.to_series().diff().eq(pd.Timedelta("1h")))
        do = (tr["outside"] - tr["outside"].shift(1)).where(tr.index.to_series().diff().eq(pd.Timedelta("1h")))
        dd = pd.concat([dl, do], axis=1).loc[j.index].dropna()
        rows.append({"subset": name, "n": len(j), "pearson": j.L.corr(j.outside),
                     "spearman": j.L.corr(j.outside, method="spearman"),
                     "mean_diff_aviles_minus_pandacan_m": (j.outside - j.L).mean(),
                     "sd_diff_m": (j.outside - j.L).std(), "corr_hourly_changes": dd.iloc[:, 0].corr(dd.iloc[:, 1]),
                     "n_changes": len(dd)})
    out = pd.DataFrame(rows)
    out.to_csv(_p("tables") / "link_first_assumption.csv", index=False)
    return out


# ----------------------------------------------------------------------------- 8.3 grid tables

def grid_levels() -> np.ndarray:
    lo, hi = load("link")["grid_range_m"]
    step = load("link")["grid_step_m"]
    return np.round(np.arange(lo, hi + 1e-9, step), 2)


def step_grid(f: pd.DataFrame, split: str = "train", tag: str = "aviles") -> pd.DataFrame:
    cfg = load("link")
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    sub = f[f.split.eq(split) & f.L.notna()]
    rows = []
    for L in grid_levels():
        h = sub[sub.L >= L]
        if h.empty:
            continue
        s = h["stress"].dropna()
        p, lo, hi = st.wilson(int(s.sum()), len(s))
        g = h["gate_open"].dropna()
        dry = h["dry"][h["R24"].notna()]
        eid = st.event_ids(h.index, cfg["event_merge_gap_h"])
        ev = pd.DataFrame({"eid": eid, "stress": h["stress"].to_numpy(), "dry": h["dry"].to_numpy(),
                           "R24": h["R24"].to_numpy()})
        e_stress = ev.dropna(subset=["stress"]).groupby("eid")["stress"].max()
        e_dry = ev.dropna(subset=["R24"]).groupby("eid")["dry"].all()
        es_lo, es_hi = st.bootstrap_share(e_stress.to_numpy(), cfg["bootstrap_reps"], rng)
        ed_lo, ed_hi = st.bootstrap_share(e_dry.to_numpy().astype(float), cfg["bootstrap_reps"], rng)
        rows.append({"level_m": L, "hours": len(h), "hours_with_status": len(s), "stress_share": p,
                     "stress_ci_low": lo, "stress_ci_high": hi,
                     "open_gate_share": g.mean() if len(g) else np.nan, "dry_share": dry.mean() if len(dry) else np.nan,
                     "events": int(eid.max() + 1), "events_with_status": len(e_stress),
                     "event_stress_share": e_stress.mean() if len(e_stress) else np.nan,
                     "event_stress_ci_low": es_lo, "event_stress_ci_high": es_hi,
                     "event_dry_share": e_dry.mean() if len(e_dry) else np.nan,
                     "event_dry_ci_low": ed_lo, "event_dry_ci_high": ed_hi})
    out = pd.DataFrame(rows)
    out.to_csv(_p("tables") / f"link_grid_{tag}_{split}.csv", index=False)
    return out


# ----------------------------------------------------------------------------- 8.4 logistic regression, H1

def _center(f: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    g = f.copy()
    g["Lc"] = g["L"] - ref["L"].mean()
    g["lnRc"] = g["lnR"] - ref["lnR"].mean()
    g["LxlnR"] = g["Lc"] * g["lnRc"]
    return g


def logit_models(f: pd.DataFrame, years: list[int], y: str = "stress") -> dict:
    per = load("link")["logit"]["or_per_m"]
    d = f[f.season.isin(years)].dropna(subset=[y, "L", "lnR"])
    d = _center(d, d)
    xs = {"M0": ["lnRc"], "M1": ["lnRc", "Lc"], "M2": ["lnRc", "Lc", "LxlnR"]}
    cl = {k: st.fit_logit(d, y, v, cluster="day") for k, v in xs.items()}
    un = {k: st.fit_logit(d, y, v, cluster=None) for k, v in xs.items()}
    res = {"n_hours": int(len(d)), "n_days": int(d["day"].nunique()), "positives": int(d[y].sum()),
           "years": years, "centering": {"L_mean": float(d["L"].mean()), "lnR_mean": float(d["lnR"].mean())},
           "or_L_per_0.1m_M1": st.odds_ratio(cl["M1"], "Lc", per),
           "or_L_per_0.1m_M2_at_mean_rain": st.odds_ratio(cl["M2"], "Lc", per),
           "interaction_M2": st.odds_ratio(cl["M2"], "LxlnR", 1.0),
           "or_lnR_M0": st.odds_ratio(cl["M0"], "lnRc", 1.0), "or_lnR_M1": st.odds_ratio(cl["M1"], "lnRc", 1.0),
           "lr_M0_vs_M1_unclustered": st.lr_test(un["M0"], un["M1"]),
           "lr_M1_vs_M2_unclustered": st.lr_test(un["M1"], un["M2"]),
           "aic": {k: float(v.aic) for k, v in un.items()}}
    res["_fits"] = {"cl": cl, "ref": d}
    return res


def auc_gain_2023(f: pd.DataFrame, fits: dict) -> dict:
    cfg = load("link")
    ref = fits["ref"]
    v = f[f.split.eq("val")].dropna(subset=["stress", "L", "lnR"])
    v = _center(v, ref)   # centred with TRAINING means
    p0 = st.predict(fits["cl"]["M0"], v, ["lnRc"])
    p1 = st.predict(fits["cl"]["M1"], v, ["lnRc", "Lc"])
    return st.auc_gain_day_bootstrap(v["stress"].to_numpy().astype(int), p0, p1, v["day"].to_numpy(),
                                     cfg["bootstrap_reps"], cfg["bootstrap_seed"])


def h1_verdict(or_res: dict, auc: dict) -> dict:
    or_ok = or_res["ci_low"] > 1 or or_res["ci_high"] < 1
    auc_ok = auc["gain_ci_low"] > 0 or auc["gain_ci_high"] < 0
    verdict = "supported" if (or_ok and auc_ok) else ("partly supported" if (or_ok or auc_ok) else "not supported")
    return {"or_ci_excludes_1": bool(or_ok), "or_direction": "positive" if or_res["or"] > 1 else "negative",
            "auc_gain_ci_excludes_0": bool(auc_ok), "auc_gain_direction": "positive" if auc["gain"] > 0 else "negative",
            "verdict": verdict}


# ----------------------------------------------------------------------------- Quiapo secondary check (D18)

def quiapo_check(inp: dict, g20_passed: bool, fields_q: dict, alert: dict | None = None) -> dict:
    tr = train_years()
    conf = load("link")["quiapo_g20"]["confirmed_years"]
    statuses = ("use", "pending_G20") if g20_passed else ("use",)
    q = station_frame(inp["pump"], "quiapo", statuses)
    f = analysis_frame(inp, q, fields_q["pumps_running"], fields_q["gate_open"])
    out = {"fields_used": fields_q, "g20_passed": g20_passed}
    if not fields_q["pumps_running"]:
        out["note"] = "Quiapo pump field dropped by G4: no Quiapo stress label"
        return out
    sets = {"without_2021": [y for y in conf if y in tr]}
    if g20_passed:
        sets["with_2021"] = sorted(sets["without_2021"] + [2021])
    for name, yrs in sets.items():
        r = logit_models(f, yrs)
        r.pop("_fits")
        out[name] = r
    # Descriptive only (decision (a), 6 Nov 2026): Quiapo 2023 sheet
    q23 = station_frame(inp["pump"], "quiapo", ("descriptive_only",))
    f23 = analysis_frame(inp, q23, fields_q["pumps_running"], fields_q["gate_open"])
    v = f23[f23.split.eq("val")].dropna(subset=["stress", "L"])
    d = {"n_hours_with_status": int(len(v)), "stress_hours": int(v["stress"].sum()),
         "note": "decision (a): descriptive only, not part of the H1 verdict"}
    if alert:
        for k, lvl in alert.items():
            h = v[v.L >= lvl]
            d[f"stress_share_at_or_above_{k}_{lvl}"] = (float(h["stress"].mean()) if len(h) else None, int(len(h)))
    out["quiapo_2023_descriptive"] = d
    return out


# ----------------------------------------------------------------------------- 8.5 alert levels, G8

def alert_levels(f: pd.DataFrame) -> dict:
    """G8 on training seasons only. f: analysis frame with L (observed), stress, split."""
    cfg = load("link")["alert"]
    tr = f[f.split.eq("train")]
    hw = round(float(tr["L"].dropna().quantile(cfg["high_water_quantile"])), 2)
    table = []
    fr = None
    for L in grid_levels():
        if L < hw:
            continue
        s = tr.loc[tr.L >= L, "stress"].dropna()
        share = float(s.mean()) if len(s) else np.nan
        table.append((float(L), int(len(s)), share))
        if fr is None and len(s) and share >= cfg["flood_risk_cutoff"]:
            fr = float(L)
    return {"high_water_m": hw, "flood_risk_m": fr, "scan": table}


def alert_sensitivity(inp: dict, station_df: pd.DataFrame) -> pd.DataFrame:
    cfg = load("link")["alert"]["sensitivity"]
    f = analysis_frame(inp, station_df, True, True)
    tr = f[f.split.eq("train")]
    rows = []
    for k in cfg["min_pumps"]:
        stress = (tr["pumps"] >= k).astype(float).where(tr["pumps"].notna())
        for L in cfg["levels_m"]:
            s = stress[tr.L >= L].dropna()
            rows.append({"min_pumps": k, "level_m": L, "hours": len(s), "stress_share": s.mean() if len(s) else np.nan})
        for c in cfg["cutoffs"]:
            lvl = next((float(L) for L in grid_levels() if L >= 11.69 and len(stress[tr.L >= L].dropna())
                        and stress[tr.L >= L].dropna().mean() >= c), None)
            rows.append({"min_pumps": k, "cutoff": c, "derived_flood_risk_level_m": lvl})
    out = pd.DataFrame(rows)
    out.to_csv(_p("tables") / "alert_sensitivity.csv", index=False)
    return out


def check_2023(f: pd.DataFrame, levels: dict) -> dict:
    v = f[f.split.eq("val") & f.L.notna()]
    out = {}
    for k, lvl in levels.items():
        ex = v[v.L >= lvl]
        s = ex["stress"].dropna()
        out[k] = {"level_m": lvl, "exceedance_rate": float(len(ex) / len(v)), "hours": int(len(ex)),
                  "dry_alert_share": float(ex.loc[ex.R24.notna(), "dry"].mean()) if len(ex) else None,
                  "stress_share_at_or_above": float(s.mean()) if len(s) else None, "hours_with_status": int(len(s))}
    return out


# ----------------------------------------------------------------------------- run

def run() -> dict:
    require_gates("G6")
    inp = load_inputs()
    leg = step_legend(inp)
    g20 = step_g20(inp)
    av = station_frame(inp["pump"], "aviles")
    fa = leg["fields"]["aviles"]
    f = analysis_frame(inp, av, fa["pumps_running"], fa["gate_open"])
    first = step_first_assumption(f)
    if not fa["pumps_running"]:
        raise GateFailure("G4 dropped the Aviles pump field: the stress label cannot be built (H1 untestable)")
    grid = step_grid(f, "train", "aviles")
    grid23 = step_grid(f, "val", "aviles")
    lm = logit_models(f, train_years())
    fits = lm.pop("_fits")
    auc = auc_gain_2023(f, fits)
    h1 = h1_verdict(lm["or_L_per_0.1m_M1"], auc)
    record_gate("G7", "h1_result", True,
                "H1 supported if the adjusted OR (per 0.1 m, M1, day-clustered SE) 95% CI excludes 1 AND the 2023 ROC-AUC "
                "gain (M1 vs M0) day-block bootstrap 95% CI excludes 0; partly supported if one holds; not supported if neither",
                {"logit": lm, "auc_2023": auc, "rule_outcome": h1}, decision=h1["verdict"])
    al = alert_levels(f)
    sens = alert_sensitivity(inp, av)
    levels = {"high_water": al["high_water_m"], "flood_risk": al["flood_risk_m"]}
    chk = check_2023(f, levels)
    qc = quiapo_check(inp, g20["passed"], leg["fields"]["quiapo"], levels)
    record_gate("G8", "alert_levels", True,
                "High water = P95 of training Pandacan (observed, unfilled, unflagged), rounded 0.01 m; flood risk = "
                "lowest 0.05 m grid level (>= high water) where >= 50% of training hours at/above are stress hours "
                "(>= 2 Aviles pumps)",
                {"levels": levels, "scan": al["scan"], "check_2023_recorded_not_used": chk},
                decision=f"high water {levels['high_water']:.2f} m; flood risk {levels['flood_risk']:.2f} m")
    config.write_yaml("alert_levels", {
        "high_water_m": levels["high_water"], "flood_risk_m": levels["flood_risk"],
        "rule": {"high_water": "P95 of training Pandacan (observed, unfilled, unflagged), rounded to 0.01 m",
                 "flood_risk": "lowest 0.05 m grid level >= high water where >= 50% of training hours at/above "
                               "have >= 2 Aviles pumps running"},
        "training_seasons": train_years(), "git_sha": git_sha(), "date": dt.date.today().isoformat(),
        "check_2023": chk,
    }, header="# GENERATED by G8 (NB02). Frozen: later phases read these levels; never edit by hand.")
    plot_grid(grid, levels, _p("figures") / "fig_link_stress_by_level.png")
    summary = {"legend_fields": leg["fields"], "g20": g20["conditions"] | {"passed": g20["passed"]},
               "first_assumption": first.to_dict(orient="records"), "logit": lm, "auc_2023": auc, "h1": h1,
               "alert_levels": levels, "check_2023": chk, "quiapo": qc}
    (_p("tables") / "phase2_summary.json").write_text(json.dumps(summary, indent=2, default=_js))
    return {"inputs": inp, "legend": leg, "g20": g20, "frame": f, "first": first, "grid": grid, "grid_2023": grid23,
            "logit": lm, "auc": auc, "h1": h1, "alert": al, "sensitivity": sens, "check_2023": chk, "quiapo": qc}


def plot_grid(grid: pd.DataFrame, levels: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = grid[grid.hours_with_status >= 10]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
    ax.fill_between(g.level_m, g.stress_ci_low * 100, g.stress_ci_high * 100, color="#9ecae1", alpha=0.6,
                    label="hours: 95% Wilson CI")
    ax.plot(g.level_m, g.stress_share * 100, color="#08519c", lw=1.8, label="hours at/above level")
    ax.plot(g.level_m, g.event_stress_share * 100, color="#d95f0e", lw=1.2, ls="--", label="events with a stress hour")
    for k, lvl in levels.items():
        ax.axvline(lvl, color="0.3", lw=0.8, ls=":")
        ax.text(lvl - 0.01, 97, f"{k.replace('_', ' ')} {lvl:.2f} m", rotation=90, va="top", ha="right", fontsize=7,
                color="0.25")
    ax.axhline(50, color="0.6", lw=0.6)
    ax.set_xlabel("Pandacan level L (m)")
    ax.set_ylabel("share with >= 2 Aviles pumps (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Drainage stress at or above each Pandacan level, training wet seasons 2018-2022", fontsize=9)
    ax.legend(fontsize=7, loc="center left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


__all__ = ["run", "load_inputs", "step_legend", "step_g20", "analysis_frame", "station_frame", "step_first_assumption",
           "step_grid", "logit_models", "auc_gain_2023", "h1_verdict", "alert_levels", "alert_sensitivity",
           "check_2023", "quiapo_check", "sps"]
