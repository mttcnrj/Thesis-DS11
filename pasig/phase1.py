"""Phase 1 / Notebook 01: load and quality-check (Section 7). Gates G1, G2, G3, G5, G6.

Each step is a function so the notebook can call and display them one by one; `run()` executes all
of them in order (scripts/run_phase.py --phase 1).
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pasig import config
from pasig.config import load, path
from pasig.gates import GateFailure, read_gate, record_gate
from pasig.io import manifest as mf
from pasig.io import mmda_pumping as mp
from pasig.io import mmda_waterlevel as mw
from pasig.io import pagasa as pg
from pasig.qc.coverage import coverage_table, eligibility
from pasig.qc.gaps import fill_short_gaps
from pasig.qc.rules import apply_qc

TARGET = "pandacan"
CANDIDATES = ("sto_nino", "fort_santiago")
PRELIM_LEVELS = {"high_water": 11.69, "flood_risk": 12.0}   # Section 0.3; re-derived and frozen by G8 (NB02)


def _p(key: str) -> Path:
    p = path(key)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _season_index(years) -> pd.DatetimeIndex:
    parts = [pd.date_range(f"{y}-{min(config.wet_months()):02d}-01", f"{y}-{max(config.wet_months()):02d}-30 23:00", freq="h")
             for y in years]
    idx = parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]
    return idx[idx.month.isin(config.wet_months())]


def all_years() -> list[int]:
    s = load("splits")
    return sorted(s["train_years"] + s["val_years"] + s["test_years"])


# ----------------------------------------------------------------------------- 1.1 manifest, G1

def step_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    man, bl = mf.build_manifest(path("raw"))
    mdir = _p("manifest")
    man.drop(columns=["dup_group"]).to_csv(mdir / "raw_manifest.csv", index=False)
    bl.to_csv(mdir / "pumping_blocks.csv", index=False)
    ok, ev = mf.g1_check(man, bl)
    record_gate("G1", "raw_files", ok,
                "Every raw file has a SHA-256 and a status (use/exclude/pending_G20); every identical-hash group has "
                "exactly one non-excluded copy; Taguig files excluded; the single Quiapo copy dated by sheet headers; "
                "Quiapo 2021 data marked pending_G20", ev)
    if not ok:
        raise GateFailure(f"G1 failed: {ev['problems']}")
    return man, bl


# ----------------------------------------------------------------------------- 1.2 EFCOS, G2 part

def rollover_selftest() -> dict:
    """Build a tiny FORM 2-1 workbook (30-day month with a day-31 template row) and parse it."""
    import datetime as dt

    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "PANDACAN"
    ws.append(["FORM 2-1  HOURLY  WATERLEVEL CHART"])
    ws.append(["JUNE"])
    ws.append(["LOCATION :", "PANDACAN", "MONTH :", "JUNE", "YEAR :", 2021])
    ws.append([])
    ws.append([None] + [dt.time((9 + i) % 24, 0) for i in range(24)] + ["AVE.", "DATE", "HIGHEST", "LOWEST"])
    for d in range(1, 32):
        vals = [11.0 + d / 100 + i / 1000 for i in range(24)]
        ws.append([d] + vals + [None, d, max(vals), min(vals)])
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "6. June.xlsx"
        wb.save(f)
        res = mw.parse_file(f, ("MMDA Dataset", "Waterlevel Data 2018-2024", "Pandacan 2018-2024", "2021", "6. June.xlsx"),
                            "pandacan", "Pandacan")
    d = res["data"].set_index("time_pht")["value_m"]
    checks = {
        "day30_col15_is_1jul_0000": bool(np.isclose(d.get(pd.Timestamp("2021-07-01 00:00"), np.nan), 11.30 + 0.015)),
        "day1_col0_is_1jun_0900": bool(np.isclose(d.get(pd.Timestamp("2021-06-01 09:00"), np.nan), 11.01)),
        "day31_set_aside": bool(len(res["set_aside"]) == 24 and (res["set_aside"]["day"] == 31).all()),
        "no_hl_mismatch": bool(not res["hl"]["mismatch"].any()),
    }
    checks["passed"] = all(checks.values())
    return checks


def step_waterlevel(man: pd.DataFrame) -> dict:
    use = set(man.loc[man.source.eq("mmda_efcos") & man.status.eq("use"), "path"])
    res = mw.parse_all(path("raw"), use)
    long, hl, files = res["long"], res["hl"], res["files"]
    logs = _p("logs")
    res["set_aside"].to_csv(logs / "impossible_dates.csv", index=False)
    files.to_csv(logs / "efcos_files.csv", index=False)
    tok = long[long.raw_token.notna()].groupby(["station", "raw_token"]).size().rename("n").reset_index()
    tok.sort_values("n", ascending=False).to_csv(logs / "efcos_missing_tokens.csv", index=False)
    # HIGHEST/LOWEST cross-check (G2): per-file day-level mismatch rate, formula-aware diagnosis above 1%
    comp = hl[hl.comparable]
    per = comp.groupby("file").agg(days=("mismatch", "size"), mismatches=("mismatch", "sum")).reset_index()
    per["rate"] = per.mismatches / per.days
    diag = []
    for f in per.loc[per.rate > 0.01, "file"]:
        diag.append(mw.diagnose_hl(path("raw") / f, hl[hl.file.eq(f)], long[long.source_file.eq(f)]))
    diag = pd.concat(diag, ignore_index=True) if diag else pd.DataFrame(columns=["file", "explained_by_formula"])
    expl = diag[diag.explained_by_formula].groupby("file").size() if len(diag) else pd.Series(dtype=int)
    per["explained_by_formula"] = per.file.map(expl).fillna(0).astype(int)
    per["rate_unexplained"] = (per.mismatches - per.explained_by_formula) / per.days
    mm = hl[hl.mismatch].merge(diag, on=["file", "date"], how="left") if len(diag) else hl[hl.mismatch]
    mm.to_csv(logs / "highest_lowest_mismatch.csv", index=False)
    per.to_csv(logs / "highest_lowest_by_file.csv", index=False)
    out_dir = _p("interim")
    long.drop(columns=["col"]).to_parquet(out_dir / "waterlevel_long.parquet", index=False)
    return {"long": long, "hl_by_file": per, "hl_diag": diag, "files": files, "set_aside": res["set_aside"]}


# ----------------------------------------------------------------------------- 1.3 pumping

def step_pumping(bl: pd.DataFrame) -> dict:
    rng = tuple(load("stations")["pumping_value_range_m"])
    res = mp.parse_from_blocks(path("raw"), bl, statuses=mp.LIVE_BLOCK_STATUS, value_range=rng)
    long = res["long"]
    logs = _p("logs")
    res["log"].to_csv(logs / "pumping_blocks_skipped.csv", index=False)
    res["out_of_range"].to_csv(logs / "pumping_out_of_range.csv", index=False)
    keep = ["time_pht", "station", "inside_m", "outside_m", "pumps_running", "gate_open", "sky", "raw_status",
            "status_parsed", "inside_token", "outside_token", "inside_m_range_flag", "outside_m_range_flag",
            "flag_impossible_date", "flag_duplicate_time", "block_status", "block_year", "source_file", "sheet",
            "block_header"]
    long = long[keep].copy()
    long["gate_open"] = long["gate_open"].astype("float")   # True/False/NaN -> 1/0/NaN
    long.to_parquet(_p("interim") / "pumping_long.parquet", index=False)
    av = long[long.station.eq("aviles")]
    n_av = int(av[["inside_m", "outside_m"]].notna().sum().sum())
    n_bad = int(av[["inside_m_range_flag", "outside_m_range_flag"]].sum().sum())
    return {"long": long, "aviles_in_range_share": n_av / max(n_av + n_bad, 1), "aviles_out_of_range": n_bad,
            "quiapo_out_of_range": int(long[long.station.eq("quiapo")][["inside_m_range_flag", "outside_m_range_flag"]].sum().sum()),
            "skipped_blocks": res["log"]}


# ----------------------------------------------------------------------------- 1.4 PAGASA, G3

def step_pagasa() -> dict:
    raw = path("raw") / "PAGASA Dataset"
    six = pg.read_6hourly(raw / "6-Hourly Data.csv")
    daily = pg.read_daily(raw / "Science Garden Daily Data.csv")
    ev = pg.g3_convention_test(six, daily)
    table = ev.pop("table")
    table.to_csv(_p("logs") / "g3_rain_convention.csv", index=False)
    conv = ev["convention"]
    rule = ("For rain days compare the daily total with (a) the sum of the 00/06/12/18 UTC reports of that date and "
            "(b) 06/12/18 UTC + 00 UTC of the next date; convention = option matching >= 95% of rain days "
            "(+-0.2 mm); usable time = period end + 1 h")
    if conv is None:
        record_gate("G3", "rain_timing", False, rule, ev, decision=None)
        raise GateFailure("G3: neither convention matches >= 95% of rain days")
    record_gate("G3", "rain_timing", True, rule, ev, decision=conv)
    config.write_yaml("pagasa", {
        "convention": conv,
        "meaning": "6-hourly stamp marks the END of its 6 h period" if conv == "period_end" else "stamp marks the START",
        "available_after_period_end_h": 1, "trace_mm": pg.TRACE_MM, "missing_code": -999,
        "daily_available_from": "09:00 PHT the day after its label",
        "evidence": {k: v for k, v in ev.items() if k != "convention"},
    }, header="# GENERATED by G3 (NB01 step 1.4). Do not edit by hand.")
    six = pg.add_availability(six, conv)
    six.to_parquet(_p("interim") / "pagasa_6h.parquet", index=False)
    daily.to_parquet(_p("interim") / "pagasa_daily.parquet", index=False)
    return {"six": six, "daily": daily, "g3": ev}


def g2_gate(wl: dict, pump: dict, selftest: dict) -> bool:
    per = wl["hl_by_file"]
    bad = per[per.rate_unexplained > 0.01]
    ev = {
        "rollover_selftest": selftest,
        "impossible_date_rows_logged": int(wl["set_aside"][["station", "year", "month", "day"]].drop_duplicates().shape[0]) if len(wl["set_aside"]) else 0,
        "impossible_dates_log": "reports/logs/impossible_dates.csv",
        "hl_files_checked": int(len(per)), "hl_days_checked": int(per.days.sum()),
        "hl_files_raw_rate_above_1pct": per.loc[per.rate > 0.01, ["file", "days", "mismatches", "explained_by_formula"]].values.tolist(),
        "hl_files_unexplained_rate_above_1pct": bad.file.tolist(),
        "hl_note": "Mismatches above 1% were all traced to HIGHEST/LOWEST formulas that aggregate another day's row "
                   "(source-sheet template error); the readings are in day order (continuity checked). A parser column "
                   "shift would leave unexplained mismatches.",
        "efcos_files_set_aside": wl["files"].loc[wl["files"].status.ne("ok"), ["file", "reason"]].values.tolist(),
        "duplicate_timestamps": int(wl["long"].flag_duplicate_time.sum()),
        "aviles_in_range_share_after_cm_to_m": round(pump["aviles_in_range_share"], 5),
        "aviles_values_out_of_9_14_m": pump["aviles_out_of_range"],
        "quiapo_values_out_of_9_14_m": pump["quiapo_out_of_range"],
    }
    ok = (selftest["passed"] and bad.empty and ev["duplicate_timestamps"] == 0
          and pump["aviles_in_range_share"] >= 0.99)
    record_gate("G2", "parsing", ok,
                "09:00 rollover test passes; impossible-date rows logged; HIGHEST/LOWEST mismatch <= 1% of complete days "
                "per file (after tracing misreferenced sheet formulas); Aviles values in 9-14 m after /100 "
                "(>= 99% of readings; out-of-range readings set to NaN and logged)", ev)
    if not ok:
        raise GateFailure("G2 failed: see gates/G02_parsing.json")
    return ok


# ----------------------------------------------------------------------------- 1.5-1.6 QC and gap fill

def step_qc(long: pd.DataFrame, pump_long: pd.DataFrame, six: pd.DataFrame) -> dict:
    H, runs = apply_qc(long)
    H_noex, _ = apply_qc(long[long.station.eq("sto_nino") | long.station.eq("montalban")], flat_exemption=False)
    logs, tables = _p("logs"), _p("tables")
    rem = H[H.qc_flag.ne("")].copy()
    rem["season"] = rem.time_pht.dt.year
    rem["wet"] = rem.time_pht.dt.month.isin(config.wet_months())
    rem["split"] = rem.season.map(config.split_of_year)
    rem.to_csv(logs / "qc_removals.csv", index=False)
    H.to_parquet(_p("interim") / "waterlevel_qc.parquet", index=False)   # all months, QC flags + value_qc (tide fit, D7)
    runs.to_csv(logs / "qc_flat_runs.csv", index=False)
    summary = (rem[rem.wet].groupby(["station", "split", "qc_flag"]).size().rename("hours").reset_index())
    summary.to_csv(tables / "qc_summary_wet.csv", index=False)

    # Hand check (Section 7.5): every flagged training Pandacan value > 11.69 m vs neighbours, Aviles river side, rain
    W = H.pivot(index="time_pht", columns="station", values="value_m")
    av = pump_long[pump_long.station.eq("aviles")].drop_duplicates("time_pht").set_index("time_pht")["outside_m"]
    rain = pg.hourly_rain_features(six, W.index)["sg_rain_24h"]
    p = H[H.station.eq(TARGET)].set_index("time_pht")
    tr = p.index.year.isin(load("splits")["train_years"]) & p.index.month.isin(config.wet_months())
    hc = p[tr & p.qc_flag.ne("") & (p.value_m > PRELIM_LEVELS["high_water"])].copy()
    for k in (-1, 1):
        hc[f"pandacan_t{k:+d}"] = W[TARGET].reindex(hc.index + pd.Timedelta(hours=k)).to_numpy()
    hc["fort_santiago_t"] = W.get("fort_santiago", pd.Series(dtype=float)).reindex(hc.index).to_numpy()
    hc["aviles_outside_t"] = av.reindex(hc.index).to_numpy()
    hc["sg_rain_24h"] = rain.reindex(hc.index).to_numpy()
    nb_max = hc[["pandacan_t-1", "pandacan_t+1"]].max(axis=1)
    hc["neighbour_support"] = ((hc["aviles_outside_t"] >= hc["value_m"] - 0.3) |
                               (nb_max >= hc["value_m"] - 0.1) |
                               (hc["qc_flag"].str.startswith("flat")))
    hc["verdict"] = np.where(hc.qc_flag.eq("spike"),
                             np.where(hc.neighbour_support, "CHECK: neighbour supports the value",
                                      "isolated one-hour jump not seen at neighbours/Aviles: removed"),
                             "stuck flat run at 11.89 m (sensor hold after a peak): removed")
    hc.reset_index().to_csv(tables / "qc_handcheck_pandacan_train.csv", index=False)
    return {"H": H, "H_sto_nino_no_exemption": H_noex, "removals": rem, "runs": runs, "handcheck": hc}


def hourly_master(H: pd.DataFrame, six: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    gap = load("qc")["gap_fill_max_h"]
    idx = _season_index(all_years())
    W = H.pivot(index="time_pht", columns="station", values="value_qc").reindex(idx)
    F = H.pivot(index="time_pht", columns="station", values="qc_flag").reindex(idx)
    out = pd.DataFrame(index=idx)
    out.index.name = "time_pht"
    out["season"] = idx.year
    out["split"] = out["season"].map(config.split_of_year)
    for st in W.columns:
        vals, filled = [], []
        for y in all_years():
            s = W.loc[W.index.year == y, st]
            f, m = fill_short_gaps(s, gap)
            vals.append(f)
            filled.append(m)
        out[st] = pd.concat(vals)
        out[f"{st}_filled"] = pd.concat(filled).astype(bool)
        out[f"{st}_qcflag"] = F[st].fillna("")
        out.loc[out[f"{st}_qcflag"].eq("") & W[st].isna() & ~out[f"{st}_filled"], f"{st}_qcflag"] = "missing"
    rain = pg.hourly_rain_features(six, idx)
    out = out.join(rain)
    # previous-day Science Garden weather, usable from 09:00 PHT the day after its label
    d = daily.sort_values("available_from_pht")[["available_from_pht", "rh", "atm_pressure", "tmean", "rain_day_mm"]]
    m = pd.merge_asof(pd.DataFrame({"time_pht": idx}), d, left_on="time_pht", right_on="available_from_pht",
                      direction="backward")
    stale = (m["time_pht"] - m["available_from_pht"]) >= pd.Timedelta(hours=24)
    for c in ["rh", "atm_pressure", "tmean", "rain_day_mm"]:
        out[f"sg_daily_{c}"] = m[c].mask(stale).to_numpy()
    return out


# ----------------------------------------------------------------------------- 1.8 coverage, G5

def step_coverage(H: pd.DataFrame, H_noex: pd.DataFrame) -> dict:
    tables, figs = _p("tables"), _p("figures")
    years = all_years()
    idx = _season_index(years)
    full = []
    for st, g in H.groupby("station"):
        s = g.set_index("time_pht")["value_qc"].reindex(idx)
        full.append(pd.DataFrame({"station": st, "value": s.to_numpy(), "season": idx.year}))
    full = pd.concat(full)
    cov = coverage_table(full, "value", "season")
    cov.to_csv(tables / "coverage_wet_season.csv")
    # Sto. Niño without the 11.8-12.5 m exemption
    s = H_noex[H_noex.station.eq("sto_nino")].set_index("time_pht")["value_qc"].reindex(idx)
    cov_noex = pd.DataFrame({"value": s.to_numpy(), "season": idx.year, "station": "sto_nino (no exemption)"})
    cov_noex = coverage_table(cov_noex, "value", "season")
    cov_noex.to_csv(tables / "coverage_wet_season_sto_nino_no_exemption.csv")
    # per calendar year (all months)
    cal = []
    for st, g in H.groupby("station"):
        s = g.set_index("time_pht")["value_qc"]
        s = s[(s.index.year >= min(years)) & (s.index.year <= max(years))]
        full_idx = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31 23:00", freq="h")
        s = s.reindex(full_idx)
        cal.append(pd.DataFrame({"station": st, "value": s.to_numpy(), "year": full_idx.year}))
    cov_cal = coverage_table(pd.concat(cal), "value", "year")
    cov_cal.to_csv(tables / "coverage_calendar_year.csv")
    el = eligibility(cov, years, 0.80)
    el.to_csv(tables / "eligibility.csv")
    cand = {c: {"eligible": bool(el.loc[c, "eligible"]), "min_season_coverage": round(float(el.loc[c, "min_season_coverage"]), 4),
                "seasons_below_80pct": el.loc[c, "seasons_below"]} for c in CANDIDATES}
    eligible = [c for c in CANDIDATES if cand[c]["eligible"]]
    config.write_yaml("eligible_inputs", {
        "rule": "station eligible iff >= 80% usable (unflagged, before fill) hours in every wet season 2018-2024 (D2)",
        "eligible_stations": eligible, "candidates": cand,
        "descriptive_only": [c for c in CANDIDATES if c not in eligible],
        "note": "Ineligible stations still go through the Granger procedure on their available training hours, "
                "reported as descriptive results, never used as inputs (D2). No missing-indicator variant.",
    }, header="# GENERATED by G5 (NB01 step 1.8). Do not edit by hand.")
    record_gate("G5", "eligibility", True, "Station eligible iff >= 80% usable hours in every season 2018-2024",
                {"coverage_wet_season": cov.round(4).to_dict(orient="index"),
                 "sto_nino_without_exemption": cov_noex.round(4).to_dict(orient="index"),
                 "candidates": cand}, decision=("eligible: " + ", ".join(eligible)) if eligible else "no eligible upstream input")
    _plot_coverage(cov, cov_noex, figs / "fig_III-3_coverage_wet_season.png")
    return {"coverage": cov, "coverage_no_exemption": cov_noex, "coverage_calendar": cov_cal,
            "eligibility": el, "eligible": eligible}


def _plot_coverage(cov: pd.DataFrame, cov_noex: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = load("stations")["efcos"]
    order = ["pandacan", "fort_santiago", "sto_nino", "montalban", "rosario_js", "rosario_ls"]
    order = [o for o in order if o in cov.index]
    data = cov.loc[order] * 100
    names = [labels[o]["label"] for o in order]
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=150)
    im = ax.imshow(data.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(data.shape[1]), [str(c) for c in data.columns])
    ax.set_yticks(range(len(names)), names)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.iat[i, j]
            ax.text(j, i, "" if np.isnan(v) else f"{v:.0f}", ha="center", va="center",
                    color="white" if v >= 60 else "black", fontsize=8)
    ax.set_xlabel("Wet season (June-November)")
    ax.set_title("Share of usable hourly readings per wet season (%), after QC, before gap fill", fontsize=9)
    fig.colorbar(im, ax=ax, label="%", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- 1.9 evaluation hours

def eval_hours(master: pd.DataFrame, eligible: list[str]) -> pd.DataFrame:
    fc = load("features")
    win, leads = fc["window_h"], fc["leads_h"]
    out = pd.DataFrame(index=master.index)
    out["season"] = master["season"]
    out["split"] = master["split"]
    pos_in_season = master.groupby("season").cumcount().to_numpy()
    window_ok = np.ones(len(master), dtype=bool)
    for st in [TARGET] + list(eligible):
        have = master[st].notna().astype(int)
        full = have.groupby(master["season"]).transform(lambda x: x.rolling(win, min_periods=win).sum()).to_numpy()
        window_ok &= (full == win) & (pos_in_season >= win - 1)
    window_ok &= master["sg_report_ok"].to_numpy()
    obs = master[TARGET].notna() & ~master[f"{TARGET}_filled"]
    for h in leads:
        tgt = obs.groupby(master["season"]).shift(-h).fillna(False).astype(bool).to_numpy()
        out[f"lead_{h}"] = window_ok & tgt
    out.index.name = "time_pht"
    return out


def table_iii6(master: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    obs = master[TARGET].where(~master[f"{TARGET}_filled"])
    rows = []
    for sp in ["train", "val", "test"]:
        m = master["split"].eq(sp)
        o = obs[m]
        r = {"split": sp, "seasons": ",".join(map(str, sorted(master.loc[m, "season"].unique()))),
             "hours_in_season": int(m.sum()),
             "observed_unflagged_hours": int(o.notna().sum()),
             "observed_or_filled_hours": int(master.loc[m, TARGET].notna().sum()),
             "cases_lead1": int(ev.loc[m, "lead_1"].sum()), "cases_lead6": int(ev.loc[m, "lead_6"].sum())}
        for name, lvl in PRELIM_LEVELS.items():
            r[f"{name}_hours_ge_{lvl}"] = int((o >= lvl).sum())
            r[f"{name}_share"] = round(float((o >= lvl).sum() / max(o.notna().sum(), 1)), 4)
            r[f"{name}_events"] = count_events(o, lvl)
        rows.append(r)
    return pd.DataFrame(rows)


def count_events(s: pd.Series, level: float, merge_gap_h: int = 6) -> int:
    """Events = runs of hours >= level; runs separated by < merge_gap_h hours are merged (Section 8.3)."""
    t = s.index[(s >= level).to_numpy()]
    if len(t) == 0:
        return 0
    gaps = np.diff(t.to_numpy()) / np.timedelta64(1, "h")
    return int(1 + (gaps >= merge_gap_h).sum())


# ----------------------------------------------------------------------------- 1.10 G6

def g6_gate(test_result: int | None, master_path: Path, ev_path: Path) -> bool:
    have = {g: read_gate(g) is not None and read_gate(g)["passed"] for g in ("G1", "G2", "G3", "G5")}
    ok = all(have.values()) and (test_result == 0) and ev_path.exists() and master_path.exists()
    record_gate("G6", "phase1_frozen", ok,
                "G1-G3, G5 written and passed; Phase 1 tests (test_parsers, test_time_alignment, test_qc, test_splits, "
                "test_gapfill, test_gates) pass; eval_hours.parquet written",
                {"gates": have, "pytest_exit_code": test_result, "eval_hours": str(ev_path.relative_to(config.root())),
                 "hourly_master": str(master_path.relative_to(config.root()))},
                decision="tag phase1-frozen" if ok else None)
    return ok


def run(run_tests: bool = True) -> dict:
    out: dict = {}
    man, bl = step_manifest()
    selftest = rollover_selftest()
    wl = step_waterlevel(man)
    pump = step_pumping(bl)
    g2_gate(wl, pump, selftest)
    pag = step_pagasa()
    qc = step_qc(wl["long"], pump["long"], pag["six"])
    cov = step_coverage(qc["H"], qc["H_sto_nino_no_exemption"])
    master = hourly_master(qc["H"], pag["six"], pag["daily"])
    proc = _p("processed")
    master.to_parquet(proc / "hourly_master.parquet")
    ev = eval_hours(master, cov["eligible"])
    ev.to_parquet(proc / "eval_hours.parquet")
    t6 = table_iii6(master, ev)
    t6.to_csv(_p("tables") / "table_III-6_recomputed.csv", index=False)
    qc_meta = {
        "pandacan_train_wet_removed_by_rule": qc["removals"].query("station=='pandacan' and wet and split=='train'")
        .qc_flag.value_counts().to_dict(),
        "pandacan_p95_train_observed": round(float(master.loc[master.split.eq("train"), TARGET]
                                                   .where(~master[f"{TARGET}_filled"]).quantile(0.95)), 2),
        "pandacan_max_test_observed": float(master.loc[master.split.eq("test"), TARGET].max()),
        "pandacan_max_test_time": str(master.loc[master.split.eq("test"), TARGET].idxmax()),
        "pandacan_values_gt_11_69_flagged_2023_2024": qc["removals"].query(
            "station=='pandacan' and wet and split in ['val','test'] and value_m > 11.69")[["time_pht", "value_m", "qc_flag"]]
        .astype(str).values.tolist(),
    }
    (_p("tables") / "qc_meta.json").write_text(json.dumps(qc_meta, indent=2, default=str))
    out.update(manifest=man, blocks=bl, waterlevel=wl, pumping=pump, pagasa=pag, qc=qc, coverage=cov,
               master=master, eval_hours=ev, table_iii6=t6, qc_meta=qc_meta)
    if run_tests:
        import subprocess
        import sys
        tests = ["tests/test_parsers.py", "tests/test_time_alignment.py", "tests/test_qc.py", "tests/test_splits.py",
                 "tests/test_gapfill.py", "tests/test_gates.py"]
        code = subprocess.call([sys.executable, "-m", "pytest", "-q", *tests], cwd=config.root())
        out["g6"] = g6_gate(code, proc / "hourly_master.parquet", proc / "eval_hours.parquet")
    return out


def eval_hours_summary(ev: pd.DataFrame) -> pd.DataFrame:
    return ev.groupby("split")[[c for c in ev.columns if c.startswith("lead_")]].sum()


__all__ = [n for n in dir() if re.match(r"^(step_|g\d|run|eval_|table_|hourly_|rollover|count_)", n)]
