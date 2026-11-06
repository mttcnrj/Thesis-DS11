"""Phase 3 / Notebook 03: tide (D7, G9), features, windows, split, scaling (Section 9.1).

Builds TRAIN and VALIDATION windows only. The 2024 test season is read once, in Notebook 06 (Section 0.3),
with the same frozen `pasig.features.windows.build_windows`, tide coefficients and scalers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pasig import config
from pasig.config import load, path
from pasig.features import windows as W
from pasig.features.scaling import MinMax, level_scaler, save_scalers
from pasig.features.tide import HarmonicTide, floor_of, residual_spectrum_check, tide_path
from pasig.gates import record_gate, require_gates


def _p(key: str) -> Path:
    p = path(key)
    p.mkdir(parents=True, exist_ok=True)
    return p


def fit_station_tide(series: pd.Series, years: list[int] | None = None):
    """D7 + G9 for one station: fit on the training years only (all months), add constituents while the
    residual spectrum check fails (max 3). Returns (model, floor, last check, iteration log, constituents).
    `years` overrides the frozen fit years (walk-forward folds, Notebook 07)."""
    cfg = load("tide")
    s = series[series.index.year.isin(years or cfg["fit_years"])]
    fl = floor_of(s)
    cons = dict(cfg["constituents_h"])
    adds = list(cfg["additions_h"].items())
    log = []
    for k in range(cfg["max_additions"] + 1):
        m = HarmonicTide(cons, cfg["t0"]).fit(s, floor=fl, margin=cfg["floor_margin_m"])
        chk = residual_spectrum_check(m, cfg["g9"])
        log.append({"constituents": list(cons), "rmse_fit_m": m.rmse_, "check": chk})
        if chk["passed"] or k == cfg["max_additions"]:
            break
        cons[adds[k][0]] = adds[k][1]
    return m, fl, chk, log, cons


def step_tide(qc_long: pd.DataFrame) -> dict:
    cfg = load("tide")
    fits, ev = {}, {}
    rows = []
    for st in cfg["stations"]:
        s = qc_long[qc_long.station.eq(st)].set_index("time_pht")["value_qc"].sort_index()
        m, fl, chk, log, cons = fit_station_tide(s)
        d = m.to_json()
        d.update({"station": st, "floor_m": fl, "fit_years": cfg["fit_years"], "g9_passed": chk["passed"],
                  "added": list(cons)[len(cfg["constituents_h"]):]})
        tide_path(st).parent.mkdir(parents=True, exist_ok=True)
        tide_path(st).write_text(json.dumps(d, indent=1))
        fits[st] = m
        ev[st] = {"floor_m": fl, "n_fit_hours": m.n_fit_, "iterations": log,
                  "final_constituents": list(cons), "added": d["added"], "spectrum_passed": chk["passed"]}
        t = m.table()
        t.insert(0, "station", st)
        rows.append(t)
    pd.concat(rows).to_csv(_p("tables") / "tide_constituents.csv", index=False)
    dec = "; ".join(f"{st}: +{','.join(e['added']) or 'none'}" + ("" if e["spectrum_passed"] else " (residual peaks remain, logged)")
                    for st, e in ev.items())
    record_gate("G9", "tide_fit", True,
                "After fitting M2 S2 N2 K1 O1 M4 MS4 (all months of 2018-2022, hours > floor + 0.05 m), no residual "
                "periodogram peak within +-3% of 12.0, 12.42, 23.93, 24.0, 25.82 h exceeds 3x the median power nearby "
                "(3-20%); else add P1, K2, Q1, M3, 2N2 in order, refit and recheck (max 3 additions, each logged)",
                ev, decision=dec)
    _plot_tide(qc_long, fits)
    return {"fits": fits, "evidence": ev}


def _plot_tide(qc_long: pd.DataFrame, fits: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = fits["pandacan"]
    s = qc_long[qc_long.station.eq("pandacan")].set_index("time_pht")["value_qc"]
    a, b = pd.Timestamp("2022-07-01"), pd.Timestamp("2022-07-15")
    x = s.loc[a:b]
    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    ax.plot(x.index, x.to_numpy(), lw=1, color="#08519c", label="Pandacan (QC'd)")
    ax.plot(x.index, m.predict(x.index), lw=1, color="#d95f0e", label="predicted tide")
    ax.set_ylabel("m")
    ax.set_title("Pandacan and its harmonic tide, 1-15 Jul 2022 (fit: all months 2018-2022)", fontsize=9)
    ax.legend(fontsize=7)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(_p("figures") / "fig_tide_pandacan_jul2022.png")
    plt.close(fig)


def tide_check_2023(qc_long: pd.DataFrame, fits: dict) -> dict:
    """Recorded, not used for any choice: RMSE of tide-only and tide + persisted residual on 2023 wet hours."""
    out = {}
    for st, m in fits.items():
        s = qc_long[qc_long.station.eq(st)].set_index("time_pht")["value_qc"]
        v = s[(s.index.year == 2023) & s.index.month.isin(config.wet_months())].dropna()
        out[st] = {"rmse_tide_only_2023_wet_m": float(np.sqrt(np.mean((v - m.predict(v.index)) ** 2))),
                   "n_hours": int(len(v))}
    return out


def run() -> dict:
    require_gates("G6", "G7", "G8")
    proc, interim = path("processed"), path("interim")
    qc_long = pd.read_parquet(interim / "waterlevel_qc.parquet")
    master = pd.read_parquet(proc / "hourly_master.parquet")
    ev = pd.read_parquet(proc / "eval_hours.parquet")
    six = pd.read_parquet(interim / "pagasa_6h.parquet")
    eligible = load("eligible_inputs")["eligible_stations"] or []
    if (master["season"] >= min(load("splits")["test_years"])).any():
        master = master[master["season"] < min(load("splits")["test_years"])]   # the test season is not read here
        ev = ev.loc[master.index]
    tide = step_tide(qc_long)
    tchk = tide_check_2023(qc_long, tide["fits"])
    train_rows = master["split"].eq("train")
    daily_fill = master.loc[train_rows, W.DAILY_TAB].median().to_dict()
    arrays = {sp: W.build_windows(master, ev, six, tide["fits"]["pandacan"], sp, eligible, daily_fill)
              for sp in ("train", "val")}
    tr = arrays["train"]
    s_seq, s_tab, s_cons = MinMax().fit(tr["X_seq"]), MinMax().fit(tr["X_tab"]), MinMax().fit(tr["X_tab_cons"])
    s_y = level_scaler(s_seq)
    art = _p("artifacts")
    save_scalers(art / "scalers.json", seq=(s_seq, tr["seq_names"]), tab=(s_tab, tr["tab_names"]),
                 tab_cons=(s_cons, tr["cons_names"]), y=(s_y, ["pandacan"]))
    names = {"seq": tr["seq_names"], "tab": tr["tab_names"], "tab_cons": tr["cons_names"],
             "daily_fill_training_median": daily_fill, "eligible_stations": eligible,
             "note": "sg_daily_* columns enter the model only if NB04 (Granger) selects them"}
    (art / "feature_names.json").write_text(json.dumps(names, indent=1))
    summ = []
    for sp, a in arrays.items():
        np.savez_compressed(proc / f"windows_{sp}.npz", **{k: a[k] for k in ("X_seq", "X_tab", "X_tab_cons", "y", "valid")},
                            t_index=a["t_index"].astype("datetime64[ns]").astype("int64"))
        summ.append({"split": sp, "cases": len(a["y"]), **{f"valid_lead{h}": int(a["valid"][:, h - 1].sum()) for h in range(1, 7)},
                     "seq_channels": a["X_seq"].shape[2], "tab_features": a["X_tab"].shape[1],
                     "cons_features": a["X_tab_cons"].shape[1], "daily_values_imputed": a["n_daily_missing_filled"]})
    summ = pd.DataFrame(summ)
    summ.to_csv(_p("tables") / "windows_summary.csv", index=False)
    (_p("tables") / "tide_check_2023.json").write_text(json.dumps(tchk, indent=1))
    return {"tide": tide, "tide_check_2023": tchk, "arrays": arrays, "summary": summ,
            "scalers": {"seq": s_seq, "tab": s_tab, "cons": s_cons, "y": s_y}}


def load_windows(split: str) -> dict:
    """Load saved arrays (train/val). Test windows are built in NB06 only."""
    if split not in ("train", "val"):
        raise ValueError("test windows are built in Notebook 06 after G18")
    z = np.load(path("processed") / f"windows_{split}.npz")
    out = {k: z[k] for k in z.files}
    out["t_index"] = pd.to_datetime(out["t_index"])
    return out
