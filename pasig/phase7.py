"""Phase 7 / Notebook 07: robustness (Section 10.2). Runs after G18; changes no frozen setting.

Stages (each resumable; neural stages in their own process so the device is fixed before TensorFlow starts):
  walkforward  3 folds (2018-20 -> 2021, 2018-21 -> 2022, 2018-22 -> 2023). Per fold, from the fold's training
               years only: tide fit (D7/G9 rule), alert levels (G8 rule), Granger selection (G10 rule), training
               medians and scalers; the network trains on all but the last training year, which is the inner
               validation (early stopping, ridge alpha, trigger offsets). QC rules and hyperparameters stay frozen.
               The 2023 fold is reported separately (2023 chose the hyperparameters; not independent).
  ablations    Table III-5, six variants on the frozen windows and hyperparameters: no Granger selection,
               correlation selection, no rainfall, no tabular path, second LSTM instead of the GRU, weighted
               loss (D20; w chosen on 2023). Seeds per G15, determinism on; own trigger offsets on 2023;
               scored on the same 2024 hours as Notebook 06.
  casestudy    every 2024 flood-risk event (incl. 24 Jul 2024, Carina): observed level, each lead's forecast,
               alert times, hours of warning.
  aviles       deferred descriptive check of the alert levels against the Aviles June-15 Jul 2024 records.

Rehearsal profile: the 2023 validation season stands in for 2024 (ablations, case study) and Aviles 2023 for
Aviles 2024; the walk-forward never reads 2024 in either profile.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from pasig.config import load, out_dir, path, profile, root
from pasig.gates import read_gate, require_gates

STAGES = ["walkforward", "ablations", "casestudy", "aviles"]
NEURAL = {"walkforward", "ablations"}
LEADS = 6
ABLATIONS = ["no_granger", "correlation_selection", "no_rainfall", "no_tabular", "second_lstm", "weighted_loss"]
WF_BASELINES = ("persistence", "tide_only", "tide_resid", "ridge")


def rehearsal() -> bool:
    return profile() == "rehearsal"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _T():
    return out_dir("tables")


def _nb07() -> dict:
    return load("eval")["nb07"]


def seeds() -> list[int]:
    """D13 / G15: seeds_wf_ablation (3 or 5); rehearsal: the rehearsal final seeds."""
    from pasig.train.loop import hp
    if rehearsal():
        return list(hp()["final_seeds"])
    return list(range(int(load("compute")["seeds_wf_ablation"])))


def final_cfg():
    from pasig.models.hybrid import HybridConfig
    from pasig.phase5 import state
    return HybridConfig(**state()["hybrid_final"])


def _levels() -> dict:
    al = load("alert_levels")
    return {"high_water": float(al["high_water_m"]), "flood_risk": float(al["flood_risk_m"])}


def _offset_cfg() -> dict:
    return load("baselines")["trigger_offsets"]


def offsets_for(preds: list[np.ndarray], d: dict, levels: dict) -> dict:
    """D11 on the given validation data: one offset per level shared by all leads; mean F1 over seeds."""
    from pasig.models.offsets import best_offset
    return {lvl: best_offset(preds, d["y_raw"], d["valid"], v, _offset_cfg()) | {"level_m": v}
            for lvl, v in levels.items()}


def _frame(d: dict, pred: np.ndarray, model: str, seed: int, rainy: np.ndarray, **extra) -> pd.DataFrame:
    from pasig.train.loop import preds_frame
    f = preds_frame(d, pred, model, seed)
    f["rainy"] = np.repeat(rainy, LEADS)
    for k, v in extra.items():
        f[k] = v
    return f


def _rainy(w: dict) -> np.ndarray:
    names = list(w["tab_names"]) if "tab_names" in w else \
        json.loads((path("artifacts") / "feature_names.json").read_text())["tab"]
    return w["X_tab"][:, names.index("sg_rain_24h_in")] >= float(load("eval")["rainy_rain24_mm"])


def _alerts(pt: pd.DataFrame, offsets: dict, levels: dict) -> pd.DataFrame:
    """Hourly alert counts per model x level x lead x subset (mean over seeds) with each model's own offsets."""
    from pasig.eval.alerts import counts
    rows = []
    v = pt[pt["valid"]]
    for (model, lead), g in v.groupby(["model", "lead"]):
        for lvl, L in levels.items():
            off = offsets[model][lvl]["offset_m"]
            for sub, gs in {"all": g, "rainy": g[g["rainy"]], "dry": g[~g["rainy"]]}.items():
                per = [counts((x.y_pred >= L - off).to_numpy(), (x.y_true >= L).to_numpy())
                       for _, x in gs.groupby("seed") if len(x)]
                if per:
                    d = pd.DataFrame(per)
                    rows.append({"model": model, "level": lvl, "lead": lead, "subset": sub, "offset_m": off,
                                 "seeds": len(d), **d.mean(numeric_only=True).to_dict()})
    return pd.DataFrame(rows)


def _dm_holm(pairs: list[tuple], frame_for) -> pd.DataFrame:
    """pairs: (key dict, model_a, model_b, lead); frame_for(key) -> one-seed prediction frame."""
    from statsmodels.stats.multitest import multipletests

    from pasig.eval.dm_test import diebold_mariano
    from pasig.phase6 import _aligned
    rows = []
    for key, a, b, h in pairs:
        w = _aligned(frame_for(key), a, b, h, "all")
        if len(w) < 30:
            rows.append(key | {"model": a, "reference": b, "lead": h, "n": len(w), "note": "too few hours"})
            continue
        rows.append(key | {"model": a, "reference": b, "lead": h,
                           **diebold_mariano((w[a] - w["y"]).to_numpy(), (w[b] - w["y"]).to_numpy(), h)})
    d = pd.DataFrame(rows)
    ok = d["p_value"].notna() if "p_value" in d else pd.Series(False, index=d.index)
    if ok.any():
        d.loc[ok, "p_holm"] = multipletests(d.loc[ok, "p_value"], alpha=float(load("eval")["dm"]["alpha"]),
                                            method="holm")[1]
        d["a_better"] = ok & (d["p_holm"] < 0.05) & (d["mean_loss_diff"] < 0)
        d["a_worse"] = ok & (d["p_holm"] < 0.05) & (d["mean_loss_diff"] > 0)
    return d


def _median_seed(rows: pd.DataFrame) -> int:
    r = rows.sort_values(["val_rmse_mean", "seed"])
    return int(r.iloc[(len(r) - 1) // 2]["seed"])


class RunLog:
    """Resumable record of trained networks (one CSV per stage)."""

    def __init__(self, name: str):
        self.p = out_dir("runs") / f"{name}.csv"
        self.df = pd.read_csv(self.p) if self.p.exists() else pd.DataFrame()

    def get(self, tag: str, seed: int):
        if self.df.empty:
            return None
        hit = self.df[(self.df["tag"] == tag) & (self.df["seed"] == seed)]
        return None if hit.empty else hit.iloc[-1].to_dict()

    def add(self, row: dict) -> None:
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.p, index=False)


def train_or_load(log: RunLog, tag: str, seed: int, inp, tr, va, loss=None, **variant):
    """Train with determinism on (D15) unless this (tag, seed) already finished; return (model, row)."""
    import keras

    import pasig.models.hybrid  # noqa: F401  (registers custom layers/losses)
    from pasig.models.hybrid import masked_mse
    from pasig.train.loop import train_run
    have = log.get(tag, seed)
    if have is not None:
        return keras.models.load_model(f"{have['run_dir']}/model.keras", compile=False), have
    r = train_run("hybrid", final_cfg(), inp, tr, va, seed, True, tag, loss=loss or masked_mse, **variant)
    log.add(r["row"])
    _log(f"  {tag} seed {seed}: val RMSE {r['row']['val_rmse_mean']:.4f} m ({r['row']['epochs']} epochs)")
    return r["model"], r["row"]


def _seq_only(d: dict) -> dict:
    o = dict(d)
    o["X"] = {"seq": d["X"]["seq"]}
    return o


# ============================================================================= ablations (Table III-5, D20)

def ablation_inputs() -> dict:
    from pasig.train.data import Inputs
    base = Inputs.from_config()
    rain = load("granger")["feature_groups"]["sg_rain"]
    drop = set(rain.get("seq", [])) | set(rain.get("tab", []))
    nr = Inputs(seq=[c for c in base.seq if c not in drop], tab=[c for c in base.tab if c not in drop],
                use_constituents=base.use_constituents, add_back=base.add_back, tag="no_rainfall")
    ng = Inputs.from_config("no_granger_ablation")
    ng.tag = "no_granger"
    cs = Inputs.from_config("correlation_selection_ablation")
    cs.tag = "correlation_selection"
    return {"main": base, "no_granger": ng, "correlation_selection": cs, "no_rainfall": nr, "no_tabular": base,
            "second_lstm": base, "weighted_loss": base}


def _variant_kw(name: str) -> dict:
    return {"no_tabular": {"use_tabular": False}, "second_lstm": {"second_lstm_instead_of_gru": True}}.get(name, {})


def choose_d20_weight(log: RunLog, inp, tr, va) -> dict:
    """D20: w in {2, 5, 10} with the final hyperparameters, seed 0, by high-water F1 after own offset tuning."""
    from pasig.models.hybrid import weighted_masked_mse
    from pasig.train.loop import predict_levels
    hw = _levels()["high_water"]
    ys = va["y_scaler"]
    thr_s = float(((hw - ys.lo) / ys.span).ravel()[0])
    s0 = seeds()[0]
    res = {}
    for w in _nb07()["d20_weights"]:
        m, row = train_or_load(log, f"abl_weighted_w{w}", s0, inp, tr, va, loss=weighted_masked_mse(float(w), thr_s))
        p = predict_levels(m, va["X"], va["y_scaler"], "hybrid")
        off = offsets_for([p], va, {"high_water": hw})["high_water"]
        res[int(w)] = {"f1_high_water_val": off["f1_val"], "offset_m": off["offset_m"],
                       "val_rmse_mean": float(row["val_rmse_mean"])}
    best = max(res, key=lambda k: (res[k]["f1_high_water_val"], -k))
    return {"weight": best, "threshold_scaled": thr_s, "search": res}


def stage_ablations() -> dict:
    from pasig.models.hybrid import weighted_masked_mse
    from pasig.phase5 import _data
    from pasig.phase6 import test_windows
    from pasig.train.data import load_split
    from pasig.train.loop import predict_levels
    require_gates("G18")
    log = RunLog("ablations")
    inputs = ablation_inputs()
    wt = test_windows()
    rainy_te = _rainy(wt)
    levels = _levels()
    d20 = None
    val_frames, test_frames, rows, offsets = [], [], [], {}
    for name in ABLATIONS:
        inp = inputs[name]
        tr, va = _data(inp)
        te = load_split("test", inp, windows=wt)
        kw = _variant_kw(name)
        if name == "no_tabular":
            tr, va, te = _seq_only(tr), _seq_only(va), _seq_only(te)
        loss = None
        if name == "weighted_loss":
            d20 = choose_d20_weight(log, inp, tr, va)
            loss = weighted_masked_mse(float(d20["weight"]), d20["threshold_scaled"])
            _log(f"  D20 weight chosen: w = {d20['weight']}")
        tag = f"abl_weighted_w{d20['weight']}" if name == "weighted_loss" else f"abl_{name}"
        preds_val = []
        rainy_va = _rainy(_raw_val())
        for s in seeds():
            m, row = train_or_load(log, tag, s, inp, tr, va, loss=loss, **kw)
            pv = predict_levels(m, va["X"], va["y_scaler"], "hybrid")
            pt = predict_levels(m, te["X"], te["y_scaler"], "hybrid")
            preds_val.append(pv)
            val_frames.append(_frame(va, pv, f"abl_{name}", s, rainy_va))
            test_frames.append(_frame(te, pt, f"abl_{name}", s, rainy_te))
            rows.append({"variant": name, "tag": tag, "seed": s, "val_rmse_mean": float(row["val_rmse_mean"]),
                         "epochs": int(row["epochs"]), "seconds": float(row["seconds"])})
        offsets[f"abl_{name}"] = offsets_for(preds_val, va, levels)
    T = _T()
    pd.concat(val_frames, ignore_index=True).to_parquet(T / "preds_ablation_val.parquet", index=False)
    pd.concat(test_frames, ignore_index=True).to_parquet(T / "preds_ablation_test.parquet", index=False)
    runs = pd.DataFrame(rows)
    runs.to_csv(T / "ablation_runs.csv", index=False)
    (T / "ablation_offsets.json").write_text(json.dumps({"offsets": offsets, "d20": d20}, indent=1, default=float))
    return evaluate_ablations()


def _raw_val() -> dict:
    from pasig.phase3 import load_windows
    return load_windows("val")


def evaluate_ablations() -> dict:
    from pasig.phase6 import level_table, test_seed
    T = _T()
    levels = _levels()
    pa = pd.read_parquet(T / "preds_ablation_test.parquet")
    runs = pd.read_csv(T / "ablation_runs.csv")
    offs = json.loads((T / "ablation_offsets.json").read_text())
    s_ab = seeds()
    main = pd.read_parquet(T / "preds_test.parquet")
    main = main[main["model"].eq("hybrid")]
    main_same = main[main["seed"].isin(s_ab)].assign(model="hybrid_same_seeds")
    pt = pd.concat([pa, main.assign(model="hybrid_all_seeds"), main_same], ignore_index=True)
    lv = level_table(pt)
    # difference vs the main hybrid on the same seeds, seed-noise rule
    a = lv[lv["subset"].eq("all")].set_index(["model", "lead"])
    ref = a.loc["hybrid_same_seeds"]
    diff = []
    for name in ABLATIONS:
        m = f"abl_{name}"
        if m not in a.index.get_level_values(0):
            continue
        for h in range(1, LEADS + 1):
            d = a.loc[(m, h), "rmse"] - ref.loc[h, "rmse"]
            sd = ref.loc[h, "rmse_sd"]
            diff.append({"variant": name, "lead": h, "rmse": a.loc[(m, h), "rmse"], "rmse_main": ref.loc[h, "rmse"],
                         "rmse_diff": d, "main_seed_sd": sd, "beyond_seed_noise": bool(abs(d) > sd) if np.isfinite(sd) else None})
    diff = pd.DataFrame(diff)
    th = load("thresholds")["trigger_offsets"]
    all_off = offs["offsets"] | {"hybrid_all_seeds": th["hybrid"], "hybrid_same_seeds": th["hybrid"]}
    al = _alerts(pt, all_off, levels)
    # DM (Holm, descriptive): each variant's median-validation seed vs the main hybrid's NB06 test seed
    med = {f"abl_{v}": _median_seed(g) for v, g in runs.groupby("variant")}
    hs = test_seed("hybrid")
    one = pd.concat([pa[pa["model"].eq(m) & pa["seed"].eq(s)] for m, s in med.items()] +
                    [main[main["seed"].eq(hs)]], ignore_index=True)
    dm = _dm_holm([({}, m, "hybrid", h) for m in med for h in range(1, LEADS + 1)], lambda key: one)
    d20 = d20_report(pt, all_off, levels, offs.get("d20"))
    for n, df in (("ablation_level_metrics", lv), ("ablation_rmse_diff", diff), ("ablation_alerts", al),
                  ("ablation_dm_holm", dm)):
        df.to_csv(T / f"{n}.csv", index=False)
    (T / "ablation_d20.json").write_text(json.dumps(d20, indent=1, default=float))
    return {"level": lv, "diff": diff, "alerts": al, "dm": dm, "d20": d20, "seeds": s_ab}


def _event_peaks(pt: pd.DataFrame, level: float) -> pd.DataFrame:
    from pasig.eval.events import events
    obs = pt[pt["valid"]].assign(target=lambda d: d["t"] + pd.to_timedelta(d["lead"], "h"))
    lo = obs.drop_duplicates("target").set_index("target")["y_true"].sort_index()
    ev = events(lo, level, int(load("eval")["event_merge_gap_h"]))
    if ev.empty:
        return ev.assign(peak_time=pd.Series(dtype="datetime64[ns]"))
    ev["peak_time"] = [lo.loc[r["first"]:r["last"]].idxmax() for _, r in ev.iterrows()]
    return ev


def d20_report(pt: pd.DataFrame, offsets: dict, levels: dict, d20: dict | None) -> dict:
    """D20 measures for the weighted-loss variant and the main hybrid (same seeds): overall RMSE, RMSE on hours
    >= high water, mean error at flood-risk event peaks, Recall at both levels."""
    from pasig.eval.alerts import counts
    out = {"weight": None if d20 is None else d20["weight"], "search": None if d20 is None else d20["search"]}
    peaks = _event_peaks(pt[pt["model"].eq("hybrid_same_seeds")], levels["flood_risk"])
    for m in ("abl_weighted_loss", "hybrid_same_seeds"):
        g = pt[pt["model"].eq(m) & pt["valid"]]
        if g.empty:
            continue
        r = {}
        for h, gl in g.groupby("lead"):
            e = gl.y_pred - gl.y_true
            hw = gl.y_true >= levels["high_water"]
            tgt = gl["t"] + pd.Timedelta(hours=int(h))
            at_peak = gl[tgt.isin(peaks["peak_time"])] if len(peaks) else gl.iloc[:0]
            rec = {}
            for lvl, L in levels.items():
                off = offsets[m][lvl]["offset_m"]
                rec[lvl] = float(np.mean([counts((x.y_pred >= L - off).to_numpy(), (x.y_true >= L).to_numpy())["recall"]
                                          for _, x in gl.groupby("seed")]))
            r[int(h)] = {"rmse": float(np.sqrt(np.mean(e ** 2))), "rmse_high_water_hours": float(np.sqrt(np.mean(e[hw] ** 2)))
                         if hw.any() else None, "mean_error_at_flood_risk_peaks": float((at_peak.y_pred - at_peak.y_true).mean())
                         if len(at_peak) else None, "n_peaks": int(at_peak["t"].nunique()),
                         "recall_high_water": rec["high_water"], "recall_flood_risk": rec["flood_risk"]}
        out[m] = r
    out["flood_risk_events"] = int(len(peaks))
    return out


# ============================================================================= walk-forward

def _fields_aviles() -> dict:
    return read_gate("G4")["evidence"]["aviles"]["fields"]


def _aviles_frame(master: pd.DataFrame, pump: pd.DataFrame, statuses=("use",)) -> pd.DataFrame:
    from pasig import phase2
    pan = master["pandacan"].where(~master["pandacan_filled"])
    av = phase2.station_frame(pump, "aviles", statuses)
    fa = _fields_aviles()
    return phase2.analysis_frame({"master": master, "pandacan": pan}, av, fa["pumps_running"], fa["gate_open"])


def _pumping_live() -> pd.DataFrame:
    pump = pd.read_parquet(path("interim") / "pumping_long.parquet")
    return pump[~pump.flag_impossible_date & pump.time_pht.notna()]


def fold_setup(fold: dict, master_all: pd.DataFrame, ev_all: pd.DataFrame) -> dict:
    """Everything a fold learns from data, from its training years only."""
    from pasig import phase2, phase4
    from pasig.features import windows as W
    from pasig.features.scaling import MinMax, level_scaler
    from pasig.phase3 import fit_station_tide
    years = [int(y) for y in fold["train"]]
    ev_year, inner = int(fold["eval"]), years[-1]
    net = years[:-1]
    keep = master_all["season"].isin(years + [ev_year])
    master = master_all[keep].copy()
    ev = ev_all.loc[master.index].copy()
    lab = np.select([master["season"].isin(net), master["season"].eq(inner), master["season"].eq(ev_year)],
                    ["train", "val", "test"], "none").astype(object)
    master["split"], ev["split"] = lab, lab
    qc = pd.read_parquet(path("interim") / "waterlevel_qc.parquet")
    s = qc[qc.station.eq("pandacan")].set_index("time_pht")["value_qc"].sort_index()
    tide, floor, chk, _, cons = fit_station_tide(s, years)
    f = _aviles_frame(master, _pumping_live())
    f["split"] = np.where(f["season"].isin(years), "train", "other")
    al = phase2.alert_levels(f)
    six = pd.read_parquet(path("interim") / "pagasa_6h.parquet")
    daily = pd.read_parquet(path("interim") / "pagasa_daily.parquet")
    sel = phase4.select_candidates(master, six, daily, years, {"pandacan": tide}, eligible=[])
    fill = master.loc[master["season"].isin(years), W.DAILY_TAB].median().to_dict()
    win = {sp: W.build_windows(master, ev, six, tide, sp, [], fill) for sp in ("train", "val", "test")}
    for w in win.values():
        w["t_index"] = pd.DatetimeIndex(w["t_index"])
    fit_seq = np.concatenate([win["train"]["X_seq"], win["val"]["X_seq"]])
    s_seq = MinMax().fit(fit_seq)
    sc = {"seq": s_seq, "tab": MinMax().fit(np.concatenate([win["train"]["X_tab"], win["val"]["X_tab"]])),
          "tab_cons": MinMax().fit(np.concatenate([win["train"]["X_tab_cons"], win["val"]["X_tab_cons"]])),
          "y": level_scaler(s_seq)}
    cols = phase4.model_columns(sel["selected"], list(win["train"]["tab_names"]))
    info = {"train_years": years, "network_years": net, "inner_validation_year": inner, "eval_year": ev_year,
            "tide_added": list(cons)[len(load("tide")["constituents_h"]):], "tide_g9_passed": bool(chk["passed"]),
            "tide_floor_m": float(floor), "high_water_m": al["high_water_m"], "flood_risk_m": al["flood_risk_m"],
            "granger_selected": sel["selected"], "granger_p": {k: float(v["p_value"]) for k, v in sel["tests"].items()},
            "cases": {sp: int(len(w["y"])) for sp, w in win.items()}, "inputs": cols}
    return {"windows": win, "scalers": sc, "tide": tide, "master": master, "info": info,
            "levels": {"high_water": al["high_water_m"], "flood_risk": al["flood_risk_m"]}}


def stage_walkforward() -> dict:
    from pasig.models import baselines as B
    from pasig.train.data import Inputs, load_split, rehearsal_cases
    from pasig.train.loop import predict_levels
    require_gates("G18")
    master_all = pd.read_parquet(path("processed") / "hourly_master.parquet")
    master_all = master_all[master_all["season"] < min(load("splits")["test_years"])]   # 2024 never read here
    ev_all = pd.read_parquet(path("processed") / "eval_hours.parquet").loc[master_all.index]
    base = Inputs.from_config()
    log = RunLog("walkforward")
    frames, infos, offsets = [], {}, {}
    for fold in load("splits")["walk_forward"]:
        fy = int(fold["eval"])
        _log(f"== walk-forward fold -> {fy}")
        fs = fold_setup(fold, master_all, ev_all)
        w, sc = fs["windows"], fs["scalers"]
        inp = Inputs(seq=fs["info"]["inputs"]["seq"], tab=fs["info"]["inputs"]["tab"],
                     use_constituents=base.use_constituents, add_back=base.add_back, tag=f"wf{fy}")
        tr = load_split("train", inp, windows=w["train"], scalers=sc, max_cases=rehearsal_cases())
        va = load_split("val", inp, windows=w["val"], scalers=sc)
        te = load_split("test", inp, windows=w["test"], scalers=sc)
        rainy = _rainy(w["test"])
        pv, rows = {}, []
        for s in seeds():
            m, row = train_or_load(log, f"wf{fy}_hybrid", s, inp, tr, va)
            pv.setdefault("hybrid", []).append(predict_levels(m, va["X"], va["y_scaler"], "hybrid"))
            frames.append(_frame(te, predict_levels(m, te["X"], te["y_scaler"], "hybrid"), "hybrid", s, rainy, fold=fy))
            rows.append({"seed": s, "val_rmse_mean": float(row["val_rmse_mean"])})
        # baselines fitted on the fold's training data, tuned on its inner validation year
        obs = fs["master"]["pandacan"].where(~fs["master"]["pandacan_filled"])
        to = B.TideOnly(fs["tide"], load("baselines")["tide_only"]["seasonal_harmonics"]).fit(
            obs[fs["master"]["season"].isin(fs["info"]["network_years"])])
        ridge = B.fit_ridge(tr, va)
        Fte = B.flat(te["X"])
        base_te = {"persistence": B.persistence(te), "tide_only": to.predict(te), "tide_resid": B.tide_resid(te, fs["tide"]),
                   "ridge": np.column_stack([mm.predict(Fte) for mm in ridge["models"]])}
        base_va = {"persistence": B.persistence(va), "tide_only": to.predict(va), "tide_resid": B.tide_resid(va, fs["tide"]),
                   "ridge": ridge["pred_val"]}
        for k, p in base_te.items():
            frames.append(_frame(te, p, k, -1, rainy, fold=fy))
            pv[k] = [base_va[k]]
        offsets[fy] = {k: offsets_for(p, va, fs["levels"]) for k, p in pv.items()}
        fs["info"].update({"ridge_alpha": ridge["alpha"], "hybrid_seed_for_dm": _median_seed(pd.DataFrame(rows)),
                           "hybrid_inner_val_rmse": rows})
        infos[fy] = fs["info"]
    T = _T()
    pd.concat(frames, ignore_index=True).to_parquet(T / "preds_walkforward.parquet", index=False)
    (T / "walkforward_folds.json").write_text(json.dumps({"folds": infos, "offsets": offsets}, indent=1, default=float))
    return evaluate_walkforward()


def evaluate_walkforward() -> dict:
    from pasig.eval.metrics import skill
    from pasig.phase6 import level_table
    T = _T()
    pw = pd.read_parquet(T / "preds_walkforward.parquet")
    meta = json.loads((T / "walkforward_folds.json").read_text())
    lv, al, pairs, ones = [], [], [], {}
    for fy, g in pw.groupby("fold"):
        info = meta["folds"][str(fy)]
        t = level_table(g).assign(fold=fy)
        a = t[t["subset"].eq("all")].set_index(["model", "lead"])["rmse"]
        t["skill_vs_persistence"] = [skill(r.rmse, a.get(("persistence", r.lead), np.nan)) if r.subset == "all" else np.nan
                                     for r in t.itertuples()]
        lv.append(t)
        levels = {"high_water": info["high_water_m"], "flood_risk": info["flood_risk_m"]}
        al.append(_alerts(g, meta["offsets"][str(fy)], levels).assign(fold=fy))
        s = info["hybrid_seed_for_dm"]
        ones[int(fy)] = g[g["model"].ne("hybrid") | g["seed"].eq(s)]
        pairs += [({"fold": int(fy)}, "hybrid", b, h) for b in WF_BASELINES for h in range(1, LEADS + 1)]
    lv, al = pd.concat(lv, ignore_index=True), pd.concat(al, ignore_index=True)
    dm = _dm_holm(pairs, lambda key: ones[key["fold"]])
    last = max(int(f) for f in meta["folds"])
    for d in (lv, al, dm):
        d["independent"] = d["fold"].astype(int) != last      # the 2023 fold chose the hyperparameters
    for n, df in (("walkforward_level_metrics", lv), ("walkforward_alerts", al), ("walkforward_dm_holm", dm)):
        df.to_csv(T / f"{n}.csv", index=False)
    return {"level": lv, "alerts": al, "dm": dm, "folds": meta["folds"]}


# ============================================================================= case study

def stage_casestudy() -> dict:
    from pasig.eval.events import detect
    from pasig.phase6 import _one_seed, chosen_input_sets, strongest_simple, test_seed
    require_gates("G18")
    T = _T()
    pt = pd.read_parquet(T / "preds_test.parquet")
    pr = pd.read_parquet(T / "probs_test.parquet")
    levels = _levels()
    th = load("thresholds")["trigger_offsets"]
    c = _nb07()
    seeds_t = {k: test_seed(k) for k in ("hybrid", "lstm")}
    p1 = _one_seed(pt, seeds_t)
    ev = _event_peaks(p1[p1["model"].eq("persistence")], levels["flood_risk"])
    carina = pd.Timestamp(c["carina_date"])
    ev["carina"] = [bool(r["first"].normalize() <= carina <= r["last"].normalize()) for _, r in ev.iterrows()]
    ev["event"] = range(1, len(ev) + 1)
    models = ["hybrid", "lstm", "persistence", strongest_simple(), "ridge", "knn"]
    clfs = list(chosen_input_sets().values())
    rows = []
    for _, e in ev.iterrows():
        one = pd.DataFrame([e])
        for m in dict.fromkeys(models):
            off = th[m]["flood_risk"]["offset_m"]
            g = p1[p1["model"].eq(m) & p1["valid"]]
            for h in range(1, LEADS + 1):
                gl = g[g["lead"].eq(h)]
                tgt = gl["t"] + pd.Timedelta(hours=h)
                al = gl[gl.y_pred >= levels["flood_risk"] - off]
                d = detect(one, pd.DatetimeIndex(al["t"]) + pd.Timedelta(hours=h), h, int(load("eval")["event_detect_window_h"])).iloc[0]
                pk = gl[tgt.eq(e["peak_time"])]
                near = al[(al["t"] + pd.Timedelta(hours=h) >= e["first"] - pd.Timedelta(hours=12)) &
                          (al["t"] <= e["last"])]
                rows.append({"event": e["event"], "model": m, "lead": h, "detected": bool(d["detected"]),
                             "warning_h": d["warning_h"],
                             "first_alert_issue": near["t"].min() if len(near) else pd.NaT,
                             "forecast_at_peak": float(pk["y_pred"].iloc[0]) if len(pk) else np.nan,
                             "error_at_peak": float(pk["y_pred"].iloc[0] - pk["y_true"].iloc[0]) if len(pk) else np.nan})
        for k in clfs:
            g = pr[pr["model"].eq(k) & pr["level"].eq("flood_risk")]
            for h, gl in g.groupby("lead"):
                a = gl[gl["alert"]]
                d = detect(one, pd.DatetimeIndex(a["t"]) + pd.Timedelta(hours=int(h)), int(h), int(load("eval")["event_detect_window_h"])).iloc[0]
                rows.append({"event": e["event"], "model": k, "lead": int(h), "detected": bool(d["detected"]),
                             "warning_h": d["warning_h"], "first_alert_issue": pd.NaT, "forecast_at_peak": np.nan,
                             "error_at_peak": np.nan})
    det = pd.DataFrame(rows, columns=["event", "model", "lead", "detected", "warning_h", "first_alert_issue",
                                      "forecast_at_peak", "error_at_peak"])
    ev.to_csv(T / "case_study_events.csv", index=False)
    det.to_csv(T / "case_study_detail.csv", index=False)
    _case_figures(p1, ev, levels, th)
    summ = det.groupby(["model", "lead"]).agg(detected=("detected", "sum"), events=("detected", "size"),
                                              median_warning_h=("warning_h", "median"),
                                              mean_error_at_peak=("error_at_peak", "mean")).reset_index()
    summ.to_csv(T / "case_study_summary.csv", index=False)
    return {"events": ev, "detail": det, "summary": summ}


def _case_figures(p1: pd.DataFrame, ev: pd.DataFrame, levels: dict, th: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = out_dir("figures")
    for old in F.glob("case_event_*.png"):
        old.unlink()
    c = _nb07()
    pad = pd.Timedelta(hours=int(c["case_study_window_h"]))
    g = p1[p1["model"].eq("hybrid") & p1["valid"]].assign(target=lambda d: d["t"] + pd.to_timedelta(d["lead"], "h"))
    obs = g.drop_duplicates("target").set_index("target")["y_true"].sort_index()
    off = th["hybrid"]["flood_risk"]["offset_m"]
    for _, e in ev.iterrows():
        lo, hi = e["first"] - pad, e["last"] + pad
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(obs.loc[lo:hi].index, obs.loc[lo:hi], color="k", lw=1.4, label="observed")
        for h in c["case_study_leads"]:
            s = g[g["lead"].eq(h)].set_index("target")["y_pred"].sort_index().loc[lo:hi]
            ax.plot(s.index, s, lw=0.9, label=f"hybrid lead {h}")
            a = s[s >= levels["flood_risk"] - off]
            ax.plot(a.index, np.full(len(a), levels["flood_risk"] + 0.03 * h), "|", ms=6)
        for lvl, L in levels.items():
            ax.axhline(L, ls="--", lw=0.7, color="r" if lvl == "flood_risk" else "orange")
        ttl = f"Event {e['event']}: {e['first']:%d %b %Y %H:%M}, peak {e['peak_m']:.2f} m"
        ax.set(title=ttl + (" (Carina)" if e.get("carina") else ""), ylabel="Pandacan (m)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(F / f"case_event_{int(e['event']):02d}.png", dpi=130)
        plt.close(fig)


# ============================================================================= Aviles 2024 (deferred check)

def stage_aviles() -> dict:
    from pasig.io import mmda_pumping as mp
    require_gates("G18")
    c = _nb07()
    master = pd.read_parquet(path("processed") / "hourly_master.parquet")
    if rehearsal():
        year, pump, statuses = 2023, _pumping_live(), ("use",)
    else:
        year, statuses = int(min(load("splits")["test_years"])), ("post_eval_only",)
        bl = pd.read_csv(path("manifest") / "pumping_blocks.csv")
        res = mp.parse_from_blocks(path("raw"), bl, statuses=statuses,
                                   value_range=tuple(load("stations")["pumping_value_range_m"]))
        pump = res["long"]
        pump["gate_open"] = pump["gate_open"].astype("float")
        pump = pump[~pump.flag_impossible_date & pump.time_pht.notna()]
    f = _aviles_frame(master, pump, statuses)
    lo, hi = (pd.Timestamp(f"{year}-{d}") for d in c["aviles_check_window"])
    win = f[(f.index >= lo) & (f.index < hi + pd.Timedelta(days=1)) & f["L"].notna()]
    trn = _aviles_frame(master[master["split"].eq("train")], _pumping_live())
    out = {"year": year, "window": [str(lo.date()), str(hi.date())], "hours_with_level": int(len(win)),
           "hours_with_status": int(win["stress"].notna().sum()), "levels": {}}
    rows = []
    for lvl, L in _levels().items():
        for name, d in (("check", win), ("training", trn[trn["L"].notna()])):
            at, below = d[d.L >= L], d[d.L < L]
            r = {"level": lvl, "level_m": L, "data": name, "hours_at_or_above": int(len(at)),
                 "stress_share_at_or_above": float(at["stress"].dropna().mean()) if at["stress"].notna().any() else None,
                 "stress_share_below": float(below["stress"].dropna().mean()) if below["stress"].notna().any() else None,
                 "gate_open_share_at_or_above": float(at["gate_open"].dropna().mean()) if at["gate_open"].notna().any() else None,
                 "dry_share_at_or_above": float(at.loc[at.R24.notna(), "dry"].mean()) if at["R24"].notna().any() else None}
            rows.append(r)
            if name == "check":
                out["levels"][lvl] = r
    tab = pd.DataFrame(rows)
    tab.to_csv(_T() / "aviles_check.csv", index=False)
    out["note"] = "descriptive only (Section 10.2); levels are not changed"
    (_T() / "aviles_check.json").write_text(json.dumps(out, indent=1, default=float))
    return {"table": tab, "summary": out}


# ============================================================================= driver

def run_stage(name: str):
    fn = {"walkforward": stage_walkforward, "ablations": stage_ablations, "casestudy": stage_casestudy,
          "aviles": stage_aviles}[name]
    if name in NEURAL:
        from pasig.utils.hardware import configure_device
        configure_device(load("compute").get("device_final", "cpu"))
    return fn()


def run(stages: list[str] | None = None) -> int:
    """Run stages in order, each in its own process (device fixed before TensorFlow starts)."""
    require_gates("G18")
    for s in stages or STAGES:
        env = dict(os.environ)
        if s in NEURAL and load("compute").get("device_final", "cpu") == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        _log(f"== NB07 stage {s} ({profile()})")
        code = subprocess.call([sys.executable, "scripts/run_phase.py", "--phase", "7", "--stage", s], cwd=root(), env=env)
        if code != 0:
            _log(f"stage {s} failed (exit {code}); fix and re-run from this stage: --stage {s}")
            return code
    return 0
