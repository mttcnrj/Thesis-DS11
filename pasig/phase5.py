"""Phase 5 / Notebook 05: models and training (Section 9.3, SOP 3). Gates G11-G17, G19.

Stages (each resumable; neural stages run in their own process so the device is fixed before TensorFlow starts):
  compute    G14 determinism -> G13 device -> G15 seed budget (bench_compute twice per device)  -> compute.yaml
  variants   G11 tidal-input variant, G12 output form on the reference config (3 seeds each)  -> model_inputs.yaml
  grid       two-stage hybrid grid (24 + 36, D12), G19 edge check, standalone LSTM grid (18)
  finals     final hybrid and LSTM configs x 5 seeds (determinism on, D15) -> frozen checkpoints
  baselines  persistence, tide-only, tide + persisted residual, ridge, KNN regressor, LR + KNN classifiers
  thresholds trigger offsets (D11) and classifier thresholds -> thresholds.yaml
  checks     G16 persistence sanity (+ bug checklist, Section 9.7)
  freeze     G17: frozen_manifest.json (SHA-256 of every frozen file); tag settings-frozen (full profile, --tag)
Run with `python scripts/run_phase.py --phase 5 [--stage NAME]`; PASIG_PROFILE=rehearsal for the tiny dry run.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pasig import config
from pasig.config import load, out_dir, path, profile, root
from pasig.gates import GateFailure, read_gate, record_gate, require_gates
from pasig.utils.gitinfo import git_sha

STAGES = ["compute", "variants", "grid", "finals", "baselines", "thresholds", "checks", "freeze"]
NEURAL = {"variants", "grid", "finals"}


# ----------------------------------------------------------------------------- shared state

def _state_path() -> Path:
    return out_dir("tables") / "phase5_state.json"


def state() -> dict:
    p = _state_path()
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(**kw) -> None:
    s = state() | kw
    _state_path().write_text(json.dumps(s, indent=1, default=str))


def machine() -> dict:
    p = root() / "configs" / "compute.yaml"           # G0 block always lives in the real configs/compute.yaml
    return (yaml.safe_load(p.read_text()) or {}).get("machine", {}) if p.exists() else {}


def hcfg() -> dict:
    return load("model_hybrid")


def rehearsal() -> bool:
    return profile() == "rehearsal"


def _log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------- stage: compute (G14, G13, G15)

def stage_compute() -> dict:
    from pasig.train.compute import call_module, plan_hours, run_bench
    mach = machine()
    name = (mach.get("name") or "unknown-machine") if not rehearsal() else "rehearsal"
    probe = call_module(["--probe"], device="gpu")   # not the CPU env (CUDA_VISIBLE_DEVICES=-1), or no GPU is ever seen
    gpu = probe["nvidia_gpu"]
    devices = (["gpu"] if gpu else []) + ["cpu"]
    det = {d: call_module(["--determinism", "--device", d], device=d) for d in devices}
    record_gate("G14", "determinism", True,
                "With enable_op_determinism(), a 1-epoch run repeated twice on the device gives identical predictions "
                "(atol 1e-6); pass -> final runs on that device with determinism on; fail -> final runs on CPU with "
                "determinism on, the grid may stay on GPU without determinism",
                {"results": det, "nvidia_gpu": gpu},
                decision=("gpu reproducible" if gpu and det["gpu"]["identical"] else
                          ("gpu NOT reproducible -> finals on cpu" if gpu else "no NVIDIA GPU: cpu")))
    from pasig.phase3 import load_windows
    target = int(load_windows("train")["valid"][:, 0].sum())
    bench_cfg = {"quick": rehearsal(), "train_cases": 800 if rehearsal() else None,
                 "measure_epochs": 1 if rehearsal() else 5}
    bench, plans = {}, {}
    for d in devices:
        runs = []
        for k in (1, 2):                 # run twice back to back; use the second (Section 12)
            runs.append(run_bench(d, out_dir("compute") / name / f"{d}_run{k}", target, **bench_cfg))
        bench[d] = runs[-1]
        plans[d] = plan_hours(runs[-1]["summary"], runs[-1]["projection"])
    nodet = None
    if gpu:
        nodet = run_bench("gpu", out_dir("compute") / name / "gpu_nodet", target, no_determinism=True, **bench_cfg)
    use_gpu = bool(gpu and det["gpu"]["identical"] and plans["gpu"]["plan_hours_5_seeds"] < plans["cpu"]["plan_hours_5_seeds"])
    device_final = "gpu" if use_gpu else "cpu"
    record_gate("G13", "device", True,
                "GPU iff TF lists an NVIDIA GPU and G14 passes on GPU and the bench_compute GPU plan hours < CPU plan "
                "hours (second of two back-to-back runs)",
                {"gpu": gpu, "plan_hours": plans, "g14": det, "target_cases": target}, decision=device_final)
    # grid device: GPU (determinism off) is allowed when the GPU is faster even if G14 failed on it (D15)
    grid_on_gpu = bool(gpu and (use_gpu or (nodet is not None and nodet["summary"]["median_epoch_s"] < bench["cpu"]["summary"]["median_epoch_s"])))
    device_grid = "gpu" if grid_on_gpu else "cpu"
    grid_det = device_grid == "cpu" or use_gpu
    hours = plans[device_final]["plan_hours_5_seeds"]
    seeds_wf = 5 if hours <= 10 else 3
    record_gate("G15", "seed_budget", True,
                "Recommended-plan projection on the chosen device <= 10 h -> 5 seeds for walk-forward and ablations, "
                "else 3 (final configs stay 5); projection includes 6 ablation variants, the D20 weight choice and "
                "the G11/G12 runs",
                {"device": device_final, "projection": plans[device_final], "limit_h": 10},
                decision=f"seeds_wf_ablation: {seeds_wf}" + ("" if seeds_wf == 5 else " (flag Chapter rows 4 and 26)"))
    comp = {"machine": mach, "device": device_final, "device_final": device_final, "device_grid": device_grid,
            "grid_deterministic": bool(grid_det), "final_deterministic": True, "seeds_final": 5,
            "seeds_wf_ablation": seeds_wf, "plan_hours": plans, "target_cases": target, "profile": profile()}
    config.write_yaml("compute", comp, header="# GENERATED by G0 (machine), G13, G14, G15. Do not edit by hand.")
    for d, b in bench.items():   # keep summary + projection with the laptop's name (Section 12)
        dst = out_dir("compute") / name
        shutil.copy(Path(b["dir"]) / "summary.json", dst / f"summary_{d}.json")
        shutil.copy(Path(b["dir"]) / "projection.csv", dst / f"projection_{d}.csv")
    return comp


# ----------------------------------------------------------------------------- neural helpers

def _data(inp):
    from pasig.train.data import load_split, rehearsal_cases
    return load_split("train", inp, max_cases=rehearsal_cases()), load_split("val", inp)


def _seeds(key: str) -> list[int]:
    from pasig.train.loop import hp
    return list(hp()[key])


def _variant_runs(name: str, inp, cfg, seeds, det) -> pd.DataFrame:
    from pasig.train.grid import run_grid
    rows = []
    tr, va = _data(inp)
    for s in seeds:
        t = run_grid(f"{name}_seed{s}", "hybrid", [cfg], inp, tr, va, s, det, log=_log)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def _switch(default: pd.Series, alt: pd.Series) -> tuple[bool, dict]:
    sd = float(default.std(ddof=1)) if len(default) > 1 else 0.0
    sw = bool(alt.mean() < default.mean() - sd)
    return sw, {"default_mean": float(default.mean()), "default_sd": sd, "alt_mean": float(alt.mean()),
                "default_runs": default.round(6).tolist(), "alt_runs": alt.round(6).tolist()}


# ----------------------------------------------------------------------------- stage: variants (G11, G12)

def stage_variants() -> dict:
    from pasig.models.hybrid import HybridConfig
    from pasig.train.data import Inputs
    require_gates("G13", "G14", "G15")
    c = hcfg()
    cfg = HybridConfig(**c["reference"], head_units=c["head_units"])
    seeds = _seeds("gate_seeds")
    base = Inputs.from_config()
    base.use_constituents, base.add_back, base.tag = False, True, "tide_addback"
    cons = Inputs(base.seq, base.tab, use_constituents=True, add_back=True, tag="cons_addback")
    r_def = _variant_runs("g11_default", base, cfg, seeds, True)
    r_con = _variant_runs("g11_constituents", cons, cfg, seeds, True)
    sw11, ev11 = _switch(r_def["val_rmse_mean"], r_con["val_rmse_mean"])
    record_gate("G11", "tidal_input_variant", True, hcfg()["gate_switch_rule"] + " (predicted tide only vs + constituent pairs)",
                ev11 | {"n_constituent_columns": int(_data(cons)[0]["X"]["tab"].shape[1] - _data(base)[0]["X"]["tab"].shape[1])},
                decision="keep constituent pairs" if sw11 else "keep predicted tide only")
    default_g12 = cons if sw11 else base
    r_g12_def = r_con if sw11 else r_def
    direct = Inputs(base.seq, base.tab, use_constituents=sw11, add_back=False, tag="direct")
    r_dir = _variant_runs("g12_direct", direct, cfg, seeds, True)
    sw12, ev12 = _switch(r_g12_def["val_rmse_mean"], r_dir["val_rmse_mean"])
    record_gate("G12", "output_form", True, hcfg()["gate_switch_rule"] + " (add-back default vs direct level)", ev12,
                decision="switch to direct level" if sw12 else "keep add-back")
    config.write_yaml("model_inputs", {"use_constituents": bool(sw11), "add_back": not sw12,
                                       "g11": ev11, "g12": ev12, "seeds": seeds, "reference_config": cfg.as_dict()},
                      header="# GENERATED by G11 and G12 (NB05). Do not edit by hand.")
    return {"g11": ev11, "g12": ev12, "use_constituents": sw11, "add_back": not sw12, "default_inputs": default_g12.tag}


# ----------------------------------------------------------------------------- stage: grid (D12, G19)

def stage_grid() -> dict:
    from pasig.train.data import Inputs
    from pasig.train.grid import best, lstm_configs, run_grid, stage1_configs, stage2_configs
    require_gates("G11", "G12")
    comp = load("compute")
    det, seed = bool(comp["grid_deterministic"]), int(hcfg()["grid_seed"])
    inp = Inputs.from_config()
    tr, va = _data(inp)
    s1 = run_grid("grid_stage1", "hybrid", stage1_configs(), inp, tr, va, seed, det, log=_log)
    w1 = best(s1)
    s2 = run_grid("grid_stage2", "hybrid", stage2_configs(w1), inp, tr, va, seed, det, log=_log)
    w2 = best(pd.concat([s2], ignore_index=True))
    edges = g19_edges(w2)
    record_gate("G19", "hyperparameter_edge", True,
                "Record whether the winner sits on a range boundary (D14 ranges; never widened after seeing results)",
                {"winner": w2.as_dict(), "on_edge": edges, "stage1_winner": w1.as_dict()},
                decision=("winner on edge: " + ", ".join(k for k, v in edges.items() if v)) if any(edges.values())
                else "winner interior in every hyperparameter")
    ls = run_grid("lstm_grid", "lstm", lstm_configs(w1.batch), inp, tr, va, seed, det, log=_log)
    wl = best(ls)
    save_state(hybrid_final=w2.as_dict(), hybrid_stage1=w1.as_dict(), lstm_final=wl.as_dict())
    return {"stage1": s1, "stage2": s2, "lstm": ls, "hybrid": w2.as_dict(), "lstm_winner": wl.as_dict(), "edges": edges}


def g19_edges(w) -> dict:
    r = hcfg()["ranges"]
    d = w.as_dict()
    return {k: bool(d[k] in (lo, hi)) for k, (lo, hi) in r.items()}


# ----------------------------------------------------------------------------- stage: finals

def stage_finals() -> dict:
    from pasig.models.hybrid import HybridConfig
    from pasig.train.data import Inputs
    from pasig.train.loop import train_run
    require_gates("G19")
    st = state()
    inp = Inputs.from_config()
    tr, va = _data(inp)
    seeds = _seeds("final_seeds")
    frozen = out_dir("frozen")
    path_csv = out_dir("runs") / "finals.csv"
    done = pd.read_csv(path_csv) if path_csv.exists() else pd.DataFrame()
    preds = []
    for kind, key in (("hybrid", "hybrid_final"), ("lstm", "lstm_final")):
        cfg = HybridConfig(**st[key])
        for s in seeds:
            tag = f"final_{kind}"
            have = len(done) and ((done["tag"] == tag) & (done["seed"] == s)).any()
            if have:
                rd = done[(done["tag"] == tag) & (done["seed"] == s)].iloc[-1]["run_dir"]
                preds.append(pd.read_parquet(Path(rd) / "preds_val.parquet"))
                continue
            r = train_run(kind, cfg, inp, tr, va, s, True, tag)
            shutil.copy(Path(r["row"]["run_dir"]) / "model.keras", frozen / f"{kind}_seed{s}.keras")
            done = pd.concat([done, pd.DataFrame([r["row"]])], ignore_index=True)
            done.to_csv(path_csv, index=False)
            preds.append(pd.read_parquet(Path(r["row"]["run_dir"]) / "preds_val.parquet"))
            _log(f"  final {kind} seed {s}: val RMSE {r['row']['val_rmse_mean']:.4f} m")
    pv = pd.concat(preds, ignore_index=True)
    pv["model"] = pv["model"].str.split(":").str[0]
    pv.to_parquet(out_dir("tables") / "preds_val_neural.parquet", index=False)
    return {"finals": done}


# ----------------------------------------------------------------------------- stage: baselines

def stage_baselines() -> dict:
    import joblib

    from pasig.features.tide import load_tide
    from pasig.models import baselines as B
    from pasig.train.data import Inputs, load_split
    from pasig.train.loop import preds_frame, rmse_by_lead
    inp = Inputs.from_config()
    tr, va = load_split("train", inp), load_split("val", inp)
    tide = load_tide("pandacan")
    master = pd.read_parquet(path("processed") / "hourly_master.parquet")
    obs = master["pandacan"].where(~master["pandacan_filled"])
    obs = obs[master["split"].eq("train")]
    to = B.TideOnly(tide, load("baselines")["tide_only"]["seasonal_harmonics"]).fit(obs)
    preds = {"persistence": B.persistence(va), "tide_only": to.predict(va), "tide_resid": B.tide_resid(va, tide)}
    ridge = B.fit_ridge(tr, va)
    knn = B.fit_knn_regressor(tr, va)
    preds["ridge"], preds["knn"] = ridge["pred_val"], knn["pred_val"]
    rows = [{"model": k, **{f"val_rmse_lead{i + 1}": v for i, v in enumerate(rmse_by_lead(p, va["y_raw"], va["valid"]))}}
            for k, p in preds.items()]
    tab = pd.DataFrame(rows)
    tab["val_rmse_mean"] = tab.filter(like="val_rmse_lead").mean(axis=1)
    tab.to_csv(out_dir("tables") / "baselines_val_rmse.csv", index=False)
    pv = pd.concat([preds_frame(va, p, k) for k, p in preds.items()], ignore_index=True)
    pv.to_parquet(out_dir("tables") / "preds_val_baselines.parquet", index=False)
    levels = {"high_water": load("alert_levels")["high_water_m"], "flood_risk": load("alert_levels")["flood_risk_m"]}
    clf_lr = B.fit_classifiers(tr, va, levels, "logistic")
    clf_knn = B.fit_classifiers(tr, va, levels, "knn")
    clf = pd.concat([clf_lr, clf_knn], ignore_index=True)
    clf.to_csv(out_dir("tables") / "classifiers_val.csv", index=False)
    frozen = out_dir("frozen")
    joblib.dump({"ridge": ridge["models"], "logistic": {k: v["model"] for k, v in clf_lr.attrs["fitted"].items()},
                 "tide_only_coef": to.coef_}, frozen / "baselines.joblib")
    probs = []
    for kind, c in (("logistic", clf_lr), ("knn", clf_knn)):
        for (inset, lvl, lead), v in c.attrs["fitted"].items():
            mv = va["valid"][:, lead - 1]
            probs.append(pd.DataFrame({"t": va["t_index"][mv], "lead": lead, "model": f"{kind}_{inset}", "level": lvl,
                                       "prob": v["prob_val"], "truth": va["y_raw"][mv, lead - 1] >= levels[lvl]}))
    pd.concat(probs, ignore_index=True).to_parquet(out_dir("tables") / "probs_val_classifiers.parquet", index=False)
    save_state(baselines={"ridge_alpha": ridge["alpha"], "ridge_search": ridge["search"], "knn": knn["params"],
                          "knn_search": knn["search"], "tide_only_coef": to.coef_.tolist()})
    return {"rmse": tab, "classifiers": clf, "ridge": ridge["alpha"], "knn": knn["params"]}


# ----------------------------------------------------------------------------- stage: thresholds (D11)

def _val_preds() -> pd.DataFrame:
    parts = [pd.read_parquet(out_dir("tables") / "preds_val_baselines.parquet")]
    p = out_dir("tables") / "preds_val_neural.parquet"
    if p.exists():
        parts.append(pd.read_parquet(p))
    return pd.concat(parts, ignore_index=True)


def _wide(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = g.pivot_table(index="t", columns="lead", values=["y_pred", "y_true"], aggfunc="first").sort_index()
    v = g.pivot_table(index="t", columns="lead", values="valid", aggfunc="first").sort_index().astype(bool)
    return w["y_pred"].to_numpy(), w["y_true"].to_numpy(), v.to_numpy()


def stage_thresholds() -> dict:
    from pasig.models.offsets import best_offset
    cfg = load("baselines")["trigger_offsets"]
    al = load("alert_levels")
    levels = {"high_water": al["high_water_m"], "flood_risk": al["flood_risk_m"]}
    pv = _val_preds()
    offsets = {}
    for model, g in pv.groupby("model"):
        seeds = sorted(g["seed"].unique())
        arrays = [_wide(g[g.seed.eq(s)]) for s in seeds]
        y_true, valid = arrays[0][1], arrays[0][2]
        offsets[model] = {lvl: best_offset([a[0] for a in arrays], y_true, valid, v, cfg) | {"level_m": v}
                          for lvl, v in levels.items()}
    clf = pd.read_csv(out_dir("tables") / "classifiers_val.csv")
    clf_out = {}
    for _, r in clf.dropna(subset=["threshold"]).iterrows():
        key = f"{r['model']}_{r['input_set']}"
        clf_out.setdefault(key, {}).setdefault(r["level"], {})[int(r["lead"])] = {
            "threshold": float(r["threshold"]), "f1_val": float(r["f1_val"]),
            "params": {k: (None if pd.isna(r.get(k)) else r.get(k)) for k in ("C", "class_weight", "k", "weights") if k in r}}
    config.write_yaml("thresholds", {"trigger_offsets": offsets, "classifiers": _plain(clf_out), "levels": levels,
                                     "rule": cfg["objective"], "git_sha": git_sha()},
                      header="# GENERATED in NB05 (D11 trigger offsets, classifier thresholds). Frozen by G17.")
    return {"offsets": offsets, "classifiers": clf_out}


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


# ----------------------------------------------------------------------------- stage: checks (G16)

def stage_checks() -> dict:
    pv = _val_preds()
    pv = pv[pv["valid"]]
    rm = (pv.assign(se=(pv.y_pred - pv.y_true) ** 2).groupby(["model", "seed", "lead"])["se"].mean() ** 0.5)
    rm = rm.groupby(["model", "lead"]).mean().unstack("lead")
    pers = rm.loc["persistence"]
    later = [c for c in rm.columns if c >= 2]                 # leads >= 2 (lead 1 may tie)
    beats = {m: bool((rm.loc[m, later] < pers.loc[later]).all()) for m in rm.index if m != "persistence"}
    checklist = bug_checklist(pv)
    bug = [k for k, v in checklist.items() if not v["ok"]]
    passed = not bug
    record_gate("G16", "persistence_sanity", passed,
                "Each model's validation RMSE < persistence at leads >= 2 (lead 1 may tie). If not, run the bug "
                "checklist (Section 9.7): pass if no bug is found (record 'beats persistence: no') or a bug is fixed",
                {"val_rmse_by_lead": rm.round(5).to_dict(orient="index"), "beats_persistence": beats,
                 "checklist": checklist},
                decision=("bug found: " + ", ".join(bug)) if bug else
                ("all models beat persistence" if all(beats.values()) else
                 "beats persistence: no for " + ", ".join(k for k, v in beats.items() if not v) + " (no bug found)"))
    rm.to_csv(out_dir("tables") / "val_rmse_by_model_lead.csv")
    if not passed:
        raise GateFailure("G16: bug checklist found a problem; fix before freezing")
    return {"rmse": rm, "beats": beats, "checklist": checklist}


def bug_checklist(pv: pd.DataFrame) -> dict:
    """Section 9.7 checks that can be run automatically."""
    from pasig.features.scaling import load_scalers
    from pasig.phase3 import load_windows
    out = {}
    w = load_windows("val")
    m = pd.read_parquet(path("processed") / "hourly_master.parquet")
    t = pd.DatetimeIndex(w["t_index"])
    obs = m["pandacan"].where(~m["pandacan_filled"])
    ok = all(np.allclose(w["y"][w["valid"][:, h - 1], h - 1],
                         obs.reindex(t + pd.Timedelta(hours=h)).to_numpy()[w["valid"][:, h - 1]], atol=1e-4) for h in range(1, 7))
    out["1_target_alignment"] = {"ok": bool(ok)}
    sc = load_scalers(path("artifacts") / "scalers.json")
    out["2_target_scaler_is_pandacan_channel"] = {"ok": bool(np.allclose(sc["y"].lo, sc["seq"].lo[0]) and
                                                            np.allclose(sc["y"].span, sc["seq"].span[0]))}
    from pasig.features.scaling import encode_targets
    Y = encode_targets(w["y"], w["valid"], sc["y"])
    out["3_masked_loss_invalid_minus1"] = {"ok": bool(np.all(Y[~w["valid"]] == -1.0) and np.all(Y[w["valid"]] > -0.5))}
    six = pd.read_parquet(path("interim") / "pagasa_6h.parquet")
    lag = (six["available_from_pht"] - six["period_end_pht"]).dropna().unique()
    out["4_rain_available_after_period_end"] = {"ok": bool(len(lag) == 1 and lag[0] == pd.Timedelta(hours=1))}
    fin = out_dir("runs") / "finals.csv"
    if fin.exists():
        f = pd.read_csv(fin)
        out["5_learning"] = {"ok": bool((f["train_loss_last"] < f["train_loss_first"]).all()),
                             "best_epoch_gt_1_share": float((f["best_epoch"] > 1).mean())}
    hours = pv.groupby("model").apply(lambda g: frozenset(zip(g["t"], g["lead"], strict=True)))
    out["6_same_scored_hours"] = {"ok": bool(hours.nunique() == 1)}
    return out


# ----------------------------------------------------------------------------- stage: freeze (G17)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def frozen_files() -> list[Path]:
    gen = config._config_file          # Phase 5 GENERATED configs live in runs/rehearsal/configs in rehearsal
    files = [gen("alert_levels"), gen("selected_features"), gen("model_inputs"), gen("thresholds"), gen("compute"),
             path("artifacts") / "scalers.json", path("artifacts") / "feature_names.json",
             root() / "configs" / "eval.yaml"]      # H3/H4 rules and NB06/NB07 settings: pre-registered before G18
    files += sorted(path("artifacts").glob("tide_*.json"))
    files += sorted(out_dir("frozen").glob("*"))
    return files


def stage_freeze(tag: bool = False) -> dict:
    require_gates("G16")
    files = frozen_files()
    missing = [str(f) for f in files if not f.exists()]
    ckpt = [f for f in files if f.suffix == ".keras"]
    ok = not missing and len(ckpt) > 0
    manifest = {str(f.relative_to(root())): sha256(f) for f in files if f.exists()}
    mpath = config._config_file("compute").parent / "frozen_manifest.json"
    mpath.write_text(json.dumps({"files": manifest, "git_sha": git_sha(), "profile": profile()}, indent=1))
    record_gate("G17", "settings_frozen", ok,
                "alert_levels, selected_features, model_inputs, thresholds, compute, scalers, tide coefficients and final "
                "checkpoints exist; their SHA-256 written to configs/frozen_manifest.json; tag settings-frozen",
                {"n_files": len(manifest), "missing": missing, "checkpoints": [str(c.name) for c in ckpt],
                 "manifest": str(mpath.relative_to(root()))},
                decision="tag settings-frozen" if ok else None)
    if not ok:
        raise GateFailure(f"G17: missing {missing}")
    if tag and not rehearsal():
        subprocess.check_call(["git", "tag", "-f", "settings-frozen"], cwd=root())
    return {"manifest": manifest}


# ----------------------------------------------------------------------------- driver

def run_stage(name: str, tag: bool = False):
    fn = {"compute": stage_compute, "variants": stage_variants, "grid": stage_grid, "finals": stage_finals,
          "baselines": stage_baselines, "thresholds": stage_thresholds, "checks": stage_checks,
          "freeze": lambda: stage_freeze(tag)}[name]
    if name in NEURAL:
        from pasig.utils.hardware import configure_device
        comp = load("compute")
        dev = comp.get("device_grid" if name == "grid" else "device_final", "cpu")
        configure_device(dev)
    return fn()


def run(stages: list[str] | None = None, tag: bool = False) -> int:
    """Run stages in order, each in its own process (device fixed before TensorFlow starts)."""
    require_gates("G0", "G6", "G8", "G9", "G10")
    for s in stages or STAGES:
        env = dict(os.environ)
        if s in NEURAL:
            dev = load("compute").get("device_grid" if s == "grid" else "device_final", "cpu")
            if dev == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = "-1"
        cmd = [sys.executable, "scripts/run_phase.py", "--phase", "5", "--stage", s] + (["--tag"] if tag else [])
        _log(f"== stage {s} ({profile()})")
        code = subprocess.call(cmd, cwd=root(), env=env)
        if code != 0:
            _log(f"stage {s} failed (exit {code}); fix and re-run from this stage: --stage {s}")
            return code
    return 0


def summary() -> dict:
    out = {g: (read_gate(g) or {}).get("decision") for g in ("G13", "G14", "G15", "G11", "G12", "G19", "G16", "G17")}
    return out
