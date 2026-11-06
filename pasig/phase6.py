"""Phase 6 / Notebook 06: the one look at the 2024 test season (Section 10.1, SOP 4). Gate G18; H3, H4.

Order (nothing here may change a frozen setting):
  unseal     G18: tag settings-frozen is an ancestor of HEAD, every file in frozen_manifest.json has its frozen
             SHA-256, and the test has not been opened before (TEST_UNSEALED marker) -> write the marker
  predict    2024 windows built here with the frozen tide, scalers and feature builder; predictions of every
             model (5 neural seeds, baselines, classifiers) -> preds_test.parquet, probs_test.parquet
  evaluate   level metrics and skill, hourly alerts with the frozen offsets/thresholds, equal-recall false alarms
             (D19), PR-AUC, events, Diebold-Mariano + Holm, moving-block bootstrap CIs; H3 and H4 verdicts

Rehearsal profile (PASIG_PROFILE=rehearsal): the 2023 validation windows stand in for the test season so every
line runs end to end; 2024 is never read, the marker and all outputs go under runs/rehearsal/. Rehearsal numbers
are meaningless (offsets and thresholds were tuned on the same 2023 hours).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pasig import config
from pasig.config import load, out_dir, path, profile, root
from pasig.eval import alerts as A
from pasig.eval import events as E
from pasig.eval.bootstrap import block_ci
from pasig.eval.dm_test import diebold_mariano
from pasig.eval.metrics import level_metrics, skill
from pasig.gates import GateFailure, read_gate, record_gate, require_gates
from pasig.utils.gitinfo import git_sha

PHT = timezone(timedelta(hours=8))
LEADS = 6
NEURAL = ("hybrid", "lstm")
BASELINES = ("persistence", "tide_only", "tide_resid", "ridge", "knn")
SIMPLE = ("persistence", "tide_only", "tide_resid")
STAGES = ["unseal", "predict", "evaluate"]


def rehearsal() -> bool:
    return profile() == "rehearsal"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _tables() -> Path:
    return out_dir("tables")


def _levels() -> dict:
    al = load("alert_levels")
    return {"high_water": float(al["high_water_m"]), "flood_risk": float(al["flood_risk_m"])}


# ----------------------------------------------------------------------------- G18 (unseal)

def marker_path() -> Path:
    return out_dir("gates") / "TEST_UNSEALED.json"


def _tag_is_ancestor() -> tuple[bool, str]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "settings-frozen^{commit}"], cwd=root(),
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return False, "tag settings-frozen not found"
    anc = subprocess.call(["git", "merge-base", "--is-ancestor", "settings-frozen", "HEAD"], cwd=root()) == 0
    return anc, sha


def manifest_check() -> dict:
    from pasig.phase5 import sha256
    mpath = config._config_file("compute").parent / "frozen_manifest.json"
    if not mpath.exists():
        return {"ok": False, "manifest": str(mpath), "missing": ["frozen_manifest.json"], "changed": []}
    man = json.loads(mpath.read_text())["files"]
    missing = [f for f in man if not (root() / f).exists()]
    changed = [f for f, h in man.items() if (root() / f).exists() and sha256(root() / f) != h]
    return {"ok": not missing and not changed, "manifest": str(mpath.relative_to(root())), "n_files": len(man),
            "missing": missing, "changed": changed}


def stage_unseal(rerun_reason: str | None = None) -> dict:
    """G18 (blocking). A second opening needs a written reason; it is appended to the marker and must be reported
    in the transparency log (Section 0.3)."""
    require_gates("G17")
    tag_ok, tag_sha = (True, "rehearsal: tag not required") if rehearsal() else _tag_is_ancestor()
    man = manifest_check()
    mk = marker_path()
    prior = json.loads(mk.read_text()) if mk.exists() else None
    if prior and rehearsal() and not rerun_reason:
        rerun_reason = "rehearsal re-run (2023 stand-in)"
    first = prior is None
    reopen_ok = first or bool(rerun_reason)
    ok = tag_ok and man["ok"] and reopen_ok
    record_gate("G18", "test_unsealed", ok,
                "settings-frozen tag is an ancestor of HEAD; every frozen file matches its SHA-256 in "
                "frozen_manifest.json; the test season has not been opened before (TEST_UNSEALED marker), or a "
                "re-run reason is logged",
                {"tag_ancestor": tag_ok, "tag_commit": tag_sha, "manifest": man,
                 "previous_openings": (prior or {}).get("openings", []), "rerun_reason": rerun_reason},
                decision=("test season opened" if first else "test season re-opened: " + str(rerun_reason)) if ok else None)
    if not ok:
        why = [] if tag_ok else [tag_sha if tag_sha.startswith("tag") else "settings-frozen is not an ancestor of HEAD"]
        why += [] if man["ok"] else [f"frozen files changed/missing: {man['changed'] + man['missing']}"]
        why += [] if reopen_ok else ["test already opened (pass --rerun-reason and log it in the transparency log)"]
        raise GateFailure("G18: " + "; ".join(why))
    opening = {"time": datetime.now(PHT).isoformat(timespec="seconds"), "git_sha": git_sha(),
               "reason": rerun_reason or "first opening", "profile": profile()}
    mk.write_text(json.dumps({"openings": (prior or {}).get("openings", []) + [opening]}, indent=1))
    return {"tag": tag_sha, "manifest": man, "opening": opening}


# ----------------------------------------------------------------------------- test windows

def build_frozen_windows(split: str) -> dict:
    """Windows for one split with the frozen builder, tide coefficients and training medians (Phase 3 code path).
    Outside the test split the 2024 rows are dropped first, exactly as Phase 3 does, so 2024 is never read."""
    from pasig.features import windows as W
    from pasig.features.tide import load_tide
    names = json.loads((path("artifacts") / "feature_names.json").read_text())
    master = pd.read_parquet(path("processed") / "hourly_master.parquet")
    ev = pd.read_parquet(path("processed") / "eval_hours.parquet")
    six = pd.read_parquet(path("interim") / "pagasa_6h.parquet")
    if split != "test":
        master = master[master["season"] < min(load("splits")["test_years"])]
        ev = ev.loc[master.index]
    w = W.build_windows(master, ev, six, load_tide("pandacan"), split, names["eligible_stations"],
                        names["daily_fill_training_median"])
    if list(w["tab_names"]) != names["tab"] or list(w["seq_names"]) != names["seq"]:
        raise AssertionError("windows do not have the frozen feature layout")
    w["t_index"] = pd.DatetimeIndex(w["t_index"])
    return w


def test_windows() -> dict:
    """2024 arrays (after G18). Rehearsal: the 2023 windows rebuilt through the same code path, checked to equal the
    arrays Phase 3 saved (so the builder used for 2024 is the frozen one)."""
    require_gates("G18")
    if not rehearsal():
        return build_frozen_windows("test")
    from pasig.phase3 import load_windows
    w, saved = build_frozen_windows("val"), load_windows("val")
    for k in ("X_seq", "X_tab", "X_tab_cons", "y", "valid"):
        if not np.array_equal(w[k], saved[k], equal_nan=k != "valid"):
            raise AssertionError(f"rebuilt 2023 windows differ from Phase 3 in {k}")
    if not (pd.DatetimeIndex(saved["t_index"]) == w["t_index"]).all():
        raise AssertionError("rebuilt 2023 windows differ from Phase 3 in t_index")
    return w


def rainy_flag(w: dict) -> np.ndarray:
    """Rainy hour: >= rainy_rain24_mm of Science Garden rain in the previous 24 h, from reports available at t."""
    names = json.loads((path("artifacts") / "feature_names.json").read_text())
    return w["X_tab"][:, names["tab"].index("sg_rain_24h_in")] >= float(load("eval")["rainy_rain24_mm"])


# ----------------------------------------------------------------------------- predict

def _finals() -> pd.DataFrame:
    return pd.read_csv(out_dir("runs") / "finals.csv")


def test_seed(kind: str) -> int:
    """The seed with the median validation RMSE (lower median for an even count): used for DM and H4."""
    f = _finals()
    f = f[f["tag"].eq(f"final_{kind}")].drop_duplicates("seed", keep="last").sort_values(["val_rmse_mean", "seed"])
    return int(f.iloc[(len(f) - 1) // 2]["seed"])


def stage_predict() -> dict:
    import joblib
    import keras

    import pasig.models.hybrid  # noqa: F401  (registers LevelAtT and the losses for loading)
    from pasig.features.tide import load_tide
    from pasig.models import baselines as B
    from pasig.phase5 import state
    from pasig.train.data import Inputs, load_split
    from pasig.train.loop import predict_levels, preds_frame
    require_gates("G18")
    inp = Inputs.from_config()
    w = test_windows()
    te = load_split("test", inp, windows=w)
    tr = load_split("train", inp)
    rainy = rainy_flag(w)
    st = state()
    tide = load_tide("pandacan")
    frozen = out_dir("frozen")
    preds = []
    for kind in NEURAL:
        fin = _finals()
        for s in sorted(fin.loc[fin["tag"].eq(f"final_{kind}"), "seed"].unique()):
            m = keras.models.load_model(frozen / f"{kind}_seed{s}.keras", compile=False)
            preds.append(preds_frame(te, predict_levels(m, te["X"], te["y_scaler"], kind), kind, int(s)))
            _log(f"  {kind} seed {s}: predicted")
    bj = joblib.load(frozen / "baselines.joblib")
    to = B.TideOnly(tide, load("baselines")["tide_only"]["seasonal_harmonics"])
    to.coef_ = np.asarray(bj["tide_only_coef"])
    Fte, Ftr = B.flat(te["X"]), B.flat(tr["X"])
    kp = st["baselines"]["knn"]
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    knn = [KNeighborsRegressor(n_neighbors=kp["k"], weights=kp["weights"])
           .fit(Ftr[tr["valid"][:, h]], tr["y_raw"][tr["valid"][:, h], h]) for h in range(LEADS)]
    base = {"persistence": B.persistence(te), "tide_only": to.predict(te), "tide_resid": B.tide_resid(te, tide),
            "ridge": np.column_stack([m.predict(Fte) for m in bj["ridge"]]),
            "knn": np.column_stack([m.predict(Fte) for m in knn])}
    preds += [preds_frame(te, p, k) for k, p in base.items()]
    pt = pd.concat(preds, ignore_index=True)
    pt["rainy"] = np.tile(np.repeat(rainy, LEADS), len(preds))
    pt.to_parquet(_tables() / "preds_test.parquet", index=False)
    # classifiers: LR from the frozen joblib; KNN refitted on training with its frozen k/weights (deterministic)
    thr = load("thresholds")["classifiers"]
    levels = _levels()
    probs = []
    for key, by_level in thr.items():
        kind, inset = key.split("_", 1)
        cur = inset == "a_current_hour"
        Fte_c, Ftr_c = B.flat(te["X"], cur), B.flat(tr["X"], cur)
        for lvl, by_lead in by_level.items():
            for lead, spec in by_lead.items():
                h = int(lead) - 1
                if kind == "logistic":
                    m = bj["logistic"][(inset, lvl, h + 1)]
                else:
                    mt = tr["valid"][:, h]
                    m = KNeighborsClassifier(n_neighbors=int(spec["params"]["k"]), weights=spec["params"]["weights"]) \
                        .fit(Ftr_c[mt], tr["y_raw"][mt, h] >= levels[lvl])
                mv = te["valid"][:, h]
                p = m.predict_proba(Fte_c[mv])[:, 1]
                probs.append(pd.DataFrame({"t": te["t_index"][mv], "lead": h + 1, "model": key, "level": lvl, "prob": p,
                                           "alert": p >= float(spec["threshold"]),
                                           "truth": te["y_raw"][mv, h] >= levels[lvl], "rainy": rainy[mv]}))
    pd.concat(probs, ignore_index=True).to_parquet(_tables() / "probs_test.parquet", index=False)
    summ = {"cases": int(len(te["Y"])), **{f"valid_lead{h + 1}": int(te["valid"][:, h].sum()) for h in range(LEADS)},
            "rainy_cases": int(rainy.sum()), "stand_in": "2023 validation (rehearsal)" if rehearsal() else "2024 test",
            "seeds": {k: test_seed(k) for k in NEURAL}}
    (_tables() / "test_windows_summary.json").write_text(json.dumps(summ, indent=1))
    return summ


# ----------------------------------------------------------------------------- evaluate helpers

def _subsets(g: pd.DataFrame) -> dict:
    return {"all": g, "rainy": g[g["rainy"]], "dry": g[~g["rainy"]]}


def level_table(pt: pd.DataFrame) -> pd.DataFrame:
    """RMSE/MAE/R2/bias per model x lead x subset; neural models as mean +- SD over seeds."""
    rows = []
    v = pt[pt["valid"]]
    for (model, lead), g in v.groupby(["model", "lead"]):
        for sub, gs in _subsets(g).items():
            per = [level_metrics(x.y_pred.to_numpy(), x.y_true.to_numpy()) for _, x in gs.groupby("seed") if len(x)]
            if not per:
                continue
            d = pd.DataFrame(per)
            rows.append({"model": model, "lead": lead, "subset": sub, "n": int(d["n"].iloc[0]), "seeds": len(d),
                         **{k: float(d[k].mean()) for k in ("rmse", "mae", "r2", "bias")},
                         **{f"{k}_sd": float(d[k].std(ddof=1)) if len(d) > 1 else np.nan for k in ("rmse", "mae")}})
    return pd.DataFrame(rows)


def strongest_simple() -> str:
    b = pd.read_csv(_tables() / "baselines_val_rmse.csv").set_index("model")
    return str(b.loc[list(SIMPLE), "val_rmse_mean"].idxmin())


def skill_table(lv: pd.DataFrame, ref_simple: str) -> pd.DataFrame:
    rows = []
    for (lead, sub), g in lv.groupby(["lead", "subset"]):
        r = g.set_index("model")["rmse"]
        for m in r.index:
            rows.append({"model": m, "lead": lead, "subset": sub, "rmse": r[m],
                         "skill_vs_persistence": skill(r[m], r["persistence"]),
                         f"skill_vs_{ref_simple}": skill(r[m], r[ref_simple])})
    return pd.DataFrame(rows)


def _one_seed(pt: pd.DataFrame, seeds: dict) -> pd.DataFrame:
    keep = pt["model"].isin(BASELINES) | pd.Series(
        [s == seeds.get(m, s) for m, s in zip(pt["model"], pt["seed"], strict=True)], index=pt.index)
    return pt[keep]


def _aligned(pt: pd.DataFrame, a: str, b: str, lead: int, sub: str) -> pd.DataFrame:
    g = pt[pt["lead"].eq(lead) & pt["valid"] & pt["model"].isin([a, b])]
    if sub == "rainy":
        g = g[g["rainy"]]
    w = g.pivot_table(index="t", columns="model", values="y_pred", aggfunc="first").dropna()
    y = g.drop_duplicates("t").set_index("t")["y_true"].reindex(w.index)
    return w.assign(y=y).sort_index()


def dm_table(p1: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.multitest import multipletests
    rows = []
    for sub in ("all", "rainy"):
        for other in BASELINES + ("lstm",):
            for h in range(1, LEADS + 1):
                w = _aligned(p1, "hybrid", other, h, sub)
                if len(w) < 30:
                    rows.append({"subset": sub, "baseline": other, "lead": h, "n": len(w), "note": "too few hours"})
                    continue
                r = diebold_mariano((w["hybrid"] - w["y"]).to_numpy(), (w[other] - w["y"]).to_numpy(), h)
                rows.append({"subset": sub, "baseline": other, "lead": h, **r})
    d = pd.DataFrame(rows)
    ok = d["p_value"].notna()
    alpha = float(load("eval")["dm"]["alpha"])
    d.loc[ok, "p_holm"] = multipletests(d.loc[ok, "p_value"], alpha=alpha, method="holm")[1]
    d["hybrid_better"] = ok & (d["p_holm"] < alpha) & (d["mean_loss_diff"] < 0)
    d["hybrid_worse"] = ok & (d["p_holm"] < alpha) & (d["mean_loss_diff"] > 0)
    return d


def h3_verdict(dm: pd.DataFrame) -> dict:
    a = dm[dm["subset"].eq("all")]
    n_sig, n = int(a["hybrid_better"].sum()), len(a)
    v = "supported" if n_sig == n and n > 0 else ("partly supported" if n_sig > 0 else "not supported")
    return {"verdict": v, "significant_pairs_all_hours": n_sig, "pairs": n,
            "pairs_hybrid_significantly_worse": int(a["hybrid_worse"].sum()),
            "rainy_significant_pairs": int(dm.loc[dm["subset"].eq("rainy"), "hybrid_better"].sum())}


def hourly_alerts(pt: pd.DataFrame, pr: pd.DataFrame) -> pd.DataFrame:
    """Alerts with the frozen trigger offsets (forecast >= level - offset) and classifier thresholds."""
    th = load("thresholds")
    levels = _levels()
    rows = []
    v = pt[pt["valid"]]
    for (model, lead), g in v.groupby(["model", "lead"]):
        for lvl, L in levels.items():
            off = th["trigger_offsets"][model][lvl]["offset_m"]
            for sub, gs in _subsets(g).items():
                per = [A.counts((x.y_pred >= L - off).to_numpy(), (x.y_true >= L).to_numpy())
                       for _, x in gs.groupby("seed") if len(x)]
                if per:
                    d = pd.DataFrame(per)
                    rows.append({"model": model, "level": lvl, "lead": lead, "subset": sub, "offset_m": off,
                                 "seeds": len(d), **d.mean(numeric_only=True).to_dict(),
                                 "f1_sd": float(d["f1"].std(ddof=1)) if len(d) > 1 else np.nan})
    for (model, lvl, lead), g in pr.groupby(["model", "level", "lead"]):
        for sub, gs in _subsets(g).items():
            if len(gs):
                rows.append({"model": model, "level": lvl, "lead": lead, "subset": sub, "seeds": 1,
                             **A.counts(gs["alert"].to_numpy(), gs["truth"].to_numpy())})
    return pd.DataFrame(rows)


def chosen_input_sets() -> dict:
    """Per classifier family, the input set with the higher mean 2023 F1 at high water (fixed before the test)."""
    clf = load("thresholds")["classifiers"]
    out = {}
    for fam in ("logistic", "knn"):
        f1 = {k.split("_", 1)[1]: np.mean([v["f1_val"] for v in clf[k].get("high_water", {}).values()] or [np.nan])
              for k in clf if k.startswith(fam + "_")}
        f1 = {k: v for k, v in f1.items() if np.isfinite(v)}
        if f1:
            out[fam] = f"{fam}_{max(f1, key=f1.get)}"
    return out


def _scores(p1: pd.DataFrame, pr: pd.DataFrame, clf_key: str, level: str, lead: int, sub: str) -> pd.DataFrame:
    """Hybrid score (forecast level) and classifier probability on the same hours, in time order."""
    L = _levels()[level]
    hy = p1[p1["model"].eq("hybrid") & p1["lead"].eq(lead) & p1["valid"]].set_index("t")
    c = pr[pr["model"].eq(clf_key) & pr["level"].eq(level) & pr["lead"].eq(lead)].set_index("t")
    j = hy[["y_pred", "y_true", "rainy"]].join(c[["prob"]], how="inner").sort_index()
    if sub == "rainy":
        j = j[j["rainy"]]
    return j.assign(truth=j["y_true"] >= L)


def equal_recall(p1: pd.DataFrame, pr: pd.DataFrame, families: dict) -> pd.DataFrame:
    """FP at Recall >= target (D19) and PR-AUC for the hybrid vs each classifier family, with moving-block CIs of
    the differences (hybrid - classifier)."""
    ec = load("eval")
    reps, block, seed = int(ec["bootstrap_reps"]), int(ec["block_h"]), int(ec["seed"])
    targets = [float(ec["recall_point"])] + [float(x) for x in ec["recall_sensitivity"]]
    if rehearsal():
        reps = min(reps, 200)
    rows = []
    for level in _levels():
        for fam, key in families.items():
            for lead in range(1, LEADS + 1):
                for sub in ("all", "rainy"):
                    j = _scores(p1, pr, key, level, lead, sub)
                    sh, sc, tr = j["y_pred"].to_numpy(), j["prob"].to_numpy(), j["truth"].to_numpy()
                    base = {"level": level, "family": fam, "classifier": key, "lead": lead, "subset": sub,
                            "n": len(j), "positives": int(tr.sum())}
                    if tr.sum() < 2 or (~tr).sum() < 2:
                        rows.append(base | {"note": "too few positives"})
                        continue
                    r = base | {"pr_auc_hybrid": A.pr_auc(sh, tr), "pr_auc_clf": A.pr_auc(sc, tr)}
                    r["pr_auc_diff"] = r["pr_auc_hybrid"] - r["pr_auc_clf"]

                    def d_auc(i, sh=sh, sc=sc, tr=tr):
                        t = tr[i]
                        return A.pr_auc(sh[i], t) - A.pr_auc(sc[i], t) if t.any() and (~t).any() else None
                    r["pr_auc_diff_lo"], r["pr_auc_diff_hi"], _ = block_ci(d_auc, len(j), reps, block, seed)
                    for tg in targets:
                        a, b = A.fp_at_recall(sh, tr, tg), A.fp_at_recall(sc, tr, tg)
                        k = f"r{int(round(tg * 100))}"
                        r |= {f"fp_hybrid_{k}": a["fp"], f"recall_hybrid_{k}": a["recall"],
                              f"fp_clf_{k}": b["fp"], f"recall_clf_{k}": b["recall"], f"fp_diff_{k}": a["fp"] - b["fp"]}
                        if tg == targets[0]:
                            def d_fp(i, sh=sh, sc=sc, tr=tr, tg=tg):
                                t = tr[i]
                                if not t.any():
                                    return None
                                return A.fp_at_recall(sh[i], t, tg)["fp"] - A.fp_at_recall(sc[i], t, tg)["fp"]
                            r[f"fp_diff_{k}_lo"], r[f"fp_diff_{k}_hi"], _ = block_ci(d_fp, len(j), reps, block, seed)
                    rows.append(r)
    return pd.DataFrame(rows)


def h4_verdict(er: pd.DataFrame) -> dict:
    k = f"r{int(round(float(load('eval')['recall_point']) * 100))}"
    g = er[er["level"].eq("high_water") & er["subset"].eq("all")].copy()
    if f"fp_diff_{k}_hi" not in g:
        return {"verdict": "not testable", "reason": "no high-water positives"}
    g["win"] = (g[f"fp_diff_{k}"] < 0) & (g[f"fp_diff_{k}_hi"] < 0) & (g["pr_auc_diff"] > 0) & (g["pr_auc_diff_lo"] > 0)
    n_win, n = int(g["win"].sum()), int(g[f"fp_diff_{k}"].notna().sum())
    fams = sorted(g["family"].unique())
    v = "supported" if n == LEADS * len(fams) and n_win == n and len(fams) == 2 else \
        ("partly supported" if n_win > 0 else "not supported")
    return {"verdict": v, "wins": n_win, "pairs": n, "families": fams,
            "wins_by_pair": {f"{r.family}_lead{r.lead}": bool(r.win) for r in g.itertuples()}}


def event_table(pt: pd.DataFrame, pr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observed events (from the target hours) and per-model detection rate / warning time with event-bootstrap
    CIs. Neural models: every seed (rows keyed by seed)."""
    ec = load("eval")
    th = load("thresholds")
    obs = pt[pt["valid"] & pt["model"].eq("persistence")].assign(target=lambda d: d["t"] + pd.to_timedelta(d["lead"], "h"))
    level_obs = obs.drop_duplicates("target").set_index("target")["y_true"].sort_index()
    ev_rows, det_rows = [], []
    for lvl, L in _levels().items():
        ev = E.events(level_obs, L, int(ec["event_merge_gap_h"]))
        ev_rows.append(ev.assign(level=lvl))
        if ev.empty:
            continue
        for model, g in pt[pt["valid"]].groupby("model"):
            off = th["trigger_offsets"][model][lvl]["offset_m"]
            for lead, gl in g.groupby("lead"):
                for s, gs in gl.groupby("seed"):
                    tgt = pd.DatetimeIndex(gs.loc[gs.y_pred >= L - off, "t"]) + pd.Timedelta(hours=int(lead))
                    det_rows.append(E.detect(ev, tgt, int(lead), int(ec["event_detect_window_h"]))
                                    .assign(model=model, level=lvl, lead=lead, seed=s))
        for (model, lead), gs in pr[pr["level"].eq(lvl)].groupby(["model", "lead"]):
            tgt = pd.DatetimeIndex(gs.loc[gs["alert"], "t"]) + pd.Timedelta(hours=int(lead))
            det_rows.append(E.detect(ev, tgt, int(lead), int(ec["event_detect_window_h"]))
                            .assign(model=model, level=lvl, lead=lead, seed=-1))
    evs = pd.concat(ev_rows, ignore_index=True) if ev_rows else pd.DataFrame()
    if not det_rows:
        return evs, pd.DataFrame()
    det = pd.concat(det_rows, ignore_index=True)
    rows = []
    for (model, lvl, lead, s), g in det.groupby(["model", "level", "lead", "seed"]):
        lo, hi = E.event_bootstrap_ci(g["detected"].to_numpy(), int(ec["bootstrap_reps"]), int(ec["seed"]))
        rows.append({"model": model, "level": lvl, "lead": lead, "seed": s, "events": len(g),
                     "detected": int(g["detected"].sum()), "detection_rate": float(g["detected"].mean()),
                     "rate_lo": lo, "rate_hi": hi, "median_warning_h": float(g["warning_h"].median())
                     if g["detected"].any() else np.nan})
    return evs, pd.DataFrame(rows)


def rmse_ci(p1: pd.DataFrame, ref: str) -> pd.DataFrame:
    """Moving-block 95% CI of RMSE(hybrid) - RMSE(ref) per lead (all hours)."""
    ec = load("eval")
    reps = min(int(ec["bootstrap_reps"]), 200) if rehearsal() else int(ec["bootstrap_reps"])
    rows = []
    for h in range(1, LEADS + 1):
        w = _aligned(p1, "hybrid", ref, h, "all")
        ea, eb = (w["hybrid"] - w["y"]).to_numpy(), (w[ref] - w["y"]).to_numpy()

        def d(i, ea=ea, eb=eb):
            return float(np.sqrt(np.mean(ea[i] ** 2)) - np.sqrt(np.mean(eb[i] ** 2)))
        lo, hi, _ = block_ci(d, len(w), reps, int(ec["block_h"]), int(ec["seed"]))
        rows.append({"reference": ref, "lead": h, "rmse_diff": d(np.arange(len(w))), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- stage: evaluate

def stage_evaluate() -> dict:
    require_gates("G18")
    T = _tables()
    pt = pd.read_parquet(T / "preds_test.parquet")
    pr = pd.read_parquet(T / "probs_test.parquet")
    seeds = {k: test_seed(k) for k in NEURAL}
    p1 = _one_seed(pt, seeds)
    ref = strongest_simple()
    lv = level_table(pt)
    sk = skill_table(lv, ref)
    dm = dm_table(p1)
    ci = pd.concat([rmse_ci(p1, "persistence"), rmse_ci(p1, ref)] if ref != "persistence" else [rmse_ci(p1, ref)])
    al = hourly_alerts(pt, pr)
    fams = chosen_input_sets()
    er = equal_recall(p1, pr, fams)
    evs, det = event_table(pt, pr)
    for name, df in (("test_level_metrics", lv), ("test_skill", sk), ("test_dm_holm", dm), ("test_rmse_diff_ci", ci),
                     ("test_alerts_hourly", al), ("test_equal_recall", er), ("test_events", evs),
                     ("test_event_detection", det)):
        df.to_csv(T / f"{name}.csv", index=False)
    h3, h4 = h3_verdict(dm), h4_verdict(er)
    res = {"profile": profile(), "stand_in": "2023 validation (rehearsal)" if rehearsal() else "2024 test",
           "git_sha": git_sha(), "test_seeds": seeds, "strongest_simple_baseline": ref,
           "classifier_input_sets": fams, "H3": h3, "H4": h4}
    (T / "test_results.json").write_text(json.dumps(res, indent=1, default=str))
    _figures(pt, lv, ref)
    return res | {"tables": {"level": lv, "skill": sk, "dm": dm, "alerts": al, "equal_recall": er,
                             "events": evs, "detection": det}}


def _figures(pt: pd.DataFrame, lv: pd.DataFrame, ref: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = out_dir("figures")
    a = lv[lv["subset"].eq("all")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, g in a.groupby("model"):
        g = g.sort_values("lead")
        ax.errorbar(g["lead"], g["rmse"], yerr=g["rmse_sd"].fillna(0), marker="o", capsize=2, label=m,
                    lw=2.2 if m == "hybrid" else 1.1)
    ax.set(xlabel="lead (h)", ylabel="RMSE (m)", title=f"Test RMSE by lead ({'rehearsal stand-in' if rehearsal() else '2024'})")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(F / "test_rmse_by_lead.png", dpi=150)
    plt.close(fig)
    seeds = {k: test_seed(k) for k in NEURAL}
    p = _one_seed(pt, seeds)
    g = p[p["lead"].eq(3) & p["model"].isin(["hybrid", ref])]
    w = g.pivot_table(index="t", columns="model", values="y_pred", aggfunc="first").sort_index()
    y = g.drop_duplicates("t").set_index("t")["y_true"].reindex(w.index)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    tt = w.index + pd.Timedelta(hours=3)
    ax.plot(tt, y, color="k", lw=1, label="observed")
    for c in w.columns:
        ax.plot(tt, w[c], lw=0.8, label=f"{c} (lead 3)")
    for lvl, L in _levels().items():
        ax.axhline(L, ls="--", lw=0.7, color="r" if lvl == "flood_risk" else "orange")
    ax.set(ylabel="Pandacan level (m)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(F / "test_series_lead3.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- driver

def run(stages: list[str] | None = None, rerun_reason: str | None = None) -> dict:
    require_gates("G17")
    out = {}
    for s in stages or STAGES:
        _log(f"== NB06 stage {s} ({profile()})")
        out[s] = {"unseal": lambda: stage_unseal(rerun_reason), "predict": stage_predict,
                  "evaluate": stage_evaluate}[s]()
    return out


def summary() -> dict:
    p = _tables() / "test_results.json"
    return {"G18": (read_gate("G18") or {}).get("decision"),
            **({k: v for k, v in json.loads(p.read_text()).items() if k in ("H3", "H4")} if p.exists() else {})}
