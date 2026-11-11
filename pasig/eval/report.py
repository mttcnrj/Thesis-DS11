"""Chapter IV assembler: one command rebuilds every Chapter IV table and the two summary figures from the
Notebook 05-07 outputs (Section 15: "every Chapter IV table and figure regenerates from one run of Notebooks
06-07"). It only reads finished outputs; it never opens the test season, trains or re-scores anything.

    python -m pasig.eval.report          # or: make report
    PASIG_PROFILE=rehearsal python -m pasig.eval.report   # rehearsal outputs -> runs/rehearsal/reports/chapter4

Writes reports/chapter4/: table_IV-NN_<name>.csv (numbers at full precision), chapter4.md (every table formatted
for the manuscript, the H3/H4 verdicts and the figure list), fig_walkforward_rmse.png, fig_ablation_rmse_diff.png,
and compute_runtime.csv (hardware and measured training time, for Chapter III).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pasig.config import load, out_dir, profile
from pasig.utils.gitinfo import git_sha

LEADS = list(range(1, 7))
LEVEL_MODELS = ["hybrid", "lstm", "ridge", "knn", "tide_resid", "tide_only", "persistence"]
NEURAL = ("hybrid", "lstm")
LABEL = {"hybrid": "Hybrid LSTM-GRU", "lstm": "Standalone LSTM", "ridge": "Ridge", "knn": "KNN regressor",
         "tide_resid": "Tide + persisted residual", "tide_only": "Tide only", "persistence": "Persistence",
         "logistic_b_window": "Logistic regression (window)", "knn_b_window": "KNN classifier (window)",
         "logistic_a_current_hour": "Logistic regression (current hour)",
         "knn_a_current_hour": "KNN classifier (current hour)", "logistic": "Logistic regression", "knn_clf": "KNN classifier"}
ABL_LABEL = {"no_granger": "No Granger selection", "correlation_selection": "Correlation selection",
             "no_rainfall": "No rainfall", "no_tabular": "No tabular path", "second_lstm": "GRU -> second LSTM",
             "weighted_loss": "Weighted loss (D20)"}
RUN_GROUPS = [("g11_|g12_", "Notebook 05 variants (G11, G12)"), ("grid_stage1", "Hybrid grid, stage 1"),
              ("grid_stage2", "Hybrid grid, stage 2"), ("lstm_grid", "Standalone LSTM grid"),
              ("final_hybrid", "Final hybrid seeds"), ("final_lstm", "Final LSTM seeds"),
              ("abl_weighted_w", "Ablation: weighted loss incl. D20 weight choice"), ("abl_", "Ablations (other five)"),
              ("wf20", "Walk-forward folds")]


# ----------------------------------------------------------------------------- small helpers

def _dirs() -> tuple[Path, Path, Path, Path]:
    tables, figures, gates = out_dir("tables"), out_dir("figures"), out_dir("gates")
    dest = tables.parent / "chapter4"
    dest.mkdir(parents=True, exist_ok=True)
    return tables, figures, gates, dest


def _csv(tables: Path, name: str) -> pd.DataFrame | None:
    p = tables / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _json(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def _gate(gates: Path, code: str) -> dict | None:
    hits = sorted(gates.glob(f"{code}_*.json"))
    return _json(hits[0]) if hits else None


def _pm(mean: float, sd: float | None, nd: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return ""
    if sd is None or (isinstance(sd, float) and np.isnan(sd)) or sd == 0:
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {sd:.{nd}f}"


def _ci(lo: float, hi: float, nd: int = 3) -> str:
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def _md(df: pd.DataFrame, index: bool = True) -> str:
    """GitHub-flavoured Markdown table (no tabulate dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    rows = [" | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) for r in d.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + [f"| {r} |" for r in rows])


def _fmt(df: pd.DataFrame, nd: int) -> pd.DataFrame:
    return df.apply(lambda c: c.map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}"))


def _lead_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [f"{c} h" if isinstance(c, (int, np.integer)) else c for c in df.columns]
    return df


def _order(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    keep = [m for m in models if m in df.index] + [m for m in df.index if m not in models]
    out = df.loc[keep]
    out.index = [LABEL.get(m, m) for m in out.index]
    out.index.name = "Model"
    return out


# ----------------------------------------------------------------------------- tables

def t_validation(tables: Path, gates: Path) -> list[tuple[str, str, pd.DataFrame, pd.DataFrame, str]]:
    out = []
    v = _csv(tables, "val_rmse_by_model_lead")
    if v is not None:
        raw = v.set_index("model")
        raw.columns = [int(c) for c in raw.columns]
        out.append(("validation_rmse", "Validation (2023) RMSE (m) by lead; neural models: mean of the final seeds",
                    raw, _lead_cols(_order(_fmt(raw, 4), LEVEL_MODELS)), ""))
    rows = []
    for code, what in [("G11", "Tidal input"), ("G12", "Output form"), ("G13", "Training device"),
                       ("G14", "GPU determinism"), ("G15", "Seeds for walk-forward and ablations"),
                       ("G16", "Persistence sanity"), ("G19", "Hyperparameter at a range edge"),
                       ("G17", "Settings frozen"), ("G18", "Test season opened")]:
        g = _gate(gates, code)
        if g:
            rows.append({"Gate": code, "Question": what, "Decision": g.get("decision"), "Passed": g.get("passed")})
    if rows:
        d = pd.DataFrame(rows)
        out.append(("nb05_decisions", "Notebook 05-06 gate decisions (rules fixed before the results)", d, d.set_index("Gate"), ""))
    st = _json(tables / "phase5_state.json")
    if st and "hybrid_final" in st:
        hp = pd.DataFrame({"Hybrid": st.get("hybrid_final", {}), "Standalone LSTM": st.get("lstm_final", {})})
        hp = hp.map(lambda v: "" if pd.isna(v) else f"{v:g}" if isinstance(v, (int, float)) else str(v))
        out.append(("final_hyperparameters", "Selected hyperparameters (two-stage grid on 2023)", hp, hp.rename_axis("Setting"), ""))
    return out


def t_test_levels(tables: Path) -> list:
    lv = _csv(tables, "test_level_metrics")
    if lv is None:
        return []
    out = []
    for metric, nd, title in [("rmse", 4, "RMSE (m)"), ("mae", 4, "MAE (m)"), ("r2", 3, "R²")]:
        for sub, sub_t in [("all", "all hours"), ("rainy", "rainy hours (≥ 5 mm in the previous 24 h)"), ("dry", "dry hours")]:
            if metric != "rmse" and sub != "all":
                continue
            a = lv[lv["subset"].eq(sub)]
            raw = a.pivot(index="model", columns="lead", values=metric)
            sdcol = f"{metric}_sd" if f"{metric}_sd" in a.columns else None
            sd = a.pivot(index="model", columns="lead", values=sdcol) if sdcol else None
            fmt = raw.copy().astype(object)
            for m in raw.index:
                for h in raw.columns:
                    s = sd.loc[m, h] if (sd is not None and m in NEURAL) else None
                    fmt.loc[m, h] = _pm(raw.loc[m, h], s, nd)
            n = int(a["n"].max())
            out.append((f"test_{metric}_{sub}", f"Test (2024) {title}, {sub_t}, by lead (n ≈ {n} hours; neural: mean ± SD over 5 seeds)",
                        raw, _lead_cols(_order(fmt, LEVEL_MODELS)), ""))
    sk = _csv(tables, "test_skill")
    if sk is not None:
        a = sk[sk["subset"].eq("all") & sk["model"].isin(["hybrid", "lstm", "ridge", "knn", "tide_resid"])]
        for col, ref in [("skill_vs_persistence", "persistence"), ("skill_vs_tide_resid", "tide + persisted residual")]:
            raw = a.pivot(index="model", columns="lead", values=col)
            out.append((f"test_{col}", f"Test skill (1 − RMSE/RMSE_ref) vs {ref}, all hours",
                        raw, _lead_cols(_order(_fmt(raw, 3), LEVEL_MODELS)), ""))
    ci = _csv(tables, "test_rmse_diff_ci")
    if ci is not None:
        fmt = ci.assign(cell=[f"{d:+.4f} {_ci(lo, hi, 4)}" for d, lo, hi in zip(ci.rmse_diff, ci.lo, ci.hi, strict=True)])
        w = fmt.pivot(index="reference", columns="lead", values="cell")
        out.append(("test_rmse_diff_ci", "Hybrid RMSE minus reference RMSE (m), moving-block bootstrap 95% CI (24 h blocks)",
                    ci, _lead_cols(_order(w, LEVEL_MODELS)), "Negative = hybrid better; a CI that excludes 0 is a clear difference."))
    return out


def t_dm(tables: Path, res: dict | None) -> list:
    dm = _csv(tables, "test_dm_holm")
    if dm is None:
        return []
    out = []
    for sub, sub_t in [("all", "all hours"), ("rainy", "rainy hours")]:
        a = dm[dm["subset"].eq(sub)].copy()
        a["cell"] = [("better" if b else "worse" if w else "n.s.") + f" (p={p:.3g})"
                     for b, w, p in zip(a.hybrid_better, a.hybrid_worse, a.p_holm, strict=True)]
        w = a.pivot(index="baseline", columns="lead", values="cell")
        note = ""
        if res and sub == "all":
            h3 = res["H3"]
            note = (f"**H3: {h3['verdict']}** — hybrid significantly better in {h3['significant_pairs_all_hours']} of "
                    f"{h3['pairs']} (baseline, lead) pairs on all hours, significantly worse in {h3['pairs_hybrid_significantly_worse']}; "
                    f"rainy hours: {h3['rainy_significant_pairs']} pairs. DM on squared errors, Newey-West variance, HLN "
                    f"correction, Holm over the 72 tests; neural models use the median-validation seed.")
        out.append((f"test_dm_{sub}", f"Diebold-Mariano: hybrid vs each baseline, {sub_t} (Holm-adjusted p)",
                    a.drop(columns="cell"), _lead_cols(_order(w, LEVEL_MODELS)), note))
    return out


def t_alerts(tables: Path, res: dict | None) -> list:
    out = []
    al = _csv(tables, "test_alerts_hourly")
    if al is not None:
        for lvl in ("high_water", "flood_risk"):
            a = al[al["level"].eq(lvl) & al["subset"].eq("all")].copy()
            a["cell"] = [f"{f:.2f} (P {p:.2f}, R {r:.2f}; FP {fp:.0f}, FN {fn:.0f})"
                         for f, p, r, fp, fn in zip(a.f1, a.precision, a.recall, a.fp, a.fn, strict=True)]
            w = a.pivot(index="model", columns="lead", values="cell")
            order = LEVEL_MODELS + ["logistic_b_window", "knn_b_window", "logistic_a_current_hour", "knn_a_current_hour"]
            out.append((f"test_alerts_{lvl}", f"Hourly alerts at the {lvl.replace('_', '-')} level, 2024, all hours: F1 (precision, "
                        "recall; false alarms, misses) with the frozen offsets/thresholds; neural: mean over seeds",
                        a.drop(columns="cell"), _lead_cols(_order(w, order)), ""))
    er = _csv(tables, "test_equal_recall")
    if er is not None:
        a = er[er["level"].eq("high_water") & er["subset"].eq("all")].copy()
        rows = []
        wins = (res or {}).get("H4", {}).get("wins_by_pair", {})
        for _, r in a.iterrows():
            rows.append({"Classifier": LABEL.get(r.classifier, r.classifier), "Lead": f"{int(r.lead)} h",
                         "False alarms at R≥0.90 (hybrid / clf)": f"{int(r.fp_hybrid_r90)} / {int(r.fp_clf_r90)}",
                         "FP difference 95% CI": _ci(r.fp_diff_r90_lo, r.fp_diff_r90_hi, 0),
                         "PR-AUC (hybrid / clf)": f"{r.pr_auc_hybrid:.3f} / {r.pr_auc_clf:.3f}",
                         "PR-AUC difference 95% CI": _ci(r.pr_auc_diff_lo, r.pr_auc_diff_hi),
                         "Hybrid wins": "yes" if wins.get(f"{r.family}_lead{int(r.lead)}") else "no"})
        note = ""
        if res:
            h4 = res["H4"]
            note = (f"**H4: {h4['verdict']}** — the hybrid wins {h4['wins']} of {h4['pairs']} (family, lead) pairs. A win needs fewer "
                    "false alarms at Recall ≥ 0.90 with the CI of the difference below 0 AND a higher PR-AUC with its CI above 0 (D19).")
        out.append(("test_equal_recall_h4", "H4: hybrid vs classifiers at equal recall, high-water level, 2024",
                    a, pd.DataFrame(rows).set_index("Classifier"), note))
        cols = ["classifier", "lead", "fp_hybrid_r80", "fp_clf_r80", "fp_hybrid_r95", "fp_clf_r95"]
        sens = a[cols].copy()
        sens["classifier"] = sens["classifier"].map(lambda c: LABEL.get(c, c))
        sens["lead"] = sens["lead"].map(lambda h: f"{int(h)} h")
        sens = sens.rename(columns={"classifier": "Classifier", "lead": "Lead",
                                    "fp_hybrid_r80": "FP at R 0.80, hybrid", "fp_clf_r80": "FP at R 0.80, classifier",
                                    "fp_hybrid_r95": "FP at R 0.95, hybrid", "fp_clf_r95": "FP at R 0.95, classifier"})
        out.append(("test_equal_recall_sensitivity", "False alarms at Recall 0.80 and 0.95 (descriptive, D19), high-water level",
                    a[cols], sens.set_index("Classifier"), "Recall is reached on the test year for this comparison only (D19)."))
    ev = _csv(tables, "test_event_detection")
    if ev is not None:
        g = ev.groupby(["model", "level", "lead"]).agg(events=("events", "first"), detected=("detected", "mean"),
                                                       rate=("detection_rate", "mean"), warning=("median_warning_h", "median")).reset_index()
        for lvl in ("high_water", "flood_risk"):
            a = g[g["level"].eq(lvl)].copy()
            a["cell"] = [f"{d:g}/{e} ({r:.0%}; {w:.1f} h)" if not np.isnan(w) else f"{d:g}/{e} ({r:.0%})"
                         for d, e, r, w in zip(a.detected, a.events, a.rate, a.warning, strict=True)]
            w = a.pivot(index="model", columns="lead", values="cell")
            order = LEVEL_MODELS + ["logistic_b_window", "knn_b_window", "logistic_a_current_hour", "knn_a_current_hour"]
            out.append((f"test_events_{lvl}", f"Event detection, {lvl.replace('_', '-')} events in 2024: detected/events (rate; median warning)",
                        a.drop(columns="cell"), _lead_cols(_order(w, order)),
                        "Detected if an alert's target hour falls from 2 h before to 2 h after the first exceedance; neural: mean over seeds."))
    return out


def t_walkforward(tables: Path) -> list:
    lv = _csv(tables, "walkforward_level_metrics")
    if lv is None:
        return []
    a = lv[lv["subset"].eq("all")]
    raw = a.pivot_table(index=["fold", "model"], columns="lead", values="rmse")
    fmt = raw.copy().astype(object)
    sd = a.pivot_table(index=["fold", "model"], columns="lead", values="rmse_sd")
    for idx in raw.index:
        for h in raw.columns:
            fmt.loc[idx, h] = _pm(raw.loc[idx, h], sd.loc[idx, h] if idx[1] in NEURAL and idx in sd.index else None, 4)
    fmt = fmt.reset_index()
    fmt["fold"] = [f"{f}{'' if f != a['fold'].max() else ' (not independent)'}" for f in fmt["fold"]]
    fmt["model"] = fmt["model"].map(lambda m: LABEL.get(m, m))
    out = [("walkforward_rmse", "Walk-forward RMSE (m), all hours: train on 2018..(Y−1), evaluate on Y (hybrid: mean ± SD over seeds)",
            raw.reset_index(), _lead_cols(fmt.set_index(["fold", "model"])),
            "The 2023 fold chose the hyperparameters, so it is reported but not counted as independent evidence.")]
    dm = _csv(tables, "walkforward_dm_holm")
    if dm is not None:
        s = dm.groupby(["fold", "reference"]).agg(better=("a_better", "sum"), worse=("a_worse", "sum"), tests=("lead", "count")).reset_index()
        s["Result (of 6 leads)"] = [f"better {b}, worse {w}" for b, w in zip(s.better, s.worse, strict=True)]
        w = s.pivot(index="reference", columns="fold", values="Result (of 6 leads)")
        out.append(("walkforward_dm", "Walk-forward Diebold-Mariano: hybrid vs each baseline per fold (Holm over 72 tests; descriptive)",
                    dm, _order(w, LEVEL_MODELS), ""))
    return out


def t_ablations(tables: Path) -> list:
    out = []
    d = _csv(tables, "ablation_rmse_diff")
    if d is not None:
        d2 = d.copy()
        d2["cell"] = [f"{x:+.4f}{' *' if b else ''}" for x, b in zip(d2.rmse_diff, d2.beyond_seed_noise, strict=True)]
        w = d2.pivot(index="variant", columns="lead", values="cell")
        w = w.loc[[v for v in ABL_LABEL if v in w.index]]
        w.index = [ABL_LABEL[v] for v in w.index]
        w.index.name = "Variant"
        main = d.drop_duplicates("lead").set_index("lead")["rmse_main"]
        note = ("Variant RMSE minus main-hybrid RMSE over the same seeds (m), 2024 all hours; * = |difference| larger than the "
                "main hybrid's seed SD (the pre-registered noise rule). Main hybrid RMSE: "
                + ", ".join(f"{h} h {main[h]:.4f}" for h in main.index) + ".")
        out.append(("ablations_rmse_diff", "Ablations (Table III-5): change in test RMSE vs the main hybrid", d, _lead_cols(w), note))
    dm = _csv(tables, "ablation_dm_holm")
    if dm is not None:
        s = dm.groupby("model").agg(better=("a_better", "sum"), worse=("a_worse", "sum")).reset_index()
        s["Variant"] = s["model"].str.replace("abl_", "", regex=False).map(lambda v: ABL_LABEL.get(v, v))
        s = s.rename(columns={"better": "Leads where variant is significantly better", "worse": "Leads where variant is significantly worse"})
        out.append(("ablations_dm", "Ablations: Diebold-Mariano vs the main hybrid (Holm over 36 tests; descriptive)",
                    dm, s.drop(columns="model").set_index("Variant"), ""))
    runs = _csv(tables, "ablation_runs")
    if runs is not None:
        g = runs.groupby("variant").agg(seeds=("seed", "nunique"), val_rmse=("val_rmse_mean", "mean"),
                                        val_rmse_sd=("val_rmse_mean", "std"), epochs=("epochs", "mean")).reset_index()
        g["Validation RMSE (m)"] = [_pm(m, s, 4) for m, s in zip(g.val_rmse, g.val_rmse_sd, strict=True)]
        g["variant"] = g["variant"].map(lambda v: ABL_LABEL.get(v, v))
        out.append(("ablations_validation", "Ablation runs on 2023 validation (mean ± SD over seeds)", runs,
                    g[["variant", "seeds", "Validation RMSE (m)", "epochs"]].set_index("variant").round(1), ""))
    d20 = _json(tables / "ablation_d20.json")
    if d20:
        rows = []
        for h in map(str, LEADS):
            m, w = d20.get("hybrid_same_seeds", {}).get(h), d20.get("abl_weighted_loss", {}).get(h)
            if not (m and w):
                continue
            for key, lab in [("rmse", "RMSE (m)"), ("rmse_high_water_hours", "RMSE at high-water hours (m)"),
                             ("mean_error_at_flood_risk_peaks", "Mean error at flood-risk peaks (m)"),
                             ("recall_high_water", "Recall, high water"), ("recall_flood_risk", "Recall, flood risk")]:
                rows.append({"lead": int(h), "measure": lab, "main": m[key], "weighted": w[key]})
        if rows:
            r = pd.DataFrame(rows)
            r["cell"] = [f"{a:.3f} → {b:.3f}" for a, b in zip(r.main, r.weighted, strict=True)]
            w = r.pivot(index="measure", columns="lead", values="cell")
            search = d20.get("search", {})
            note = (f"Weight chosen on 2023 by high-water F1: w = {d20.get('weight')} (searched "
                    + ", ".join(f"w={k}: F1 {v['f1_high_water_val']:.3f}" for k, v in search.items()) + ").")
            out.append(("d20_weighted_loss", "D20 weighted loss: main hybrid → weighted-loss hybrid, 2024 (same seeds)",
                        r, _lead_cols(w), note))
    return out


def t_case_aviles(tables: Path) -> list:
    out = []
    e = _csv(tables, "case_study_events")
    if e is not None:
        f = e.copy()
        f["carina"] = f["carina"].map({True: "Carina", False: ""})
        out.append(("case_study_events", "2024 flood-risk events (≥ 12.0 m) used in the case study", e,
                    f.set_index("event"), ""))
    s = _csv(tables, "case_study_summary")
    if s is not None:
        a = s[s["lead"].isin([1, 3, 6])].copy()
        a["cell"] = [f"{d}/{n}" + (f", {w:.1f} h" if not np.isnan(w) else "") + (f", {p:+.2f} m" if not np.isnan(p) else "")
                     for d, n, w, p in zip(a.detected, a.events, a.median_warning_h, a.mean_error_at_peak, strict=True)]
        w = a.pivot(index="model", columns="lead", values="cell")
        order = LEVEL_MODELS + ["logistic_b_window", "knn_b_window"]
        out.append(("case_study_summary", "Case study, leads 1/3/6 h: events detected, median warning, mean forecast error at the peak",
                    s, _lead_cols(_order(w, order)), "Neural models: the median-validation seed. Negative error = forecast below the observed peak."))
    av = _csv(tables, "aviles_check")
    if av is not None:
        out.append(("aviles_check", "Deferred check: alert levels vs Aviles pumping-station records (descriptive; levels unchanged)",
                    av, av.set_index(["level", "data"]).round(3), ""))
    return out


def compute_runtime(dest: Path) -> pd.DataFrame | None:
    """Hardware (G0/G13 benchmark) and measured training time from runs/*/timing.json (Chapter III, Section 15)."""
    runs = out_dir("runs")
    rows = []
    for d in sorted(p for p in runs.iterdir() if p.is_dir() and (p / "timing.json").exists()):
        t = json.loads((d / "timing.json").read_text())
        name = re.sub(r"^\d{8}_\d{4}_", "", d.name)
        grp = next((lab for pat, lab in RUN_GROUPS if re.match(pat, name)), "other")
        rows.append({"group": grp, "seconds": t.get("seconds", np.nan), "epochs": t.get("epochs", np.nan)})
    if not rows:
        return None
    r = pd.DataFrame(rows)
    order = [lab for _, lab in RUN_GROUPS] + ["other"]
    g = r.groupby("group").agg(runs=("seconds", "size"), hours=("seconds", lambda s: s.sum() / 3600),
                               mean_epochs=("epochs", "mean")).reindex([o for o in order if o in set(r.group)])
    g.loc["Total"] = [g["runs"].sum(), g["hours"].sum(), r["epochs"].mean()]
    g = g.round(2)
    g.to_csv(dest / "compute_runtime.csv")
    return g


# ----------------------------------------------------------------------------- figures

def figures(tables: Path, dest: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    made = []
    lv = _csv(tables, "walkforward_level_metrics")
    if lv is not None:
        a = lv[lv["subset"].eq("all")]
        folds = sorted(a["fold"].unique())
        fig, axes = plt.subplots(1, len(folds), figsize=(4 * len(folds), 3.6), sharey=True)
        for ax, f in zip(np.atleast_1d(axes), folds, strict=True):
            for m, g in a[a["fold"].eq(f)].groupby("model"):
                if m == "persistence":
                    continue
                g = g.sort_values("lead")
                ax.errorbar(g["lead"], g["rmse"], yerr=g["rmse_sd"].fillna(0), marker="o", capsize=2,
                            label=LABEL.get(m, m), lw=2.2 if m == "hybrid" else 1.1)
            ax.set(title=f"{f}{' (not independent)' if f == max(folds) else ''}", xlabel="lead (h)")
        np.atleast_1d(axes)[0].set_ylabel("RMSE (m)")
        np.atleast_1d(axes)[0].legend(fontsize=7)
        fig.suptitle("Walk-forward RMSE by evaluation year (persistence omitted for scale)")
        fig.tight_layout()
        fig.savefig(dest / "fig_walkforward_rmse.png", dpi=150)
        plt.close(fig)
        made.append("fig_walkforward_rmse.png")
    d = _csv(tables, "ablation_rmse_diff")
    if d is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        sd = d.drop_duplicates("lead").sort_values("lead")
        ax.fill_between(sd["lead"], -sd["main_seed_sd"], sd["main_seed_sd"], color="0.85", label="± main-hybrid seed SD")
        for v, g in d.groupby("variant"):
            g = g.sort_values("lead")
            ax.plot(g["lead"], g["rmse_diff"], marker="o", label=ABL_LABEL.get(v, v))
        ax.axhline(0, color="k", lw=0.8)
        ax.set(xlabel="lead (h)", ylabel="RMSE difference vs main hybrid (m)", title="Ablations, 2024 test (positive = worse)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(dest / "fig_ablation_rmse_diff.png", dpi=150)
        plt.close(fig)
        made.append("fig_ablation_rmse_diff.png")
    return made


# ----------------------------------------------------------------------------- driver

def build() -> Path:
    tables, figs, gates, dest = _dirs()
    res = _json(tables / "test_results.json")
    blocks = (t_validation(tables, gates) + t_test_levels(tables) + t_dm(tables, res) + t_alerts(tables, res)
              + t_walkforward(tables) + t_ablations(tables) + t_case_aviles(tables))
    for old in dest.glob("table_IV-*.csv"):
        old.unlink()
    md = [f"# Chapter IV tables and figures ({'REHEARSAL — not results' if profile() == 'rehearsal' else 'chosen-laptop run'})", "",
          f"Generated by `python -m pasig.eval.report` at commit `{git_sha()}` from `{tables.relative_to(tables.parents[1])}`. "
          "Do not edit by hand; re-run the command instead. Full-precision numbers are in the CSV beside each table.", ""]
    if res:
        md += ["## Verdicts", "", f"* **H3 (level accuracy): {res['H3']['verdict']}**",
               f"* **H4 (alerts vs classifiers): {res['H4']['verdict']}**",
               f"* Strongest simple baseline (chosen on 2023): {LABEL.get(res['strongest_simple_baseline'])}",
               f"* Neural seed used for DM/H4: hybrid seed {res['test_seeds']['hybrid']}, LSTM seed {res['test_seeds']['lstm']} "
               "(median validation RMSE)", ""]
    for i, (key, title, raw, fmt, note) in enumerate(blocks, 1):
        name = f"table_IV-{i:02d}_{key}.csv"
        raw.to_csv(dest / name, index=not isinstance(raw.index, pd.RangeIndex))
        md += [f"## Table IV-{i}. {title}", "", _md(fmt), ""]
        if note:
            md += [note, ""]
        md += [f"*Data: `{name}`*", ""]
    made = figures(tables, dest)
    existing = sorted(p.name for p in figs.glob("test_*.png")) + sorted(p.name for p in figs.glob("case_event_*.png"))
    md += ["## Figures", ""]
    md += [f"* `chapter4/{f}`" for f in made]
    md += [f"* `figures/{f}`" for f in existing]
    md += [""]
    rt = compute_runtime(dest)
    hw = None
    comp = tables.parent / "compute"
    for p in sorted(comp.glob("*/summary_gpu.json")) + sorted(comp.glob("*/summary_cpu.json")):
        hw = _json(p)
        break
    if rt is not None or hw:
        md += ["## For Chapter III: hardware and measured training time", ""]
        if hw:
            h = hw.get("hardware", {})
            mach = (load("compute") or {}).get("machine", {}) if profile() != "rehearsal" else {}
            gpu = mach.get("gpu") or (", ".join(h.get("gpus", [])) or "none")
            md += [f"* Machine: {h.get('cpu')} ({h.get('physical_cores')} cores), {mach.get('ram_gb', '?')} GB RAM "
                   f"({h.get('ram_gb')} GB visible to WSL); GPU: {gpu} ({mach.get('vram_gb', '?')} GB); "
                   f"{h.get('os')}; Python {h.get('python')}, "
                   f"TensorFlow {h.get('tensorflow')} (CUDA {h.get('cuda')}, cuDNN {h.get('cudnn')})",
                   f"* Benchmark ({hw.get('device')}): median {hw.get('median_epoch_s')} s per epoch; projected "
                   f"{hw.get('recommended_total_hours')} h for the recommended plan", ""]
        if rt is not None:
            md += [_md(rt.rename_axis("Run group")), "", "*Data: `compute_runtime.csv` (sums of `runs/*/timing.json`; "
                   "excludes prediction, evaluation and the benchmark itself)*", ""]
    (dest / "chapter4.md").write_text("\n".join(md))
    return dest


def main() -> int:
    dest = build()
    print(f"Chapter IV tables and figures written to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
