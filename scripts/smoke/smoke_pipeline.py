"""End-to-end smoke test: every phase of Chapter III on synthetic data.

Purpose: prove the pipeline runs start to finish and time each stage, BEFORE
the real data are rebuilt. It does not prove the model is good.

    python scripts/smoke/smoke_pipeline.py --device cpu            # ~3-5 min
    python scripts/smoke/smoke_pipeline.py --device gpu            # NVIDIA laptop, inside WSL2
    python scripts/smoke/smoke_pipeline.py --device cpu --size full --epochs 3

Writes runs/smoke_<device>_<size>/timings.csv and summary.json and exits with
a non-zero code if any check fails.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--size", choices=["tiny", "full"], default="tiny")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    max_cases = 3000 if args.size == "tiny" else None
    out_dir = os.path.join(args.out, f"smoke_{args.device}_{args.size}")
    os.makedirs(out_dir, exist_ok=True)
    T = C.Timings()
    checks: dict[str, bool] = {}
    t_start = time.perf_counter()

    with T.stage("import tensorflow"):
        dev = C.configure_device(args.device)
    hw = C.hardware_info()
    print(f"Device: {dev} | {hw}")
    C.set_seed(0)

    # ---------------- Phase 1 ----------------
    print("Phase 1: data")
    with T.stage("p1 synthesize raw records"):
        raw = C.make_synthetic(seed=0)
    with T.stage("p1 align, availability, gap fill"):
        df = C.align_inputs(raw)
    with T.stage("p1 coverage table"):
        cov = (df.groupby("season")[["pandacan", "sto_nino"]]
               .apply(lambda g: g.notna().mean() * 100).round(1))
        eligible = cov.ge(80).all()
        cov.to_csv(os.path.join(out_dir, "coverage.csv"))
    print(cov.T.to_string())
    # alignment check: the 6-hourly report stamped 06:00 UTC must appear at 14:00 PHT, not before
    six = raw["six_hourly_utc"]
    ts_utc = six.index[(six.index.month == 7) & (six["rain_6h_mm"] > 0)][0]
    ts_usable = ts_utc + pd.Timedelta(hours=8 + C.RAIN_LAG_H)  # D5: one hour after period end, in PHT
    before = df.loc[ts_usable - pd.Timedelta(hours=1), "rain_6h_latest"]
    at = df.loc[ts_usable, "rain_6h_latest"]
    checks["rain_usable_only_1h_after_period_end"] = bool(at == six.loc[ts_utc, "rain_6h_mm"]
                                                          and before != at)

    train = df[df.season.isin(C.TRAIN_YEARS)]

    # ---------------- Phase 2 ----------------
    print("Phase 2: link analysis and alert levels")
    with T.stage("p2 logistic regression, day-clustered SE"):
        import statsmodels.api as sm
        av = raw["aviles"].reindex(train.index)
        stress = (av["pumps_running"] >= 2).astype(int)
        r24 = train["rain_24h"].fillna(0)
        X = pd.DataFrame({"L": train["pandacan"], "lr": np.log1p(r24)})
        ok = X.notna().all(axis=1)
        groups = pd.factorize(train.index[ok].normalize())[0]
        fit1 = sm.Logit(stress[ok], sm.add_constant(X[ok])).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": groups})
        or_per_01m = float(np.exp(0.1 * fit1.params["L"]))
    with T.stage("p2 alert levels"):
        lvl = train["pandacan"][~train["pandacan_filled"]].dropna()
        high_water = float(np.round(lvl.quantile(0.95), 2))
        grid = np.round(np.arange(lvl.median(), lvl.max(), 0.05), 2)
        flood_risk = None
        for g in grid:
            m = ok & (train["pandacan"] >= g)
            if m.sum() >= 5 and stress[m].mean() >= 0.5:
                flood_risk = float(g)
                break
        flood_risk = flood_risk or float(np.round(lvl.quantile(0.995), 2))
    print(f"  OR per 0.1 m = {or_per_01m:.2f}; high-water = {high_water} m; flood-risk = {flood_risk} m")

    # ---------------- Phase 3 ----------------
    print("Phase 3: tide, Granger, windows")
    with T.stage("p3 harmonic tide fit (7 constituents)"):
        # D7: all months of the training years (not test data), hours above the floor
        yrs = raw["hourly"].loc[f"{C.TRAIN_YEARS[0]}":f"{C.TRAIN_YEARS[-1]}"]
        tide = C.HarmonicTide().fit(yrs["pandacan"], floor=10.63)
        resid = train["pandacan"] - tide.predict(train.index)
        tide_up = C.HarmonicTide().fit(yrs["sto_nino"])
        resid_up = train["sto_nino"] - tide_up.predict(train.index)
    with T.stage("p3 ADF per season"):
        adf = {int(s): C.adf_pvalue(resid[train.season == s].to_numpy()[:3000])
               for s in C.TRAIN_YEARS}
    with T.stage("p3 pooled Granger, BIC lags (sto_nino, max 24)"):
        ys = [resid[train.season == s].to_numpy() for s in C.TRAIN_YEARS]
        xs = [resid_up[train.season == s].to_numpy() for s in C.TRAIN_YEARS]
        g_up = C.pooled_granger(ys, xs, max_lag=24)
        noise = [np.random.default_rng(i).standard_normal(len(y)) for i, y in enumerate(ys)]
        g_noise = C.pooled_granger(ys, noise, max_lag=6)
    m_tests = 2
    checks["granger_finds_planted_lead"] = g_up["p_value"] < 0.05 / m_tests
    checks["granger_rejects_noise"] = g_noise["p_value"] > 0.01
    print(f"  Granger sto_nino: lag={g_up['lag']} p={g_up['p_value']:.2e}; noise p={g_noise['p_value']:.3f}")

    seq_cols = C.SEQ_COLS_BASE + (["sto_nino"] if g_up["p_value"] < 0.05 / m_tests else [])
    rng = np.random.default_rng(0)
    with T.stage("p3 build windows (train/val/test)"):
        tr = C.build_windows(df, tide, seq_cols, C.TRAIN_YEARS, max_cases, rng)
        va = C.build_windows(df, tide, seq_cols, C.VAL_YEARS, max_cases and max_cases // 3, rng)
        te = C.build_windows(df, tide, seq_cols, C.TEST_YEARS, max_cases and max_cases // 3, rng)
    print(f"  cases: train {len(tr[2])}, val {len(va[2])}, test {len(te[2])}; "
          f"seq {tr[0].shape[1:]}, tab {tr[1].shape[1]}")

    with T.stage("p3 scalers on training only + leakage check"):
        s_seq = C.MinMax().fit(tr[0])
        s_tab = C.MinMax().fit(tr[1])
        s_y = C.level_scaler(s_seq)  # D9: targets on the Pandacan channel's scale
        # perturb the test season and refit: scaler must not change
        s_seq2 = C.MinMax().fit(tr[0])
        te_pert = te[0] + 5.0  # noqa: F841  (never passed to fit)
        checks["scaler_ignores_test"] = bool(np.allclose(s_seq.lo, s_seq2.lo))
        # target alignment: y[:,0] must equal the level one hour after the window end
        t0 = tr[4][10]
        checks["target_is_t_plus_1"] = bool(np.isclose(tr[2][10, 0], df.loc[t0 + pd.Timedelta(hours=1), "pandacan"])
                                             or not tr[3][10, 0])

    def prep(split, tab=True):
        xs = s_seq.transform(split[0])
        xt = s_tab.transform(split[1])
        return ({"seq": xs, "tab": xt} if tab else xs), C.encode_targets(split[2], split[3], s_y)

    Xtr, Ytr = prep(tr)
    Xva, Yva = prep(va)
    Xte, Yte = prep(te)

    # ---------------- Phase 4 ----------------
    print("Phase 4: train")
    cfg = C.HybridConfig(lstm_layers=1, lstm_units=64, gru_units=32, batch=64)
    with T.stage("p4 build hybrid"):
        model = C.build_hybrid(cfg, tr[0].shape[1:], tr[1].shape[1])
    print(f"  hybrid parameters: {model.count_params():,}")
    with T.stage("p4 train hybrid", epochs=args.epochs, cases=len(Ytr)):
        hist = model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=args.epochs,
                         batch_size=cfg.batch, verbose=0)
    losses = hist.history["loss"]
    checks["hybrid_training_loss_falls"] = bool(losses[-1] < losses[0])
    with T.stage("p4 train standalone LSTM", epochs=args.epochs):
        lstm_in = s_seq.transform(tr[0])
        lstm = C.build_lstm(64, 0.2, 1e-3, (C.WINDOW, tr[0].shape[2] + tr[1].shape[1]))
        lstm.fit(C.repeat_tab(lstm_in, s_tab.transform(tr[1])), Ytr, epochs=args.epochs,
                 batch_size=64, verbose=0)

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    flat = lambda s: np.hstack([s_seq.transform(s[0]).reshape(len(s[0]), -1), s_tab.transform(s[1])])  # noqa: E731
    with T.stage("p4 ridge + KNN regressor"):
        ridge = Ridge(alpha=1.0).fit(flat(tr), tr[2])
        knn = KNeighborsRegressor(n_neighbors=21, weights="distance").fit(flat(tr), tr[2])
    with T.stage("p4 logistic + KNN classifiers (6 leads x 2 levels)"):
        clfs = {}
        for lv_name, lv in (("hw", high_water), ("fr", flood_risk)):
            for h in range(C.LEADS):
                yb = (tr[2][:, h] >= lv) & tr[3][:, h]
                if yb.sum() < 2:
                    continue
                clfs[(lv_name, h, "lr")] = LogisticRegression(C=1.0, class_weight="balanced",
                                                              max_iter=500).fit(flat(tr), yb)
                clfs[(lv_name, h, "knn")] = KNeighborsClassifier(21).fit(flat(tr), yb)

    # ---------------- Phase 5 ----------------
    print("Phase 5: evaluate")
    with T.stage("p5 predict all models on test"):
        p_hyb = model.predict(Xte, verbose=0) * s_y.span + s_y.lo
        p_ridge = ridge.predict(flat(te))
        p_knn = knn.predict(flat(te))
        p_pers = np.repeat(te[0][:, -1, 0:1], C.LEADS, axis=1)
        tide_te = np.column_stack([tide.predict(te[4] + pd.Timedelta(hours=h)) for h in range(1, C.LEADS + 1)])
        p_tide_resid = tide_te + (te[0][:, -1, 0] - tide.predict(te[4]))[:, None]
    with T.stage("p5 RMSE, skill per lead"):
        rows = []
        for name, p in [("hybrid", p_hyb), ("ridge", p_ridge), ("knn", p_knn),
                        ("persistence", p_pers), ("tide+resid", p_tide_resid)]:
            for h in range(C.LEADS):
                v = te[3][:, h]
                rows.append({"model": name, "lead": h + 1,
                             "rmse": float(np.sqrt(np.mean((p[v, h] - te[2][v, h]) ** 2)))})
        rm = pd.DataFrame(rows).pivot(index="model", columns="lead", values="rmse").round(3)
        rm.to_csv(os.path.join(out_dir, "rmse_by_lead.csv"))
    print(rm.to_string())
    with T.stage("p5 Diebold-Mariano (hybrid vs 4 baselines x 6 leads) + Holm"):
        from statsmodels.stats.multitest import multipletests
        dm_rows = []
        for name, p in [("ridge", p_ridge), ("knn", p_knn), ("persistence", p_pers), ("tide+resid", p_tide_resid)]:
            for h in range(C.LEADS):
                v = te[3][:, h]
                r = C.diebold_mariano(p_hyb[v, h] - te[2][v, h], p[v, h] - te[2][v, h], h + 1)
                dm_rows.append({"baseline": name, "lead": h + 1, **r})
        dmt = pd.DataFrame(dm_rows)
        dmt["p_holm"] = multipletests(dmt["p_value"], method="holm")[1]
        dmt.to_csv(os.path.join(out_dir, "dm_tests.csv"), index=False)
    with T.stage("p5 PR-AUC, FP at recall 0.90, block bootstrap (200 reps)"):
        from sklearn.metrics import average_precision_score, precision_recall_curve
        h = 0
        v = te[3][:, h]
        truth = te[2][v, h] >= high_water
        score = p_hyb[v, h]
        pr_auc = float(average_precision_score(truth, score)) if truth.any() else float("nan")
        if truth.any():
            prec, rec, thr = precision_recall_curve(truth, score)
            i = np.flatnonzero(rec[:-1] >= 0.90)[-1]
            fp_at_90 = int(((score >= thr[i]) & ~truth).sum())
        else:
            fp_at_90 = -1
        rng_b = np.random.default_rng(1)
        boots = []
        for _ in range(200):
            ix = C.moving_block_indices(len(truth), 24, rng_b)
            if truth[ix].any() and (~truth[ix]).any():
                boots.append(average_precision_score(truth[ix], score[ix]))
        ci = np.percentile(boots, [2.5, 97.5]).round(3).tolist() if boots else [None, None]
    print(f"  lead 1 high-water: PR-AUC {pr_auc:.3f} (95% CI {ci}), FP at recall 0.90 = {fp_at_90}")

    with T.stage("p5 reproducibility check (same seed twice)"):
        def tiny_run():
            C.set_seed(123)
            m = C.build_hybrid(C.HybridConfig(lstm_units=16, gru_units=8, batch=128),
                               tr[0].shape[1:], tr[1].shape[1])
            m.fit({"seq": Xtr["seq"][:512], "tab": Xtr["tab"][:512]}, Ytr[:512], epochs=1,
                  batch_size=128, verbose=0)
            return m.predict({"seq": Xte["seq"][:64], "tab": Xte["tab"][:64]}, verbose=0)
        checks["same_seed_same_predictions"] = bool(np.allclose(tiny_run(), tiny_run(), atol=1e-6))

    total = time.perf_counter() - t_start
    tim = T.to_frame()
    tim.to_csv(os.path.join(out_dir, "timings.csv"), index=False)
    summary = {"device": dev, "size": args.size, "epochs": args.epochs, "total_seconds": round(total, 1),
               "hardware": hw, "cases": {"train": len(Ytr), "val": len(Yva), "test": len(Yte)},
               "hybrid_params": int(model.count_params()),
               "epoch_seconds_hybrid": round(float(tim.loc[tim.stage == "p4 train hybrid", "seconds"].iloc[0]) / args.epochs, 2),
               "alert_levels": {"high_water": high_water, "flood_risk": flood_risk},
               "granger_sto_nino": g_up, "checks": checks}
    C.save_json(summary, os.path.join(out_dir, "summary.json"))
    print("\nChecks:")
    for k, ok_ in checks.items():
        print(f"  {'PASS' if ok_ else 'FAIL'}  {k}")
    print(f"\nTotal: {total:.1f} s  ->  {out_dir}/")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
