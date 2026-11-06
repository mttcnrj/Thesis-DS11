"""Compute benchmark: how long will Phase 4 and Phase 5 take on this machine?

Measures seconds per epoch for every hybrid architecture x batch size in the
Chapter III search space, the standalone LSTM, the scikit-learn baselines and
inference, then projects wall-clock time for the whole experiment plan.

    python scripts/smoke/bench_compute.py --device cpu --train-cases 4000
    python scripts/smoke/bench_compute.py --device gpu            # NVIDIA laptop, inside WSL2
    python scripts/smoke/bench_compute.py --device cpu --quick    # 4 corners only

Epoch time scales linearly with the number of training cases, so a CPU run
may subsample (--train-cases) and the script rescales to --target-cases
(16,158 = training cases at lead 1 in Table III-6).

Outputs runs/bench_<device>[_nodet]/{epoch_times.csv, baselines.csv, projection.csv, summary.json}.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

ARCH = {"lstm_layers": [1, 2], "lstm_units": [64, 128], "gru_units": [32, 64]}
BATCHES = [32, 64, 128]
# Remaining Chapter III ranges: they change accuracy, not run time
DROPOUTS = [0.2, 0.35, 0.5]
LRS = [1e-3, 5e-4, 1e-4]
DENSE = [(1, 16), (1, 32), (2, 16), (2, 32)]
LSTM_UNITS = [64, 128]


def time_epochs(build, X, Y, batch, warmup=1, epochs=2):
    m = build()
    t0 = time.perf_counter()
    m.fit(X, Y, epochs=warmup, batch_size=batch, verbose=0)
    t_warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    m.fit(X, Y, epochs=epochs, batch_size=batch, verbose=0)
    t_ep = (time.perf_counter() - t0) / epochs
    return m, t_warm, t_ep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--train-cases", type=int, default=None,
                    help="cases actually trained on during the benchmark (default: all synthetic)")
    ap.add_argument("--target-cases", type=int, default=16158, help="Table III-6 training cases")
    ap.add_argument("--measure-epochs", type=int, default=2)
    ap.add_argument("--expected-epochs", type=float, default=60,
                    help="mean epochs per run with early stopping (prototype: 32-73)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no-determinism", action="store_true")
    ap.add_argument("--quick", action="store_true", help="time only 4 corners of the grid")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    t_bench = time.perf_counter()
    out = os.path.join(args.out, f"bench_{args.device}" + ("_nodet" if args.no_determinism else ""))
    os.makedirs(out, exist_ok=True)
    C.configure_device(args.device)
    C.set_seed(0, deterministic=not args.no_determinism)
    hw = C.hardware_info()
    print("Hardware:", hw)

    # ---------- data at realistic shape ----------
    raw = C.make_synthetic(seed=0)
    df = C.align_inputs(raw)
    train = df[df.season.isin(C.TRAIN_YEARS)]
    yrs = raw["hourly"].loc[f"{C.TRAIN_YEARS[0]}":f"{C.TRAIN_YEARS[-1]}"]
    tide = C.HarmonicTide().fit(yrs["pandacan"], floor=10.63)  # D7
    seq_cols = C.SEQ_COLS_BASE + ["sto_nino"]
    tr = C.build_windows(df, tide, seq_cols, C.TRAIN_YEARS, args.train_cases, np.random.default_rng(0))
    te = C.build_windows(df, tide, seq_cols, C.TEST_YEARS, 3752, np.random.default_rng(1))
    s_seq, s_tab = C.MinMax().fit(tr[0]), C.MinMax().fit(tr[1])
    s_y = C.level_scaler(s_seq)  # D9
    X = {"seq": s_seq.transform(tr[0]), "tab": s_tab.transform(tr[1])}
    Y = C.encode_targets(tr[2], tr[3], s_y)
    Xte = {"seq": s_seq.transform(te[0]), "tab": s_tab.transform(te[1])}
    n = len(Y)
    scale = args.target_cases / n
    print(f"Benchmark cases: {n:,} (rescaled x{scale:.2f} to {args.target_cases:,}); "
          f"seq {tr[0].shape[1:]}, tab {tr[1].shape[1]}")

    # ---------- hybrid: every arch x batch ----------
    combos = list(itertools.product(ARCH["lstm_layers"], ARCH["lstm_units"], ARCH["gru_units"], BATCHES))
    if args.quick:
        combos = [(1, 64, 32, 128), (1, 64, 32, 32), (2, 128, 64, 128), (2, 128, 64, 32)]
    rows = []
    for i, (ll, lu, gu, b) in enumerate(combos, 1):
        cfg = C.HybridConfig(lstm_layers=ll, lstm_units=lu, gru_units=gu, batch=b)
        m, tw, te_ = time_epochs(lambda: C.build_hybrid(cfg, tr[0].shape[1:], tr[1].shape[1]),
                                 X, Y, b, epochs=args.measure_epochs)
        t0 = time.perf_counter()
        m.predict(Xte, batch_size=512, verbose=0)
        t_inf = time.perf_counter() - t0
        rows.append({"model": "hybrid", "lstm_layers": ll, "lstm_units": lu, "gru_units": gu, "batch": b,
                     "params": m.count_params(), "warmup_s": round(tw, 2),
                     "epoch_s_measured": round(te_, 3), "epoch_s_target": round(te_ * scale, 3),
                     "predict_test_s": round(t_inf, 3)})
        print(f"  [{i:2d}/{len(combos)}] hybrid L{ll}x{lu} G{gu} b{b}: "
              f"{te_ * scale:7.2f} s/epoch at {args.target_cases:,} cases ({m.count_params():,} params)", flush=True)

    # ---------- standalone LSTM ----------
    Xl = C.repeat_tab(X["seq"], X["tab"])
    for lu, b in itertools.product(LSTM_UNITS, BATCHES if not args.quick else [64]):
        m, tw, te_ = time_epochs(lambda: C.build_lstm(lu, 0.2, 1e-3, Xl.shape[1:]), Xl, Y, b,
                                 epochs=args.measure_epochs)
        rows.append({"model": "lstm", "lstm_layers": 1, "lstm_units": lu, "gru_units": 0, "batch": b,
                     "params": m.count_params(), "warmup_s": round(tw, 2),
                     "epoch_s_measured": round(te_, 3), "epoch_s_target": round(te_ * scale, 3)})
        print(f"  LSTM {lu} b{b}: {te_ * scale:7.2f} s/epoch", flush=True)
    ep = pd.DataFrame(rows)
    ep.to_csv(os.path.join(out, "epoch_times.csv"), index=False)

    # ---------- scikit-learn baselines (CPU regardless of --device) ----------
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    F = np.hstack([X["seq"].reshape(n, -1), X["tab"]])
    Fte = np.hstack([Xte["seq"].reshape(len(te[2]), -1), Xte["tab"]])
    yb = (tr[2][:, 0] >= np.quantile(tr[2][tr[3][:, 0], 0], 0.95))
    base = []

    def bench(name, fn):
        t0 = time.perf_counter()
        fn()
        base.append({"baseline": name, "seconds_measured": round(time.perf_counter() - t0, 3)})

    bench("ridge fit+predict (1 alpha)", lambda: Ridge(1.0).fit(F, tr[2]).predict(Fte))
    bench("knn regressor fit+predict (1 k)", lambda: KNeighborsRegressor(21).fit(F, tr[2]).predict(Fte))
    bench("logistic fit (1 C, 1 lead, 1 level)",
          lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(F, yb))
    bench("knn classifier fit+predict (1 k, 1 lead, 1 level)",
          lambda: KNeighborsClassifier(21).fit(F, yb).predict_proba(Fte))
    bl = pd.DataFrame(base)
    bl["seconds_target"] = (bl["seconds_measured"] * scale).round(3)  # KNN predict is ~linear in n too
    bl.to_csv(os.path.join(out, "baselines.csv"), index=False)

    # ---------- projection ----------
    hyb = ep[ep.model == "hybrid"]
    E = args.expected_epochs
    S = args.seeds

    def mean_epoch(df_):
        return float(df_["epoch_s_target"].mean())

    if args.quick:  # fill the grid by averaging the measured corners
        arch_batch_mean = mean_epoch(hyb)
        n_arch_batch = len(list(itertools.product(*ARCH.values(), BATCHES)))
        full_arch_cost = arch_batch_mean * n_arch_batch
    else:
        full_arch_cost = float(hyb["epoch_s_target"].sum())
    other = len(DROPOUTS) * len(LRS) * len(DENSE)
    median_cfg = float(hyb["epoch_s_target"].median())
    fastest = float(hyb["epoch_s_target"].min())
    lstm_ep = float(ep.loc[ep.model == "lstm", "epoch_s_target"].median())
    lr_fit = float(bl.loc[bl.baseline.str.startswith("logistic"), "seconds_target"].iloc[0])
    knnc = float(bl.loc[bl.baseline.str.startswith("knn classifier"), "seconds_target"].iloc[0])

    plan = []

    def add(item, runs, sec_per_run, note):
        plan.append({"item": item, "runs": runs, "hours": round(runs * sec_per_run / 3600, 2), "note": note})

    add("Hybrid grid, Chapter III full factorial (1 seed)", int(len(combos) if not args.quick else 24) * other,
        full_arch_cost / (24 if not args.quick else 1) * E if not args.quick else arch_batch_mean * E,
        f"{24 * other} configs x {E:.0f} epochs")
    add("Hybrid grid, two-stage (recommended, 1 seed)", 24 + other,
        (full_arch_cost / 24) * E, "stage 1: 24 arch x batch; stage 2: 36 dropout x lr x dense on the winner")
    add("Hybrid final config, 5 seeds", S, median_cfg * E, "median-cost config")
    add("Standalone LSTM grid + 5 seeds", 2 * len(DROPOUTS) * len(LRS) + S, lstm_ep * E, "18 configs + 5 seeds")
    add("Ablations (5 variants x 5 seeds)", 5 * S, median_cfg * E, "Table III-5")
    add("Walk-forward (3 folds x hybrid + LSTM x 5 seeds)", 3 * 2 * S, (median_cfg + lstm_ep) / 2 * E * 0.8,
        "folds train on 3/5, 4/5, 5/5 of the seasons")
    add("Logistic + KNN classifiers (6 leads x 2 levels x 2 inputs x ~5 settings)", 6 * 2 * 2 * 5,
        lr_fit + knnc, "scikit-learn, CPU")
    proj = pd.DataFrame(plan)
    proj.to_csv(os.path.join(out, "projection.csv"), index=False)

    summary = {"device": args.device, "quick": args.quick, "hardware": hw,
               "bench_wall_seconds": round(time.perf_counter() - t_bench, 1), "benchmark_cases": n, "target_cases": args.target_cases,
               "expected_epochs": E, "seeds": S, "deterministic": not args.no_determinism,
               "fastest_epoch_s": fastest, "median_epoch_s": median_cfg,
               "recommended_total_hours": round(float(proj["hours"].iloc[1:].sum()), 2),
               "full_factorial_total_hours": round(float(proj["hours"].iloc[[0, 2, 3, 4, 5, 6]].sum()), 2)}
    C.save_json(summary, os.path.join(out, "summary.json"))
    print("\nProjection (wall clock, one machine, sequential):")
    print(proj.to_string(index=False))
    print(f"\nRecommended plan total: {summary['recommended_total_hours']} h | "
          f"with full-factorial grid: {summary['full_factorial_total_hours']} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
