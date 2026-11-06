"""Table III-4 baselines (Section 9.3). All tuned on 2023; nothing is fitted on 2023 targets except the choice
of hyperparameters and thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from pasig.config import load
from pasig.models.offsets import best_threshold

LEADS = 6


def flat(X: dict, current_hour: bool = False) -> np.ndarray:
    s = X["seq"][:, -1, :] if current_hour else X["seq"].reshape(len(X["seq"]), -1)
    return np.hstack([s, X["tab"]])


def persistence(d: dict) -> np.ndarray:
    return np.repeat(d["level_t"][:, None], LEADS, axis=1)


def tide_resid(d: dict, tide) -> np.ndarray:
    """tide(t+h) + (L_t - tide(t))."""
    t = d["t_index"]
    now = tide.predict(t)
    fut = np.column_stack([tide.predict(t + pd.Timedelta(hours=h)) for h in range(1, LEADS + 1)])
    return fut + (d["level_t"] - now)[:, None]


class TideOnly:
    """level ~ a + b * predicted tide + annual harmonics, OLS on training wet-season observed hours."""

    def __init__(self, tide, harmonics: int):
        self.tide, self.k = tide, harmonics

    def _X(self, t: pd.DatetimeIndex) -> np.ndarray:
        doy = t.dayofyear.to_numpy()
        cols = [np.ones(len(t)), self.tide.predict(t)]
        for j in range(1, self.k + 1):
            cols += [np.sin(2 * np.pi * j * doy / 365.25), np.cos(2 * np.pi * j * doy / 365.25)]
        return np.column_stack(cols)

    def fit(self, level: pd.Series):
        s = level.dropna()
        self.coef_, *_ = np.linalg.lstsq(self._X(s.index), s.to_numpy(), rcond=None)
        return self

    def predict(self, d: dict) -> np.ndarray:
        t = d["t_index"]
        return np.column_stack([self._X(t + pd.Timedelta(hours=h)) @ self.coef_ for h in range(1, LEADS + 1)])


def _rmse_by_lead(P, Y, V):
    return [float(np.sqrt(np.mean((P[V[:, h], h] - Y[V[:, h], h]) ** 2))) for h in range(LEADS)]


def fit_ridge(tr: dict, va: dict) -> dict:
    import warnings
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    Ftr, Fva = flat(tr["X"]), flat(va["X"])
    res = {}
    for a in load("baselines")["ridge_alphas"]:
        models = [Ridge(alpha=a).fit(Ftr[tr["valid"][:, h]], tr["y_raw"][tr["valid"][:, h], h]) for h in range(LEADS)]
        P = np.column_stack([m.predict(Fva) for m in models])
        rm = _rmse_by_lead(P, va["y_raw"], va["valid"])
        res[a] = {"models": models, "pred_val": P, "val_rmse_by_lead": rm, "val_rmse_mean": float(np.mean(rm))}
    a = min(res, key=lambda k: res[k]["val_rmse_mean"])
    out = res[a] | {"alpha": a, "search": {str(k): v["val_rmse_mean"] for k, v in res.items()}}
    out["params"] = {"alpha": a, "coef": [m.coef_.tolist() for m in out["models"]],
                     "intercept": [float(m.intercept_) for m in out["models"]]}
    return out


def fit_knn_regressor(tr: dict, va: dict) -> dict:
    c = load("baselines")
    Ftr, Fva = flat(tr["X"]), flat(va["X"])
    ks = list(c["knn_k"])
    res = {}

    def run(k, w):
        models = [KNeighborsRegressor(n_neighbors=k, weights=w).fit(Ftr[tr["valid"][:, h]], tr["y_raw"][tr["valid"][:, h], h])
                  for h in range(LEADS)]
        P = np.column_stack([m.predict(Fva) for m in models])
        rm = _rmse_by_lead(P, va["y_raw"], va["valid"])
        res[(k, w)] = {"pred_val": P, "val_rmse_by_lead": rm, "val_rmse_mean": float(np.mean(rm))}

    for k in ks:
        for w in c["knn_weights"]:
            run(k, w)
    while True:   # extend until the best k is interior (Table III-4)
        kb, _ = min(res, key=lambda z: res[z]["val_rmse_mean"])
        if kb != max(k for k, _ in res) or kb * c["knn_k_extend_factor"] > int(tr["valid"][:, 0].sum()) // 2:
            break
        knew = kb * c["knn_k_extend_factor"]
        for w in c["knn_weights"]:
            run(knew, w)
    kb, wb = min(res, key=lambda z: res[z]["val_rmse_mean"])
    return res[(kb, wb)] | {"params": {"k": int(kb), "weights": wb},
                            "search": {f"{k}_{w}": v["val_rmse_mean"] for (k, w), v in res.items()}}


def fit_classifiers(tr: dict, va: dict, levels: dict, kind: str) -> pd.DataFrame:
    """One classifier per lead x level x input set; hyperparameters + threshold chosen by F1 on 2023.
    LR: C, class weight, threshold. KNN: k, weighting, threshold only (C4)."""
    c = load("baselines")
    rows, fitted = [], {}
    for inset, cur in (("a_current_hour", True), ("b_window", False)):
        Ftr, Fva = flat(tr["X"], cur), flat(va["X"], cur)
        for lname, lvl in levels.items():
            for h in range(LEADS):
                mt, mv = tr["valid"][:, h], va["valid"][:, h]
                ytr = tr["y_raw"][mt, h] >= lvl
                yva = va["y_raw"][mv, h] >= lvl
                best = None
                if ytr.sum() < 2:
                    rows.append({"model": kind, "input_set": inset, "level": lname, "lead": h + 1, "note": "too few positives"})
                    continue
                grid = ([{"C": C, "class_weight": cw} for C in c["logistic"]["C"] for cw in c["logistic"]["class_weight"]]
                        if kind == "logistic" else
                        [{"k": k, "weights": w} for k in c["knn_classifier"]["k"] for w in c["knn_classifier"]["weights"]])
                for g in grid:
                    m = (LogisticRegression(C=g["C"], class_weight=g["class_weight"], max_iter=c["logistic"]["max_iter"])
                         if kind == "logistic" else KNeighborsClassifier(n_neighbors=g["k"], weights=g["weights"]))
                    m.fit(Ftr[mt], ytr)
                    prob = m.predict_proba(Fva[mv])[:, 1]
                    thr, f1 = best_threshold(prob, yva)
                    if best is None or f1 > best["f1_val"]:
                        best = {"params": g, "threshold": thr, "f1_val": f1, "model": m, "prob_val": prob}
                key = (inset, lname, h + 1)
                fitted[key] = best
                rows.append({"model": kind, "input_set": inset, "level": lname, "lead": h + 1, **best["params"],
                             "threshold": best["threshold"], "f1_val": best["f1_val"],
                             "train_positives": int(ytr.sum()), "val_positives": int(yva.sum())})
    out = pd.DataFrame(rows)
    out.attrs["fitted"] = fitted
    return out
