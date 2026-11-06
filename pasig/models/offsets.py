"""Alert thresholds (Section 9.3, D11): trigger offsets for level forecasts and F1-tuned classifier thresholds."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def f1_score_counts(alert: np.ndarray, truth: np.ndarray) -> float:
    tp = np.sum(alert & truth)
    fp = np.sum(alert & ~truth)
    fn = np.sum(~alert & truth)
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0


def offset_curve(pred: np.ndarray, y_true: np.ndarray, valid: np.ndarray, level: float, offsets: np.ndarray) -> np.ndarray:
    """F1 over all (hour, lead) pairs with a valid target, for each offset (alert iff forecast >= level - offset)."""
    p, t = pred[valid], y_true[valid] >= level
    return np.array([f1_score_counts(p >= level - o, t) for o in offsets])


def best_offset(preds: list[np.ndarray], y_true: np.ndarray, valid: np.ndarray, level: float, cfg: dict) -> dict:
    """One offset per model per level, shared by all six leads; neural models: maximise the mean F1 over seeds;
    ties -> the offset closest to 0."""
    offs = np.round(np.arange(cfg["min_m"], cfg["max_m"] + 1e-9, cfg["step_m"]), 2)
    curves = np.vstack([offset_curve(p, y_true, valid, level, offs) for p in preds])
    mean = curves.mean(axis=0)
    cand = np.flatnonzero(np.isclose(mean, mean.max()))
    i = cand[np.argmin(np.abs(offs[cand]))]
    return {"offset_m": float(offs[i]), "f1_val": float(mean[i]), "f1_at_zero": float(mean[np.argmin(np.abs(offs))]),
            "edge": bool(i in (0, len(offs) - 1))}


def best_threshold(prob: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """Decision threshold maximising F1 on the validation set: returns (threshold, F1)."""
    if truth.sum() == 0:
        return 0.5, 0.0
    prec, rec, thr = precision_recall_curve(truth, prob)
    den = prec[:-1] + rec[:-1]
    f1 = np.divide(2 * prec[:-1] * rec[:-1], den, out=np.zeros_like(den), where=den > 0)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])
