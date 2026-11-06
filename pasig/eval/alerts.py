"""Hourly alert metrics and the equal-recall comparison (Section 10.1, D19)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def counts(alert: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(np.sum(alert & truth))
    fp = int(np.sum(alert & ~truth))
    fn = int(np.sum(~alert & truth))
    tn = int(np.sum(~alert & ~truth))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1), "alert_share": float(np.mean(alert)) if len(alert) else float("nan"),
            "event_hour_share": float(np.mean(truth)) if len(truth) else float("nan")}


def pr_auc(score: np.ndarray, truth: np.ndarray) -> float:
    return float(average_precision_score(truth, score)) if truth.any() and (~truth).any() else float("nan")


def fp_at_recall(score: np.ndarray, truth: np.ndarray, target: float) -> dict:
    """Move the threshold down from the top until Recall >= target (first such threshold); report FP and the
    achieved Recall (D19: KNN probabilities move in steps, so exactly 0.90 may be unreachable)."""
    P = int(truth.sum())
    if P == 0:
        return {"fp": None, "recall": None, "threshold": None}
    order = np.argsort(-score, kind="mergesort")
    s, t = score[order], truth[order]
    # group ties: a threshold admits all rows with score >= it
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(t)[ends]
    fp = (ends + 1) - tp
    rec = tp / P
    k = int(np.flatnonzero(rec >= target - 1e-12)[0])
    return {"fp": int(fp[k]), "recall": float(rec[k]), "threshold": float(s[ends[k]])}
