"""Level metrics, alert counts, equal-recall false alarms (D19) and the moving-block bootstrap (Section 10.1)."""
import numpy as np
import pytest

from pasig.eval import alerts as A
from pasig.eval.bootstrap import block_ci, moving_block_indices
from pasig.eval.metrics import level_metrics, skill

pytestmark = pytest.mark.fast


def test_level_metrics_known_values():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    m = level_metrics(truth + 0.5, truth)
    assert m["rmse"] == pytest.approx(0.5) and m["mae"] == pytest.approx(0.5) and m["bias"] == pytest.approx(0.5)
    assert m["r2"] == pytest.approx(1 - 4 * 0.25 / 5.0)
    assert skill(0.5, 1.0) == pytest.approx(0.5) and skill(1.0, 1.0) == 0.0


def test_counts():
    alert = np.array([True, True, False, False])
    truth = np.array([True, False, True, False])
    c = A.counts(alert, truth)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 1)
    assert c["precision"] == c["recall"] == c["f1"] == 0.5


def test_fp_at_recall_first_threshold_reaching_target():
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    truth = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0], dtype=bool)   # 4 positives
    r = A.fp_at_recall(score, truth, 0.90)                           # needs all 4 -> threshold 0.2
    assert r["recall"] == 1.0 and r["fp"] == 4 and r["threshold"] == pytest.approx(0.2)
    r = A.fp_at_recall(score, truth, 0.50)                           # 2 of 4 at 0.7
    assert r["recall"] == 0.5 and r["fp"] == 1


def test_fp_at_recall_handles_ties_as_one_threshold():
    """KNN probabilities move in steps: tied scores are admitted together, so the achieved recall can exceed the
    target (reported, D19)."""
    score = np.array([1.0, 0.5, 0.5, 0.5, 0.0])
    truth = np.array([True, True, False, True, False])
    r = A.fp_at_recall(score, truth, 0.60)
    assert r["recall"] == 1.0 and r["fp"] == 1


def test_perfect_ranking_has_pr_auc_one_and_no_false_alarms():
    score = np.linspace(1, 0, 100)
    truth = np.arange(100) < 10
    assert A.pr_auc(score, truth) == pytest.approx(1.0)
    assert A.fp_at_recall(score, truth, 0.90)["fp"] == 0


def test_moving_blocks_keep_time_order_inside_blocks():
    idx = moving_block_indices(100, 24, np.random.default_rng(0))
    assert len(idx) == 100 and idx.min() >= 0 and idx.max() < 100
    starts = np.flatnonzero(np.r_[True, np.diff(idx) != 1])
    assert len(starts) <= 5                                          # ceil(100 / 24) blocks


def test_block_ci_covers_a_known_mean_difference():
    rng = np.random.default_rng(1)
    d = 0.3 + rng.normal(0, 1, 2000)
    lo, hi, n = block_ci(lambda i: d[i].mean(), len(d), 300, 24, 7)
    assert n == 300 and lo < 0.3 < hi and lo > 0
