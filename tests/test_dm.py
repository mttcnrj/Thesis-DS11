"""Diebold-Mariano with Newey-West variance and the HLN correction; Holm adjustment (Section 10.1)."""
import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from pasig.eval.dm_test import diebold_mariano

pytestmark = pytest.mark.fast


def test_equal_models_are_not_significant():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 1, (2, 3000))
    r = diebold_mariano(e[0], e[1], h=3)
    assert r["p_value"] > 0.01 and r["nw_lags"] >= 2


def test_better_model_has_negative_statistic_and_small_p():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 0.8, 3000), rng.normal(0, 1.0, 3000)
    r = diebold_mariano(a, b, h=1)
    assert r["mean_loss_diff"] < 0 and r["dm_hln"] < 0 and r["p_value"] < 1e-6


def test_hln_shrinks_the_statistic_and_nans_are_dropped():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 0.9, 500), rng.normal(0, 1.0, 500)
    a[:10] = np.nan
    r = diebold_mariano(a, b, h=6)
    assert r["n"] == 490 and abs(r["dm_hln"]) < abs(r["dm"])


def test_holm_is_monotone_and_not_below_raw():
    p = np.array([0.001, 0.01, 0.02, 0.04, 0.3])
    adj = multipletests(p, method="holm")[1]
    assert np.all(adj >= p) and np.all(np.diff(adj[np.argsort(p)]) >= 0)
