"""Granger procedure (Section 9.2): finds a planted 3 h lead; rejects noise at alpha = 0.01; lags stay in season."""
import numpy as np
import pandas as pd
import pytest

from pasig.phase4 import granger_test
from pasig.selection.granger import pooled_granger

pytestmark = pytest.mark.fast


def seasons(seed, n=1500, lead=3, coupled=True):
    rng = np.random.default_rng(seed)
    ys, xs = [], []
    for _ in range(4):
        x = rng.standard_normal(n)
        e = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + (0.6 * x[t - lead] if coupled and t >= lead else 0.0) + e[t]
        ys.append(y)
        xs.append(x)
    return ys, xs


def test_finds_planted_3h_lead():
    ys, xs = seasons(0)
    r = pooled_granger(ys, xs, max_lag=8)
    assert r["p_value"] < 1e-6 and r["lag"] >= 3


def test_rejects_noise_at_001():
    ys, _ = seasons(1, coupled=False)
    rng = np.random.default_rng(2)
    noise = [rng.standard_normal(len(y)) for y in ys]
    assert pooled_granger(ys, noise, max_lag=6)["p_value"] > 0.01


def test_lags_do_not_cross_seasons():
    # x is pure noise inside each season, but the END of season k is copied into the START of season k+1 of y:
    # a lag matrix that crossed the season boundary would pick that up; within-season lags must not.
    rng = np.random.default_rng(5)
    ys, xs = [], []
    prev_x = None
    for _ in range(4):
        x = rng.standard_normal(400)
        y = rng.standard_normal(400)
        if prev_x is not None:
            y[:3] = prev_x[-3:] * 10
        ys.append(y)
        xs.append(x)
        prev_x = x
    assert pooled_granger(ys, xs, max_lag=3)["p_value"] > 0.01


def test_granger_test_wrapper_with_dates():
    ys, xs = seasons(3, n=800)
    idx = [pd.date_range(f"{2018 + i}-06-01", periods=800, freq="h") for i in range(4)]
    y = pd.concat([pd.Series(v, index=i) for v, i in zip(ys, idx, strict=True)])
    x = pd.concat([pd.Series(v, index=i) for v, i in zip(xs, idx, strict=True)])
    r = granger_test(y, x, 6)
    assert r["p_value"] < 1e-6 and r["seasons"] == [2018, 2019, 2020, 2021]
