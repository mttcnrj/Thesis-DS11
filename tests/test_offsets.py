"""D11 trigger offsets and F1-tuned classifier thresholds; baseline formulas (Table III-4)."""
import numpy as np
import pandas as pd
import pytest

from pasig.models import baselines as B
from pasig.models.offsets import best_offset, best_threshold, f1_score_counts

pytestmark = pytest.mark.fast
CFG = {"min_m": -0.30, "max_m": 0.30, "step_m": 0.01}


def test_offset_recovers_a_known_under_forecast():
    rng = np.random.default_rng(0)
    y = 11.0 + rng.random((2000, 6)) * 1.2
    pred = y - 0.10                                # every forecast is 0.10 m low
    v = np.ones_like(y, dtype=bool)
    r = best_offset([pred], y, v, 12.0, CFG)
    assert np.isclose(r["offset_m"], 0.10) and np.isclose(r["f1_val"], 1.0)


def test_offset_is_shared_by_all_leads_and_ignores_invalid():
    y = np.full((10, 6), 12.5)
    pred = np.full((10, 6), 12.5)
    v = np.ones_like(y, dtype=bool)
    v[:, 3] = False
    pred[:, 3] = 0.0                               # garbage at an invalid lead must not matter
    assert best_offset([pred], y, v, 12.0, CFG)["offset_m"] == 0.0


def test_threshold_and_f1():
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    truth = np.array([False, False, True, True])
    thr, f1 = best_threshold(prob, truth)
    assert f1 == 1.0 and 0.2 < thr <= 0.8
    assert f1_score_counts(np.array([True, False]), np.array([True, True])) == pytest.approx(2 / 3)


def test_persistence_and_tide_residual():
    class T:
        def predict(self, t):
            return np.sin(np.asarray(t.hour, dtype=float))
    t = pd.date_range("2021-07-01", periods=4, freq="h")
    d = {"t_index": t, "level_t": np.array([11.0, 11.1, 11.2, 11.3])}
    assert np.allclose(B.persistence(d), np.repeat(d["level_t"][:, None], 6, axis=1))
    tr = B.tide_resid(d, T())
    expect = np.sin((t + pd.Timedelta(hours=3)).hour.to_numpy()) + d["level_t"] - np.sin(t.hour.to_numpy())
    assert np.allclose(tr[:, 2], expect)
