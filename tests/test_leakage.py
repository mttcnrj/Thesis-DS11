"""Required test 2: +5 m on every 2024 value leaves fitted quantities unchanged (tide coefficients, scalers,
alert levels, Granger results, thresholds)."""
import numpy as np
import pandas as pd
import pytest

from pasig import phase2, phase3
from pasig.config import root
from pasig.features import windows as W
from pasig.features.scaling import MinMax
from tests.test_windows import synthetic


@pytest.mark.fast
def test_tide_fit_ignores_non_training_years():
    idx = pd.date_range("2018-01-01", "2024-12-31 23:00", freq="h")
    s = pd.Series(11 + 0.3 * np.sin(np.arange(len(idx)) * 2 * np.pi / 12.4206012), index=idx)
    m1 = phase3.fit_station_tide(s)[0]
    s2 = s.copy()
    s2[s2.index.year >= 2023] += 5.0
    m2 = phase3.fit_station_tide(s2)[0]
    assert np.array_equal(m1.coef_, m2.coef_)


@pytest.mark.fast
def test_scalers_ignore_validation_and_test():
    master, ev, six, tide = synthetic()
    tr1 = W.build_windows(master, ev, six, tide, "train")
    m2 = master.copy()
    m2.loc[m2.split.ne("train"), "pandacan"] += 5.0
    tr2 = W.build_windows(m2, ev, six, tide, "train")
    a, b = MinMax().fit(tr1["X_seq"]), MinMax().fit(tr2["X_seq"])
    assert np.array_equal(a.lo, b.lo) and np.array_equal(a.hi, b.hi)


@pytest.mark.data
def test_real_tide_coefficients_unchanged_by_2024_shift():
    q = pd.read_parquet(root() / "data" / "interim" / "waterlevel_qc.parquet")
    s = q[q.station.eq("pandacan")].set_index("time_pht")["value_qc"].sort_index()
    m1 = phase3.fit_station_tide(s)[0]
    s2 = s.copy()
    s2[s2.index.year == 2024] += 5.0
    assert np.array_equal(m1.coef_, phase3.fit_station_tide(s2)[0].coef_)


@pytest.mark.data
def test_real_alert_levels_unchanged_by_2024_shift():
    inp = phase2.load_inputs()
    av = phase2.station_frame(inp["pump"], "aviles")
    f = phase2.analysis_frame(inp, av, True, True)
    f2 = f.copy()
    f2.loc[f2.index.year == 2024, "L"] += 5.0
    assert phase2.alert_levels(f) == phase2.alert_levels(f2)


@pytest.mark.data
def test_real_granger_results_unchanged_by_2024_shift():
    from pasig.config import load
    from pasig.phase4 import detided_hourly, granger_test, six_hourly_pair
    m = pd.read_parquet(root() / "data" / "processed" / "hourly_master.parquet")
    six = pd.read_parquet(root() / "data" / "interim" / "pagasa_6h.parquet")
    m2 = m.copy()
    m2.loc[m2.split.eq("test"), "pandacan"] += 5.0
    lag = load("granger")["max_lag"]["six_hourly"]
    r1 = granger_test(*six_hourly_pair(detided_hourly(m, "pandacan"), six), lag)
    r2 = granger_test(*six_hourly_pair(detided_hourly(m2, "pandacan"), six), lag)
    assert r1["p_value"] == r2["p_value"] and r1["lag"] == r2["lag"]


@pytest.mark.fast
def test_thresholds_ignore_test_season():
    """Trigger offsets (D11) and classifier thresholds are tuned on train/val windows only: +5 m on every 2024 value
    changes neither. (Phase 3 also drops the test season before any window is built.)"""
    from sklearn.linear_model import LogisticRegression

    from pasig.config import load
    from pasig.models.offsets import best_offset, best_threshold
    from pasig.phase1 import eval_hours
    master, _, six, tide = synthetic(((2021, "train"), (2022, "train"), (2023, "val"), (2024, "test")))
    shifted = master.copy()
    shifted.loc[shifted.split.eq("test"), "pandacan"] += 5.0
    cfg = load("baselines")["trigger_offsets"]
    out = []
    for m in (master, shifted):
        ev = eval_hours(m, [])
        tr, va = (W.build_windows(m, ev, six, tide, sp) for sp in ("train", "val"))
        level = float(np.nanquantile(tr["y"], 0.9))
        persistence = np.repeat(va["X_seq"][:, -1, :1], 6, axis=1)
        offset = best_offset([persistence], va["y"], va["valid"], level, cfg)
        ok_tr, ok_va = tr["valid"][:, 0], va["valid"][:, 0]
        clf = LogisticRegression(max_iter=1000).fit(tr["X_tab"][ok_tr], tr["y"][ok_tr, 0] >= level)
        threshold = best_threshold(clf.predict_proba(va["X_tab"][ok_va])[:, 1], va["y"][ok_va, 0] >= level)
        out.append((level, offset, threshold))
    assert out[0] == out[1]
