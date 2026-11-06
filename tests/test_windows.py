"""Required test 4: y[:, h-1] is the level at t+h; tabular t+h terms come from the clock; windows stay in season."""
import numpy as np
import pandas as pd
import pytest

from pasig.features import windows as W
from pasig.features.tide import HarmonicTide
from pasig.phase1 import eval_hours

pytestmark = pytest.mark.fast


def synthetic(seasons=((2021, "train"), (2022, "train"), (2023, "val"))):
    parts = []
    for y, sp in seasons:
        idx = pd.date_range(f"{y}-06-01", f"{y}-11-30 23:00", freq="h")
        m = pd.DataFrame(index=idx)
        m["season"], m["split"] = y, sp
        m["pandacan"] = 11.0 + 0.3 * np.sin(np.arange(len(idx)) * 2 * np.pi / 12.42) + (y - 2020) * 0.01
        m["pandacan_filled"] = False
        m["sg_report_ok"] = True
        m["sg_hours_since_report"] = (np.arange(len(idx)) % 6).astype(float)
        for c in W.DAILY_TAB:
            m[c] = 1.0
        parts.append(m)
    master = pd.concat(parts)
    master.iloc[100, master.columns.get_loc("pandacan_filled")] = True
    stamps = pd.date_range("2021-01-01", "2023-12-31", freq="6h")
    six = pd.DataFrame({"rain_6h_mm": np.arange(len(stamps), dtype=float) % 7, "available_from_pht": stamps + pd.Timedelta(hours=1)})
    tide = HarmonicTide({"M2": 12.4206012, "K1": 23.93447213}, "2018-01-01").fit(master["pandacan"])
    return master, eval_hours(master, []), six, tide


def test_targets_are_level_at_t_plus_h():
    master, ev, six, tide = synthetic()
    a = W.build_windows(master, ev, six, tide, "train")
    t = pd.DatetimeIndex(a["t_index"])
    for h in range(1, 7):
        ok = a["valid"][:, h - 1]
        expect = master["pandacan"].reindex(t + pd.Timedelta(hours=h)).to_numpy()
        assert np.allclose(a["y"][ok, h - 1], expect[ok], atol=1e-5)
        assert np.isnan(a["y"][~ok, h - 1]).all()
    assert np.allclose(a["X_seq"][:, -1, 0], master["pandacan"].reindex(t).to_numpy(), atol=1e-5)
    # the filled hour (row 100) is never a valid target
    filled_t = master.index[100]
    for h in range(1, 7):
        hit = t == filled_t - pd.Timedelta(hours=h)
        assert not a["valid"][hit, h - 1].any()


def test_tabular_future_terms_come_from_the_clock_and_tide():
    master, ev, six, tide = synthetic()
    a = W.build_windows(master, ev, six, tide, "val")
    t = pd.DatetimeIndex(a["t_index"])
    names = a["tab_names"]
    for h in (1, 6):
        hs = a["X_tab"][:, names.index(f"hour_sin_t{h}")]
        assert np.allclose(hs, np.sin(2 * np.pi * (t + pd.Timedelta(hours=h)).hour / 24), atol=1e-5)
        tc = a["X_tab"][:, names.index(f"tide_change_t{h}")]
        assert np.allclose(tc, tide.predict(t + pd.Timedelta(hours=h)) - tide.predict(t), atol=1e-4)
    assert a["X_tab_cons"].shape[1] == 2 * 2 * 6


def test_windows_never_cross_seasons_and_rain_lags():
    master, ev, six, tide = synthetic()
    a = W.build_windows(master, ev, six, tide, "train")
    t = pd.DatetimeIndex(a["t_index"])
    starts = t - pd.Timedelta(hours=23)
    assert (starts.year == t.year).all() and (starts.month >= 6).all()
    # latest rain at t is the last report whose availability time is <= t
    r = W.rain_inputs(six, t[:50])
    for ti, v in zip(t[:50], r["sg_rain_6h_latest"], strict=True):
        last = six[six.available_from_pht <= ti].iloc[-1]
        assert v == last.rain_6h_mm
