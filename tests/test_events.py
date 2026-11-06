"""Event definition, detection window and warning time (Section 10.1)."""
import numpy as np
import pandas as pd
import pytest

from pasig.eval.events import detect, event_bootstrap_ci, events

pytestmark = pytest.mark.fast


def _series(vals, start="2024-07-01 00:00"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="h"))


def test_runs_with_short_gaps_merge():
    v = [11.0] * 5 + [12.1] * 3 + [11.0] * 4 + [12.2] * 2 + [11.0] * 10 + [12.3] * 1
    ev = events(_series(v), 12.0, merge_gap_h=6)
    assert len(ev) == 2                                   # gap of 4 h merged, gap of 10 h not
    assert ev.iloc[0]["hours"] == 5 and ev.iloc[0]["peak_m"] == pytest.approx(12.2)
    assert ev.iloc[0]["first"] == pd.Timestamp("2024-07-01 05:00")


def test_no_events():
    assert events(_series([11.0] * 10), 12.0, 6).empty


def test_detection_window_and_warning_time():
    ev = events(_series([11.0] * 10 + [12.5] * 2 + [11.0] * 5), 12.0, 6)   # first exceedance 10:00
    first = pd.Timestamp("2024-07-01 10:00")
    # lead-3 alert issued at 06:00 targets 09:00 (inside +-2 h) -> warning 4 h
    d = detect(ev, pd.DatetimeIndex([first - pd.Timedelta(hours=1)]), lead=3, window_h=2)
    assert bool(d.iloc[0]["detected"]) and d.iloc[0]["warning_h"] == 4.0
    # target 3 h early -> outside the window -> missed
    d = detect(ev, pd.DatetimeIndex([first - pd.Timedelta(hours=3)]), lead=3, window_h=2)
    assert not bool(d.iloc[0]["detected"]) and np.isnan(d.iloc[0]["warning_h"])
    # the earliest qualifying alert sets the warning time
    d = detect(ev, pd.DatetimeIndex([first + pd.Timedelta(hours=2), first - pd.Timedelta(hours=2)]), lead=1, window_h=2)
    assert d.iloc[0]["warning_h"] == 3.0


def test_event_bootstrap_ci():
    lo, hi = event_bootstrap_ci(np.array([1, 1, 1, 0] * 10), 500, 0)
    assert lo < 0.75 < hi
    assert event_bootstrap_ci(np.array([1, 1, 1]), 100, 0) == (1.0, 1.0)
