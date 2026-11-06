"""Required test 5: only gaps <= 3 h are filled and flagged; filled hours are never valid targets."""
import numpy as np
import pandas as pd
import pytest

from pasig.phase1 import eval_hours
from pasig.qc.gaps import fill_short_gaps


@pytest.mark.fast
def test_only_short_gaps_filled():
    s = pd.Series([1.0, np.nan, 2.0, np.nan, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 4.0, np.nan])
    f, m = fill_short_gaps(s, 3)
    assert f.tolist()[:7] == [1, 1, 2, 2, 2, 2, 3]
    assert f.iloc[7:11].isna().all()                        # 4-hour gap stays NaN
    assert np.isnan(f.iloc[12])                             # open gap at the end is not filled
    assert m.tolist() == [False, True, False, True, True, True, False, False, False, False, False, False, False]


@pytest.mark.fast
def test_leading_gap_not_filled():
    f, m = fill_short_gaps(pd.Series([np.nan, np.nan, 1.0]), 3)
    assert f.isna().sum() == 2 and not m.any()


def _master(n=60):
    idx = pd.date_range("2021-06-01", periods=n, freq="h")
    m = pd.DataFrame(index=idx)
    m["season"], m["split"] = 2021, "train"
    m["pandacan"] = 11.0
    m["pandacan_filled"] = False
    m["sg_report_ok"] = True
    return m


@pytest.mark.fast
def test_filled_hours_never_targets():
    m = _master()
    m.iloc[40, m.columns.get_loc("pandacan_filled")] = True
    ev = eval_hours(m, [])
    for h in range(1, 7):
        assert not ev[f"lead_{h}"].iloc[40 - h]             # target at t+h = row 40 is filled
    assert ev["lead_1"].iloc[38]
