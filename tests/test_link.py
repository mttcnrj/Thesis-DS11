"""Phase 2 (Section 8): Wilson CI, events, legend checks, H1 rule, alert-level rule, day-block AUC bootstrap."""
import numpy as np
import pandas as pd
import pytest

from pasig import phase2
from pasig.link import legend as lg
from pasig.link import stats as st

pytestmark = pytest.mark.fast


def test_wilson_known_values():
    p, lo, hi = st.wilson(5, 10)
    assert p == 0.5 and np.isclose(lo, 0.2366, atol=1e-4) and np.isclose(hi, 0.7634, atol=1e-4)
    assert np.isnan(st.wilson(0, 0)[0])


def test_event_ids_merge_gaps_below_6h():
    t = pd.to_datetime(["2021-07-01 00:00", "2021-07-01 01:00", "2021-07-01 06:00", "2021-07-01 13:00"])
    assert st.event_ids(t, 6).tolist() == [0, 0, 0, 1]      # 5 h gap merged, 7 h gap splits


def test_h1_rule():
    or_yes = {"or": 1.2, "ci_low": 1.1, "ci_high": 1.3}
    or_no = {"or": 1.05, "ci_low": 0.9, "ci_high": 1.2}
    auc_yes = {"gain": 0.03, "gain_ci_low": 0.01, "gain_ci_high": 0.05}
    auc_no = {"gain": 0.01, "gain_ci_low": -0.02, "gain_ci_high": 0.03}
    assert phase2.h1_verdict(or_yes, auc_yes)["verdict"] == "supported"
    assert phase2.h1_verdict(or_yes, auc_no)["verdict"] == "partly supported"
    assert phase2.h1_verdict(or_no, auc_yes)["verdict"] == "partly supported"
    assert phase2.h1_verdict(or_no, auc_no)["verdict"] == "not supported"


def test_auc_bootstrap_detects_real_gain():
    rng = np.random.default_rng(0)
    n = 2400
    days = np.repeat(np.arange(100), 24)
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.5, size=n) > 1.2).astype(int)
    p0 = rng.random(n)                  # useless model
    p1 = 1 / (1 + np.exp(-x))           # informative model
    r = st.auc_gain_day_bootstrap(y, p0, p1, days, reps=200, seed=1)
    assert r["gain"] > 0.3 and r["gain_ci_low"] > 0


def _legend_frame(n=400):
    idx = pd.date_range("2020-07-01", periods=n, freq="h")
    d = pd.DataFrame(index=idx)
    d["raw_status"], d["status_parsed"] = "CF", True
    d["gate_open"] = 0.0
    d["pumps_running"] = 0
    d["inside_m"], d["outside_m"] = 10.6, 11.2
    d["sky"] = "F"
    d.iloc[:50, d.columns.get_loc("gate_open")] = 1.0                 # open gate: equal levels
    d.iloc[:50, d.columns.get_loc("outside_m")] = 10.6
    d.iloc[100:160, d.columns.get_loc("pumps_running")] = 2            # pumping: closed gate, river above sump
    d.iloc[100:160, d.columns.get_loc("gate_open")] = np.nan
    d.iloc[100:160, d.columns.get_loc("inside_m")] = 10.2
    return d


def test_legend_amended_checks_pass_on_physical_data():
    d = _legend_frame()
    pan = pd.Series(11.1, index=d.index)
    rain6 = pd.Series(0.0, index=d.index)
    r = lg.check_legend(d, pan, rain6)
    assert r["gate_O"]["fail_share"] == 1.0 and r["gate_O_amended"]["fail_share"] == 0.0
    assert r["pumps_river_above_sump_amended"]["fail_share"] == 0.0
    f = lg.field_decisions(r)
    assert f["gate_open"] and f["pumps_running"]


def test_legend_amended_check_catches_wrong_meaning():
    d = _legend_frame()
    d.iloc[100:160, d.columns.get_loc("outside_m")] = 10.0            # "pumping" with the river below the sump
    r = lg.check_legend(d, pd.Series(11.1, index=d.index), pd.Series(0.0, index=d.index))
    assert not lg.field_decisions(r)["pumps_running"]


def _alert_frame(test_shift=0.0):
    rng = np.random.default_rng(3)
    rows = []
    for y, sp in [(2020, "train"), (2021, "train"), (2023, "val"), (2024, "test")]:
        idx = pd.date_range(f"{y}-06-01", periods=3000, freq="h")
        L = 11.0 + 0.5 * rng.random(3000) + (rng.random(3000) < 0.05) * 0.8
        stress = ((L > 12.0) & (rng.random(3000) < 0.8)).astype(float)
        if sp == "test":
            L = L + test_shift
        rows.append(pd.DataFrame({"L": L, "stress": stress, "split": sp, "season": y}, index=idx))
    return pd.concat(rows)


def test_alert_levels_rule_and_leakage():
    a = phase2.alert_levels(_alert_frame())
    b = phase2.alert_levels(_alert_frame(test_shift=5.0))          # +5 m on every test value
    assert a == b                                                   # test data cannot move the levels
    assert a["flood_risk_m"] >= a["high_water_m"]
    scan = dict((lvl, share) for lvl, _, share in a["scan"])
    assert scan[a["flood_risk_m"]] >= 0.5
    below = [s for lvl, _, s in a["scan"] if lvl < a["flood_risk_m"]]
    assert all(s < 0.5 for s in below)
