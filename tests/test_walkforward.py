"""Notebook 07: walk-forward folds never see their evaluation year or 2024; D20 weighted loss (Section 10.2)."""
import numpy as np
import pandas as pd
import pytest

from pasig.config import load

YEARS = list(range(2018, 2024))


@pytest.mark.fast
def test_folds_are_chronological_and_never_reach_the_test_season():
    test = min(load("splits")["test_years"])
    for f in load("splits")["walk_forward"]:
        assert max(f["train"]) < f["eval"] < test
        assert list(f["train"]) == sorted(f["train"]) and len(f["train"]) >= 3   # >= 2 network years + inner validation


@pytest.mark.fast
def test_weighted_loss_reduces_to_masked_mse_at_weight_one():
    import keras  # noqa: F401
    from keras import ops

    from pasig.models.hybrid import masked_mse, weighted_masked_mse
    yt = np.array([[0.2, 0.9, -1.0], [0.95, 0.1, 0.5]], dtype="float32")
    yp = np.array([[0.1, 0.7, 0.3], [0.8, 0.2, 0.5]], dtype="float32")
    a = float(ops.convert_to_numpy(masked_mse(yt, yp)))
    b = float(ops.convert_to_numpy(weighted_masked_mse(1.0, 0.85)(yt, yp)))
    assert np.isclose(a, b)
    # weight 5 on the two targets >= 0.85; the invalid target (-1) stays excluded
    se = np.array([0.01, 0.04, 0.0, 0.0225, 0.01, 0.0])
    w = np.array([1, 5, 0, 5, 1, 1])
    c = float(ops.convert_to_numpy(weighted_masked_mse(5.0, 0.85)(yt, yp)))
    assert np.isclose(c, (se * w).sum() / 5)


@pytest.mark.data
def test_fold_fits_ignore_the_evaluation_year():
    """+5 m on every Pandacan value of the first fold's evaluation year leaves the fold's tide, alert levels,
    Granger result, training medians and scalers unchanged (test 2 applied to the walk-forward)."""
    from pasig.config import path
    from pasig.phase7 import fold_setup
    fold = load("splits")["walk_forward"][0]
    m = pd.read_parquet(path("processed") / "hourly_master.parquet")
    m = m[m["season"].isin(fold["train"] + [fold["eval"]])]
    ev = pd.read_parquet(path("processed") / "eval_hours.parquet").loc[m.index]
    a = fold_setup(fold, m, ev)
    m2 = m.copy()
    m2.loc[m2["season"].eq(fold["eval"]), "pandacan"] += 5.0
    b = fold_setup(fold, m2, ev)
    for k in ("high_water_m", "flood_risk_m", "granger_selected", "granger_p", "tide_added", "tide_floor_m"):
        assert a["info"][k] == b["info"][k], k
    for k in ("seq", "tab", "tab_cons"):
        assert np.allclose(a["scalers"][k].lo, b["scalers"][k].lo) and np.allclose(a["scalers"][k].hi, b["scalers"][k].hi)
    assert np.allclose(a["tide"].predict(m.index[:50]), b["tide"].predict(m.index[:50]))
    # and the evaluation-year targets do move
    assert not np.allclose(np.nan_to_num(a["windows"]["test"]["y"]), np.nan_to_num(b["windows"]["test"]["y"]))


@pytest.mark.data
def test_last_fold_reproduces_the_frozen_pipeline():
    """The 2018-22 -> 2023 fold has the main run's training years, so its refits must equal the frozen ones:
    alert levels (G8), Granger selection (G10), and the Phase 3 2023 windows and scalers, bit for bit."""
    from pasig.config import path
    from pasig.features.scaling import load_scalers
    from pasig.phase3 import load_windows
    from pasig.phase7 import fold_setup
    fold = load("splits")["walk_forward"][-1]
    m = pd.read_parquet(path("processed") / "hourly_master.parquet")
    m = m[m["season"] < min(load("splits")["test_years"])]
    ev = pd.read_parquet(path("processed") / "eval_hours.parquet").loc[m.index]
    fs = fold_setup(fold, m, ev)
    al = load("alert_levels")
    assert (fs["info"]["high_water_m"], fs["info"]["flood_risk_m"]) == (al["high_water_m"], al["flood_risk_m"])
    assert fs["info"]["granger_selected"] == load("selected_features")["selected_candidates"]
    w, ref = fs["windows"]["test"], load_windows("val")
    for k in ("X_seq", "X_tab", "X_tab_cons", "y", "valid"):
        assert np.array_equal(w[k], ref[k], equal_nan=k != "valid"), k
    sc = load_scalers(path("artifacts") / "scalers.json")
    for k in ("seq", "tab", "tab_cons", "y"):
        assert np.allclose(fs["scalers"][k].lo, sc[k].lo) and np.allclose(fs["scalers"][k].span, sc[k].span), k
