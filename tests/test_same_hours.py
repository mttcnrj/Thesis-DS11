"""Required test 6: every model (incl. persistence) is scored on identical (hour, lead) pairs."""
import numpy as np
import pandas as pd
import pytest

from pasig.config import root
from pasig.train.loop import preds_frame


@pytest.mark.fast
def test_preds_frame_keeps_the_valid_mask():
    t = pd.date_range("2021-07-01", periods=3, freq="h")
    v = np.array([[True] * 6, [False] + [True] * 5, [True] * 6])
    d = {"t_index": t, "y_raw": np.ones((3, 6)), "valid": v}
    a = preds_frame(d, np.zeros((3, 6)), "hybrid", 0)
    b = preds_frame(d, np.ones((3, 6)), "persistence")
    assert a[["t", "lead", "valid"]].equals(b[["t", "lead", "valid"]])


@pytest.mark.data
@pytest.mark.parametrize("base", ["reports/tables", "runs/rehearsal/reports/tables"])
def test_validation_predictions_share_hours(base):
    files = [root() / base / f for f in ("preds_val_baselines.parquet", "preds_val_neural.parquet")]
    if not all(f.exists() for f in files):
        pytest.skip(f"no Notebook 05 predictions under {base}")
    pv = pd.concat([pd.read_parquet(f) for f in files])
    pv = pv[pv["valid"]]
    keys = pv.groupby(["model", "seed"]).apply(lambda g: frozenset(zip(g["t"], g["lead"], strict=True)))
    assert keys.nunique() == 1


@pytest.mark.data
@pytest.mark.parametrize("base", ["reports/tables", "runs/rehearsal/reports/tables"])
def test_test_predictions_share_hours(base):
    f = root() / base / "preds_test.parquet"
    if not f.exists():
        pytest.skip(f"no Notebook 06 predictions under {base}")
    pt = pd.read_parquet(f)
    pt = pt[pt["valid"]]
    keys = pt.groupby(["model", "seed"]).apply(lambda g: frozenset(zip(g["t"], g["lead"], strict=True)))
    assert keys.nunique() == 1


@pytest.mark.data
def test_rehearsal_reloaded_models_reproduce_notebook05():
    """On the 2023 stand-in, NB06's reloaded checkpoints, frozen baselines and refitted KNNs must reproduce the
    validation predictions NB05 saved, exactly (proves the test season is scored with the frozen models)."""
    T = root() / "runs/rehearsal/reports/tables"
    if not (T / "preds_test.parquet").exists():
        pytest.skip("no Notebook 06 rehearsal")
    pt = pd.read_parquet(T / "preds_test.parquet")
    pv = pd.concat([pd.read_parquet(T / f) for f in ("preds_val_neural.parquet", "preds_val_baselines.parquet")])
    m = pt.merge(pv, on=["t", "lead", "model", "seed"], suffixes=("_te", "_va"))
    assert len(m) == len(pt) == len(pv)
    assert np.allclose(m["y_pred_te"], m["y_pred_va"], atol=1e-6, equal_nan=True)
    pr = pd.read_parquet(T / "probs_test.parquet")
    pc = pd.read_parquet(T / "probs_val_classifiers.parquet")
    j = pr.merge(pc, on=["t", "lead", "model", "level"], suffixes=("_te", "_va"))
    assert len(j) == len(pr) == len(pc) and np.allclose(j["prob_te"], j["prob_va"], atol=1e-9)


@pytest.mark.data
@pytest.mark.parametrize("base", ["reports/tables", "runs/rehearsal/reports/tables"])
def test_ablations_scored_on_the_test_hours(base):
    """Required test 6 for Notebook 07: every ablation is scored on exactly the hours of Notebook 06."""
    a, t = root() / base / "preds_ablation_test.parquet", root() / base / "preds_test.parquet"
    if not (a.exists() and t.exists()):
        pytest.skip(f"no Notebook 07 ablations under {base}")
    pa, pt = pd.read_parquet(a), pd.read_parquet(t)
    ref = frozenset(zip(*pt.loc[pt["valid"] & pt["model"].eq("persistence"), ["t", "lead"]].to_numpy().T, strict=True))
    for _, g in pa[pa["valid"]].groupby(["model", "seed"]):
        assert frozenset(zip(g["t"], g["lead"], strict=True)) == ref


@pytest.mark.data
@pytest.mark.parametrize("base", ["reports/tables", "runs/rehearsal/reports/tables"])
def test_walkforward_models_share_hours_per_fold(base):
    f = root() / base / "preds_walkforward.parquet"
    if not f.exists():
        pytest.skip(f"no walk-forward under {base}")
    pw = pd.read_parquet(f)
    pw = pw[pw["valid"]]
    for _, g in pw.groupby("fold"):
        keys = g.groupby(["model", "seed"]).apply(lambda x: frozenset(zip(x["t"], x["lead"], strict=True)))
        assert keys.nunique() == 1
