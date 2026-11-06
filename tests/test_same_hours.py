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
