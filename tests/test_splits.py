"""Required test 1: no train timestamp >= any val/test timestamp; wet months only; no window crosses seasons."""
import pandas as pd
import pytest

from pasig import config
from pasig.config import load, root
from pasig.phase1 import eval_hours


@pytest.mark.fast
def test_split_years_are_chronological_and_disjoint():
    s = load("splits")
    assert max(s["train_years"]) < min(s["val_years"]) <= max(s["val_years"]) < min(s["test_years"])
    assert set(s["wet_months"]) == {6, 7, 8, 9, 10, 11}
    assert config.split_of_year(2022) == "train" and config.split_of_year(2023) == "val" and config.split_of_year(2024) == "test"


@pytest.mark.fast
def test_no_window_crosses_seasons():
    idx = pd.date_range("2021-11-29", "2021-11-30 23:00", freq="h").append(pd.date_range("2022-06-01", periods=48, freq="h"))
    m = pd.DataFrame(index=idx)
    m["season"] = idx.year
    m["split"] = "train"
    m["pandacan"], m["pandacan_filled"], m["sg_report_ok"] = 11.0, False, True
    ev = eval_hours(m, [])
    s22 = ev[ev.season.eq(2022)]
    assert not s22["lead_1"].iloc[:23].any() and s22["lead_1"].iloc[23]   # needs 24 readings inside the season
    s21 = ev[ev.season.eq(2021)]
    assert not s21["lead_1"].iloc[-1]                                     # target would be in the next season


@pytest.mark.data
def test_master_is_wet_season_and_ordered():
    m = pd.read_parquet(root() / "data" / "processed" / "hourly_master.parquet")
    assert m.index.month.isin([6, 7, 8, 9, 10, 11]).all() and m.index.is_monotonic_increasing and m.index.is_unique
    tr, va, te = (m.index[m.split.eq(k)] for k in ("train", "val", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
    ev = pd.read_parquet(root() / "data" / "processed" / "eval_hours.parquet")
    assert ev.index.equals(m.index)
