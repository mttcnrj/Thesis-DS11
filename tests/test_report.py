"""Chapter IV assembler (pasig/eval/report.py): formatting helpers and the table builders on the real outputs."""
import numpy as np
import pandas as pd
import pytest

from pasig.config import out_dir
from pasig.eval import report as R


@pytest.mark.fast
def test_markdown_table_and_number_formats():
    df = pd.DataFrame({"1 h": ["0.0516 ± 0.0011", ""], "2 h": [np.nan, "x"]}, index=pd.Index(["a", "b"], name="Model"))
    md = R._md(df).splitlines()
    assert md[0] == "| Model | 1 h | 2 h |" and md[1] == "|---|---|---|" and md[2] == "| a | 0.0516 ± 0.0011 |  |"
    assert R._pm(0.05163, 0.00114, 4) == "0.0516 ± 0.0011" and R._pm(0.1, np.nan, 2) == "0.10" and R._pm(0.1, 0.0, 2) == "0.10"
    assert R._ci(-0.01234, 0.5, 3) == "[-0.012, 0.500]"
    assert R._fmt(pd.DataFrame({"x": [0.079, np.nan]}), 4)["x"].tolist() == ["0.0790", ""]


@pytest.mark.data
def test_builders_read_the_committed_outputs():
    tables = out_dir("tables")
    if not (tables / "test_results.json").exists():
        pytest.skip("Notebook 06 has not run")
    res = R._json(tables / "test_results.json")
    blocks = R.t_test_levels(tables) + R.t_dm(tables, res) + R.t_alerts(tables, res)
    keys = [b[0] for b in blocks]
    assert {"test_rmse_all", "test_dm_all", "test_equal_recall_h4"} <= set(keys)
    rmse = dict((b[0], b[2]) for b in blocks)["test_rmse_all"]
    lv = pd.read_csv(tables / "test_level_metrics.csv")
    ref = lv[(lv.subset == "all") & (lv.model == "hybrid") & (lv.lead == 3)]["rmse"].iloc[0]
    assert rmse.loc["hybrid", 3] == ref                 # the CSV beside each table keeps full precision
