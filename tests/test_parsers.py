"""Section 11 / G2: EFCOS + pumping parsers on synthetic sheets with the real quirks."""
import numpy as np
import pandas as pd
import pytest

from pasig.io import mmda_pumping as mp
from pasig.io import mmda_waterlevel as mw
from tests.conftest import make_form21

pytestmark = pytest.mark.fast
PARTS = ("MMDA Dataset", "Waterlevel Data 2018-2024", "Pandacan 2018-2024", "2021")


def _parse(tmp_path, name, **kw):
    f = make_form21(tmp_path / name, **kw)
    return mw.parse_file(f, PARTS + (name,), "pandacan", "Pandacan")


def test_0900_rollover_day31_to_first_of_next_month(tmp_path):
    res = _parse(tmp_path, "7. July.xlsx", year=2021, month_name="July", ndays=31)
    d = res["data"].set_index("time_pht")["value_m"]
    # column 15 (00:00) of the day-31 row is 1 August 00:00; column 23 (08:00) is 1 August 08:00
    assert np.isclose(d[pd.Timestamp("2021-08-01 00:00")], 11.31 + 0.015)
    assert np.isclose(d[pd.Timestamp("2021-08-01 08:00")], 11.31 + 0.023)
    assert np.isclose(d[pd.Timestamp("2021-07-01 09:00")], 11.01)
    assert d.index.is_unique and len(d) == 31 * 24


def test_impossible_dates_set_aside(tmp_path):
    res = _parse(tmp_path, "6. June.xlsx", year=2021, month_name="June", ndays=31)
    assert len(res["data"]) == 30 * 24
    assert (res["set_aside"]["day"] == 31).all() and len(res["set_aside"]) == 24
    res = _parse(tmp_path, "2. February.xlsx", year=2021, month_name="February", ndays=31)
    assert sorted(res["set_aside"]["day"].unique()) == [29, 30, 31]


def test_missing_tokens_and_zero_are_nan(tmp_path):
    toks = ["**", "****", "**.**", "*", "-", None, "#REF!", 0.0, "11..39", 0]
    def vals(d):
        row = [11.0 + i / 100 for i in range(24)]
        if d == 2:
            row[: len(toks)] = toks
        return row
    res = _parse(tmp_path, "8. August.xlsx", year=2021, month_name="August", ndays=31, values=vals)
    x = res["data"]
    day2 = x[x.day.eq(2)].sort_values("time_pht")
    assert day2["value_m"].iloc[: len(toks)].isna().all()
    assert day2["value_m"].iloc[len(toks):].notna().all()
    kept = day2["raw_token"].iloc[: len(toks)].tolist()
    assert kept[0] == "**" and kept[6] == "#REF!" and kept[7] == "0" and kept[8] == "11..39"


def test_month_from_file_name_not_header(tmp_path):
    # header text claims February (as in Pandacan 2024/4. April.xlsx); the file name wins
    res = _parse(tmp_path, "4. April.xlsx", year=2021, month_name="February", ndays=30)
    assert res["data"]["time_pht"].min() == pd.Timestamp("2021-04-01 09:00")
    assert mw.month_from_filename("ROSARIO LS - JUNE 2018.xlsx") == 6
    assert mw.month_from_filename("11. November.xlsx.xlsx") == 11


def test_year_cell_must_match_folder(tmp_path):
    f = make_form21(tmp_path / "1. January.xlsx", year=2023, month_name="January", ndays=31)
    res = mw.parse_file(f, PARTS[:3] + ("2024", "1. January.xlsx"), "pandacan", "Pandacan")
    assert res["meta"]["status"] == "set_aside" and res["data"].empty


def test_highest_lowest_crosscheck_flags_shift(tmp_path):
    res = _parse(tmp_path, "9. September.xlsx", year=2021, month_name="September", ndays=30,
                 highest=lambda d: 11.0 + (d + 1) / 100 + 0.023)   # off by one day
    assert res["hl"]["mismatch"].mean() > 0.9
    ok = _parse(tmp_path, "10. October.xlsx", year=2021, month_name="October", ndays=31)
    assert not ok["hl"]["mismatch"].any()


def _pump_sheet(units_scale=100):
    """One Aviles-style block: header, day row, 24 x (inside+status, outside) rows."""
    rows = [["WATER ELEVATION READINGS: AVILES PUMPING STATION"] + [None] * 6,
            ["DATE:_JULY 1-15, 2019____"] + [None] * 6,
            [None, 1, None, 2, None, 3, None]]
    labels = ["12MN"] + [f"{h}AM" for h in range(1, 12)] + ["12NN"] + [f"{h}PM" for h in range(1, 12)]
    for lab in labels:
        rows.append([lab, 10.50 * units_scale, "CLRS", 0, "1PLRS", "**", "O"])
        rows.append([None, 11.25 * units_scale, None, 11.30 * units_scale, None, 11.35 * units_scale, None])
    return pd.DataFrame(rows)


def test_aviles_cm_to_m_hours_and_status():
    df = _pump_sheet()
    (b, nxt), = mp.blocks_with_bounds(df, "JULY 1-15")
    out = mp.parse_block(df, b, nxt, "cm", "aviles", "AVILES 2019.xlsx", 2019)
    assert len(out) == 24 * 3
    d1 = out[out.day.eq(1)].set_index("hour")
    assert np.isclose(d1.loc[0, "inside_m"], 10.5) and np.isclose(d1.loc[0, "outside_m"], 11.25)
    assert sorted(out.hour.unique()) == list(range(24))
    assert out.loc[out.day.eq(2), "inside_m"].isna().all()          # 0 -> missing
    st = out[out.day.eq(1)].iloc[0]
    assert st.gate_open == 0 and st.sky == "LRS" and st.pumps_running == 0
    st2 = out[out.day.eq(2)].iloc[0]
    assert st2.pumps_running == 1 and st2.sky == "LRS" and np.isnan(st2.gate_open)
    assert out[out.day.eq(3)].iloc[0].gate_open == 1
    assert out.time_pht.min() == pd.Timestamp("2019-07-01 00:00")


@pytest.mark.parametrize("code,pumps,gate,sky", [
    ("CF", 0, False, "F"), ("OC", 0, True, "C"), ("2PLRS", 2, None, "LRS"), ("C", 0, False, None),
    ("O", 0, True, None), ("2P’s", 2, None, None), ("1PF", 1, None, "F")])
def test_status_codes(code, pumps, gate, sky):
    s = mp.parse_status(code)
    assert s["pumps_running"] == pumps and s["sky"] == sky
    assert (np.isnan(s["gate_open"]) if gate is None else s["gate_open"] == gate)


@pytest.mark.parametrize("label,hour", [("12MN", 0), ("1AM", 1), ("12NN", 12), ("11PM", 23), ("01:00:00", 1),
                                        ("12:00:00 AM", 0), ("1:00:00 PM", 13), ("2AM  ", 2)])
def test_hour_labels(label, hour):
    assert mp.parse_hour(label) == hour
