"""Required test 3: a 6 h total is invisible until 1 h after its period end; daily values at 09:00 PHT
the next day; the G3 convention still holds on the real files."""
import numpy as np
import pandas as pd
import pytest

from pasig.config import load, root
from pasig.io import pagasa as pg


def _six():
    stamps = pd.date_range("2021-07-01 00:00", periods=12, freq="6h")     # UTC
    df = pd.DataFrame({"stamp_utc": stamps, "rain_6h_mm": np.arange(1, 13, dtype=float)})
    df["stamp_pht"] = df["stamp_utc"] + pd.Timedelta(hours=8)
    return pg.add_availability(df, "period_end")


@pytest.mark.fast
def test_6h_total_invisible_until_one_hour_after_period_end():
    six = _six()
    # report stamped 06 UTC (= 14:00 PHT, end of 08:00-14:00 PHT) holds 2.0 mm
    idx = pd.date_range("2021-07-01 13:00", "2021-07-01 16:00", freq="h")
    f = pg.hourly_rain_features(six, idx)
    assert f.loc["2021-07-01 14:00", "sg_rain_6h_latest"] == 1.0     # period ended, not yet usable
    assert f.loc["2021-07-01 15:00", "sg_rain_6h_latest"] == 2.0     # usable 1 h after period end
    assert f.loc["2021-07-01 15:00", "sg_hours_since_report"] == 0


@pytest.mark.fast
def test_rolling_totals_use_only_available_reports():
    six = _six()
    idx = pd.date_range("2021-07-02 09:00", periods=1, freq="h")   # reports available up to 00 UTC 2 Jul + 1 h
    f = pg.hourly_rain_features(six, idx)
    # available at 09:00 PHT 2 Jul: stamps up to 00 UTC 2 Jul (5th report, value 5), last 4 = 2+3+4+5
    assert f["sg_rain_24h"].iloc[0] == 2 + 3 + 4 + 5
    assert np.isnan(f["sg_rain_72h"].iloc[0])                        # fewer than 12 reports so far


@pytest.mark.fast
def test_period_start_convention_shifts_by_six_hours():
    df = _six().drop(columns=["period_end_pht", "period_start_pht", "available_from_pht"])
    a = pg.add_availability(df, "period_start")
    assert (a["available_from_pht"] - a["stamp_pht"]).eq(pd.Timedelta(hours=7)).all()


@pytest.mark.fast
def test_daily_value_usable_from_0900_next_day(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("YEAR,MONTH,DAY,RAINFALL,TMAX,TMIN,TMEAN,RH,WIND_SPEED,WIND_DIRECTION,ATM PRESSURE\n"
                 "2021,7,1,-1,30,24,27,80,2,90,1005\n2021,7,2,-999,30,24,27,80,2,90,-999\n")
    d = pg.read_daily(p)
    assert d.loc[0, "available_from_pht"] == pd.Timestamp("2021-07-02 09:00")
    assert d.loc[0, "rain_day_mm"] == pg.TRACE_MM                     # D4 trace
    assert np.isnan(d.loc[1, "rain_day_mm"]) and np.isnan(d.loc[1, "atm_pressure"])


@pytest.mark.fast
def test_trace_T_is_005mm(tmp_path):
    p = tmp_path / "6h.csv"
    p.write_text('Date(UTC),"Port Area, Manila",,,"Science Garden Quezon City, Metro Manila",,\n'
                 ",RAINFALL [mm],MAX TEMP [degC],MIN TEMP [degC],RAINFALL [mm],MAX TEMP [degC],MIN TEMP [degC]\n"
                 "7/1/2021 0:00,0,30,25,T,30,25\n7/1/2021 6:00,0,30,25,,30,25\n")
    s = pg.read_6hourly(p)
    assert s.loc[0, "rain_6h_mm"] == 0.05 and np.isnan(s.loc[1, "rain_6h_mm"])


@pytest.mark.data
def test_g3_convention_still_holds():
    raw = root() / "data" / "raw" / "PAGASA Dataset"
    ev = pg.g3_convention_test(pg.read_6hourly(raw / "6-Hourly Data.csv"), pg.read_daily(raw / "Science Garden Daily Data.csv"))
    assert ev["convention"] == load("pagasa")["convention"]


@pytest.mark.data
def test_master_rain_lags_real_reports():
    m = pd.read_parquet(root() / "data" / "processed" / "hourly_master.parquet")
    six = pd.read_parquet(root() / "data" / "interim" / "pagasa_6h.parquet")
    t = pd.Timestamp("2021-07-20 10:00")
    usable = six[six.available_from_pht <= t].iloc[-1]
    assert m.loc[t, "sg_rain_6h_latest"] == usable.rain_6h_mm or (np.isnan(usable.rain_6h_mm))
