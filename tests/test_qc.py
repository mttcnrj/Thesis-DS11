"""QC rules (Section 7.5): planted spikes and flat runs flagged; floor runs kept; 11.89 m rule."""
import numpy as np
import pandas as pd
import pytest

from pasig.qc.rules import apply_qc, spike_flags


def _long(station, values, start="2021-07-01"):
    t = pd.date_range(start, periods=len(values), freq="h")
    return pd.DataFrame({"time_pht": t, "station": station, "value_m": values})


def _tide(n, base=11.0, amp=0.4):
    return base + amp * np.sin(np.arange(n) * 2 * np.pi / 12.42)


@pytest.mark.fast
def test_planted_spike_flagged_but_ramp_kept():
    v = _tide(200)
    v[50] += 0.9           # one-hour spike
    v[120:126] += np.array([0.3, 0.6, 0.9, 0.9, 0.6, 0.3])   # multi-hour real rise
    H, _ = apply_qc(_long("pandacan", v))
    f = H.set_index("time_pht")["qc_flag"]
    assert f.iloc[50] == "spike"
    assert (f.iloc[118:128] == "").all()


@pytest.mark.fast
def test_range_flag():
    v = _tide(60)
    v[10] = 1.07           # typo
    H, _ = apply_qc(_long("pandacan", v))
    assert H["qc_flag"].iloc[10] == "range" and np.isnan(H["value_qc"].iloc[10])


@pytest.mark.fast
def test_pandacan_1189_rule_reproduces():
    v = _tide(300)
    v[100:105] = 11.89     # 5 h stuck -> removed
    v[200:203] = 11.89     # 3 h -> kept
    v[250:262] = 10.65     # low-tide floor plateau -> kept (no generic flat rule for Pandacan)
    H, runs = apply_qc(_long("pandacan", v))
    f = H["qc_flag"]
    assert (f.iloc[100:105] == "flat_11.89").all()
    assert (f.iloc[200:203] == "").all() and (f.iloc[250:262] == "").all()


@pytest.mark.fast
def test_fort_santiago_flat_run_flagged_floor_exempt():
    v = _tide(300)
    v[100:108] = 11.30     # 8 h identical mid-tide -> stuck
    v[200:210] = 10.65     # floor plateau -> exempt
    H, _ = apply_qc(_long("fort_santiago", v))
    f = H["qc_flag"]
    assert (f.iloc[100:108] == "flat").all() and (f.iloc[200:210] == "").all()


@pytest.mark.fast
def test_sto_nino_band_exemption_and_montalban_override():
    n = 200
    sn = np.full(n, 12.0)                                   # pool level, flat, inside 11.8-12.5
    mo = np.full(n, 21.0)
    a = pd.concat([_long("sto_nino", sn), _long("montalban", mo)])
    H, _ = apply_qc(a)
    assert (H[H.station.eq("sto_nino")]["qc_flag"] == "").all()       # exempt
    mo2 = mo.copy()
    mo2[60:] = 22.0                                         # Montalban rises 1 m while Sto. Niño stays flat
    H2, _ = apply_qc(pd.concat([_long("sto_nino", sn), _long("montalban", mo2)]))
    assert (H2[H2.station.eq("sto_nino")]["qc_flag"] == "flat_upstream_rise").all()
    H3, _ = apply_qc(a, flat_exemption=False)
    assert (H3[H3.station.eq("sto_nino")]["qc_flag"] == "flat").all()


@pytest.mark.fast
def test_spike_needs_opposite_jumps():
    s = pd.Series([11.0, 11.0, 11.8, 11.8, 11.0])
    assert not spike_flags(s, 0.5).any()                    # step up then down over two hours: not a spike
