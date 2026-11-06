"""G4 (D3): check each part of the assumed Aviles/Quiapo status legend on 2018-2022 data (Section 8.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pasig.config import load


def _share(fail: pd.Series) -> tuple[int, float]:
    n = int(len(fail))
    return n, (float(fail.mean()) if n else np.nan)


def check_legend(st: pd.DataFrame, pandacan: pd.Series, rain6: pd.Series) -> dict:
    """`st`: one station's training wet-season hours (inside_m, outside_m, pumps_running, gate_open, sky,
    raw_status, status_parsed), indexed by time. Returns {check: {n, fail_share, passed}} plus the S verdict."""
    cfg = load("link")["legend"]
    mx = cfg["max_fail_share"]
    d = st.copy()
    d["pandacan"] = pandacan.reindex(d.index)
    d["rain6"] = rain6.reindex(d.index)
    both = d["inside_m"].notna() & d["outside_m"].notna()
    has = d["raw_status"].notna() & d["status_parsed"].astype(bool)
    out = {}

    g_open = d[has & both & d["gate_open"].eq(1)]
    out["gate_O"] = _share(g_open["outside_m"] >= g_open["inside_m"])
    g_closed = d[has & both & d["gate_open"].eq(0)]
    out["gate_C"] = _share(g_closed["outside_m"] < g_closed["inside_m"])
    pump = has & d["pumps_running"].ge(1)
    nog = d[pump & both & d["gate_open"].isna()]
    out["pumps_no_gate_closed"] = _share(nog["outside_m"] < nog["inside_m"])

    nopump = d[has & d["pumps_running"].eq(0)]
    med_sump, med_pan = nopump["inside_m"].median(), nopump["pandacan"].median()
    ph = d[pump & d["inside_m"].notna() & d["pandacan"].notna()]
    out["pumps_high_sump_or_river"] = _share(~((ph["inside_m"] > med_sump) | (ph["pandacan"] > med_pan)))
    days = d[pump].groupby("pumps_running").apply(lambda x: x.index.normalize().nunique())
    max_ok = int(days[days >= cfg["pumps_digit_min_days"]].index.max()) if (days >= cfg["pumps_digit_min_days"]).any() else 0
    out["pumps_digit"] = _share(d.loc[pump, "pumps_running"] > max_ok)

    sky = d["sky"].fillna("")
    dry = d[has & sky.isin(cfg["sky_dry_codes"]) & d["rain6"].notna()]
    out["sky_dry"] = _share(dry["rain6"] >= cfg["sky_dry_fail_mm"])
    wet = d[has & sky.isin(cfg["sky_rain_codes"]) & d["rain6"].notna()]
    out["sky_rain"] = _share(wet["rain6"] == 0)
    s = d[has & sky.eq(cfg["sky_ambiguous"]) & d["rain6"].notna()]
    rainy = lambda x: float((x["rain6"] > 0).mean()) if len(x) else np.nan  # noqa: E731
    p_s, p_wet, p_dry = rainy(s), rainy(wet), rainy(dry)
    s_meaning = None if np.isnan(p_s) else ("shower" if abs(p_s - p_wet) < abs(p_s - p_dry) else "sunny")

    # Post-hoc amendment approved by the group on 6 Nov 2026 (decision P2-a, docs/decisions.md). The literal checks
    # above assume the wrong physics: an open gate equalises sump and river (99% equal readings), and pumping lowers
    # the sump. Amended checks: 'O' fails if the river side is ABOVE the sump; 'nP' fails if the river side is not
    # above the sump (pumping is needed only against a closed gate and a higher river). Literal results stay reported.
    out["gate_O_amended"] = _share(g_open["outside_m"] > g_open["inside_m"] + 1e-9)
    ph2 = d[pump & both]
    out["pumps_river_above_sump_amended"] = _share(ph2["outside_m"] <= ph2["inside_m"])
    res = {k: {"n": n, "fail_share": f, "passed": bool(n > 0 and f <= mx)} for k, (n, f) in out.items()}
    for k in ("gate_O", "pumps_high_sump_or_river"):
        res[k]["superseded_by_amendment"] = True
    res["S_meaning"] = {"rainy_share_S": p_s, "rainy_share_rain_codes": p_wet, "rainy_share_dry_codes": p_dry,
                        "verdict": s_meaning, "n": int(len(s))}
    res["reference_no_pump_medians"] = {"sump": float(med_sump) if pd.notna(med_sump) else None,
                                        "pandacan": float(med_pan) if pd.notna(med_pan) else None,
                                        "max_valid_pumps": max_ok}
    return res


def field_decisions(res: dict) -> dict:
    """Which fields Phase 2 may use (a field is dropped if any of its checks fails). Uses the amended gate-'O' and
    pump checks (decision P2-a); the literal results are kept in the G4 evidence."""
    gate_ok = res["gate_O_amended"]["passed"] and res["gate_C"]["passed"]
    return {
        "gate_open": gate_ok,
        "gate_closed_while_pumping": res["pumps_no_gate_closed"]["passed"],
        "pumps_running": res["pumps_river_above_sump_amended"]["passed"] and res["pumps_digit"]["passed"],
        "sky": res["sky_dry"]["passed"] and res["sky_rain"]["passed"],
    }
