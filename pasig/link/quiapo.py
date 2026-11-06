"""G20 (D18): is the Quiapo 2021 block really Quiapo? Four conditions (Section 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pasig.config import load


def _corr(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    j = pd.concat([a, b], axis=1).dropna()
    return (float(j.iloc[:, 0].corr(j.iloc[:, 1])) if len(j) > 2 else np.nan), int(len(j))


def _is_copy(a: pd.Series, b: pd.Series, cmax: float) -> dict:
    j = pd.concat([a, b], axis=1).dropna()
    if len(j) < 3:
        return {"n_overlap": int(len(j)), "corr": None, "equal_share": None, "copy": False}
    r = float(j.iloc[:, 0].corr(j.iloc[:, 1]))
    eq = float(np.isclose(j.iloc[:, 0], j.iloc[:, 1]).mean())
    return {"n_overlap": int(len(j)), "corr": r, "equal_share": eq, "copy": bool(r >= cmax or eq >= 0.95)}


def g20(quiapo: pd.DataFrame, aviles: pd.DataFrame, taguig: pd.DataFrame, blocks: pd.DataFrame,
        pandacan: pd.Series, wet_months: list[int]) -> dict:
    """quiapo: pumping_long rows for Quiapo (all statuses); pandacan: observed hourly series (all hours available)."""
    cfg = load("link")["quiapo_g20"]
    q21 = quiapo[quiapo.block_year.eq(2021)]
    copies = {f: g.set_index("time_pht").sort_index() for f, g in q21.groupby("source_file")}
    ev = {"copies": list(copies)}
    # copies must agree; identical -> one series
    if len(copies) == 2:
        a, b = list(copies.values())
        cols = ["inside_m", "outside_m"]
        same = a[cols].reindex(b.index).fillna(-9).equals(b[cols].fillna(-9))
        ev["copies_identical"] = bool(same)
    else:
        ev["copies_identical"] = True
    s21 = next(iter(copies.values())) if copies else pd.DataFrame()

    # (a) layout and units: metres, O/C (+ nP) codes, two rows per hour
    vals = s21[["inside_m", "outside_m"]].notna().sum().sum()
    flagged = s21[["inside_m_range_flag", "outside_m_range_flag"]].sum().sum()
    in_range = float(vals / max(vals + flagged, 1))
    st = s21["raw_status"].dropna().astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    oc_like = st.str.fullmatch(r"[OC]|[OC]?\d+P.*").mean() if len(st) else np.nan
    two_rows = float((s21["outside_m"].notna() | ~s21["inside_m"].notna()).mean())
    conf = quiapo[quiapo.block_year.isin(cfg["confirmed_years"]) & quiapo.block_status.eq("use")]
    cst = conf["raw_status"].dropna().astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    conf_oc_like = float(cst.str.fullmatch(r"[OC]|[OC]?\d+P.*").mean())
    ev["a"] = {"in_9_14_m_share": in_range, "status_OC_or_nP_share": float(oc_like),
               "confirmed_years_status_OC_or_nP_share": conf_oc_like, "outside_row_present_share": two_rows}
    a_ok = in_range >= cfg["units_in_range_min_share"] and oc_like >= 0.95 and two_rows >= 0.95

    # (b) sheets dated 2021
    bq = blocks[blocks.file.isin(copies.keys()) & blocks.status.eq("pending_G20")]
    ev["b"] = {"block_years": sorted(set(bq.year.dropna().astype(int))), "n_blocks": int(len(bq)),
               "date_source": sorted(set(bq.date_source))}
    b_ok = ev["b"]["block_years"] == [2021] and set(bq.date_source) == {"header"}

    # (c) correlation of the outside elevation with Pandacan vs confirmed years (wet seasons)
    def wet(s):
        return s[s.index.month.isin(wet_months)]
    rc = {}
    for y in cfg["confirmed_years"]:
        g = conf[conf.block_year.eq(y)].drop_duplicates("time_pht").set_index("time_pht")["outside_m"]
        rc[y] = _corr(wet(g), pandacan)
    r21 = _corr(wet(s21["outside_m"]), pandacan)
    lowest = min(v[0] for v in rc.values())
    ev["c"] = {"corr_confirmed": {str(k): {"r": v[0], "n": v[1]} for k, v in rc.items()},
               "corr_2021": {"r": r21[0], "n": r21[1]}, "threshold": lowest - cfg["corr_margin"]}
    c_ok = r21[0] >= lowest - cfg["corr_margin"]

    # (d) not a copy of Aviles or of the Taguig files over the same hours
    av = aviles.drop_duplicates("time_pht").set_index("time_pht")
    tg = taguig.set_index("time_pht")
    ev["d"] = {"vs_aviles_outside": _is_copy(s21["outside_m"], av["outside_m"], cfg["copy_corr_max"]),
               "vs_aviles_inside": _is_copy(s21["inside_m"], av["inside_m"], cfg["copy_corr_max"]),
               "vs_taguig_land": _is_copy(s21["outside_m"], tg["land_m"], cfg["copy_corr_max"]),
               "vs_taguig_lake": _is_copy(s21["outside_m"], tg["lake_m"], cfg["copy_corr_max"]),
               "vs_taguig_land_inside": _is_copy(s21["inside_m"], tg["land_m"], cfg["copy_corr_max"])}
    d_ok = not any(v["copy"] for v in ev["d"].values())
    ev["conditions"] = {"a": bool(a_ok), "b": bool(b_ok), "c": bool(c_ok), "d": bool(d_ok),
                        "copies_agree": ev["copies_identical"]}
    ev["passed"] = bool(all(ev["conditions"].values()))
    return ev
