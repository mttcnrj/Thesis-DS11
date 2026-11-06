"""MMDA EFCOS "FORM 2-1 HOURLY WATERLEVEL CHART" parser (Section 3.2, 7.2; gate G2).

* One row per day; 24 hourly columns 09:00 -> 08:00; columns 00:00-08:00 belong to day d+1:
  column i (0-based, 09:00 first) of day d -> Timestamp(year, month, d, 9) + i hours.
* Month from the file name; year from the `YEAR :` cell, which must equal the folder year. Header text
  ("APRIL 1-30", "February 1-31") is unreliable and never used for dating.
* Rows for impossible dates (31 June, 29-31 Feb in non-leap years) are set aside and logged.
* Missing tokens (`**`, `****`, `**.**`, `*`, `-`, `#REF!`, blanks, any non-numeric cell) -> NaN, raw token
  kept. A reading of exactly 0.0 m is missing.
* Every complete day (no missing hour) is cross-checked against the sheet's HIGHEST / LOWEST.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from pasig.config import load

MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august",
               "september", "october", "november", "december"]
HL_TOL_M = 0.005


def month_from_filename(name: str) -> int:
    """'7. July.xlsx' -> 7; 'ROSARIO LS - JUNE 2018.xlsx' -> 6; '1. January 2024.xlsx' -> 1."""
    stem = Path(name).stem.strip().lower()
    m = re.match(r"^\s*(\d{1,2})\s*\.", stem)
    if m:
        return int(m.group(1))
    for i, mn in enumerate(MONTH_NAMES, start=1):
        if mn in stem or (len(stem) >= 3 and re.search(rf"\b{mn[:3]}\b", stem)):
            return i
    raise ValueError(f"cannot read month from file name {name!r}")


def year_from_folder(rel_parts: tuple[str, ...]) -> int:
    for p in rel_parts:
        m = re.search(r"(20\d\d)", p)
        if m and not re.search(r"20\d\d\s*-\s*20\d\d", p):
            return int(m.group(1))
    raise ValueError(f"no year folder in {rel_parts}")


def _cell_number(v) -> tuple[float, str | None]:
    if v is None:
        return np.nan, None
    if isinstance(v, bool):
        return np.nan, str(v)
    if isinstance(v, (int, float)):
        x = float(v)
        if np.isnan(x):
            return np.nan, None
        return (np.nan, "0") if x == 0.0 else (x, None)
    s = str(v).strip()
    return np.nan, (s if s else None)


def _sheet_meta(rows: list[tuple]) -> dict:
    year = month_cell = None
    for r in rows[:6]:
        vals = list(r)
        for j, c in enumerate(vals):
            if isinstance(c, str) and "YEAR" in c.upper() and year is None:
                year = next((x for x in vals[j + 1:] if x is not None and str(x).strip()), None)
            if isinstance(c, str) and "MONTH" in c.upper() and month_cell is None:
                month_cell = next((x for x in vals[j + 1:] if x is not None and str(x).strip()), None)
    try:
        year = int(float(year)) if year is not None else None
    except (TypeError, ValueError):
        year = None
    return {"year_cell": year, "month_cell": month_cell}


def _layout_ok(rows: list[tuple], lay: dict) -> bool:
    if len(rows) <= lay["header_row"]:
        return False
    h = rows[lay["header_row"]]
    c = lay["first_hour_col"]
    return len(h) > c and h[c] == dt.time(lay["first_hour"], 0)


def parse_file(path: Path, rel_parts: tuple[str, ...], station: str, station_label: str) -> dict:
    """Parse one monthly EFCOS workbook. Returns dict(data=long DataFrame, set_aside=DataFrame,
    hl=DataFrame of HIGHEST/LOWEST checks, meta=dict)."""
    lay = load("stations")["efcos_layout"]
    folder_year = year_from_folder(rel_parts)
    month = month_from_filename(path.name)
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    candidates = []
    for sn in wb.sheetnames:
        rows = list(wb[sn].iter_rows(values_only=True))
        if _layout_ok(rows, lay):
            candidates.append((sn, rows, _sheet_meta(rows)))
    wb.close()
    meta = {"file": str(Path(*rel_parts)), "station": station, "folder_year": folder_year, "month": month,
            "n_sheets_with_layout": len(candidates), "sheet": None, "year_cell": None, "status": "ok", "reason": ""}
    empty = {"data": pd.DataFrame(), "set_aside": pd.DataFrame(), "hl": pd.DataFrame(), "meta": meta}
    if not candidates:
        meta.update(status="set_aside", reason="no sheet with the FORM 2-1 layout")
        return empty
    if len(candidates) > 1:
        # multi-sheet workbook: keep the sheet named after the station, if any
        lab = re.sub(r"[^A-Z]", "", station_label.upper())
        named = [c for c in candidates if re.sub(r"[^A-Z]", "", c[0].upper()) == lab]
        candidates = named or candidates[:1]
    sn, rows, sm = candidates[0]
    meta.update(sheet=sn, year_cell=sm["year_cell"], month_cell=sm["month_cell"])
    if sm["year_cell"] != folder_year:
        meta.update(status="set_aside", reason=f"YEAR cell {sm['year_cell']} != folder year {folder_year}")
        return empty

    year = folder_year
    recs, aside, hl = [], [], []
    c0, nh = lay["first_hour_col"], lay["n_hours"]
    for r in rows[lay["header_row"] + 1:]:
        d = r[lay["day_col"]] if r else None
        if not isinstance(d, (int, float)) or not float(d).is_integer() or not 1 <= int(d) <= 31:
            if isinstance(d, str) and d.strip().upper().startswith("AVE"):
                break
            continue
        d = int(d)
        cells = list(r[c0:c0 + nh]) + [None] * max(0, nh - len(r[c0:c0 + nh]))
        try:
            base = pd.Timestamp(year, month, d, lay["first_hour"])
        except ValueError:
            for i, v in enumerate(cells):
                val, tok = _cell_number(v)
                aside.append({"station": station, "year": year, "month": month, "day": d, "col": i,
                              "value_m": val, "raw_token": tok, "source_file": meta["file"], "sheet": sn})
            continue
        vals = []
        for i, v in enumerate(cells):
            val, tok = _cell_number(v)
            vals.append(val)
            recs.append((base + pd.Timedelta(hours=i), val, tok, d, i))
        hi_v = r[lay["highest_col"]] if len(r) > lay["highest_col"] else None
        lo_v = r[lay["lowest_col"]] if len(r) > lay["lowest_col"] else None
        arr = np.array(vals, dtype=float)
        complete = bool(np.isfinite(arr).all())
        hi = float(hi_v) if isinstance(hi_v, (int, float)) and not isinstance(hi_v, bool) else np.nan
        lo = float(lo_v) if isinstance(lo_v, (int, float)) and not isinstance(lo_v, bool) else np.nan
        row = {"station": station, "file": meta["file"], "date": base.normalize(), "complete": complete,
               "parsed_max": np.nanmax(arr) if np.isfinite(arr).any() else np.nan,
               "parsed_min": np.nanmin(arr) if np.isfinite(arr).any() else np.nan,
               "highest": hi, "lowest": lo}
        row["comparable"] = complete and np.isfinite(hi) and np.isfinite(lo)
        row["mismatch"] = bool(row["comparable"] and (abs(row["parsed_max"] - hi) > HL_TOL_M
                                                      or abs(row["parsed_min"] - lo) > HL_TOL_M))
        hl.append(row)
    data = pd.DataFrame(recs, columns=["time_pht", "value_m", "raw_token", "day", "col"])
    data.insert(1, "station", station)
    data["source_file"] = meta["file"]
    data["sheet"] = sn
    data["flag_impossible_date"] = False
    set_aside = pd.DataFrame(aside)
    if len(set_aside):
        set_aside["flag_impossible_date"] = True
    meta["n_rows_days"] = int(len(hl))
    meta["n_impossible_rows"] = int(set_aside["day"].nunique()) if len(set_aside) else 0
    return {"data": data, "set_aside": set_aside, "hl": pd.DataFrame(hl), "meta": meta}


_FORMULA_RE = re.compile(r"^=\s*(MAX|MIN)\(\s*\$?B\$?(\d+)\s*:\s*\$?Y\$?(\d+)\s*\)\s*$", re.I)


def diagnose_hl(path: Path, hl_file: pd.DataFrame, long_file: pd.DataFrame) -> pd.DataFrame:
    """Explain HIGHEST/LOWEST mismatches of one file (run only when its mismatch rate > 1%).

    For each mismatching day, read the HIGHEST/LOWEST formulas. If a formula aggregates a different
    row than its own (a template error in the source sheet), re-check the cached value against the
    readings of the row it points to. Also test continuity across the day boundary (08:00 of day d vs
    09:00 of day d+1) to make sure the reading rows themselves are in day order.
    A day is `explained` when the formula points to another row and its cached value matches that row."""
    lay = load("stations")["efcos_layout"]
    wb = openpyxl.load_workbook(path, data_only=False, read_only=True)
    ws = wb[long_file["sheet"].iloc[0]]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    by_sheet_row = {}
    for i, r in enumerate(rows):
        d = r[lay["day_col"]] if r else None
        if isinstance(d, (int, float)) and float(d).is_integer():
            by_sheet_row[i + 1] = int(d)          # Excel row number -> day label
    vals = {d: g.sort_values("col")["value_m"].to_numpy() for d, g in long_file.groupby("day")}
    out = []
    for _, h in hl_file[hl_file["mismatch"]].iterrows():
        day = h["date"].day
        excel_row = next((k for k, v in by_sheet_row.items() if v == day), None)
        r = rows[excel_row - 1] if excel_row else ()
        f_hi = r[lay["highest_col"]] if len(r) > lay["highest_col"] else None
        m = _FORMULA_RE.match(str(f_hi or ""))
        ref_row = int(m.group(2)) if m else None
        ref_day = by_sheet_row.get(ref_row) if ref_row else None
        explained = False
        if ref_day is not None and ref_day != day and ref_day in vals:
            v = vals[ref_day]
            # Excel MAX/MIN skip text cells, so compare with the available readings of that row
            explained = bool(np.isfinite(v).any() and abs(np.nanmax(v) - h["highest"]) <= HL_TOL_M
                             and abs(np.nanmin(v) - h["lowest"]) <= HL_TOL_M)
        cont = np.nan
        if day in vals and (day + 1) in vals:
            cont = float(abs(vals[day][-1] - vals[day + 1][0]))
        out.append({"file": h["file"], "date": h["date"], "excel_row": excel_row, "highest_formula": f_hi,
                    "formula_row": ref_row, "formula_day": ref_day, "explained_by_formula": explained,
                    "jump_to_next_day_m": cont})
    return pd.DataFrame(out)


def station_files(raw_root: Path, folder: str) -> list[tuple[Path, tuple[str, ...]]]:
    base = raw_root / "MMDA Dataset" / "Waterlevel Data 2018-2024" / folder
    out = []
    for p in sorted(base.rglob("*.xls*")):
        if p.name.startswith("~$"):
            continue
        out.append((p, p.relative_to(raw_root).parts))
    return out


def parse_all(raw_root: Path, files_in_use: set[str] | None = None) -> dict[str, pd.DataFrame]:
    """Parse every EFCOS station in configs/stations.yaml. `files_in_use` (manifest paths with status
    'use') restricts parsing to G1-approved files."""
    st = load("stations")["efcos"]
    data, aside, hl, metas = [], [], [], []
    for sid, cfg in st.items():
        for p, parts in station_files(raw_root, cfg["folder"]):
            rel = str(Path(*parts))
            if files_in_use is not None and rel not in files_in_use:
                continue
            res = parse_file(p, parts, sid, cfg["label"])
            metas.append(res["meta"])
            if len(res["data"]):
                data.append(res["data"])
            if len(res["set_aside"]):
                aside.append(res["set_aside"])
            if len(res["hl"]):
                hl.append(res["hl"])
    long = pd.concat(data, ignore_index=True)
    dup = long.duplicated(["station", "time_pht"], keep=False)
    long["flag_duplicate_time"] = dup
    return {
        "long": long.sort_values(["station", "time_pht"]).reset_index(drop=True),
        "set_aside": pd.concat(aside, ignore_index=True) if aside else pd.DataFrame(),
        "hl": pd.concat(hl, ignore_index=True) if hl else pd.DataFrame(),
        "files": pd.DataFrame(metas),
    }
