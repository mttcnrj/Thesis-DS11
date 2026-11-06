"""Raw-file manifest and gate G1 (Section 7.1, decisions D1, D6, D18; contradiction C5).

Every raw file gets a SHA-256 and a status (`use` / `exclude` / `pending_G20`). Identical-hash groups keep
exactly one non-excluded copy. Pumping workbooks are also dated block by block from their sheet headers
(D1: never from the file name) into data/manifest/pumping_blocks.csv, with a status per block.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from pasig.config import load, path
from pasig.io import mmda_pumping as mp

PUMP_DIR = "MMDA Dataset/WOCR and Water elevation reading of Aviles and Quiapo Pumping Station"
STUDY_YEARS = range(2018, 2025)
CONFIRMED_QUIAPO_YEARS = (2018, 2019, 2022)
QUIAPO_PENDING_YEAR = 2021


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(rel: str) -> dict:
    """Source, station and folder year of a raw file from its relative path."""
    parts = rel.split("/")
    name = parts[-1]
    out = {"source": "other", "station": None, "year": None}
    if rel.startswith("MMDA Dataset/Waterlevel Data"):
        out["source"] = "mmda_efcos"
        folder = parts[2]
        for sid, cfg in load("stations")["efcos"].items():
            if cfg["folder"] == folder:
                out["station"] = sid
        m = re.search(r"(20\d\d)", "/".join(parts[3:]))
        out["year"] = int(m.group(1)) if m else None
    elif rel.startswith(PUMP_DIR):
        m = re.search(r"(20\d\d)", name)
        out["year"] = int(m.group(1)) if m else None
        if name.upper().startswith("WOCR"):
            out["source"] = "mmda_wocr"
        elif name.upper().startswith("AVILES"):
            out.update(source="mmda_pumping", station="aviles")
        elif name.upper().startswith("QUIAPO"):
            out.update(source="mmda_pumping", station="quiapo")
    elif rel.startswith("PAGASA Dataset"):
        out["source"] = "pagasa"
        out["station"] = "port_area" if "Port Area" in name else ("science_garden" if "Science Garden" in name or "6-Hourly" in name else None)
    elif rel.startswith("Kaggle"):
        out["source"] = "kaggle"
    return out


def date_pumping_blocks(raw_root: Path, rel: str) -> tuple[list[dict], bool]:
    """Date every block of a pumping workbook by its sheet header. Returns (blocks, is_taguig)."""
    book = mp.read_book(raw_root / rel)
    if mp.is_taguig(book):
        return [], True
    out = []
    m = re.search(r"(20\d\d)", Path(rel).name)
    file_year = int(m.group(1)) if m else None
    for sn, df in book.items():
        for b, nxt in mp.blocks_with_bounds(df, sn):
            year, src = b.year, b.date_source
            if year is None and src == "sheet_name":   # header text missing: sheet name + file year
                year, src = file_year, "sheet_name+file_year"
            out.append({"file": rel, "sheet": sn, "header_row": b.header_row, "header_text": b.header_text,
                        "year": year, "month": b.month, "start_day": b.start_day, "end_day": b.end_day,
                        "date_source": src, "next_header_row": nxt})
    return out, False


def build_manifest(raw_root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_root = raw_root or path("raw")
    rows = []
    for p in sorted(raw_root.rglob("*")):
        if not p.is_file() or p.name.startswith((".", "~$")) or p.name.lower() == "desktop.ini":
            continue
        rel = p.relative_to(raw_root).as_posix()
        rows.append({"path": rel, "sha256": sha256(p), "bytes": p.stat().st_size, **classify(rel)})
    man = pd.DataFrame(rows)
    man["status"], man["reason"] = "use", ""

    def setst(mask, status, reason):
        man.loc[mask, "status"] = status
        man.loc[mask, "reason"] = reason

    setst(man.source.eq("kaggle"), "exclude", "Kaggle file excluded by group decision (Section 0.3)")
    setst(man.station.eq("port_area"), "exclude", "D6: Port Area excluded")
    setst(man.path.str.endswith("6-Hourly Data.csv"), "use", "Science Garden columns only (D6: Port Area columns dropped)")
    setst(man.path.str.endswith("A.ReadMe.txt"), "use", "PAGASA readme: units, -999 missing, -1 trace")
    setst(man.source.eq("mmda_wocr"), "use", "context for Phase 2 (monthly pump operating hours)")

    # pumping workbooks: content-based dating (D1) and Taguig detection
    blocks = []
    for i, r in man[man.source.eq("mmda_pumping")].iterrows():
        bl, taguig = date_pumping_blocks(raw_root, r["path"])
        if taguig:
            setst(man.index == i, "exclude", "contains Taguig Pumping Station data (labelled 2019), not Quiapo")
            continue
        blocks += bl

    # identical-hash groups: keep the first copy (sorted by path), exclude the rest
    man["dup_group"] = man.groupby("sha256")["path"].transform("first")
    dup_extra = man.duplicated("sha256", keep="first")
    for i in man.index[dup_extra]:
        keep = man.loc[i, "dup_group"]
        setst(man.index == i, "exclude", f"identical SHA-256 to {keep} (one copy kept)")
    man.loc[~man.duplicated("sha256", keep=False), "dup_group"] = ""

    bl = pd.DataFrame(blocks)
    excluded_files = set(man.loc[man.status.eq("exclude"), "path"])
    bl["status"], bl["reason"] = "use", ""
    bl.loc[bl.file.isin(excluded_files), ["status", "reason"]] = ["exclude", "file excluded (duplicate copy)"]
    is_q = bl.file.str.contains("QUIAPO")
    is_a = bl.file.str.contains("AVILES")
    live = bl.status.eq("use")
    bl.loc[live & ~bl.year.isin(list(STUDY_YEARS)), ["status", "reason"]] = ["exclude", "outside the 2018-2024 study period"]
    bl.loc[live & is_q & bl.year.eq(QUIAPO_PENDING_YEAR), ["status", "reason"]] = [
        "pending_G20", "Quiapo 2021 block (D18): used in the secondary Quiapo check only if it passes G20"]
    bl.loc[live & is_q & bl.year.eq(2023), ["status", "reason"]] = [
        "descriptive_only", "Quiapo 2023 (unexpected, transparency log F1): decision (a), 6 Nov 2026 - outside the "
                            "H1 verdict; reported only as a descriptive 2023 check of the alert levels"]
    bl.loc[live & is_q & bl.year.eq(2024), ["status", "reason"]] = [
        "exclude", "Quiapo 2024 (unexpected, F1): test year, sealed until after Notebook 06 (decision (a))"]
    bl.loc[live & is_a & bl.year.eq(2024), ["status", "reason"]] = [
        "post_eval_only", "2024 Aviles: only for the descriptive check after Notebook 06 (Section 10.2)"]
    # duplicate dating inside a file (e.g. AVILES 2024 sheets whose headers repeat another half-month)
    key = ["file", "year", "month", "start_day"]
    d = bl.duplicated(key, keep=False) & bl.status.ne("exclude")
    bl.loc[d, "reason"] = (bl.loc[d, "reason"] + "; header repeats another block's half-month").str.lstrip("; ")
    bl["dup_dating"] = d

    # file status for pumping workbooks follows their blocks
    for i, r in man[man.source.eq("mmda_pumping") & man.status.eq("use")].iterrows():
        st = set(bl.loc[bl.file.eq(r["path"]), "status"])
        yrs = sorted(set(bl.loc[bl.file.eq(r["path"]), "year"].dropna().astype(int)))
        if st - {"exclude"} == {"pending_G20"}:
            setst(man.index == i, "pending_G20",
                  f"dated by sheet headers (D1): years {yrs}; only its Quiapo 2021 blocks are usable (D18/G20)")
        else:
            note = f"dated by sheet headers (D1): years {yrs}"
            if "pending_G20" in st:
                note += "; contains a Quiapo 2021 block marked pending_G20 (D18)"
            setst(man.index == i, "use", note)
    return man, bl


def g1_check(man: pd.DataFrame, bl: pd.DataFrame) -> tuple[bool, dict]:
    problems = []
    if man["sha256"].isna().any() or ~man["status"].isin(["use", "exclude", "pending_G20"]).all():
        problems.append("file without hash or with an unknown status")
    for _, g in man[man.duplicated("sha256", keep=False)].groupby("sha256"):
        if (g.status != "exclude").sum() != 1:
            problems.append(f"identical-hash group {list(g.path)} has {(g.status != 'exclude').sum()} live copies")
    q = man[man.station.eq("quiapo")]
    taguig_like = q.path.str.contains("QUIAPO 2020|QUIAPO 2021")
    if not (q.loc[taguig_like, "status"] == "exclude").all():
        problems.append("Taguig files (QUIAPO 2020/2021) not excluded")
    live_q = q[q.status.ne("exclude")]
    undated = [f for f in live_q.path if bl.loc[bl.file.eq(f), "year"].isna().any() or not bl.file.eq(f).any()]
    if undated:
        problems.append(f"Quiapo copies not dated by sheet headers: {undated}")
    q21 = bl[bl.file.str.contains("QUIAPO") & bl.year.eq(QUIAPO_PENDING_YEAR) & bl.status.ne("exclude")]
    if q21.empty or not q21.status.eq("pending_G20").all():
        problems.append("Quiapo 2021 blocks are not all marked pending_G20")
    ev = {
        "n_files": len(man),
        "status_counts": man.status.value_counts().to_dict(),
        "identical_hash_groups": {g.iloc[0]["sha256"][:12]: list(g.path) for _, g in man[man.duplicated("sha256", keep=False)].groupby("sha256")},
        "taguig_excluded": list(q.loc[taguig_like, "path"]),
        "quiapo_blocks_by_file_year_status": bl[bl.file.str.contains("QUIAPO")].groupby(["file", "year", "status"]).size().reset_index().values.tolist(),
        "quiapo_2021_pending_G20_blocks": int(len(q21)),
        "problems": problems,
    }
    return not problems, ev
