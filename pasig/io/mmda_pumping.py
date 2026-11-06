"""MMDA pumping-station water-elevation sheets (Aviles, Quiapo) -> long table (Section 7.3).

Layout (both stations): half-month blocks. A block has a date header ("DATE:_JULY 1-15, 2019",
"JANUARY 1-15, 2021"), a row of day numbers (one value column + one status column per day), then
24 hours x two rows: the hour row holds the inside/sump elevation and the status code, the next row
the outside/river elevation. Aviles keeps one block per sheet; Quiapo stacks 24 blocks per sheet.

Units: Aviles in centimetres (1135 -> 11.35 m), Quiapo in metres. Readings of exactly 0 and every
non-numeric token are missing (raw token kept). Pumping data are Phase 2 only, never model inputs.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}
# "JULY 1-15, 2019", "DATE:_JULY 1-15, 2019___", "JANUARY(1-15)", "SEPT 16-30 2020", "JAN 01-15, 2024"
HEADER_RE = re.compile(
    r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?\s*\(?\s*(\d{1,2})\s*-\s*(\d{1,2})\s*\)?"
    r"\s*,?\s*(\d{4})?")
# D3 status codes. Plain codes: optional gate letter then sky code ("CLRS" = gate C, sky LRS).
STATUS_PLAIN = re.compile(r"^(?P<gate>[OC])?(?P<sky>[A-Z]+)$")
STATUS_PUMPS = re.compile(r"(?P<pumps>\d+)\s*P\s*(?P<sky>[A-Z]*)")


@dataclass
class Block:
    sheet: str
    header_row: int          # row index of the date header (or -1 if dated from the sheet name)
    header_text: str
    year: int | None
    month: int | None
    start_day: int | None
    end_day: int | None
    date_source: str         # "header" | "sheet_name"


# ----------------------------------------------------------------------------- reading

def read_book(path: Path) -> dict[str, pd.DataFrame]:
    engine = "odf" if str(path).lower().endswith(".ods") else "openpyxl"
    return pd.read_excel(path, sheet_name=None, header=None, engine=engine)


def _row_text(df: pd.DataFrame, i: int, ncols: int = 4) -> str:
    vals = [v for v in df.iloc[i, :ncols].tolist() if isinstance(v, str)]
    return " ".join(vals)


def parse_header(text: str) -> tuple[int, int, int, int | None] | None:
    m = HEADER_RE.search(text.upper())
    if not m:
        return None
    month = MONTHS[m.group(1)[:3]]
    return month, int(m.group(2)), int(m.group(3)), (int(m.group(4)) if m.group(4) else None)


def find_blocks(df: pd.DataFrame, sheet: str) -> list[Block]:
    """Locate every half-month block header in a sheet (header text in the first 4 columns)."""
    blocks = []
    for i in range(len(df)):
        txt = _row_text(df, i)
        if not txt or len(txt) > 80:
            continue
        p = parse_header(txt)
        if p:
            blocks.append(Block(sheet, i, txt.strip(), p[3], p[0], p[1], p[2], "header"))
    if not blocks:  # e.g. AVILES 2023 "APRIL 16-30": header text missing, date from the sheet name
        p = parse_header(sheet)
        if p:
            blocks.append(Block(sheet, -1, sheet, p[3], p[0], p[1], p[2], "sheet_name"))
    return blocks


def is_taguig(book: dict[str, pd.DataFrame]) -> bool:
    for df in book.values():
        head = [str(v) for v in df.iloc[:6].to_numpy().ravel()]
        if any("TAGUIG" in s.upper() for s in head):
            return True
    return False


# ----------------------------------------------------------------------------- cells

def parse_hour(label) -> int | None:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    if isinstance(label, (dt.time, dt.datetime)):
        return label.hour
    if isinstance(label, (int, float, np.floating)) and 0 <= float(label) < 1:
        return int(round(float(label) * 24)) % 24
    s = str(label).strip().upper().replace(".", "")
    if not s:
        return None
    if s in {"12MN", "12 MN", "MN", "12 MIDNIGHT", "MIDNIGHT", "12AM", "12 AM"}:
        return 0
    if s in {"12NN", "12 NN", "NN", "12N", "NOON", "12 NOON", "12PM", "12 PM"}:
        return 12
    m = re.match(r"^(\d{1,2})\s*(AM|PM)$", s)
    if m:
        h = int(m.group(1)) % 12
        return h + (12 if m.group(2) == "PM" else 0)
    m = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?\s*(AM|PM)?$", s)
    if m:
        h = int(m.group(1))
        if m.group(3):
            h = h % 12 + (12 if m.group(3) == "PM" else 0)
        return h % 24
    return None


def to_number(v) -> tuple[float, str | None]:
    """Numeric cell -> (value, None); 0, blanks and any non-numeric token -> (nan, raw token)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan, None
    if isinstance(v, (bool, np.bool_)):
        return np.nan, str(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        return (np.nan, "0") if x == 0.0 else (x, None)
    s = str(v).strip()
    return np.nan, (s if s else None)


def parse_status(code) -> dict:
    """D3 assumed legend. Returns pumps_running (0 if no pump token), gate_open (True/False/NaN), sky.
    A bare 'O' or 'C' is a gate code (Quiapo writes only the gate)."""
    out = {"pumps_running": 0, "gate_open": np.nan, "sky": None, "status_parsed": False}
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return out
    s = re.sub(r"\s+", "", str(code).upper())
    if not s:
        return out
    if s in {"O", "C"}:
        out.update(gate_open=(s == "O"), status_parsed=True)
        return out
    m = STATUS_PUMPS.search(s)
    if m:
        pre = s[: m.start()]
        out.update(pumps_running=int(m.group("pumps")), sky=m.group("sky") or None, status_parsed=True)
        if pre in {"O", "C"}:
            out["gate_open"] = pre == "O"
        return out
    m = STATUS_PLAIN.match(s)
    if m:
        g = m.group("gate")
        out.update(gate_open=(np.nan if g is None else g == "O"), sky=m.group("sky"), status_parsed=True)
    return out


# ----------------------------------------------------------------------------- block parsing

def _day_columns(df: pd.DataFrame, start: int, block: Block, search: int = 4) -> tuple[int, dict[int, int]]:
    """Find the day-number row just below the header: returns (row, {day: value_col})."""
    for r in range(start, min(start + search, len(df))):
        cols = {}
        for c, v in enumerate(df.iloc[r].tolist()):
            if isinstance(v, (int, float, np.integer, np.floating)) and not np.isnan(v) and float(v).is_integer():
                d = int(v)
                if 1 <= d <= 31 and d not in cols:
                    cols[d] = c
        if len(cols) >= 3:
            days = sorted(cols)
            if block.start_day is None or days[0] == block.start_day or days[0] <= 16:
                return r, cols
    raise ValueError(f"no day-number row below row {start} in sheet {block.sheet!r}")


def parse_block(df: pd.DataFrame, block: Block, next_header: int, units: str,
                station: str, source_file: str, year_fallback: int | None) -> pd.DataFrame:
    year = block.year or year_fallback
    start = block.header_row + 1 if block.header_row >= 0 else 0
    day_row, day_cols = _day_columns(df, start, block)
    recs = []
    r = day_row + 1
    stop = min(next_header, len(df))
    while r < stop:
        hour = parse_hour(df.iat[r, 0])
        if hour is None:
            r += 1
            continue
        nxt = r + 1
        has_out = nxt < stop and parse_hour(df.iat[nxt, 0]) is None
        for day, c in day_cols.items():
            ins, ins_tok = to_number(df.iat[r, c])
            st_raw = df.iat[r, c + 1] if c + 1 < df.shape[1] else None
            out_v, out_tok = (to_number(df.iat[nxt, c]) if has_out else (np.nan, None))
            recs.append((day, hour, ins, ins_tok, out_v, out_tok,
                         None if (st_raw is None or (isinstance(st_raw, float) and np.isnan(st_raw))) else str(st_raw).strip()))
        r = nxt + 1 if has_out else nxt
    out = pd.DataFrame(recs, columns=["day", "hour", "inside_raw", "inside_token", "outside_raw",
                                      "outside_token", "raw_status"])
    scale = 0.01 if units == "cm" else 1.0
    out["inside_m"] = out["inside_raw"] * scale
    out["outside_m"] = out["outside_raw"] * scale
    out["year"], out["month"] = year, block.month
    valid = [_valid_date(year, block.month, d) for d in out["day"]]
    out["flag_impossible_date"] = ~np.array(valid, dtype=bool)
    ts = [pd.Timestamp(year, block.month, d, h) if ok else pd.NaT
          for d, h, ok in zip(out["day"], out["hour"], valid, strict=True)]
    out["time_pht"] = ts
    st = out["raw_status"].map(parse_status).apply(pd.Series)
    out = pd.concat([out, st], axis=1)
    out["station"] = station
    out["source_file"] = source_file
    out["sheet"] = block.sheet
    out["block_header"] = block.header_text
    out["block_row"] = block.header_row
    return out


def _valid_date(y, m, d) -> bool:
    try:
        dt.date(int(y), int(m), int(d))
        return True
    except (TypeError, ValueError):
        return False


def blocks_with_bounds(df: pd.DataFrame, sheet: str) -> list[tuple[Block, int]]:
    bl = find_blocks(df, sheet)
    bounds = [b.header_row for b in bl[1:]] + [len(df)]
    return list(zip(bl, bounds, strict=True))


# ----------------------------------------------------------------------------- whole files

LIVE_BLOCK_STATUS = ("use", "pending_G20")


def parse_from_blocks(raw_root: Path, blocks: pd.DataFrame, statuses=LIVE_BLOCK_STATUS,
                      value_range=(9.0, 14.0)) -> dict[str, pd.DataFrame]:
    """Parse every block of data/manifest/pumping_blocks.csv whose status is in `statuses`.

    Blocks whose readings are all blank (empty templates) are skipped and logged. Values outside
    `value_range` (m, after the cm -> m conversion) are set to NaN and logged (G2)."""
    books: dict[str, dict] = {}
    parts, log = [], []
    sel = blocks[blocks["status"].isin(statuses)]
    for _, b in sel.iterrows():
        f = b["file"]
        if f not in books:
            books[f] = read_book(raw_root / f)
        df = books[f][b["sheet"]]
        station = "aviles" if Path(f).name.upper().startswith("AVILES") else "quiapo"
        units = "cm" if station == "aviles" else "m"
        blk = Block(b["sheet"], int(b["header_row"]), b["header_text"], int(b["year"]), int(b["month"]),
                    b["start_day"], b["end_day"], b["date_source"])
        try:
            out = parse_block(df, blk, int(b["next_header_row"]), units, station, f, int(b["year"]))
        except ValueError as e:
            log.append({"file": f, "sheet": b["sheet"], "header": b["header_text"], "event": f"unparsable: {e}"})
            continue
        n_read = int(out[["inside_raw", "outside_raw"]].notna().any(axis=1).sum())
        if n_read == 0:
            log.append({"file": f, "sheet": b["sheet"], "header": b["header_text"], "event": "all readings blank: skipped"})
            continue
        out["block_status"] = b["status"]
        out["block_year"] = int(b["year"])
        parts.append(out)
    long = pd.concat(parts, ignore_index=True)
    rng = []
    for col in ("inside_m", "outside_m"):
        bad = long[col].notna() & ~long[col].between(*value_range)
        if bad.any():
            r = long.loc[bad, ["station", "time_pht", "source_file", "sheet", col]].rename(columns={col: "value_m"})
            r["column"] = col
            rng.append(r)
        long[f"{col}_range_flag"] = bad
        long.loc[bad, col] = np.nan
    dup = long.dropna(subset=["time_pht"]).duplicated(["station", "time_pht", "block_status"], keep=False)
    long["flag_duplicate_time"] = False
    long.loc[dup[dup].index, "flag_duplicate_time"] = True
    return {"long": long, "log": pd.DataFrame(log),
            "out_of_range": pd.concat(rng, ignore_index=True) if rng else pd.DataFrame()}
