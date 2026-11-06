import datetime as dt
from pathlib import Path

import openpyxl
import pytest

from pasig.config import root

DATA_READY = (root() / "data" / "processed" / "hourly_master.parquet").exists()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "data" in item.keywords and not DATA_READY:
            item.add_marker(pytest.mark.skip(reason="processed data not built (run `make phase1`)"))


def make_form21(path: Path, year: int, month_name: str, ndays: int, values=None, station="PANDACAN",
                highest=None, lowest=None):
    """Write a synthetic FORM 2-1 workbook: rows for days 1..ndays, 24 hourly columns 09:00 -> 08:00."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = station
    ws.append(["FORM 2-1  HOURLY  WATERLEVEL CHART"])
    ws.append([month_name.upper()])
    ws.append(["LOCATION :", station, "MONTH :", month_name.upper(), "YEAR :", year])
    ws.append([])
    ws.append([None] + [dt.time((9 + i) % 24, 0) for i in range(24)] + ["AVE.", "DATE", "HIGHEST", "LOWEST"])
    for d in range(1, ndays + 1):
        row = values(d) if values else [11.0 + d / 100 + i / 1000 for i in range(24)]
        nums = [v for v in row if isinstance(v, (int, float)) and v != 0]
        hi = highest(d) if highest else (max(nums) if nums else None)
        lo = lowest(d) if lowest else (min(nums) if nums else None)
        ws.append([d] + list(row) + [None, d, hi, lo])
    ws.append(["AVE.", 11.0])
    ws.append(["*PTA", 11.0, None, "AVE"])
    wb.save(path)
    return path
