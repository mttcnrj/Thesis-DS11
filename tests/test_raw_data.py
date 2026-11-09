"""The raw data are committed (decision R1): every manifest file is in data/raw with its recorded SHA-256, so
G1's evidence holds in any clone and Phase 1 reproduces the committed results without Google Drive."""
import pandas as pd
import pytest

from pasig.config import path
from pasig.local import check_raw, prepare

pytestmark = pytest.mark.fast


def test_committed_raw_data_match_the_manifest():
    r = check_raw(verbose=False)
    assert r["files"] == len(pd.read_csv(path("manifest") / "raw_manifest.csv")) == 465
    assert not r["missing"] and not r["changed"]


def test_only_manifest_files_are_in_data_raw():
    man = set(pd.read_csv(path("manifest") / "raw_manifest.csv")["path"])
    on_disk = {p.relative_to(path("raw")).as_posix() for p in path("raw").rglob("*") if p.is_file()}
    assert on_disk == man


def test_prepare_phase1_only_checks_the_raw_data():
    assert set(prepare(1, build=False)) == {"raw"}
