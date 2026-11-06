"""G18: the test season opens only with every frozen file unchanged (Section 0.3)."""
import json

import pytest

from pasig import config, phase6
from pasig.phase5 import sha256

pytestmark = pytest.mark.fast


def _setup(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    f = tmp_path / "artifacts" / "scalers.json"
    f.parent.mkdir()
    f.write_text("{}")
    man = {"files": {"artifacts/scalers.json": sha256(f)}}
    (tmp_path / "configs" / "frozen_manifest.json").write_text(json.dumps(man))
    monkeypatch.setattr(phase6, "root", lambda: tmp_path)
    monkeypatch.setattr(config, "_config_file", lambda name: tmp_path / "configs" / f"{name}.yaml")
    return f


def test_unchanged_files_pass(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    r = phase6.manifest_check()
    assert r["ok"] and r["n_files"] == 1


def test_changed_or_missing_file_blocks(tmp_path, monkeypatch):
    f = _setup(tmp_path, monkeypatch)
    f.write_text('{"edited": true}')
    r = phase6.manifest_check()
    assert not r["ok"] and r["changed"] == ["artifacts/scalers.json"]
    f.unlink()
    r = phase6.manifest_check()
    assert not r["ok"] and r["missing"] == ["artifacts/scalers.json"]


def test_missing_manifest_blocks(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    (tmp_path / "configs" / "frozen_manifest.json").unlink()
    assert not phase6.manifest_check()["ok"]
