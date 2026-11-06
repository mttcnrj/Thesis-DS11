"""require_gates raises on missing or failed gates."""
import pytest

from pasig import gates


@pytest.fixture
def gate_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "_gate_dir", lambda: tmp_path)
    monkeypatch.setattr(gates, "git_sha", lambda: "test")
    return tmp_path


@pytest.mark.fast
def test_require_gates(gate_dir):
    with pytest.raises(RuntimeError, match="G01 missing"):
        gates.require_gates("G1")
    gates.record_gate("G1", "raw_files", True, "rule", {"n": 1})
    gates.require_gates("G1")
    gates.record_gate("G2", "parsing", False, "rule", {})
    with pytest.raises(RuntimeError, match="G02 failed"):
        gates.require_gates("G1", "G2")
    rec = gates.read_gate("g02")
    assert rec["gate"] == "G2" and rec["passed"] is False and set(rec) >= {"rule", "evidence", "timestamp", "git_sha"}


@pytest.mark.fast
def test_rerecording_replaces_file(gate_dir):
    gates.record_gate("G13", "device", True, "r", {}, decision="cpu")
    gates.record_gate("G13", "device", True, "r", {}, decision="gpu")
    assert len(list(gate_dir.glob("G13_*.json"))) == 1 and gates.read_gate("G13")["decision"] == "gpu"
