"""G13/G14: the GPU probe must run without CUDA_VISIBLE_DEVICES=-1, or a GPU is never detected (F34)."""
import pytest

from pasig import phase5
from pasig.train import compute

pytestmark = pytest.mark.fast


def test_cpu_env_hides_gpu_and_gpu_env_does_not(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert compute.sub_env("cpu")["CUDA_VISIBLE_DEVICES"] == "-1"
    assert "CUDA_VISIBLE_DEVICES" not in compute.sub_env("gpu")


def test_probe_is_not_run_in_cpu_env(monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    def fake_call(args, device="cpu"):
        if args == ["--probe"]:
            seen["device"] = device
            raise Stop
        raise AssertionError(args)

    monkeypatch.setattr(compute, "call_module", fake_call)
    monkeypatch.setattr(phase5, "machine", lambda: {"name": "test"})
    with pytest.raises(Stop):
        phase5.stage_compute()
    assert seen["device"] == "gpu"
