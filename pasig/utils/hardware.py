"""Hardware and environment records (Section 2.3: every run writes env.txt). Adapted from scripts/smoke/common.py."""
from __future__ import annotations

import os
import platform


def _cpu_model() -> str:
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


def hardware_info() -> dict:
    info = {"machine_name": platform.node(), "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(), "cpu": _cpu_model(), "logical_cpus": os.cpu_count()}
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        info["gpus"] = [g.name for g in gpus]
        if gpus:
            try:
                info["gpu_name"] = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "unknown")
            except Exception:  # noqa: BLE001
                info["gpu_name"] = "unknown"
        b = tf.sysconfig.get_build_info()
        info["cuda"], info["cudnn"] = b.get("cuda_version"), b.get("cudnn_version")
    except ImportError:
        info["tensorflow"] = None
    return info


def env_text(extra: dict | None = None) -> str:
    d = hardware_info() | (extra or {})
    return "\n".join(f"{k}: {v}" for k, v in d.items()) + "\n"


def configure_device(device: str) -> str:
    """Pin TensorFlow to CPU or GPU before any model is built."""
    import tensorflow as tf
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        return "cpu"
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("device=gpu requested but TensorFlow sees no GPU (NVIDIA + WSL2 + tensorflow[and-cuda])")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    return "gpu"


def visible_nvidia_gpu() -> str | None:
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return None
        return tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
    except Exception:  # noqa: BLE001
        return None
