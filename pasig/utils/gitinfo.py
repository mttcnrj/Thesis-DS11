from __future__ import annotations

import subprocess

from pasig.config import root


def git_sha(short: bool = True) -> str:
    try:
        args = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
        return subprocess.check_output(args, cwd=root(), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"
