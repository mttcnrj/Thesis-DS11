"""Min-max scaling fitted on training cases only; target scaler = the Pandacan sequence channel's scaler (D9).

`MinMax` and `level_scaler` are moved from scripts/smoke/common.py; JSON save/load added.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MinMax:
    def fit(self, X):
        axes = tuple(range(X.ndim - 1))
        self.lo = np.nanmin(X, axis=axes)
        self.hi = np.nanmax(X, axis=axes)
        self.span = np.where(self.hi - self.lo > 0, self.hi - self.lo, 1.0)
        return self

    def transform(self, X):
        return ((X - self.lo) / self.span).astype(np.float32)

    def inverse(self, Xs):
        return Xs * self.span + self.lo

    def to_dict(self, names=None) -> dict:
        d = {"lo": self.lo.tolist(), "hi": self.hi.tolist(), "span": self.span.tolist()}
        if names is not None:
            d["names"] = list(names)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> MinMax:
        m = cls()
        m.lo, m.hi, m.span = (np.asarray(d[k], dtype=float) for k in ("lo", "hi", "span"))
        return m


def level_scaler(seq_scaler: MinMax, channel: int = 0) -> MinMax:
    """Target scaler = the scaler of the Pandacan sequence channel (D9 add-back needs both on one scale)."""
    s = MinMax()
    s.lo = seq_scaler.lo[channel:channel + 1]
    s.hi = seq_scaler.hi[channel:channel + 1]
    s.span = seq_scaler.span[channel:channel + 1]
    return s


def encode_targets(Y: np.ndarray, V: np.ndarray, y_scaler: MinMax) -> np.ndarray:
    """Scale targets and mark invalid ones as -1 for the masked MSE."""
    Ys = (np.nan_to_num(Y) - y_scaler.lo) / y_scaler.span
    return np.where(V, Ys, -1.0).astype(np.float32)


def save_scalers(path: Path, **scalers: tuple[MinMax, list]) -> None:
    path.write_text(json.dumps({k: s.to_dict(n) for k, (s, n) in scalers.items()}, indent=1))


def load_scalers(path: Path) -> dict[str, MinMax]:
    return {k: MinMax.from_dict(v) for k, v in json.loads(path.read_text()).items()}
