"""Harmonic tide model (D7) and the G9 residual-spectrum check.

`HarmonicTide` is moved from scripts/smoke/common.py unchanged in behaviour (least-squares cos/sin pairs plus
a mean, fit on hours above floor + 0.05 m), with standard errors, amplitudes and phases added for reporting,
and JSON save/load so `predict_tide(station, times)` works for any hour.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pasig.config import load, path


class HarmonicTide:
    def __init__(self, constituents: dict[str, float], t0="2018-01-01"):
        self.periods = dict(constituents)
        self.t0 = pd.Timestamp(t0)
        self.coef_ = None
        self.cov_ = None

    def _hours(self, times) -> np.ndarray:
        return ((pd.DatetimeIndex(times) - self.t0) / pd.Timedelta(hours=1)).to_numpy(dtype=float)

    def _design(self, times) -> np.ndarray:
        th = self._hours(times)
        cols = [np.ones_like(th)]
        for p in self.periods.values():
            w = 2 * np.pi / p
            cols += [np.cos(w * th), np.sin(w * th)]
        return np.column_stack(cols)

    def fit(self, series: pd.Series, floor: float | None = None, margin: float = 0.05):
        s = series.dropna()
        if floor is not None:   # drop hours sitting on the low-water floor (D7)
            s = s[s > floor + margin]
        X = self._design(s.index)
        y = s.to_numpy()
        self.coef_, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ self.coef_
        dof = max(len(y) - X.shape[1], 1)
        sigma2 = float(resid @ resid / dof)
        self.cov_ = sigma2 * np.linalg.pinv(X.T @ X)
        self.n_fit_, self.rmse_ = len(y), float(np.sqrt(np.mean(resid**2)))
        self.fit_index_ = s.index
        self.resid_ = pd.Series(resid, index=s.index)
        return self

    def predict(self, times) -> np.ndarray:
        return self._design(times) @ self.coef_

    def table(self) -> pd.DataFrame:
        """Amplitude (m), phase (deg, relative to t0) and delta-method standard errors per constituent."""
        rows = [{"constituent": "Z0 (mean)", "period_h": np.nan, "amplitude_m": self.coef_[0],
                 "amplitude_se_m": np.sqrt(self.cov_[0, 0]), "phase_deg": np.nan, "phase_se_deg": np.nan}]
        for i, (k, p) in enumerate(self.periods.items()):
            a, b = self.coef_[1 + 2 * i], self.coef_[2 + 2 * i]
            c = self.cov_[1 + 2 * i:3 + 2 * i, 1 + 2 * i:3 + 2 * i]
            amp = float(np.hypot(a, b))
            g = np.array([a, b]) / amp if amp > 0 else np.zeros(2)                 # d amp / d(a,b)
            gp = np.array([-b, a]) / amp**2 if amp > 0 else np.zeros(2)           # d phase / d(a,b)
            rows.append({"constituent": k, "period_h": p, "amplitude_m": amp,
                         "amplitude_se_m": float(np.sqrt(g @ c @ g)),
                         "phase_deg": float(np.degrees(np.arctan2(b, a)) % 360),
                         "phase_se_deg": float(np.degrees(np.sqrt(gp @ c @ gp)))})
        return pd.DataFrame(rows)

    def to_json(self) -> dict:
        return {"t0": str(self.t0), "periods_h": self.periods, "coef": self.coef_.tolist(),
                "cov": self.cov_.tolist(), "n_fit": self.n_fit_, "rmse_fit_m": self.rmse_,
                "constituents": self.table().replace({np.nan: None}).to_dict(orient="records")}

    @classmethod
    def from_json(cls, d: dict) -> HarmonicTide:
        m = cls(d["periods_h"], d["t0"])
        m.coef_ = np.asarray(d["coef"])
        m.cov_ = np.asarray(d["cov"])
        return m


def residual_spectrum_check(model: HarmonicTide, cfg: dict, pad: int = 4) -> dict:
    """G9: periodogram of the fit residuals near each target period vs the nearby median power.

    The residuals of the fitted hours are placed on the full hourly grid of the fit period (mean removed,
    unfitted/missing hours set to 0) and the periodogram is |FFT|^2 with `pad`x zero padding. Ratio = largest
    power within +-window_frac of the period / median power 3%-20% either side; passes if <= max_ratio."""
    r = model.resid_
    full = pd.date_range(r.index.min(), r.index.max(), freq="h")
    y = (r - r.mean()).reindex(full).fillna(0.0).to_numpy()
    nfft = int(2 ** np.ceil(np.log2(len(y) * pad)))
    pw = np.abs(np.fft.rfft(y, n=nfft)) ** 2 / len(r)
    freqs = np.fft.rfftfreq(nfft, d=1.0)                 # cycles per hour
    out = {}
    for T in cfg["target_periods_h"]:
        with np.errstate(divide="ignore"):
            per = 1.0 / freqs
        rel = np.abs(per - T) / T
        win = rel <= cfg["window_frac"]
        near = (rel > cfg["window_frac"]) & (rel <= cfg["nearby_frac"])
        peak, med = float(pw[win].max()), float(np.median(pw[near]))
        out[f"{T:g}"] = {"peak_power": peak, "nearby_median_power": med,
                         "ratio": peak / med if med > 0 else float("inf"),
                         "peak_period_h": float(per[win][np.argmax(pw[win])]),
                         "passed": bool(med > 0 and peak / med <= cfg["max_ratio"])}
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


def tide_path(station: str) -> Path:
    return path("artifacts") / f"tide_{station}.json"


def load_tide(station: str) -> HarmonicTide:
    return HarmonicTide.from_json(json.loads(tide_path(station).read_text()))


def predict_tide(station: str, times) -> np.ndarray:
    return load_tide(station).predict(times)


def floor_of(series: pd.Series) -> float:
    q = load("tide")["floor_quantile"]
    return round(float(series.dropna().quantile(q)), 2)
