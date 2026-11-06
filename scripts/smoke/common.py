"""Shared helpers for the smoke tests and compute benchmark.

Self-contained on purpose: these scripts must run before the `pasig` package
exists, so the group can measure run time on the chosen laptop on day one.
Everything here mirrors Chapter III: 24-h window, leads 1-6, wet season
June-November, train 2018-2022 / val 2023 / test 2024.
"""
from __future__ import annotations

import json
import os
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WINDOW = 24
LEADS = 6
WET_MONTHS = (6, 7, 8, 9, 10, 11)
TRAIN_YEARS = (2018, 2019, 2020, 2021, 2022)
VAL_YEARS = (2023,)
TEST_YEARS = (2024,)

# Tidal constituent periods in hours (Chapter III, Phase 3)
CONSTITUENTS = {
    "M2": 12.4206012, "S2": 12.0, "N2": 12.65834751, "K1": 23.93447213,
    "O1": 25.81933871, "M4": 6.210300601, "MS4": 6.103339275,
}


# --------------------------------------------------------------------------
# Timing and hardware
# --------------------------------------------------------------------------
@dataclass
class Timings:
    rows: list = field(default_factory=list)

    @contextmanager
    def stage(self, name: str, **meta):
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        self.rows.append({"stage": name, "seconds": round(dt, 3), **meta})
        print(f"  [{dt:8.2f} s] {name}", flush=True)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _cpu_model() -> str:
    """Readable CPU name on Windows, macOS and Linux."""
    system = platform.system()
    try:
        if system == "Darwin":
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if system == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:  # Linux and macOS without psutil
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _physical_cores() -> int | None:
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except ImportError:
        return None


def hardware_info() -> dict:
    info = {
        "machine_name": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "physical_cores": _physical_cores(),
        "ram_gb": _ram_gb(),
    }
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        info["gpus"] = [g.name for g in gpus]
        if gpus:
            try:
                det = tf.config.experimental.get_device_details(gpus[0])
                info["gpu_name"] = det.get("device_name", "unknown")
            except Exception:  # noqa: BLE001
                info["gpu_name"] = "unknown"
        build = tf.sysconfig.get_build_info()
        info["cuda"] = build.get("cuda_version")
        info["cudnn"] = build.get("cudnn_version")
    except ImportError:
        info["tensorflow"] = None
    return info


def configure_device(device: str) -> str:
    """Pin TensorFlow to CPU or GPU before any model is built."""
    import tensorflow as tf
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        return "cpu"
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("Requested --device gpu but TensorFlow sees no GPU. TensorFlow uses only NVIDIA "
                         "GPUs, and on Windows only inside WSL2 (pip install 'tensorflow[and-cuda]'). "
                         "Check `nvidia-smi` inside WSL, or run with --device cpu.")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    return "gpu"


def set_seed(seed: int, deterministic: bool = True) -> None:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


# --------------------------------------------------------------------------
# Synthetic data with the same shape and quirks as the MMDA/PAGASA records
# --------------------------------------------------------------------------
def make_synthetic(seed: int = 0, years=range(2018, 2025)) -> dict[str, pd.DataFrame]:
    """Hourly wet-season series that mimic the real records.

    Includes: mixed tide, rain-driven surges, an upstream gauge that leads
    Pandacan by ~3 h, a low-water floor at 10.63 m, short and long gaps,
    6-hourly rainfall stamped in UTC at the END of each period, daily RH /
    pressure / temperature, and an Aviles pump count driven by river level
    and rain.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31 23:00", freq="h")
    hours = (idx - idx[0]) / pd.Timedelta(hours=1)
    hours = hours.to_numpy(dtype=float)

    amp = {"M2": 0.33, "S2": 0.11, "N2": 0.07, "K1": 0.29, "O1": 0.26, "M4": 0.03, "MS4": 0.02}
    tide = np.zeros(len(idx))
    for k, a in amp.items():
        tide += a * np.cos(2 * np.pi * hours / CONSTITUENTS[k] + rng.uniform(0, 2 * np.pi))

    # Rain: storms more frequent in wet months, hourly mm
    month = idx.month.to_numpy()
    p_storm = np.where(np.isin(month, WET_MONTHS), 0.012, 0.002)
    rain = np.zeros(len(idx))
    starts = np.flatnonzero(rng.random(len(idx)) < p_storm)
    for s in starts:
        dur = rng.integers(2, 30)
        rain[s:s + dur] += rng.gamma(1.2, 4.0, size=min(dur, len(idx) - s))

    # River response: rain convolved with a delayed gamma kernel
    k = np.arange(72)
    kernel = (k ** 2.5) * np.exp(-k / 4.0)
    kernel /= kernel.sum()
    surge = 0.012 * np.convolve(rain, kernel)[: len(idx)]
    upstream = 12.2 + 1.8 * surge + 0.05 * rng.standard_normal(len(idx))
    surge_lagged = np.r_[np.zeros(3), surge[:-3]]  # upstream leads by 3 h

    seasonal = 0.15 * np.sin(2 * np.pi * (idx.dayofyear.to_numpy() - 150) / 365.25)
    pandacan = 11.0 + tide + seasonal + 0.9 * surge_lagged + 0.03 * rng.standard_normal(len(idx))
    pandacan = np.maximum(pandacan, 10.63)  # low-water floor, as in the real gauge
    pandacan = np.round(pandacan, 2)

    # Gaps: 2% random single hours, plus some multi-day outages
    def punch(x, frac_single=0.02, n_long=20, rng=rng):
        x = x.copy()
        x[rng.random(len(x)) < frac_single] = np.nan
        for s in rng.integers(0, len(x) - 200, size=n_long):
            x[s:s + rng.integers(4, 150)] = np.nan
        return x

    hourly = pd.DataFrame({
        "pandacan": punch(pandacan),
        "sto_nino": punch(np.round(upstream, 2), frac_single=0.05, n_long=60),
        "rain_true": rain,
    }, index=idx)
    hourly.index.name = "time_pht"

    # PAGASA 6-hourly in UTC, stamped at period END (00, 06, 12, 18 UTC)
    utc = idx - pd.Timedelta(hours=8)
    r = pd.Series(rain, index=utc)
    six = r.resample("6h", label="right", closed="right").sum().round(1)
    six = six.to_frame("rain_6h_mm")
    six.index.name = "time_utc_end"

    daily = pd.DataFrame({
        "rain_mm": r.resample("D").sum().round(1),
        "rh": np.clip(75 + 0.8 * r.resample("D").sum() + rng.normal(0, 4, r.resample("D").sum().size), 50, 100),
        "pressure": 1008 - 0.05 * r.resample("D").sum() + rng.normal(0, 1.5, r.resample("D").sum().size),
        "tmean": 28 - 0.03 * r.resample("D").sum() + rng.normal(0, 0.7, r.resample("D").sum().size),
    })
    daily.index.name = "date_utc"

    rain24 = pd.Series(rain, index=idx).rolling(24, min_periods=1).sum()
    logit = -12 + 0.9 * (pandacan - 11.0) * 8 + 0.04 * rain24.to_numpy()
    p = 1 / (1 + np.exp(-logit))
    pumps = rng.binomial(3, np.clip(p, 0, 1))
    aviles = pd.DataFrame({"pumps_running": pumps,
                           "outside_m": np.round(pandacan + 0.08 + 0.02 * rng.standard_normal(len(idx)), 2)},
                          index=idx)
    return {"hourly": hourly, "six_hourly_utc": six, "daily_utc": daily, "aviles": aviles}


# --------------------------------------------------------------------------
# Phase 1-style alignment: rainfall availability, gap fill, wet-season filter
# --------------------------------------------------------------------------
RAIN_LAG_H = 1  # D5: a 6-hourly total is usable one hour after its period ends


def align_inputs(data: dict[str, pd.DataFrame], rain_lag_h: int = RAIN_LAG_H) -> pd.DataFrame:
    h = data["hourly"][["pandacan", "sto_nino"]].copy()

    # 6-hourly totals: UTC period end -> PHT (+8 h), usable rain_lag_h later (D5)
    six = data["six_hourly_utc"].copy()
    six.index = six.index + pd.Timedelta(hours=8 + rain_lag_h)
    six = six.reindex(h.index)
    h["rain_6h_latest"] = six["rain_6h_mm"].ffill()
    report_hour = pd.Series(np.where(six["rain_6h_mm"].notna(), np.arange(len(h)), np.nan), index=h.index).ffill()
    h["hours_since_report"] = np.arange(len(h)) - report_hour.to_numpy()
    reports = six["rain_6h_mm"].fillna(0.0)
    h["rain_24h"] = reports.rolling(24, min_periods=1).sum()
    h["rain_72h"] = reports.rolling(72, min_periods=1).sum()

    # Daily values cover 08:00 PHT D -> 08:00 PHT D+1; usable from 09:00 PHT D+1
    d = data["daily_utc"][["rh", "pressure", "tmean"]].copy()
    d.index = d.index + pd.Timedelta(days=1, hours=9)
    h = h.join(d.reindex(h.index).ffill())

    # Gap fill <= 3 h with a flag (never used as a target)
    for col in ("pandacan", "sto_nino"):
        miss = h[col].isna()
        run_id = (~miss).cumsum()
        run_len = miss.groupby(run_id).transform("sum")
        fillable = miss & (run_len <= 3)
        h[f"{col}_filled"] = fillable
        h[col] = h[col].where(~fillable, h[col].ffill())

    h = h[h.index.month.isin(WET_MONTHS)]
    h["season"] = h.index.year
    return h


# --------------------------------------------------------------------------
# Tide model: least-squares harmonic fit on all months of the training years,
# hours above the low-water floor only (D7)
# --------------------------------------------------------------------------
class HarmonicTide:
    def __init__(self, constituents=CONSTITUENTS, t0="2018-01-01"):
        self.periods = dict(constituents)
        self.t0 = pd.Timestamp(t0)
        self.coef_ = None

    def _design(self, times: pd.DatetimeIndex) -> np.ndarray:
        th = ((times - self.t0) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
        cols = [np.ones_like(th)]
        for p in self.periods.values():
            w = 2 * np.pi / p
            cols += [np.cos(w * th), np.sin(w * th)]
        return np.column_stack(cols)

    def fit(self, series: pd.Series, floor: float | None = None):
        s = series.dropna()
        if floor is not None:  # drop hours sitting on the low-water floor
            s = s[s > floor + 0.05]
        X = self._design(s.index)
        self.coef_, *_ = np.linalg.lstsq(X, s.to_numpy(), rcond=None)
        return self

    def predict(self, times: pd.DatetimeIndex) -> np.ndarray:
        return self._design(times) @ self.coef_


# --------------------------------------------------------------------------
# Windows: (X_seq, X_tab, y, valid, t_index)
# --------------------------------------------------------------------------
SEQ_COLS_BASE = ["pandacan", "d_pandacan", "rain_6h_latest"]
TAB_WEATHER = ["rain_24h", "rain_72h", "hours_since_report", "rh", "pressure", "tmean"]


def build_windows(df: pd.DataFrame, tide: HarmonicTide, seq_cols: list[str],
                  years: tuple[int, ...], max_cases: int | None = None,
                  rng: np.random.Generator | None = None):
    """Build windows inside each season so no window crosses the dry-season gap."""
    Xs, Xt, Y, V, T = [], [], [], [], []
    for yr in years:
        g = df[df["season"] == yr].copy()
        if g.empty:
            continue
        g["d_pandacan"] = g["pandacan"].diff()
        seq = g[seq_cols].to_numpy(dtype=np.float32)
        lvl = g["pandacan"].to_numpy(dtype=np.float32)
        filled = g["pandacan_filled"].to_numpy()
        times = g.index
        n = len(g)
        # tabular: calendar at t..t+6, tide at t+1..t+6 and change from t, weather at t
        hrs = np.arange(n)
        tide_all = tide.predict(times.append(pd.date_range(times[-1] + pd.Timedelta(hours=1),
                                                           periods=LEADS, freq="h")))
        cal = []
        for h in range(0, LEADS + 1):
            tt = times + pd.Timedelta(hours=h)
            cal += [np.sin(2 * np.pi * tt.hour / 24), np.cos(2 * np.pi * tt.hour / 24),
                    np.sin(2 * np.pi * tt.dayofyear / 365.25), np.cos(2 * np.pi * tt.dayofyear / 365.25)]
        cal = np.column_stack(cal).astype(np.float32)
        tide_fut = np.column_stack([tide_all[h:h + n] for h in range(1, LEADS + 1)])
        tide_now = tide_all[:n]
        tab_all = np.column_stack([cal, tide_fut, tide_fut - tide_now[:, None],
                                   g[TAB_WEATHER].to_numpy()]).astype(np.float32)
        for t in range(WINDOW - 1, n - LEADS):
            w = seq[t - WINDOW + 1: t + 1]
            if np.isnan(w).any() or np.isnan(tab_all[t]).any():
                continue
            fut = lvl[t + 1: t + 1 + LEADS]
            ok = ~np.isnan(fut) & ~filled[t + 1: t + 1 + LEADS]
            if not ok.any():
                continue
            Xs.append(w)
            Xt.append(tab_all[t])
            Y.append(np.nan_to_num(fut, nan=0.0))
            V.append(ok)
            T.append(times[t])
    Xs, Xt, Y, V = map(np.asarray, (Xs, Xt, Y, V))
    T = pd.DatetimeIndex(T)
    if max_cases is not None and len(Y) > max_cases:
        rng = rng or np.random.default_rng(0)
        keep = np.sort(rng.choice(len(Y), size=max_cases, replace=False))
        Xs, Xt, Y, V, T = Xs[keep], Xt[keep], Y[keep], V[keep], T[keep]
    return Xs, Xt, Y.astype(np.float32), V, T


class MinMax:
    """Min-max scaler fitted on training cases only."""

    def fit(self, X):
        axes = tuple(range(X.ndim - 1))
        self.lo = np.nanmin(X, axis=axes)
        self.hi = np.nanmax(X, axis=axes)
        self.span = np.where(self.hi - self.lo > 0, self.hi - self.lo, 1.0)
        return self

    def transform(self, X):
        return ((X - self.lo) / self.span).astype(np.float32)


def level_scaler(seq_scaler: "MinMax", channel: int = 0) -> "MinMax":
    """Target scaler = the scaler of the Pandacan sequence channel (D9 add-back needs both on one scale)."""
    s = MinMax()
    s.lo = seq_scaler.lo[channel:channel + 1]
    s.hi = seq_scaler.hi[channel:channel + 1]
    s.span = seq_scaler.span[channel:channel + 1]
    return s


# --------------------------------------------------------------------------
# Models (Figure III-4)
# --------------------------------------------------------------------------
@dataclass
class HybridConfig:
    lstm_layers: int = 1
    lstm_units: int = 64
    gru_units: int = 32
    dense_layers: int = 1
    dense_units: int = 16
    head_units: int = 32
    dropout: float = 0.2
    lr: float = 1e-3
    batch: int = 64

    def tag(self) -> str:
        return (f"L{self.lstm_layers}x{self.lstm_units}-G{self.gru_units}-"
                f"D{self.dense_layers}x{self.dense_units}-do{self.dropout}-lr{self.lr}-b{self.batch}")


def _add_back_level(seq_in, x, layers):
    """D9: the dense head predicts the change from the current level; add L_t back.

    Requires the target scaled with the same min-max as sequence channel 0 (Pandacan level),
    see level_scaler(). The output is still the level at t+1 ... t+6.
    """
    delta = layers.Dense(LEADS, activation="linear", name="change")(x)
    last = layers.Lambda(lambda s: s[:, -1, 0:1], name="level_t")(seq_in)
    return layers.Add(name="levels")([delta, layers.Concatenate()([last] * LEADS)])


def build_hybrid(cfg: HybridConfig, seq_shape, n_tab: int, use_tabular: bool = True,
                 second_lstm_instead_of_gru: bool = False, add_back_level: bool = True):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    seq_in = keras.Input(shape=seq_shape, name="seq")
    x = seq_in
    for _ in range(cfg.lstm_layers):
        # dropout BETWEEN layers keeps the fast cuDNN kernel on GPU
        x = layers.LSTM(cfg.lstm_units, return_sequences=True)(x)
        x = layers.Dropout(cfg.dropout)(x)
    if second_lstm_instead_of_gru:
        x = layers.LSTM(cfg.gru_units)(x)
    else:
        x = layers.GRU(cfg.gru_units)(x)
    x = layers.Dropout(cfg.dropout)(x)
    inputs = [seq_in]
    if use_tabular:
        tab_in = keras.Input(shape=(n_tab,), name="tab")
        z = tab_in
        for _ in range(cfg.dense_layers):
            z = layers.Dense(cfg.dense_units, activation="relu")(z)
        x = layers.Concatenate()([x, z])
        inputs.append(tab_in)
    x = layers.Dense(cfg.head_units, activation="relu")(x)
    if add_back_level:
        out = _add_back_level(seq_in, x, layers)
    else:
        out = layers.Dense(LEADS, activation="linear", name="levels")(x)
    model = keras.Model(inputs, out)
    model.compile(optimizer=keras.optimizers.Adam(cfg.lr), loss=masked_mse)
    return model


def build_lstm(units: int, dropout: float, lr: float, seq_shape, add_back_level: bool = True):
    """Standalone LSTM: tabular features are repeated at every time step upstream."""
    from tensorflow import keras
    from tensorflow.keras import layers
    seq_in = keras.Input(shape=seq_shape)
    x = layers.LSTM(units)(seq_in)
    x = layers.Dropout(dropout)(x)
    out = _add_back_level(seq_in, x, layers) if add_back_level else layers.Dense(LEADS)(x)
    m = keras.Model(seq_in, out)
    m.compile(optimizer=keras.optimizers.Adam(lr), loss=masked_mse)
    return m


def masked_mse(y_true, y_pred):
    """MSE over valid targets only. Invalid targets are encoded as y_true = -1."""
    import tensorflow as tf
    mask = tf.cast(y_true > -0.5, y_pred.dtype)
    se = tf.square(y_true - y_pred) * mask
    return tf.reduce_sum(se) / tf.maximum(tf.reduce_sum(mask), 1.0)


def encode_targets(Y: np.ndarray, V: np.ndarray, y_scaler: MinMax) -> np.ndarray:
    """Scale targets to [0,1] and mark invalid ones as -1 for masked_mse."""
    Ys = (Y - y_scaler.lo) / y_scaler.span
    return np.where(V, Ys, -1.0).astype(np.float32)


def repeat_tab(Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    return np.concatenate([Xs, np.repeat(Xt[:, None, :], Xs.shape[1], axis=1)], axis=2)


# --------------------------------------------------------------------------
# Statistics used in Phases 3 and 5
# --------------------------------------------------------------------------
def pooled_granger(y_by_season: list[np.ndarray], x_by_season: list[np.ndarray], max_lag: int):
    """Pooled-within-season Granger F-test with BIC lag choice (Chapter III, F.5)."""
    from scipy import stats

    def design(p, with_x):
        rows_y, rows_X = [], []
        for si, (y, x) in enumerate(zip(y_by_season, x_by_season)):
            n = len(y)
            if n <= p + 1:
                continue
            cols = [y[p - i - 1: n - i - 1] for i in range(p)]
            if with_x:
                cols += [x[p - i - 1: n - i - 1] for i in range(p)]
            Xm = np.column_stack(cols)
            dummies = np.zeros((n - p, len(y_by_season)))
            dummies[:, si] = 1.0
            Xm = np.column_stack([dummies, Xm])
            yy = y[p:]
            ok = ~np.isnan(Xm).any(axis=1) & ~np.isnan(yy)
            rows_y.append(yy[ok])
            rows_X.append(Xm[ok])
        return np.concatenate(rows_y), np.vstack(rows_X)

    best = None
    for p in range(1, max_lag + 1):
        yy, XX = design(p, True)
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        rss = float(np.sum((yy - XX @ beta) ** 2))
        n, k = XX.shape
        bic = n * np.log(rss / n) + k * np.log(n)
        if best is None or bic < best[1]:
            best = (p, bic)
    p = best[0]
    yu, Xu = design(p, True)
    yr, Xr = design(p, False)
    rss_u = float(np.sum((yu - Xu @ np.linalg.lstsq(Xu, yu, rcond=None)[0]) ** 2))
    rss_r = float(np.sum((yr - Xr @ np.linalg.lstsq(Xr, yr, rcond=None)[0]) ** 2))
    df1, df2 = p, len(yu) - Xu.shape[1]
    F = ((rss_r - rss_u) / df1) / (rss_u / df2)
    return {"lag": p, "F": F, "p_value": float(stats.f.sf(F, df1, df2)), "n": len(yu)}


def adf_pvalue(x: np.ndarray) -> float:
    import warnings
    from statsmodels.tsa.stattools import adfuller
    x = x[~np.isnan(x)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return float(adfuller(x, autolag="BIC")[1])


def diebold_mariano(e_a: np.ndarray, e_b: np.ndarray, h: int):
    """DM test on squared errors with Newey-West variance and the HLN correction."""
    from scipy import stats
    d = e_a ** 2 - e_b ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    lags = max(h - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    gamma0 = np.dot(dc, dc) / n
    v = gamma0
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * np.dot(dc[k:], dc[:-k]) / n
    dm = dbar / np.sqrt(v / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return {"dm": float(dm), "dm_hln": float(hln), "p_value": float(p), "n": n}


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
