"""Model-ready arrays from the Phase 3 windows and the frozen input choices (G10 selected_features, G11/G12)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pasig.config import load, path, profile
from pasig.features.scaling import MinMax, encode_targets, load_scalers
from pasig.phase3 import load_windows


@dataclass
class Inputs:
    seq: list[str]
    tab: list[str]
    use_constituents: bool = False        # G11 (D8)
    add_back: bool = True                 # G12 (D9)
    tag: str = "main"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, selection: str = "model_inputs") -> Inputs:
        sf = load("selected_features")
        mi = load("model_inputs") or {}
        src = sf["model_inputs"] if selection == "model_inputs" else sf[selection]
        return cls(seq=list(src["seq"]), tab=list(src["tab"]),
                   use_constituents=bool(mi.get("use_constituents", False)), add_back=bool(mi.get("add_back", True)))


def _sub(s: MinMax, idx) -> MinMax:
    m = MinMax()
    m.lo, m.hi, m.span = s.lo[idx], s.hi[idx], s.span[idx]
    return m


def load_split(split: str, inp: Inputs, windows: dict | None = None, max_cases: int | None = None,
               seed: int = 0, scalers: dict | None = None) -> dict:
    """X = {'seq': (N,24,C), 'tab': (N,F)} scaled with the TRAINING scalers; Y encoded (-1 = invalid);
    y_raw (m) and valid kept for scoring. `windows` lets Notebook 06 pass the 2024 arrays it builds itself;
    `scalers` lets a walk-forward fold use the scalers fitted on its own training years (Notebook 07)."""
    w = windows if windows is not None else load_windows(split)
    sc = scalers if scalers is not None else load_scalers(path("artifacts") / "scalers.json")
    import json
    names = json.loads((path("artifacts") / "feature_names.json").read_text())
    if "seq_names" in w:
        names = {"seq": list(w["seq_names"]), "tab": list(w["tab_names"])}
    si = [names["seq"].index(n) for n in inp.seq]
    ti = [names["tab"].index(n) for n in inp.tab]
    if si[0] != names["seq"].index("pandacan"):
        raise ValueError("sequence channel 0 must be the Pandacan level (D9 add-back)")
    s_seq, s_tab = _sub(sc["seq"], si), _sub(sc["tab"], ti)
    Xs = s_seq.transform(w["X_seq"][:, :, si])
    Xt = s_tab.transform(w["X_tab"][:, ti])
    if inp.use_constituents:
        Xt = np.hstack([Xt, sc["tab_cons"].transform(w["X_tab_cons"])])
    Y = encode_targets(w["y"], w["valid"], sc["y"])
    out = {"X": {"seq": Xs, "tab": Xt}, "Y": Y, "y_raw": w["y"], "valid": w["valid"].astype(bool),
           "t_index": pd.DatetimeIndex(w["t_index"]), "level_t": w["X_seq"][:, -1, 0], "y_scaler": sc["y"]}
    if max_cases is not None and len(Y) > max_cases:
        keep = np.sort(np.random.default_rng(seed).choice(len(Y), size=max_cases, replace=False))
        out = _take(out, keep)
    return out


def _take(d: dict, idx) -> dict:
    o = dict(d)
    o["X"] = {k: v[idx] for k, v in d["X"].items()}
    for k in ("Y", "y_raw", "valid", "level_t"):
        o[k] = d[k][idx]
    o["t_index"] = d["t_index"][idx]
    return o


def repeat_tab(X: dict) -> np.ndarray:
    """Standalone LSTM input: tabular vector repeated at every time step, appended to the sequence channels."""
    s, t = X["seq"], X["tab"]
    return np.concatenate([s, np.repeat(t[:, None, :], s.shape[1], axis=1)], axis=2)


def rehearsal_cases() -> int | None:
    return load("model_hybrid")["rehearsal"]["train_cases"] if profile() == "rehearsal" else None
