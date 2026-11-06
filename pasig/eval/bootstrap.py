"""Moving-block bootstrap (24 h blocks) for differences between models, and the event bootstrap (Section 10.1)."""
from __future__ import annotations

import numpy as np


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, max(n - block + 1, 1), size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]


def block_ci(stat, n: int, reps: int, block: int, seed: int) -> tuple[float, float, int]:
    """Percentile 95% CI of stat(idx) over moving-block resamples of the time-ordered rows 0..n-1."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        v = stat(moving_block_indices(n, block, rng))
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)
