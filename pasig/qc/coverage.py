"""Coverage (Section 7.8) and input eligibility (G5, D2)."""
from __future__ import annotations

import pandas as pd


def coverage_table(hourly: pd.DataFrame, value_col: str, by: str = "season") -> pd.DataFrame:
    """Share of hours with an unflagged reading (before fill), per station and season/year.
    `hourly` holds every expected hour (missing readings as NaN)."""
    g = hourly.groupby(["station", by])
    return (g[value_col].apply(lambda x: x.notna().mean()).rename("coverage").reset_index()
            .pivot(index="station", columns=by, values="coverage"))


def eligibility(cov: pd.DataFrame, years: list[int], threshold: float = 0.80) -> pd.DataFrame:
    sub = cov.reindex(columns=years)
    out = pd.DataFrame({"min_season_coverage": sub.min(axis=1),
                        "seasons_below": sub.apply(lambda r: [int(y) for y in years if not (r[y] >= threshold)], axis=1)})
    out["eligible"] = out["seasons_below"].str.len().eq(0)
    return out
