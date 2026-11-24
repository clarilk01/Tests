"""Turn per-batch KPI records into model features.

Everything at batch ``t`` uses only batches ``<= t`` of the same resin lot, so
the same features can be computed for a column that is still in use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .simulator import KPIS

BASELINE_CYCLES = 5   # first batches of a lot define its "fresh" reference
SLOPE_WINDOW = 8      # batches used for the recent trend
HORIZONS = (1, 2)     # forecast the next two batches


def _rolling_slope(s: pd.Series, window: int) -> pd.Series:
    """Least-squares slope (per batch) over the trailing ``window`` points (vectorised)."""
    y = s.to_numpy(float)
    i = np.arange(len(y), dtype=float)
    r = lambda a: pd.Series(a).rolling(window, min_periods=1).sum().to_numpy()
    n = r(np.ones_like(y))
    sx, sy, sxx, sxy = r(i), r(y), r(i * i), r(i * y)
    den = n * sxx - sx**2
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where((n >= 3) & (den > 0), (n * sxy - sx * sy) / den, 0.0)
    return pd.Series(slope, index=s.index)


def feature_names(use_yield_history: bool = True) -> list[str]:
    names = ["cycle"]
    signals = KPIS + (["yield"] if use_yield_history else [])
    for k in signals:
        names += [k, f"{k}_mean3", f"{k}_slope", f"{k}_rel"]
    return names


def build_features(df: pd.DataFrame, use_yield_history: bool = True) -> pd.DataFrame:
    """Add feature columns and the future-yield targets ``yield_h1``/``yield_h2``."""
    df = df.sort_values(["resin_lot", "cycle"]).reset_index(drop=True)
    signals = KPIS + (["yield"] if use_yield_history else [])
    parts = []
    for _, g in df.groupby("resin_lot", sort=False):
        g = g.copy()
        for k in signals:
            base = g[k].iloc[:BASELINE_CYCLES].mean()
            g[f"{k}_mean3"] = g[k].rolling(3, min_periods=1).mean()
            g[f"{k}_slope"] = _rolling_slope(g[k], SLOPE_WINDOW)
            g[f"{k}_rel"] = g[f"{k}_mean3"] / base if base else np.nan
        for h in HORIZONS:
            g[f"yield_h{h}"] = g["yield"].shift(-h)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)
