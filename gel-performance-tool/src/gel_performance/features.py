"""Step 2 - Standardise every feature against an early-life reference, oriented so higher = worse.

For each chromatography type separately:

1. Reference window: all cycles with resin_age <= first_n_cycles, pooled over every gel batch
   of that type. This is "what fresh resin looks like" for that column set-up.
2. Reference statistics per feature:
     method "mean_std": centre = mean,   spread = standard deviation
     method "robust":   centre = median, spread = 1.4826 * MAD  (MAD = median absolute deviation;
                        the 1.4826 factor makes it comparable to a standard deviation for normal data)
   Missing values are ignored when computing these statistics.
3. Oriented z-score:
     z = sign * (value - centre) / spread
   where sign = +1 for "higher_is_worse" (pressures) and -1 for "lower_is_worse" (flows).
   So z = +2 always means "2 reference spreads worse than fresh resin", whatever the feature.
4. resin_age is a special case: inside the reference window it only runs 1..N, so its spread is
   tiny and a normal z-score would explode (about +20 by cycle 300). Instead:
     z_age = (resin_age - centre of ages in the window) / scale_cycles
   i.e. every `scale_cycles` cycles beyond the reference adds one unit of "worse".
5. Every z is capped to [-z_clip, +z_clip] so that a single sensor spike cannot dominate.

Why per chromatography type? Different column set-ups run at different pressures and flows
(in the synthetic data type_B runs at about 2x the pressure of type_A). If both were standardised
together, every type_B cycle would look "high pressure = worse" simply because of its set-up,
not because its resin is ageing. Standardising per type compares each cycle with fresh resin
of the same type.

Missing raw values stay missing: their z-score is NaN, never 0 or any other guess.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import FEATURES, Config, TypeConfig

log = logging.getLogger(__name__)

MAD_TO_STD = 1.4826


class ReferenceWindowError(ValueError):
    """Raised when a reference window is too small or has no spread."""


def z_col(feature: str) -> str:
    return f"z_{feature}"


def reference_stats(df: pd.DataFrame, tc: TypeConfig) -> pd.DataFrame:
    """Centre and spread of each feature in the early-life window of ONE chromatography type.

    Returns a table with one row per feature: centre, spread, n (number of reference values used).
    """
    ref = df[(df["chromatography_type"] == tc.name) & (df["resin_age"] <= tc.first_n_cycles)]
    rows = []
    for feature in FEATURES:
        values = ref[feature].dropna().to_numpy(dtype=float)
        if len(values) < 2:
            raise ReferenceWindowError(
                f"{tc.name}: only {len(values)} reference value(s) for {feature}; "
                f"need at least 2 cycles with resin_age <= {tc.first_n_cycles}"
            )
        centre, spread = _centre_spread(values, tc.method)
        if feature == "resin_age":
            spread = tc.features["resin_age"].scale_cycles
        if not np.isfinite(spread) or spread <= 0:
            raise ReferenceWindowError(f"{tc.name}: reference spread of {feature} is zero; cannot standardise")
        rows.append({"chromatography_type": tc.name, "feature": feature,
                     "centre": centre, "spread": spread, "n": len(values)})
    return pd.DataFrame(rows)


def _centre_spread(values: np.ndarray, method: str) -> tuple[float, float]:
    if method == "mean_std":
        return float(np.mean(values)), float(np.std(values, ddof=1))
    if method == "robust":
        median = float(np.median(values))
        return median, float(MAD_TO_STD * np.median(np.abs(values - median)))
    raise ValueError(f"Unknown reference method: {method}")


def standardise(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add oriented, clipped z-score columns (z_<feature>) for every cycle.

    Returns (cycles with z columns, reference statistics table for all types).
    """
    out = df.copy()
    for feature in FEATURES:
        out[z_col(feature)] = np.nan

    all_stats = []
    for type_name, tc in cfg.types.items():
        mask = out["chromatography_type"] == type_name
        if not mask.any():
            log.info("No cycles for %s in this input; skipped", type_name)
            continue
        stats = reference_stats(out, tc)
        all_stats.append(stats)
        for _, s in stats.iterrows():
            sign = tc.features[s["feature"]].sign
            z = sign * (out.loc[mask, s["feature"]] - s["centre"]) / s["spread"]
            out.loc[mask, z_col(s["feature"])] = z.clip(-cfg.z_clip, cfg.z_clip)
        log.info(
            "%s: reference = first %d cycles per batch (%s), %d reference cycles",
            type_name, tc.first_n_cycles, tc.method, int(stats["n"].max()),
        )

    n_clipped = int((out[[z_col(f) for f in FEATURES]].abs() >= cfg.z_clip).sum().sum())
    if n_clipped:
        log.info("%d z-score value(s) capped at +/-%.1f", n_clipped, cfg.z_clip)

    stats_table = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    return out, stats_table
