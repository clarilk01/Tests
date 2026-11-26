"""Build the three standardised features the decision tree uses.

Per cycle:
    z_resin_age               - gel age
    z_rm_elution_flow_median  - rolling mean of elution flow
    z_rm_pt_80_percentile     - rolling mean of the elution pressure (PT 80th percentile)

Steps:
1. Rolling mean of each KPI over the last `rolling_window` cycles of the SAME gel batch (ordered by
   resin_age). A missing value is simply left out of the average; it is never filled in. If the
   whole window is missing, the rolling mean is missing too.
   Why a rolling mean? It smooths cycle-to-cycle noise, so one odd run cannot flip the decision.
2. Reference window (what fresh resin looks like): cycles with resin_age <= first_n_cycles.
   Statistics come from the RAW KPI values in that window:
     method "mean_std": centre = mean,   spread = standard deviation
     method "robust":   centre = median, spread = 1.4826 * MAD
   scope "pooled":    one centre per chromatography type (all batches of the type together)
   scope "per_batch": each batch's own centre; spread still pooled (a batch with fewer than
                      MIN_BATCH_REF_CYCLES early values falls back to the pooled centre, logged)
3. Oriented z-score of the rolling mean:
     z = sign * (rolling mean - centre) / spread      sign = +1 higher_is_worse, -1 lower_is_worse
   so a higher z always means "worse than fresh resin". The spread is the single-cycle spread,
   so z reads as "how many single-cycle spreads the rolling mean has moved".
4. Gel age: z = (resin_age - mean age in the reference window) / resin_age_scale_cycles.
   (A normal z-score of age would explode, because ages in the window only run 1..N.)
5. Every z is capped to [-z_clip, +z_clip].

Why per chromatography type? Different column set-ups run at different pressures and flows (type_B
runs at about twice the pressure of type_A). Standardising per type compares every cycle with fresh
resin of the SAME type, so one decision tree can serve both types.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import KPIS, Config, TypeConfig

log = logging.getLogger(__name__)

MAD_TO_STD = 1.4826
MIN_BATCH_REF_CYCLES = 5


class ReferenceWindowError(ValueError):
    """Raised when a reference window is too small or has no spread."""


def rm_col(kpi: str) -> str:
    return f"rm_{kpi}"


def z_col(name: str) -> str:
    return f"z_{name}"


def add_rolling_means(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling mean of each KPI within each gel batch, ordered by resin_age. NaN values are skipped."""
    out = df.sort_values(["gel_batch", "resin_age"], kind="mergesort").copy()
    for kpi in KPIS:
        out[rm_col(kpi)] = (
            out.groupby("gel_batch", sort=False)[kpi]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
    return out.loc[df.index]  # back to the caller's row order


def _centre_spread(values: np.ndarray, method: str) -> tuple[float, float]:
    if method == "mean_std":
        return float(np.mean(values)), float(np.std(values, ddof=1))
    if method == "robust":
        median = float(np.median(values))
        return median, float(MAD_TO_STD * np.median(np.abs(values - median)))
    raise ValueError(f"Unknown reference method: {method}")


def reference_stats(df: pd.DataFrame, tc: TypeConfig) -> pd.DataFrame:
    """Pooled centre and spread of gel age and each KPI in the early-life window of ONE type."""
    ref = df[(df["chromatography_type"] == tc.name) & (df["resin_age"] <= tc.first_n_cycles)]
    rows = []
    for name in ["resin_age", *KPIS]:
        values = ref[name].dropna().to_numpy(dtype=float)
        if len(values) < 2:
            raise ReferenceWindowError(
                f"{tc.name}: only {len(values)} reference value(s) for {name}; "
                f"need at least 2 cycles with resin_age <= {tc.first_n_cycles}"
            )
        centre, spread = _centre_spread(values, tc.method)
        if name == "resin_age":
            centre, spread = float(np.mean(values)), tc.resin_age_scale_cycles
        if not np.isfinite(spread) or spread <= 0:
            raise ReferenceWindowError(f"{tc.name}: reference spread of {name} is zero; cannot standardise")
        rows.append({"chromatography_type": tc.name, "gel_batch": "ALL", "feature": name,
                     "centre": centre, "spread": spread, "n": len(values), "source": "pooled"})
    return pd.DataFrame(rows)


def batch_reference_stats(df: pd.DataFrame, tc: TypeConfig, pooled: pd.DataFrame) -> pd.DataFrame:
    """Per-batch centres (scope per_batch). Spread from the pooled table; centre falls back to pooled
    when a batch has fewer than MIN_BATCH_REF_CYCLES early-life values for a KPI."""
    pooled = pooled.set_index("feature")
    rows = []
    for batch, g in df[df["chromatography_type"] == tc.name].groupby("gel_batch", sort=True):
        ref = g[g["resin_age"] <= tc.first_n_cycles]
        for name in ["resin_age", *KPIS]:
            values = ref[name].dropna().to_numpy(dtype=float)
            if name != "resin_age" and len(values) >= MIN_BATCH_REF_CYCLES:
                centre, source = _centre_spread(values, tc.method)[0], "batch"
            elif name == "resin_age":
                centre, source = pooled.loc[name, "centre"], "pooled"
            else:
                centre, source = pooled.loc[name, "centre"], "pooled_fallback"
            rows.append({"chromatography_type": tc.name, "gel_batch": batch, "feature": name,
                         "centre": centre, "spread": pooled.loc[name, "spread"], "n": len(values),
                         "source": source})
    stats = pd.DataFrame(rows)
    for batch, f in stats[stats["source"] == "pooled_fallback"].groupby("gel_batch")["feature"]:
        log.warning("%s: batch %s has < %d early-life values for %s; pooled centre used",
                    tc.name, batch, MIN_BATCH_REF_CYCLES, ", ".join(f))
    return stats


def standardise(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add rolling means and the three z columns. Returns (cycles, reference statistics table)."""
    out = add_rolling_means(df, cfg.rolling_window)
    for name in ["resin_age", *[rm_col(k) for k in KPIS]]:
        out[z_col(name)] = np.nan

    all_stats = []
    for type_name, tc in cfg.types.items():
        mask = out["chromatography_type"] == type_name
        if not mask.any():
            log.info("No cycles for %s in this input; skipped", type_name)
            continue
        stats = reference_stats(out, tc)
        if tc.scope == "per_batch":
            stats = batch_reference_stats(out, tc, stats)
        all_stats.append(stats)

        for name in ["resin_age", *KPIS]:
            st = stats[stats["feature"] == name]
            if tc.scope == "per_batch":
                lookup = st.set_index("gel_batch")
                centre = out.loc[mask, "gel_batch"].map(lookup["centre"])
                spread = out.loc[mask, "gel_batch"].map(lookup["spread"])
            else:
                centre, spread = st["centre"].iloc[0], st["spread"].iloc[0]
            if name == "resin_age":
                source_col, sign = "resin_age", +1
            else:
                source_col, sign = rm_col(name), tc.sign(name)
            z = sign * (out.loc[mask, source_col] - centre) / spread
            out.loc[mask, z_col(source_col)] = z.clip(-cfg.z_clip, cfg.z_clip)
        log.info("%s: reference = first %d cycles per batch (%s, %s); rolling window %d cycles",
                 type_name, tc.first_n_cycles, tc.method, tc.scope, cfg.rolling_window)

    stats_table = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    return out, stats_table
