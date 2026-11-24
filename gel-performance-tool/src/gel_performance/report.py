"""Build the tables people read: one row per cycle, plus a one-row-per-gel-batch summary.

Every cycle row carries:
  * identifiers (gel_batch, chromatography_type, resin_age, cycle_end_time)
  * the raw features and their oriented z-scores
  * the weight actually used for each feature (re-normalised if something was missing)
  * the contribution of each feature in score points (pts_*: points lost; negative = points gained)
  * score, smoothed_score, recommendation, flag, missing_features
  * a plain-English `reason` and the config_version that produced the row
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES, Config
from .features import z_col
from .scoring import NO_SCORE, OK, REPLACE, WATCH, contrib_col, weight_col

ID_COLS = ["gel_batch", "chromatography_type", "resin_age", "cycle_end_time"]
RESULT_COLS = ["score", "smoothed_score", "recommendation", "flag", "missing_features", "reason"]
SHORT = {
    "pt_80_percentile": "PT 80th pct pressure",
    "max_pressure_displacement": "max displacement pressure",
    "elution_flow_median": "elution flow",
    "flow_median_displacement": "displacement flow",
    "resin_age": "resin age",
}


def output_columns() -> list[str]:
    return (
        ID_COLS + RESULT_COLS
        + [contrib_col(f) for f in FEATURES]
        + [weight_col(f) for f in FEATURES]
        + [z_col(f) for f in FEATURES]
        + [f for f in FEATURES if f != "resin_age"]
        + ["composite", "n_features_available", "config_version"]
    )


def build_cycle_table(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = scored.copy()
    out["config_version"] = cfg.config_version
    out["reason"] = [_reason(row, cfg) for _, row in out.iterrows()]
    out = out[output_columns()]
    out = out.sort_values(["chromatography_type", "gel_batch", "resin_age"], kind="mergesort")
    num = out.select_dtypes("number").columns.drop(["resin_age", "n_features_available"])
    out[num] = out[num].round(4)
    out["cycle_end_time"] = pd.to_datetime(out["cycle_end_time"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out.reset_index(drop=True)


def top_drivers(row: pd.Series, n: int = 2) -> list[tuple[str, float]]:
    """The n features that took the most points off this cycle (only those that took points off)."""
    pts = {f: row[contrib_col(f)] for f in FEATURES if pd.notna(row[contrib_col(f)])}
    worst = sorted(pts.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [(f, p) for f, p in worst if p > 0.5]


def _reason(row: pd.Series, cfg: Config) -> str:
    if row["recommendation"] == NO_SCORE:
        return (f"No score: only {row['n_features_available']} of {len(FEATURES)} features available "
                f"(missing: {row['missing_features']}).")

    tc = cfg.types[row["chromatography_type"]]
    s = row["smoothed_score"]
    if row["recommendation"] == OK:
        rule = f"smoothed score {s:.0f} >= {tc.ok_min:.0f}"
    elif row["recommendation"] == WATCH:
        rule = f"smoothed score {s:.0f} is between {tc.watch_min:.0f} and {tc.ok_min:.0f}"
    else:
        rule = f"smoothed score {s:.0f} < {tc.watch_min:.0f}"

    parts = [f"{row['recommendation']}: {rule}."]
    drivers = top_drivers(row)
    if drivers and row["recommendation"] in (WATCH, REPLACE):
        parts.append("Main drivers: " + ", ".join(f"{SHORT[f]} (-{p:.0f} pts)" for f, p in drivers) + ".")
    if row["flag"] == "incomplete":
        parts.append(f"Incomplete cycle, scored without: {row['missing_features']}.")
    return " ".join(parts)


def build_batch_summary(cycle_table: pd.DataFrame) -> pd.DataFrame:
    """Latest scored cycle of every gel batch: the current status of each resin lot."""
    scored = cycle_table[cycle_table["recommendation"] != NO_SCORE]
    latest = scored.sort_values("resin_age").groupby("gel_batch", sort=True).tail(1)
    counts = cycle_table.groupby("gel_batch").agg(
        cycles=("resin_age", "size"),
        incomplete_cycles=("flag", lambda f: int((f != "complete").sum())),
    )
    first_replace = (
        cycle_table[cycle_table["recommendation"] == REPLACE].groupby("gel_batch")["resin_age"].min()
        .rename("first_replace_soon_age")
    )
    summary = (
        latest[["gel_batch", "chromatography_type", "resin_age", "smoothed_score", "recommendation", "reason"]]
        .rename(columns={"resin_age": "latest_resin_age"})
        .set_index("gel_batch").join(counts).join(first_replace).reset_index()
    )
    summary["first_replace_soon_age"] = summary["first_replace_soon_age"].astype("Int64")
    return summary.sort_values(["chromatography_type", "gel_batch"]).reset_index(drop=True)
