"""Build the tables people read: one row per cycle, plus a one-row-per-gel-batch summary.

Every cycle row carries the identifiers, the decision (GO / NO-GO), the replacement score (0..1),
the rolling KPIs and the pressure limit, the standardised features, the tree rule that produced the
score, a plain-English `reason`, and the config and model versions that produced the row.
"""

from __future__ import annotations

import pandas as pd

from .config import KPIS, MODEL_FEATURES, PRESSURE_KPI, Config
from .features import rm_col
from .scoring import GO, NO_GO, NO_SCORE

ID_COLS = ["gel_batch", "chromatography_type", "resin_age", "cycle_end_time"]


def output_columns() -> list[str]:
    return (
        ID_COLS
        + ["decision", "score", "flag", "reason"]
        + [rm_col(PRESSURE_KPI), "pressure_limit_bar", "at_pressure_limit", rm_col("elution_flow_median")]
        + KPIS
        + MODEL_FEATURES
        + ["rule_path", "leaf", "missing_features", "model_version", "config_version"]
    )


def build_cycle_table(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = scored.copy()
    out["config_version"] = cfg.config_version
    out["reason"] = [_reason(row, cfg) for _, row in out.iterrows()]
    out = out[output_columns()]
    out = out.sort_values(["chromatography_type", "gel_batch", "resin_age"], kind="mergesort")
    num = out.select_dtypes("number").columns.drop(["resin_age", "leaf"])
    out[num] = out[num].round(4)
    out["cycle_end_time"] = pd.to_datetime(out["cycle_end_time"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out.reset_index(drop=True)


def _reason(row: pd.Series, cfg: Config) -> str:
    if row["decision"] == NO_SCORE:
        return f"No score: not enough data in the rolling window (missing: {row['missing_features']})."
    thr = cfg.decision_threshold
    p, lim, s = row[rm_col(PRESSURE_KPI)], row["pressure_limit_bar"], row["score"]
    pressure = f"rolling elution pressure {p:.2f} bar {'>=' if row['at_pressure_limit'] else '<'} limit {lim:.2f} bar"
    score = f"replacement score {s:.2f} {'>=' if s >= thr else '<'} {thr:.2f}"

    if row["decision"] == NO_GO:
        text = f"NO-GO (replace): {pressure} and {score}."
    elif s >= thr:
        text = f"GO: {score} is high, but {pressure} - keep watching."
    elif row["at_pressure_limit"]:
        text = f"GO: {pressure}, but {score} is low - check the run / sensor."
    else:
        text = f"GO: {score} and {pressure}."
    text += f" Tree rule: {row['rule_path']}."
    if row["flag"] == "incomplete":
        text += f" This cycle was missing {row['missing_features']}; the rolling mean used the other cycles."
    return text


def build_batch_summary(cycle_table: pd.DataFrame) -> pd.DataFrame:
    """Latest scored cycle of every gel batch: the current status of each resin lot."""
    scored = cycle_table[cycle_table["decision"] != NO_SCORE]
    latest = scored.sort_values("resin_age").groupby("gel_batch", sort=True).tail(1)
    counts = cycle_table.groupby("gel_batch").agg(
        cycles=("resin_age", "size"),
        incomplete_cycles=("flag", lambda f: int((f != "complete").sum())),
    )
    first_nogo = (cycle_table[cycle_table["decision"] == NO_GO].groupby("gel_batch")["resin_age"].min()
                  .rename("first_no_go_age"))
    cols = ["gel_batch", "chromatography_type", "resin_age", "cycle_end_time", "decision", "score",
            rm_col(PRESSURE_KPI), "pressure_limit_bar", "reason"]
    summary = (latest[cols].rename(columns={"resin_age": "latest_resin_age", "cycle_end_time": "latest_cycle_end"})
               .set_index("gel_batch").join(counts).join(first_nogo).reset_index())
    summary["first_no_go_age"] = summary["first_no_go_age"].astype("Int64")
    return summary.sort_values(["chromatography_type", "gel_batch"]).reset_index(drop=True)


__all__ = ["build_cycle_table", "build_batch_summary", "output_columns", "GO", "NO_GO", "NO_SCORE"]
