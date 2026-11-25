"""Replacement score (0..1) from the decision tree, and the GO / NO-GO decision.

Per cycle:
1. score = the decision tree's output for the three standardised features (0 = no sign of needing
   replacement, 1 = all similar training cycles were "replace" cases). If a feature is missing the
   cycle gets no score ("NO SCORE", flag "insufficient") - nothing is guessed.
2. at_pressure_limit = rolling mean of the elution pressure (PT 80th percentile) >= the
   chromatography type's pressure_limit_bar (from config.yaml).
3. Decision:
       NO-GO (replace, red)  if at_pressure_limit AND score >= decision_threshold
       GO    (keep,   green) otherwise
Flags: "complete" (all inputs present this cycle), "incomplete" (an input was missing this cycle, but the
rolling means still had data), "insufficient" (no score).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import MODEL_FEATURES, PRESSURE_KPI, Config
from .features import rm_col
from .model import predict

log = logging.getLogger(__name__)

GO, NO_GO, NO_SCORE = "GO", "NO-GO", "NO SCORE"


def score_cycles(df: pd.DataFrame, cfg: Config, model: dict) -> pd.DataFrame:
    """df = output of features.standardise(). Adds score, rule path, pressure check, decision, flag."""
    out = df.join(predict(model, df[MODEL_FEATURES]))
    out["pressure_limit_bar"] = out["chromatography_type"].map(
        {name: tc.pressure_limit_bar for name, tc in cfg.types.items()})
    out["at_pressure_limit"] = out[rm_col(PRESSURE_KPI)] >= out["pressure_limit_bar"]
    high = out["score"] >= cfg.decision_threshold

    out["decision"] = np.select(
        [out["score"].isna(), high & out["at_pressure_limit"]], [NO_SCORE, NO_GO], default=GO)
    out["flag"] = np.select(
        [out["score"].isna(), out["incomplete"].astype(bool)], ["insufficient", "incomplete"], default="complete")
    out["model_version"] = model["model_version"]

    counts = out["decision"].value_counts()
    log.info("Decisions: %d GO, %d NO-GO, %d NO SCORE (%d incomplete cycles scored from rolling means)",
             counts.get(GO, 0), counts.get(NO_GO, 0), counts.get(NO_SCORE, 0), (out["flag"] == "incomplete").sum())
    return out
