"""Step 3 - Combine the oriented z-scores into one 0-100 performance score and a recommendation.

Per cycle (all settings come from config.yaml, per chromatography type):

1. Available features = those whose z-score is not missing.
   If fewer than `min_features_for_score` are available -> no score, flag "insufficient".
2. Re-normalised weights (only when something is missing):
       w'_i = w_i / (sum of w_j over the AVAILABLE features)
   so the weights used always add up to 1. Missing values are never filled in.
3. Composite (weighted z-sum):      S = sum_i  w'_i * z_i
4. Contribution of each feature, in score points (points lost; negative = points gained):
       penalty_i = 100 * w'_i * z_i / composite_at_zero
5. Score:
       score = clip(100 - sum_i penalty_i, 0, 100)  =  clip(100 * (1 - S / composite_at_zero), 0, 100)
   A cycle that looks exactly like fresh reference resin (all z = 0) scores 100;
   a cycle whose composite reaches `composite_at_zero` scores 0.
   The penalties add up exactly to (100 - score) unless the 0/100 cap was hit.
6. Smoothed score = rolling median of the score over the last `smoothing_window` scored cycles
   of the same gel batch (ordered by resin_age). One noisy cycle cannot flip the recommendation.
7. Recommendation from the smoothed score:
       smoothed >= ok_min               -> OK
       watch_min <= smoothed < ok_min   -> WATCH
       smoothed < watch_min             -> REPLACE SOON
   A cycle without a score gets "NO SCORE".

Flags: "complete" (all features), "incomplete" (scored from fewer features), "insufficient" (no score).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import FEATURES, Config, TypeConfig
from .features import z_col

log = logging.getLogger(__name__)

OK, WATCH, REPLACE, NO_SCORE = "OK", "WATCH", "REPLACE SOON", "NO SCORE"


def contrib_col(feature: str) -> str:
    return f"pts_{feature}"


def weight_col(feature: str) -> str:
    return f"w_{feature}"


def score_type(df: pd.DataFrame, tc: TypeConfig, min_features: int) -> pd.DataFrame:
    """Score cycles of ONE chromatography type. `df` must already contain the z_<feature> columns."""
    z = df[[z_col(f) for f in FEATURES]].to_numpy(dtype=float)           # rows x features
    w = np.array([tc.features[f].weight for f in FEATURES], dtype=float)  # config weights

    available = ~np.isnan(z)
    n_avail = available.sum(axis=1)
    w_avail = np.where(available, w, 0.0)
    w_total = w_avail.sum(axis=1, keepdims=True)
    scorable = (n_avail >= min_features) & (w_total[:, 0] > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        w_norm = np.where(scorable[:, None], w_avail / w_total, np.nan)       # re-normalised weights
        penalty = 100.0 * w_norm * np.nan_to_num(z) / tc.composite_at_zero  # NaN z -> weight 0 anyway
    penalty = np.where(available & scorable[:, None], penalty, np.nan)
    composite = np.where(scorable, np.nansum(w_norm * np.nan_to_num(z), axis=1), np.nan)
    score = np.where(scorable, np.clip(100.0 - np.nansum(penalty, axis=1), 0.0, 100.0), np.nan)

    out = pd.DataFrame(index=df.index)
    out["composite"] = composite
    out["score"] = score
    for i, f in enumerate(FEATURES):
        out[weight_col(f)] = w_norm[:, i]
        out[contrib_col(f)] = penalty[:, i]
    out["flag"] = np.select(
        [~scorable, n_avail < len(FEATURES)], ["insufficient", "incomplete"], default="complete"
    )
    return out


def smooth_and_recommend(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add smoothed_score and recommendation. Needs columns gel_batch, chromatography_type, resin_age, score."""
    out = df.copy()
    out["smoothed_score"] = np.nan
    scored = out[out["score"].notna()].sort_values(["gel_batch", "resin_age"], kind="mergesort")
    rolled = (
        scored.groupby("gel_batch", sort=True)["score"]
        .rolling(cfg.smoothing_window, min_periods=1).median()
        .reset_index(level=0, drop=True)
    )
    out.loc[rolled.index, "smoothed_score"] = rolled

    out["recommendation"] = NO_SCORE
    for type_name, tc in cfg.types.items():
        mask = (out["chromatography_type"] == type_name) & out["smoothed_score"].notna()
        s = out.loc[mask, "smoothed_score"]
        out.loc[mask, "recommendation"] = np.select(
            [s >= tc.ok_min, s >= tc.watch_min], [OK, WATCH], default=REPLACE
        )
    return out


def score_cycles(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Score every cycle (all types). Input: output of features.standardise()."""
    parts = []
    for type_name, tc in cfg.types.items():
        sub = df[df["chromatography_type"] == type_name]
        if len(sub):
            parts.append(score_type(sub, tc, cfg.min_features_for_score))
    scores = pd.concat(parts) if parts else pd.DataFrame(index=df.index)
    out = df.join(scores)
    out = smooth_and_recommend(out, cfg)

    counts = out["flag"].value_counts()
    log.info(
        "Scored %d cycles: %d complete, %d incomplete (re-normalised weights), %d insufficient (no score)",
        len(out), counts.get("complete", 0), counts.get("incomplete", 0), counts.get("insufficient", 0),
    )
    return out
