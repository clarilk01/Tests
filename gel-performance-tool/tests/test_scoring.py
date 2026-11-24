"""Known-answer checks for the score, weight re-normalisation, smoothing and thresholds (Step 3)."""

import numpy as np
import pandas as pd
import pytest

from gel_performance.config import FEATURES, Config, FeatureSpec, TypeConfig
from gel_performance.scoring import NO_SCORE, OK, REPLACE, WATCH, score_cycles, score_type, smooth_and_recommend

# Simple weights that are easy to do by hand: 0.4, 0.2, 0.2, 0.1, 0.1
WEIGHTS = dict(zip(FEATURES, [0.4, 0.2, 0.2, 0.1, 0.1]))


def make_type(name="type_A", ok_min=70.0, watch_min=40.0, composite_at_zero=4.0):
    feats = {f: FeatureSpec(f, "higher_is_worse", w, 100.0 if f == "resin_age" else None)
             for f, w in WEIGHTS.items()}
    return TypeConfig(name, 30, "mean_std", feats, composite_at_zero, ok_min, watch_min)


def make_cfg(smoothing_window=1):
    return Config("test", 6.0, 3, smoothing_window, {"type_A": make_type()})


def z_rows(*rows, batch="GB-X01"):
    """Build a frame of z-scores. Each row is a list of 5 z-values (NaN = missing)."""
    df = pd.DataFrame(rows, columns=[f"z_{f}" for f in FEATURES])
    df["gel_batch"] = batch
    df["chromatography_type"] = "type_A"
    df["resin_age"] = range(1, len(df) + 1)
    return df


def test_fresh_resin_scores_100():
    out = score_type(z_rows([0, 0, 0, 0, 0]), make_type(), 3)
    assert out.loc[0, "score"] == pytest.approx(100.0)
    assert out.loc[0, "flag"] == "complete"


def test_score_known_answer_and_contributions():
    # z = 1 for every feature -> composite = 1 -> score = 100 * (1 - 1/4) = 75
    # each feature loses 100 * w * 1 / 4 points: 10, 5, 5, 2.5, 2.5
    out = score_type(z_rows([1, 1, 1, 1, 1]), make_type(), 3)
    assert out.loc[0, "composite"] == pytest.approx(1.0)
    assert out.loc[0, "score"] == pytest.approx(75.0)
    assert out.loc[0, "pts_pt_80_percentile"] == pytest.approx(10.0)
    assert out.loc[0, "pts_resin_age"] == pytest.approx(2.5)
    pts = out[[f"pts_{f}" for f in FEATURES]].sum(axis=1)
    assert 100 - pts.loc[0] == pytest.approx(out.loc[0, "score"])  # contributions add up to the score


def test_score_is_capped_between_0_and_100():
    out = score_type(z_rows([6, 6, 6, 6, 6], [-3, -3, -3, -3, -3]), make_type(), 3)
    assert out.loc[0, "score"] == 0.0     # composite 6 > 4
    assert out.loc[1, "score"] == 100.0   # better than reference is capped at 100


def test_weights_are_renormalised_when_a_feature_is_missing():
    # pt_80 (weight 0.4) missing -> remaining weights 0.2, 0.2, 0.1, 0.1 sum to 0.6
    # re-normalised: 1/3, 1/3, 1/6, 1/6
    out = score_type(z_rows([np.nan, 2, 2, 2, 2]), make_type(), 3)
    row = out.loc[0]
    assert row["flag"] == "incomplete"
    assert row["w_pt_80_percentile"] == 0                        # missing feature gets weight 0
    assert row["w_max_pressure_displacement"] == pytest.approx(1 / 3)
    assert row["w_resin_age"] == pytest.approx(1 / 6)
    assert np.isnan(row["pts_pt_80_percentile"])                 # no contribution invented
    assert row["composite"] == pytest.approx(2.0)                # all available z = 2
    assert row["score"] == pytest.approx(100 * (1 - 2 / 4))      # 50, not dragged towards 100 or 0


def test_too_few_features_gives_no_score():
    out = score_type(z_rows([np.nan, np.nan, np.nan, 1, 1]), make_type(), 3)
    assert out.loc[0, "flag"] == "insufficient"
    assert np.isnan(out.loc[0, "score"])


@pytest.mark.parametrize("score, expected", [
    (100.0, OK), (70.0, OK), (69.99, WATCH), (40.0, WATCH), (39.99, REPLACE), (0.0, REPLACE),
])
def test_thresholds(score, expected):
    df = pd.DataFrame({"gel_batch": ["B"], "chromatography_type": ["type_A"], "resin_age": [1], "score": [score]})
    out = smooth_and_recommend(df, make_cfg(smoothing_window=1))
    assert out.loc[0, "recommendation"] == expected


def test_missing_score_gives_no_score_recommendation():
    df = pd.DataFrame({"gel_batch": ["B"], "chromatography_type": ["type_A"], "resin_age": [1], "score": [np.nan]})
    out = smooth_and_recommend(df, make_cfg())
    assert out.loc[0, "recommendation"] == NO_SCORE


def test_smoothing_ignores_a_single_bad_cycle():
    scores = [80, 80, 10, 80, 80]  # one noisy cycle in the middle
    df = pd.DataFrame({"gel_batch": "B", "chromatography_type": "type_A",
                       "resin_age": range(1, 6), "score": scores})
    out = smooth_and_recommend(df, make_cfg(smoothing_window=5))
    assert (out["recommendation"] == OK).all()
    assert out.loc[2, "smoothed_score"] == pytest.approx(80.0)  # median of 80, 80, 10


def test_smoothing_stays_within_a_gel_batch():
    a = pd.DataFrame({"gel_batch": "A", "chromatography_type": "type_A", "resin_age": [1, 2], "score": [10, 10]})
    b = pd.DataFrame({"gel_batch": "B", "chromatography_type": "type_A", "resin_age": [1, 2], "score": [90, 90]})
    out = smooth_and_recommend(pd.concat([a, b], ignore_index=True), make_cfg(smoothing_window=5))
    assert out.loc[out["gel_batch"] == "B", "smoothed_score"].tolist() == [90, 90]


def test_score_cycles_end_to_end_on_z_frame():
    out = score_cycles(z_rows([0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [np.nan, 2, 2, 2, 2]), make_cfg())
    assert out["flag"].tolist() == ["complete", "complete", "incomplete"]
    assert out["score"].tolist() == pytest.approx([100.0, 75.0, 50.0])


# ---- report table -------------------------------------------------------------------

def test_reason_names_the_main_driver_and_missing_feature():
    from gel_performance.report import build_cycle_table

    df = z_rows([np.nan, 2, 6, 0, 0])
    df["cycle_end_time"] = "2024-01-01T00:00:00"
    df["n_features_available"] = 4
    df["missing_features"] = "pt_80_percentile"
    for f in FEATURES:
        if f != "resin_age":
            df[f] = 1.0
    table = build_cycle_table(score_cycles(df, make_cfg()), make_cfg())
    row = table.iloc[0]
    # weights 0.2, 0.2, 0.1, 0.1 -> 1/3, 1/3, 1/6, 1/6 ; composite = 2/3 + 2 = 2.667 ; score = 33.3
    assert row["score"] == pytest.approx(100 * (1 - (8 / 3) / 4), abs=1e-3)
    assert row["recommendation"] == REPLACE
    assert "elution flow (-50 pts)" in row["reason"]
    assert "scored without: pt_80_percentile" in row["reason"]
    assert row["config_version"] == "test"
