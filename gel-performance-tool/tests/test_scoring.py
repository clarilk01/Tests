"""Known-answer checks for the tree predictor, the GO / NO-GO rule and the reasons."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gel_performance.config import MODEL_FEATURES, load_config
from gel_performance.model import predict, rules_text
from gel_performance.report import build_batch_summary, build_cycle_table
from gel_performance.scoring import GO, NO_GO, NO_SCORE, score_cycles

CFG = load_config(Path(__file__).parents[1] / "config" / "config.yaml")

# Hand-made tree:  pressure z <= 2 -> score 0.1 ; else age z <= 1 -> 0.4 ; else 0.9
TREE = {"model_version": "test-tree", "features": MODEL_FEATURES, "nodes": [
    {"id": 0, "feature": "z_rm_pt_80_percentile", "threshold": 2.0, "left": 1, "right": 2, "n": 100},
    {"id": 1, "leaf": True, "score": 0.1, "n": 60, "n_replace": 6},
    {"id": 2, "feature": "z_resin_age", "threshold": 1.0, "left": 3, "right": 4, "n": 40},
    {"id": 3, "leaf": True, "score": 0.4, "n": 20, "n_replace": 8},
    {"id": 4, "leaf": True, "score": 0.9, "n": 20, "n_replace": 18},
]}


def frame(rows):
    """rows: (z_age, z_flow, z_pressure, rolling pressure bar, incomplete)"""
    df = pd.DataFrame(rows, columns=[*MODEL_FEATURES, "rm_pt_80_percentile", "incomplete"])
    df["gel_batch"], df["chromatography_type"] = "GB-X01", "type_A"   # type_A limit = 1.36 bar
    df["resin_age"] = range(1, len(df) + 1)
    df["cycle_end_time"] = "2024-01-01T00:00:00"
    df["rm_elution_flow_median"], df["elution_flow_median"], df["pt_80_percentile"] = 100.0, 100.0, 1.3
    df["missing_features"] = np.where(df["incomplete"], "elution_flow_median", "")
    return df


def test_predictor_follows_the_tree():
    X = pd.DataFrame([[0, 0, 1.5], [0.5, 0, 3.0], [2.0, 0, 3.0], [2.0, 0, np.nan]], columns=MODEL_FEATURES)
    out = predict(TREE, X)
    assert out["score"].tolist()[:3] == [0.1, 0.4, 0.9]
    assert np.isnan(out["score"].iloc[3])                 # missing feature -> no score, never guessed
    assert "elution pressure (rolling) z 3.00 > 2.00" in out["rule_path"].iloc[2]


def test_threshold_boundary_goes_left():
    X = pd.DataFrame([[0, 0, 2.0]], columns=MODEL_FEATURES)
    assert predict(TREE, X)["score"].iloc[0] == 0.1       # value == threshold -> left branch


@pytest.mark.parametrize("z_age, z_p, rm_p, expected", [
    (2.0, 3.0, 1.40, NO_GO),   # score 0.9 high AND pressure 1.40 >= 1.36 -> replace
    (2.0, 3.0, 1.30, GO),      # score high but pressure below limit -> keep
    (0.0, 1.0, 1.40, GO),      # pressure at limit but score 0.1 low -> keep
    (0.0, 1.0, 1.36, GO),
    (2.0, 3.0, 1.36, NO_GO),   # exactly at the limit counts as at the limit
])
def test_go_no_go_rule(z_age, z_p, rm_p, expected):
    out = score_cycles(frame([(z_age, 0.0, z_p, rm_p, False)]), CFG, TREE)
    assert out["decision"].iloc[0] == expected


def test_no_score_and_flags():
    out = score_cycles(frame([(np.nan, 0.0, 1.0, 1.2, True), (0.0, 0.0, 1.0, 1.2, True),
                              (0.0, 0.0, 1.0, 1.2, False)]), CFG, TREE)
    assert out["decision"].tolist() == [NO_SCORE, GO, GO]
    assert out["flag"].tolist() == ["insufficient", "incomplete", "complete"]


def test_reason_and_batch_summary():
    scored = score_cycles(frame([(0.0, 0.0, 1.0, 1.20, False), (2.0, 0.0, 3.0, 1.40, True)]), CFG, TREE)
    table = build_cycle_table(scored, CFG)
    nogo = table.iloc[1]
    assert nogo["decision"] == NO_GO
    assert "rolling elution pressure 1.40 bar >= limit 1.36 bar" in nogo["reason"]
    assert "replacement score 0.90 >= 0.50" in nogo["reason"]
    assert "missing elution_flow_median" in nogo["reason"]
    summary = build_batch_summary(table)
    assert summary.loc[0, "decision"] == NO_GO and summary.loc[0, "first_no_go_age"] == 2


def test_rules_text_is_readable():
    text = rules_text(TREE)
    assert "if elution pressure (rolling) z <= 2.00:" in text
    assert "-> score 0.90  (18 of 20 training cycles were 'replace')" in text
