"""Checks for training labels, the batch split and the validation metrics."""

from pathlib import Path

import pandas as pd
import pytest

from gel_performance.config import MODEL_FEATURES, load_config
from gel_performance.simulate import SimSettings, simulate
from gel_performance.train import batch_role, label_replace, train
from gel_performance.validate import evaluate

CFG = load_config(Path(__file__).parents[1] / "config" / "config.yaml")


def test_batch_split_is_odd_vs_even():
    assert batch_role("GB-A01") == "calibration"
    assert batch_role("GB-B06") == "holdout"


def test_label_boundaries():
    df = pd.DataFrame({"gel_batch": "GB-A01", "resin_age": [50, 74, 75, 100, 120]})
    truth = pd.DataFrame({"gel_batch": ["GB-A01"], "true_replacement_age": [100]})
    out = label_replace(df, truth, horizon=25)
    assert out["label_replace"].tolist() == [0, 0, 1, 1, 1]   # <= 25 cycles to go, or past it


def test_evaluate_known_answer():
    df = pd.DataFrame({
        "gel_batch": "GB-A02", "resin_age": [1, 2, 3, 4], "true_replacement_age": 4,
        "label_replace": [0, 0, 1, 1], "decision": ["GO", "NO-GO", "GO", "NO-GO"],
    })
    r = evaluate(df)
    assert r["agreement"] == 0.5
    assert r["replace_caught"] == 0.5 and r["no_go_correct"] == 0.5
    b = r["batches"].iloc[0]
    assert b["first_no_go_age"] == 2 and b["warning_cycles_before_true_point"] == 2
    assert b["outcome"] == "in time"


def test_training_is_deterministic_and_uses_calibration_batches_only():
    cycles, batches, _ = simulate(seed=3, s=SimSettings(batches_per_type=2, n_outliers_per_type=1))
    from gel_performance.data import prepare_cycles
    prepared = prepare_cycles(cycles)
    a, b = train(prepared, batches, CFG), train(prepared, batches, CFG)
    assert a == b
    assert a["features"] == MODEL_FEATURES
    assert all(batch_role(x) == "calibration" for x in a["trained_on_batches"])
    leaves = [n for n in a["nodes"] if n.get("leaf")]
    assert all(0.0 <= n["score"] <= 1.0 for n in leaves)
