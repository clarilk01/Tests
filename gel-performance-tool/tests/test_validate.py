"""Known-answer checks for the validation logic (Step 6)."""

import pandas as pd
import pytest

from gel_performance.scoring import OK, REPLACE, WATCH
from gel_performance.validate import (
    Thresholds, balanced_accuracy, batch_role, calibrate, classify, evaluate, true_class,
)


def test_true_class_boundaries():
    to_go = pd.Series([200, 76, 75, 26, 25, 0, -10])
    assert true_class(to_go).tolist() == [OK, OK, WATCH, WATCH, REPLACE, REPLACE, REPLACE]


def test_batch_split_is_odd_vs_even():
    assert batch_role("GB-A01") == "calibration"
    assert batch_role("GB-B06") == "holdout"


def test_classify_uses_thresholds():
    s = pd.Series([80.0, 50.0, 10.0, float("nan")])
    assert classify(s, Thresholds(60, 30)).tolist() == [OK, WATCH, REPLACE, "NO SCORE"]


def test_balanced_accuracy_is_not_fooled_by_the_majority_class():
    true = pd.Series([OK] * 8 + [WATCH, REPLACE])
    always_ok = pd.Series([OK] * 10)
    assert (always_ok == true).mean() == 0.8                       # plain accuracy looks good...
    assert balanced_accuracy(true, always_ok) == pytest.approx(1 / 3)  # ...balanced accuracy does not


def toy_batch(true_age=100):
    """Score falls linearly from 100 to 0 over 100 cycles; true replacement at cycle 100."""
    ages = list(range(1, 101))
    return pd.DataFrame({
        "gel_batch": "GB-A01", "resin_age": ages, "smoothed_score": [100 - a for a in ages],
        "true_replacement_age": true_age, "true_class": true_class(pd.Series([true_age - a for a in ages])),
    })


def test_calibrate_finds_the_perfect_thresholds_on_a_toy_batch():
    # true WATCH starts at cycles_to_go 75 -> age 25 -> score 75 ; REPLACE at age 75 -> score 25
    t = calibrate(toy_batch())
    assert (t.ok_min, t.watch_min) == (76.0, 26.0)
    r = evaluate(toy_batch(), t)
    assert r["agreement"] == 1.0


def test_evaluate_reports_warning_lead_time():
    r = evaluate(toy_batch(), Thresholds(76, 26))
    b = r["batches"].iloc[0]
    assert b["first_replace_soon_age"] == 75
    assert b["warning_cycles_before_true_point"] == 25
    assert b["outcome"] == "in time"
