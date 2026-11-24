"""Checks for reading and tidying cycle data: missing values are flagged, never filled."""

import numpy as np
import pandas as pd
import pytest

from gel_performance.data import DataError, prepare_cycles


def make_rows(**overrides):
    row = {
        "gel_batch": "GB-X01",
        "chromatography_type": "type_A",
        "cycle_end_time": "2024-01-01T10:00:00",
        "resin_age": 1,
        "pt_80_percentile": 1.2,
        "max_pressure_displacement": 1.8,
        "elution_flow_median": 120.0,
        "flow_median_displacement": 100.0,
    }
    row.update(overrides)
    return row


def test_missing_value_is_flagged_and_not_filled():
    df = pd.DataFrame([make_rows(), make_rows(resin_age=2, elution_flow_median=np.nan)])
    out = prepare_cycles(df)
    second = out.iloc[1]
    assert bool(second["incomplete"]) is True
    assert second["missing_features"] == "elution_flow_median"
    assert second["n_features_available"] == 4
    assert np.isnan(second["elution_flow_median"])  # still missing, not filled
    assert bool(out.iloc[0]["incomplete"]) is False


def test_text_in_a_numeric_column_becomes_missing():
    df = pd.DataFrame([make_rows(pt_80_percentile="n/a")])
    out = prepare_cycles(df)
    assert out.loc[0, "missing_features"] == "pt_80_percentile"


def test_missing_required_column_stops_the_run():
    df = pd.DataFrame([make_rows()]).drop(columns=["resin_age"])
    with pytest.raises(DataError, match="resin_age"):
        prepare_cycles(df)


def test_unknown_type_and_duplicates_are_dropped():
    df = pd.DataFrame([
        make_rows(),
        make_rows(pt_80_percentile=1.3),          # duplicate of cycle 1 -> keep one
        make_rows(chromatography_type="type_Z"),  # not in config
    ])
    out = prepare_cycles(df, known_types={"type_A"})
    assert len(out) == 1


def test_row_order_does_not_matter():
    rows = [make_rows(resin_age=a, cycle_end_time=f"2024-01-0{a}") for a in (1, 2, 3)]
    a = prepare_cycles(pd.DataFrame(rows))
    b = prepare_cycles(pd.DataFrame(rows[::-1]))
    pd.testing.assert_frame_equal(a, b)
