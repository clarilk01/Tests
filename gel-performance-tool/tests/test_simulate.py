"""Checks for the synthetic data generator (Step 1)."""

from gel_performance.simulate import PROCESS_COLS, simulate


def test_same_seed_gives_identical_data():
    a, _, _ = simulate(seed=7)
    b, _, _ = simulate(seed=7)
    assert a.equals(b)


def test_data_has_the_intended_shape_and_problems():
    cycles, batches, truth = simulate(seed=42)
    assert cycles["gel_batch"].nunique() == 12
    assert set(cycles["chromatography_type"]) == {"type_A", "type_B"}
    # ~3% of cycles have at least one missing feature (interrupted runs)
    missing_share = cycles[PROCESS_COLS].isna().any(axis=1).mean()
    assert 0.025 < missing_share < 0.035
    # outliers are injected and recorded in the hidden truth file
    assert truth["injected_issue"].str.startswith("outlier").sum() == 16
    # every batch runs past its true replacement point, so late detection can be measured
    assert (batches["last_age_in_data"] > batches["true_replacement_age"]).all()


def test_pressures_rise_and_flows_fall_with_age():
    cycles, _, _ = simulate(seed=42)
    early = cycles[cycles["resin_age"] <= 30].groupby("gel_batch")[PROCESS_COLS].median()
    late = cycles[cycles["resin_age"] >= 300].groupby("gel_batch")[PROCESS_COLS].median()
    assert (late["pt_80_percentile"] > early["pt_80_percentile"]).all()
    assert (late["elution_flow_median"] < early["elution_flow_median"]).all()
