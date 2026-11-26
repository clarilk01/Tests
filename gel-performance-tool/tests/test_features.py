"""Known-answer checks for rolling means and standardisation (worked out by hand)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gel_performance.config import MODEL_FEATURES, load_config
from gel_performance.data import prepare_cycles
from gel_performance.features import add_rolling_means, reference_stats, standardise

CONFIG_PATH = Path(__file__).parents[1] / "config" / "config.yaml"


def small_cfg(window=1, first_n=3, scope="pooled"):
    cfg = load_config(CONFIG_PATH)
    return replace(cfg, rolling_window=window,
                   types={k: replace(t, first_n_cycles=first_n, scope=scope) for k, t in cfg.types.items()})


def cycle(age, flow, pressure, batch="GB-X01", ctype="type_A"):
    return {"gel_batch": batch, "chromatography_type": ctype, "cycle_end_time": f"2024-01-01T{age % 24:02d}:00:00",
            "resin_age": age, "elution_flow_median": flow, "pt_80_percentile": pressure}


@pytest.fixture
def tiny():
    # Reference window = ages 1..3: flow 100, 101, 102 (mean 101, std 1); pressure 1, 2, 3 (mean 2, std 1)
    return prepare_cycles(pd.DataFrame([
        cycle(1, 100.0, 1.0), cycle(2, 101.0, 2.0), cycle(3, 102.0, 3.0), cycle(103, 98.0, 5.0),
    ]))


def test_rolling_mean_known_answer_and_skips_missing():
    df = prepare_cycles(pd.DataFrame([cycle(1, 10.0, 1.0), cycle(2, np.nan, 2.0), cycle(3, 30.0, 3.0),
                                      cycle(4, 40.0, 4.0)]))
    out = add_rolling_means(df, window=3)
    assert out["rm_pt_80_percentile"].tolist() == pytest.approx([1.0, 1.5, 2.0, 3.0])
    # flow at age 2 is missing: it is left out of the average, never filled in
    assert out["rm_elution_flow_median"].tolist() == pytest.approx([10.0, 10.0, 20.0, 35.0])


def test_rolling_mean_stays_within_a_gel_batch():
    df = prepare_cycles(pd.DataFrame([cycle(1, 10.0, 1.0, batch="A"), cycle(1, 99.0, 9.0, batch="B"),
                                      cycle(2, 20.0, 2.0, batch="A")]))
    out = add_rolling_means(df, window=5).set_index(["gel_batch", "resin_age"])
    assert out.loc[("A", 2), "rm_pt_80_percentile"] == pytest.approx(1.5)


def test_reference_stats_known_answer(tiny):
    stats = reference_stats(tiny, small_cfg().types["type_A"]).set_index("feature")
    assert stats.loc["pt_80_percentile", ["centre", "spread"]].tolist() == pytest.approx([2.0, 1.0])
    assert stats.loc["elution_flow_median", ["centre", "spread"]].tolist() == pytest.approx([101.0, 1.0])
    assert stats.loc["resin_age", ["centre", "spread"]].tolist() == pytest.approx([2.0, 100.0])


def test_z_scores_known_answer_and_orientation(tiny):
    out, _ = standardise(tiny, small_cfg(window=1))
    last = out[out["resin_age"] == 103].iloc[0]
    assert last["z_rm_pt_80_percentile"] == pytest.approx((5.0 - 2.0) / 1.0)        # pressure up -> +3
    assert last["z_rm_elution_flow_median"] == pytest.approx((101.0 - 98.0) / 1.0)  # flow DOWN -> +3
    assert last["z_resin_age"] == pytest.approx((103 - 2) / 100)


@pytest.mark.parametrize("column, worse_change", [("pt_80_percentile", +1.0), ("elution_flow_median", -1.0)])
def test_each_kpi_z_goes_up_when_it_gets_worse(tiny, column, worse_change):
    worse = tiny.copy()
    worse.loc[worse.index[-1], column] += worse_change
    base, _ = standardise(tiny, small_cfg())
    bad, _ = standardise(worse, small_cfg())
    z = "z_rm_" + column
    assert bad.loc[bad.index[-1], z] > base.loc[base.index[-1], z]


def test_each_type_uses_its_own_reference():
    rows = [cycle(a, 100.0 + a, float(a)) for a in (1, 2, 3)]
    rows += [cycle(a, 10.0 + a, 10.0 * a, batch="GB-Y01", ctype="type_B") for a in (1, 2, 3)]
    out, stats = standardise(prepare_cycles(pd.DataFrame(rows)), small_cfg())
    assert np.allclose(out.loc[out["resin_age"] == 2, "z_rm_pt_80_percentile"], 0.0)
    assert set(stats["chromatography_type"]) == {"type_A", "type_B"}


def test_outlier_is_capped(tiny):
    tiny = tiny.copy()
    tiny.loc[tiny.index[-1], "pt_80_percentile"] = 1000.0
    out, _ = standardise(tiny, small_cfg())
    assert out["z_rm_pt_80_percentile"].max() == 6.0


def test_per_batch_scope_compares_each_batch_with_its_own_start():
    rows = []
    for batch, offset in [("GB-X01", 0.0), ("GB-Y01", 1.0)]:
        rows += [cycle(a, 100.0 + a, 1.0 + offset + 0.1 * (a % 2), batch=batch) for a in range(1, 6)]
        rows.append(cycle(50, 103.0, 1.5 + offset, batch=batch))
    out, stats = standardise(prepare_cycles(pd.DataFrame(rows)), small_cfg(first_n=5, scope="per_batch"))
    late = out[out["resin_age"] == 50].set_index("gel_batch")["z_rm_pt_80_percentile"]
    assert late["GB-Y01"] == pytest.approx(late["GB-X01"])   # same rise from own start -> same z
    assert set(stats.loc[stats["feature"] == "pt_80_percentile", "source"]) == {"batch"}


def test_all_model_features_are_created(tiny):
    out, _ = standardise(tiny, small_cfg())
    assert all(c in out.columns for c in MODEL_FEATURES)
