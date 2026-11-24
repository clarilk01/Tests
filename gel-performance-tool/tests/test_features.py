"""Known-answer checks for standardisation and orientation (Step 2).

The tiny datasets below are small enough to work out by hand.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gel_performance.config import FEATURES, parse_config
from gel_performance.data import prepare_cycles
from gel_performance.features import reference_stats, standardise

CONFIG_PATH = Path(__file__).parents[1] / "config" / "config.yaml"


@pytest.fixture
def cfg():
    """Shipped config, but with a 3-cycle reference window so the maths is easy to follow."""
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw = copy.deepcopy(raw)
    for t in raw["chromatography_types"].values():
        t["reference"]["first_n_cycles"] = 3
        t["reference"]["method"] = "mean_std"
    return parse_config(raw)


def cycle(age, pt80, pdisp, eflow, dflow, batch="GB-X01", ctype="type_A"):
    return {
        "gel_batch": batch, "chromatography_type": ctype, "cycle_end_time": f"2024-01-01T{age:02d}:00:00",
        "resin_age": age, "pt_80_percentile": pt80, "max_pressure_displacement": pdisp,
        "elution_flow_median": eflow, "flow_median_displacement": dflow,
    }


@pytest.fixture
def tiny():
    # Reference window = ages 1..3. Each feature has mean = middle value and std = 1 there.
    rows = [
        cycle(1, 1.0, 2.0, 100.0, 50.0),
        cycle(2, 2.0, 3.0, 101.0, 51.0),
        cycle(3, 3.0, 4.0, 102.0, 52.0),
        cycle(103, 5.0, 3.0, 98.0, 51.0),   # later cycle: pressure up, flow down
    ]
    return prepare_cycles(pd.DataFrame(rows))


def test_reference_stats_known_answer(tiny, cfg):
    stats = reference_stats(tiny, cfg.types["type_A"]).set_index("feature")
    assert stats.loc["pt_80_percentile", "centre"] == pytest.approx(2.0)
    assert stats.loc["pt_80_percentile", "spread"] == pytest.approx(1.0)  # std of 1,2,3
    assert stats.loc["elution_flow_median", "centre"] == pytest.approx(101.0)
    assert stats.loc["pt_80_percentile", "n"] == 3                         # age 103 not in the window
    # resin_age: centre = mean age in window, spread = scale_cycles from the config (100)
    assert stats.loc["resin_age", "centre"] == pytest.approx(2.0)
    assert stats.loc["resin_age", "spread"] == pytest.approx(100.0)


def test_z_scores_known_answer_and_orientation(tiny, cfg):
    out, _ = standardise(tiny, cfg)
    last = out[out["resin_age"] == 103].iloc[0]
    # Pressures: higher is worse -> z = (x - centre) / spread
    assert last["z_pt_80_percentile"] == pytest.approx((5.0 - 2.0) / 1.0)          # +3
    assert last["z_max_pressure_displacement"] == pytest.approx(0.0)
    # Flows: lower is worse -> z = (centre - x) / spread, so a flow DROP gives a POSITIVE z
    assert last["z_elution_flow_median"] == pytest.approx((101.0 - 98.0) / 1.0)    # +3
    assert last["z_flow_median_displacement"] == pytest.approx(0.0)
    # Age: (103 - 2) / 100
    assert last["z_resin_age"] == pytest.approx(1.01)


@pytest.mark.parametrize("feature, worse_change", [
    ("pt_80_percentile", +1.0),
    ("max_pressure_displacement", +1.0),
    ("elution_flow_median", -1.0),
    ("flow_median_displacement", -1.0),
    ("resin_age", +50),
])
def test_each_feature_moves_up_when_it_gets_worse(tiny, cfg, feature, worse_change):
    worse = tiny.copy()
    last = worse.index[-1]                      # the late cycle (age 103)
    worse.loc[last, feature] += worse_change
    base_z, _ = standardise(tiny, cfg)
    worse_z, _ = standardise(worse, cfg)
    assert worse_z.loc[last, f"z_{feature}"] > base_z.loc[last, f"z_{feature}"]


def test_each_type_uses_its_own_reference(cfg):
    # type_B runs at 10x the pressure. Each type's own fresh cycles should give z around 0.
    rows = [cycle(a, p, 2.0 + a, 100.0 + a, 50.0 + a, ctype="type_A") for a, p in [(1, 1.0), (2, 2.0), (3, 3.0)]]
    rows += [cycle(a, p, 20.0 + a, 10.0 + a, 5.0 + a, batch="GB-Y01", ctype="type_B")
             for a, p in [(1, 10.0), (2, 20.0), (3, 30.0)]]
    out, stats = standardise(prepare_cycles(pd.DataFrame(rows)), cfg)
    mid = out[out["resin_age"] == 2]
    assert np.allclose(mid["z_pt_80_percentile"], 0.0)
    assert set(stats["chromatography_type"]) == {"type_A", "type_B"}


def test_missing_value_gives_missing_z_not_zero(tiny, cfg):
    tiny = tiny.copy()
    tiny.loc[tiny["resin_age"] == 103, "elution_flow_median"] = np.nan
    out, _ = standardise(tiny, cfg)
    assert np.isnan(out.loc[out["resin_age"] == 103, "z_elution_flow_median"].iloc[0])


def test_outlier_is_capped(tiny, cfg):
    tiny = tiny.copy()
    tiny.loc[tiny["resin_age"] == 103, "pt_80_percentile"] = 1000.0
    out, _ = standardise(tiny, cfg)
    assert out.loc[out["resin_age"] == 103, "z_pt_80_percentile"].iloc[0] == cfg.z_clip


def test_robust_method_ignores_a_reference_outlier(cfg):
    from dataclasses import replace
    rows = [cycle(a, p, 2.0 + a, 100.0 + a, 50.0 + a) for a, p in
            [(1, 1.0), (2, 2.0), (3, 2.0), (4, 3.0), (5, 100.0)]]  # one spike inside the window
    tc = replace(cfg.types["type_A"], first_n_cycles=5, method="robust")
    stats = reference_stats(prepare_cycles(pd.DataFrame(rows)), tc).set_index("feature")
    assert stats.loc["pt_80_percentile", "centre"] == pytest.approx(2.0)             # median
    assert stats.loc["pt_80_percentile", "spread"] == pytest.approx(1.4826 * 1.0)   # MAD = 1


def test_all_z_columns_are_created(tiny, cfg):
    out, _ = standardise(tiny, cfg)
    for f in FEATURES:
        assert f"z_{f}" in out.columns


# ---- reference.scope: per_batch ------------------------------------------------------

def per_batch_cfg(cfg):
    from dataclasses import replace
    return replace(cfg, types={k: replace(t, scope="per_batch", first_n_cycles=5) for k, t in cfg.types.items()})


def two_batches():
    """Same type; batch Y starts 1.0 bar higher than batch X. Both then rise by the same amount."""
    rows = []
    for batch, offset in [("GB-X01", 0.0), ("GB-Y01", 1.0)]:
        for a in range(1, 6):
            rows.append(cycle(a, 1.0 + offset + 0.1 * (a % 2), 2.0 + 0.1 * a, 100.0 + a, 50.0 + a, batch=batch))
        rows.append(cycle(50, 1.5 + offset, 2.3, 103.0, 53.0, batch=batch))
    return prepare_cycles(pd.DataFrame(rows))


def test_pooled_scope_penalises_a_batch_that_starts_high(cfg):
    from dataclasses import replace
    pooled = replace(cfg, types={k: replace(t, first_n_cycles=5) for k, t in cfg.types.items()})
    out, _ = standardise(two_batches(), pooled)
    late = out[out["resin_age"] == 50].set_index("gel_batch")["z_pt_80_percentile"]
    assert late["GB-Y01"] > late["GB-X01"]


def test_per_batch_scope_compares_each_batch_with_its_own_start(cfg):
    out, stats = standardise(two_batches(), per_batch_cfg(cfg))
    late = out[out["resin_age"] == 50].set_index("gel_batch")["z_pt_80_percentile"]
    assert late["GB-Y01"] == pytest.approx(late["GB-X01"])   # same rise -> same z
    pt = stats[stats["feature"] == "pt_80_percentile"].set_index("gel_batch")
    assert pt.loc["GB-Y01", "centre"] - pt.loc["GB-X01", "centre"] == pytest.approx(1.0)
    assert pt.loc["GB-Y01", "spread"] == pt.loc["GB-X01", "spread"]  # spread stays pooled
    assert set(pt["source"]) == {"batch"}


def test_per_batch_falls_back_to_pooled_when_a_batch_has_no_early_cycles(cfg):
    df = two_batches()
    late_only = df[df["gel_batch"] == "GB-X01"].assign(gel_batch="GB-Z01")
    late_only = late_only[late_only["resin_age"] == 50]               # history starts at cycle 50
    out, stats = standardise(pd.concat([df, late_only], ignore_index=True), per_batch_cfg(cfg))
    z = stats[(stats["gel_batch"] == "GB-Z01") & (stats["feature"] == "pt_80_percentile")].iloc[0]
    assert z["source"] == "pooled_fallback"


def test_unknown_scope_is_rejected():
    from gel_performance.config import ConfigError
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["chromatography_types"]["type_A"]["reference"]["scope"] = "global"
    with pytest.raises(ConfigError, match="scope"):
        parse_config(raw)
