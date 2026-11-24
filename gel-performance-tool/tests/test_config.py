"""Checks that config.yaml is read correctly and that bad configs are rejected."""

import copy
from pathlib import Path

import pytest
import yaml

from gel_performance.config import ConfigError, load_config, parse_config

CONFIG_PATH = Path(__file__).parents[1] / "config" / "config.yaml"


@pytest.fixture
def raw():
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_shipped_config_loads():
    cfg = load_config(CONFIG_PATH)
    assert set(cfg.types) == {"type_A", "type_B"}
    a = cfg.types["type_A"]
    assert a.first_n_cycles == 30
    assert a.features["elution_flow_median"].sign == -1  # lower flow is worse
    assert a.features["pt_80_percentile"].sign == +1     # higher pressure is worse


def test_weights_must_add_up_to_one(raw):
    bad = copy.deepcopy(raw)
    bad["chromatography_types"]["type_A"]["features"]["resin_age"]["weight"] = 0.5
    with pytest.raises(ConfigError, match="add up to 1.0"):
        parse_config(bad)


def test_thresholds_must_be_in_order(raw):
    bad = copy.deepcopy(raw)
    bad["chromatography_types"]["type_B"]["thresholds"] = {"ok_min": 40, "watch_min": 70}
    with pytest.raises(ConfigError, match="watch_min < ok_min"):
        parse_config(bad)


def test_unknown_direction_is_rejected(raw):
    bad = copy.deepcopy(raw)
    bad["chromatography_types"]["type_A"]["features"]["pt_80_percentile"]["direction"] = "up"
    with pytest.raises(ConfigError, match="direction"):
        parse_config(bad)


def test_missing_feature_is_rejected(raw):
    bad = copy.deepcopy(raw)
    del bad["chromatography_types"]["type_A"]["features"]["pt_80_percentile"]
    with pytest.raises(ConfigError, match="missing"):
        parse_config(bad)
