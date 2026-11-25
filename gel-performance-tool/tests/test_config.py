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
    assert a.sign("elution_flow_median") == -1   # lower flow is worse
    assert a.sign("pt_80_percentile") == +1      # higher pressure is worse
    assert cfg.types["type_B"].pressure_limit_bar > a.pressure_limit_bar
    assert 0 < cfg.decision_threshold < 1


@pytest.mark.parametrize("path, value, message", [
    (("defaults", "decision_threshold"), 1.5, "decision_threshold"),
    (("defaults", "rolling_window"), 0, "rolling_window"),
    (("chromatography_types", "type_A", "pressure_limit_bar"), -1, "pressure_limit_bar"),
    (("chromatography_types", "type_A", "reference", "scope"), "global", "scope"),
    (("chromatography_types", "type_A", "kpis", "pt_80_percentile", "direction"), "up", "direction"),
])
def test_invalid_values_are_rejected(raw, path, value, message):
    bad = copy.deepcopy(raw)
    node = bad
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ConfigError, match=message):
        parse_config(bad)


def test_missing_kpi_is_rejected(raw):
    bad = copy.deepcopy(raw)
    del bad["chromatography_types"]["type_A"]["kpis"]["pt_80_percentile"]
    with pytest.raises(ConfigError, match="kpis"):
        parse_config(bad)
