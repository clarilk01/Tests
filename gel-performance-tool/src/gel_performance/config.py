"""Load and check config/config.yaml.

The rest of the code never reads the YAML directly. It receives a `Config` object whose values
have already been checked, so a typo in the config fails loudly at the start of a run instead of
silently producing wrong decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# The two process KPIs the tool uses, plus gel age.
KPIS = ["elution_flow_median", "pt_80_percentile"]
PRESSURE_KPI = "pt_80_percentile"
INPUT_FEATURES = ["resin_age", *KPIS]

# Columns produced by features.py and fed to the decision tree (always in this order).
MODEL_FEATURES = ["z_resin_age", "z_rm_elution_flow_median", "z_rm_pt_80_percentile"]

DIRECTIONS = {"higher_is_worse": +1, "lower_is_worse": -1}
METHODS = {"mean_std", "robust"}
SCOPES = {"pooled", "per_batch"}


class ConfigError(ValueError):
    """Raised when config.yaml is missing something or contains an invalid value."""


@dataclass(frozen=True)
class TypeConfig:
    name: str
    first_n_cycles: int
    method: str
    scope: str
    directions: dict[str, str]          # KPI -> "higher_is_worse" / "lower_is_worse"
    resin_age_scale_cycles: float
    pressure_limit_bar: float

    def sign(self, kpi: str) -> int:
        """+1 if a higher raw value is worse, -1 if a lower raw value is worse."""
        return DIRECTIONS[self.directions[kpi]]


@dataclass(frozen=True)
class ModelConfig:
    file: str
    max_depth: int
    min_samples_leaf: int
    replace_horizon: int


@dataclass(frozen=True)
class Config:
    config_version: str
    rolling_window: int
    z_clip: float
    decision_threshold: float
    model: ModelConfig
    types: dict[str, TypeConfig]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    return parse_config(raw)


def parse_config(raw: dict) -> Config:
    """Build a Config from an already-parsed YAML dict (also handy for tests)."""
    try:
        d, m = raw["defaults"], raw["model"]
        cfg = Config(
            config_version=str(raw["config_version"]),
            rolling_window=int(d["rolling_window"]),
            z_clip=float(d["z_clip"]),
            decision_threshold=float(d["decision_threshold"]),
            model=ModelConfig(
                file=str(m["file"]), max_depth=int(m["max_depth"]),
                min_samples_leaf=int(m["min_samples_leaf"]), replace_horizon=int(m["replace_horizon"]),
            ),
            types={name: _parse_type(name, t) for name, t in raw["chromatography_types"].items()},
        )
    except KeyError as exc:
        raise ConfigError(f"Missing key in config: {exc}") from exc

    if cfg.rolling_window < 1:
        raise ConfigError("defaults.rolling_window must be >= 1 (1 = no rolling mean)")
    if cfg.z_clip <= 0:
        raise ConfigError("defaults.z_clip must be > 0")
    if not 0 < cfg.decision_threshold < 1:
        raise ConfigError("defaults.decision_threshold must be between 0 and 1")
    if cfg.model.max_depth < 1 or cfg.model.min_samples_leaf < 1 or cfg.model.replace_horizon < 0:
        raise ConfigError("model: max_depth and min_samples_leaf must be >= 1, replace_horizon >= 0")
    if not cfg.types:
        raise ConfigError("chromatography_types is empty")
    return cfg


def _parse_type(name: str, t: dict) -> TypeConfig:
    where = f"chromatography_types.{name}"
    kpis = t["kpis"]
    if set(kpis) != set(KPIS):
        raise ConfigError(f"{where}.kpis must define exactly {KPIS}")
    tc = TypeConfig(
        name=name,
        first_n_cycles=int(t["reference"]["first_n_cycles"]),
        method=t["reference"].get("method", "mean_std"),
        scope=t["reference"].get("scope", "pooled"),
        directions={k: v["direction"] for k, v in kpis.items()},
        resin_age_scale_cycles=float(t["resin_age_scale_cycles"]),
        pressure_limit_bar=float(t["pressure_limit_bar"]),
    )
    for kpi, direction in tc.directions.items():
        if direction not in DIRECTIONS:
            raise ConfigError(f"{where}.kpis.{kpi}.direction must be one of {list(DIRECTIONS)}")
    if tc.first_n_cycles < 2:
        raise ConfigError(f"{where}.reference.first_n_cycles must be >= 2 (need a spread)")
    if tc.method not in METHODS:
        raise ConfigError(f"{where}.reference.method must be one of {sorted(METHODS)}")
    if tc.scope not in SCOPES:
        raise ConfigError(f"{where}.reference.scope must be one of {sorted(SCOPES)}")
    if tc.resin_age_scale_cycles <= 0:
        raise ConfigError(f"{where}.resin_age_scale_cycles must be > 0")
    if tc.pressure_limit_bar <= 0:
        raise ConfigError(f"{where}.pressure_limit_bar must be > 0")
    return tc
