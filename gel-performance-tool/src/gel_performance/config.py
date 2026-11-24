"""Load and check config/config.yaml.

The rest of the code never reads the YAML directly. It receives a `Config` object whose values
have already been checked, so a typo in the config fails loudly at the start of a run instead of
silently producing wrong scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

# The five features every chromatography type must define (same order everywhere in the tool).
FEATURES = [
    "pt_80_percentile",
    "max_pressure_displacement",
    "elution_flow_median",
    "flow_median_displacement",
    "resin_age",
]
DIRECTIONS = {"higher_is_worse": +1, "lower_is_worse": -1}
METHODS = {"mean_std", "robust"}
SCOPES = {"pooled", "per_batch"}


class ConfigError(ValueError):
    """Raised when config.yaml is missing something or contains an invalid value."""


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    direction: str
    weight: float
    scale_cycles: float | None = None  # only used for resin_age

    @property
    def sign(self) -> int:
        """+1 if a higher raw value is worse, -1 if a lower raw value is worse."""
        return DIRECTIONS[self.direction]


@dataclass(frozen=True)
class TypeConfig:
    name: str
    first_n_cycles: int
    method: str
    features: dict[str, FeatureSpec]
    composite_at_zero: float
    ok_min: float
    watch_min: float
    scope: str = "pooled"  # pooled: one baseline per type ; per_batch: each batch's own early cycles


@dataclass(frozen=True)
class Config:
    config_version: str
    z_clip: float
    min_features_for_score: int
    smoothing_window: int
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
        defaults = raw["defaults"]
        cfg = Config(
            config_version=str(raw["config_version"]),
            z_clip=float(defaults["z_clip"]),
            min_features_for_score=int(defaults["min_features_for_score"]),
            smoothing_window=int(defaults["smoothing_window"]),
            types={name: _parse_type(name, t) for name, t in raw["chromatography_types"].items()},
        )
    except KeyError as exc:
        raise ConfigError(f"Missing key in config: {exc}") from exc

    if cfg.z_clip <= 0:
        raise ConfigError("defaults.z_clip must be > 0")
    if not 1 <= cfg.min_features_for_score <= len(FEATURES):
        raise ConfigError(f"defaults.min_features_for_score must be between 1 and {len(FEATURES)}")
    if cfg.smoothing_window < 1:
        raise ConfigError("defaults.smoothing_window must be >= 1 (1 = no smoothing)")
    if not cfg.types:
        raise ConfigError("chromatography_types is empty")
    return cfg


def _parse_type(name: str, t: dict) -> TypeConfig:
    where = f"chromatography_types.{name}"

    features = {}
    for fname, spec in t["features"].items():
        features[fname] = FeatureSpec(
            name=fname,
            direction=spec["direction"],
            weight=float(spec["weight"]),
            scale_cycles=float(spec["scale_cycles"]) if "scale_cycles" in spec else None,
        )

    tc = TypeConfig(
        name=name,
        first_n_cycles=int(t["reference"]["first_n_cycles"]),
        method=t["reference"].get("method", "mean_std"),
        features=features,
        composite_at_zero=float(t["score"]["composite_at_zero"]),
        ok_min=float(t["thresholds"]["ok_min"]),
        watch_min=float(t["thresholds"]["watch_min"]),
        scope=t["reference"].get("scope", "pooled"),
    )

    if set(features) != set(FEATURES):
        missing, extra = set(FEATURES) - set(features), set(features) - set(FEATURES)
        raise ConfigError(f"{where}.features: missing {sorted(missing)}, unexpected {sorted(extra)}")
    for f in features.values():
        if f.direction not in DIRECTIONS:
            raise ConfigError(f"{where}.features.{f.name}.direction must be one of {list(DIRECTIONS)}")
        if f.weight < 0:
            raise ConfigError(f"{where}.features.{f.name}.weight must be >= 0")
    total = sum(f.weight for f in features.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ConfigError(f"{where}: feature weights must add up to 1.0 (they add up to {total:.4f})")
    if not features["resin_age"].scale_cycles or features["resin_age"].scale_cycles <= 0:
        raise ConfigError(f"{where}.features.resin_age.scale_cycles must be > 0")
    if tc.first_n_cycles < 2:
        raise ConfigError(f"{where}.reference.first_n_cycles must be >= 2 (need a spread)")
    if tc.method not in METHODS:
        raise ConfigError(f"{where}.reference.method must be one of {sorted(METHODS)}")
    if tc.scope not in SCOPES:
        raise ConfigError(f"{where}.reference.scope must be one of {sorted(SCOPES)}")
    if tc.composite_at_zero <= 0:
        raise ConfigError(f"{where}.score.composite_at_zero must be > 0")
    if not 0 <= tc.watch_min < tc.ok_min <= 100:
        raise ConfigError(f"{where}.thresholds: need 0 <= watch_min < ok_min <= 100")
    return tc
