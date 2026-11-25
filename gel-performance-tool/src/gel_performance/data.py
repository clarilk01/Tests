"""Read and check the cycle data.

Rules:
  * Required columns must exist, otherwise the run stops with a clear error.
  * Missing or non-numeric feature values become NaN and are recorded in `missing_features`.
    They are NEVER filled in with a guess.
  * Rows that cannot be used at all (unknown chromatography type, no gel_batch, duplicate cycle)
    are removed and counted in the log, so nothing disappears silently.
  * Rows are sorted the same way every time, which keeps runs reproducible.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import INPUT_FEATURES as FEATURES

log = logging.getLogger(__name__)

ID_COLUMNS = ["gel_batch", "chromatography_type", "cycle_end_time"]
REQUIRED_COLUMNS = ID_COLUMNS + FEATURES  # other columns in the file are ignored
SORT_KEYS = ["chromatography_type", "gel_batch", "resin_age", "cycle_end_time"]


class DataError(ValueError):
    """Raised when the input file cannot be used at all."""


def load_cycles(path: str | Path, known_types: set[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataError(f"Input file not found: {path}")
    df = pd.read_csv(path, dtype={"gel_batch": "string", "chromatography_type": "string"})
    log.info("Read %d rows from %s", len(df), path)
    return prepare_cycles(df, known_types)


def prepare_cycles(df: pd.DataFrame, known_types: set[str] | None = None) -> pd.DataFrame:
    """Check and tidy an already-loaded DataFrame (used by load_cycles and by tests)."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataError(f"Input is missing required columns: {missing_cols}")

    df = df[REQUIRED_COLUMNS].copy()
    df["gel_batch"] = df["gel_batch"].astype("string").str.strip()
    df["chromatography_type"] = df["chromatography_type"].astype("string").str.strip()
    df["cycle_end_time"] = pd.to_datetime(df["cycle_end_time"], errors="coerce")

    # Features must be numbers. Text such as "n/a" becomes NaN (= missing) and is logged.
    for col in FEATURES:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        newly_missing = df[col].isna().sum() - before
        if newly_missing:
            log.warning("%d non-numeric value(s) in %s treated as missing", newly_missing, col)

    # Rows we cannot place anywhere are dropped, and counted.
    no_id = df["gel_batch"].isna() | df["chromatography_type"].isna()
    if no_id.any():
        log.error("Dropped %d row(s) without gel_batch or chromatography_type", no_id.sum())
        df = df[~no_id]
    if known_types is not None:
        unknown = ~df["chromatography_type"].isin(known_types)
        if unknown.any():
            names = sorted(df.loc[unknown, "chromatography_type"].unique())
            log.error("Dropped %d row(s) with chromatography_type not in config: %s", unknown.sum(), names)
            df = df[~unknown]
    if df["cycle_end_time"].isna().any():
        log.warning("%d row(s) have an unreadable cycle_end_time", df["cycle_end_time"].isna().sum())

    # A cycle is identified by (gel_batch, resin_age). Keep the latest record of a duplicate.
    df = df.sort_values(SORT_KEYS, kind="mergesort", na_position="last")
    dup = df.duplicated(subset=["gel_batch", "resin_age"], keep="last") & df["resin_age"].notna()
    if dup.any():
        log.warning("Dropped %d duplicate cycle(s) (same gel_batch and resin_age)", dup.sum())
        df = df[~dup]

    # Record which features are missing per cycle; the values themselves stay NaN.
    missing = df[FEATURES].isna()
    df["n_features_available"] = (~missing).sum(axis=1).astype(int)
    df["missing_features"] = missing.apply(lambda r: ",".join(c for c in FEATURES if r[c]), axis=1)
    df["incomplete"] = missing.any(axis=1)

    log.info(
        "Prepared %d cycles across %d gel batches; %d incomplete (%.1f%%)",
        len(df), df["gel_batch"].nunique(), df["incomplete"].sum(),
        100 * df["incomplete"].mean() if len(df) else 0.0,
    )
    return df.reset_index(drop=True)
