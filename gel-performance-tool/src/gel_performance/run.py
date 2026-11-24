"""Command-line entry point. Safe to schedule (e.g. hourly).

    python -m gel_performance.run --input data/synthetic/cycles.csv --output output/ [--config config/config.yaml]

Writes to the output folder:
    cycle_scores.csv     one row per cycle: score, recommendation, contributions, reason
    batch_summary.csv    latest status per gel batch
    reference_stats.csv  the early-life baseline (centre, spread) used for each type and feature
    run.log              appended on every run: what was read, counts, warnings and errors

Idempotent: the CSV files depend only on the input and the config (no timestamps inside), and the
same input always produces byte-identical files. Each file is written to a temporary name first and
then renamed, so a crash never leaves a half-written file behind.

Exit codes: 0 = success, 1 = config problem, 2 = input data problem, 3 = unexpected error.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

from . import __version__
from .config import ConfigError, load_config
from .data import DataError, load_cycles
from .features import ReferenceWindowError, standardise
from .report import build_batch_summary, build_cycle_table
from .scoring import NO_SCORE, OK, REPLACE, WATCH, score_cycles

log = logging.getLogger("gel_performance")

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
EXIT_OK, EXIT_CONFIG, EXIT_DATA, EXIT_UNEXPECTED = 0, 1, 2, 3


def run(input_path: Path, output_dir: Path, config_path: Path) -> dict[str, pd.DataFrame]:
    """Run the whole pipeline and write the outputs. Returns the tables (handy for tests)."""
    cfg = load_config(config_path)
    log.info("Config %s (config_version %s)", config_path, cfg.config_version)

    cycles = load_cycles(input_path, known_types=set(cfg.types))
    if cycles.empty:
        raise DataError("No usable cycles in the input")
    standardised, ref_stats = standardise(cycles, cfg)
    scored = score_cycles(standardised, cfg)

    tables = {
        "cycle_scores.csv": build_cycle_table(scored, cfg),
        "batch_summary.csv": None,
        "reference_stats.csv": ref_stats.round(6),
    }
    tables["batch_summary.csv"] = build_batch_summary(tables["cycle_scores.csv"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        _write_atomic(table, output_dir / name)
        log.info("Wrote %s (%d rows)", output_dir / name, len(table))

    rec = tables["cycle_scores.csv"]["recommendation"].value_counts()
    summary = tables["batch_summary.csv"]
    log.info(
        "Cycle recommendations: %d OK, %d WATCH, %d REPLACE SOON, %d NO SCORE",
        rec.get(OK, 0), rec.get(WATCH, 0), rec.get(REPLACE, 0), rec.get(NO_SCORE, 0),
    )
    for _, b in summary.iterrows():
        log.info("Batch %s (%s): latest cycle %d -> %s (smoothed score %.1f)",
                 b["gel_batch"], b["chromatography_type"], b["latest_resin_age"],
                 b["recommendation"], b["smoothed_score"])
    return tables


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)  # atomic on the same file system


def _file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _setup_logging(output_dir: Path) -> list[logging.Handler]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    console = logging.StreamHandler(sys.stderr)
    for h in (file_handler, console):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return [file_handler, console]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score chromatography gel performance per cycle.")
    parser.add_argument("--input", required=True, type=Path, help="CSV with one row per cycle")
    parser.add_argument("--output", required=True, type=Path, help="folder for the output files")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="config.yaml")
    args = parser.parse_args(argv)

    handlers = _setup_logging(args.output)
    started = time.perf_counter()
    try:
        log.info("=== Run started (gel_performance %s) ===", __version__)
        if args.input.exists():
            log.info("Input %s (sha256 %s)", args.input, _file_fingerprint(args.input))
        run(args.input, args.output, args.config)
        log.info("=== Run finished OK in %.1f s ===", time.perf_counter() - started)
        return EXIT_OK
    except ConfigError as exc:
        log.error("Config problem: %s", exc)
        return EXIT_CONFIG
    except (DataError, ReferenceWindowError) as exc:
        log.error("Input data problem: %s", exc)
        return EXIT_DATA
    except Exception:  # noqa: BLE001 - a scheduled job must log every failure, not crash silently
        log.exception("Unexpected error")
        return EXIT_UNEXPECTED
    finally:
        for h in handlers:
            log.removeHandler(h)
            h.close()


if __name__ == "__main__":
    sys.exit(main())
