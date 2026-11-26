"""Step 1 - Synthetic data generator.

Creates fake chromatography cycle data that behaves roughly like a real resin ageing:
pressures creep up and flows creep down as the gel is reused, slowly at first and then
faster after a batch-specific "knee". Every number here is invented.

Outputs (in the chosen folder):
    cycles.csv            - the input the tool reads, one row per cycle
    true_replacement.csv  - per gel batch: the "true" replacement cycle known only to the simulator
    cycles_truth.csv      - per cycle: the hidden, noise-free degradation and which problems were injected

Usage:
    python -m gel_performance.simulate --out data/synthetic --seed 42
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Simulator settings (these describe the fake "world", not the tool, so they live in code)
# --------------------------------------------------------------------------------------

# Baseline (new-resin) level of each signal per chromatography type. type_B is a different
# column set-up: higher pressures, lower flows. This is why the tool must standardise per type.
TYPE_BASELINES = {
    "type_A": {
        "pt_80_percentile": 1.20,           # bar
        "max_pressure_displacement": 1.80,  # bar
        "elution_flow_median": 120.0,       # L/h
        "flow_median_displacement": 100.0,  # L/h
    },
    "type_B": {
        "pt_80_percentile": 2.60,
        "max_pressure_displacement": 3.50,
        "elution_flow_median": 60.0,
        "flow_median_displacement": 45.0,
    },
}

PRESSURE_COLS = ["pt_80_percentile", "max_pressure_displacement"]
FLOW_COLS = ["elution_flow_median", "flow_median_displacement"]
PROCESS_COLS = PRESSURE_COLS + FLOW_COLS


@dataclass(frozen=True)
class SimSettings:
    batches_per_type: int = 6
    linear_rate: float = 0.0004        # degradation per cycle before the knee (0.04 per 100 cycles)
    knee_range: tuple = (180, 300)     # cycle where wear starts to accelerate (differs per batch)
    accel: float = 2.5e-5              # extra degradation per (cycles past knee)^2
    replace_at_rise: float = 0.25      # TRUE replacement: noise-free pressure 25% above batch start
    overrun_range: tuple = (30, 80)    # cycles the batch keeps running after the true point
    flow_sensitivity: float = 0.5      # flows fall at half the relative rate that pressures rise
    noise_cv: float = 0.02             # 2% run-to-run measurement noise
    missing_rate: float = 0.03         # ~3% of cycles interrupted -> some features missing
    n_outliers_per_type: int = 8       # sensor spikes


def degradation(age: np.ndarray, knee: float, rate_factor: float, s: SimSettings) -> np.ndarray:
    """Noise-free relative degradation (0 = new). Slow linear drift, then quadratic after the knee."""
    past_knee = np.clip(age - knee, 0, None)
    return rate_factor * (s.linear_rate * age + s.accel * past_knee**2)


def first_age_reaching(target: float, knee: float, rate_factor: float, s: SimSettings) -> int:
    """First resin_age at which degradation (relative to age 1) reaches `target`."""
    ages = np.arange(1, 5000)
    d = degradation(ages, knee, rate_factor, s)
    d = d - d[0]
    return int(ages[np.argmax(d >= target)])


def simulate(seed: int = 42, s: SimSettings = SimSettings()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (cycles, true_replacement, cycles_truth) DataFrames. Same seed -> identical output."""
    rng = np.random.default_rng(seed)
    cycle_rows, batch_rows = [], []

    for type_idx, (chrom_type, base) in enumerate(TYPE_BASELINES.items()):
        letter = chrom_type[-1]  # "A" or "B"
        for b in range(1, s.batches_per_type + 1):
            gel_batch = f"GB-{letter}{b:02d}"

            # Batch-to-batch differences: each resin lot starts a bit different and ages at its own pace.
            pressure_offset = rng.normal(1.0, 0.05)   # +/- 5% starting pressure
            flow_offset = rng.normal(1.0, 0.04)       # +/- 4% starting flow
            rate_factor = rng.lognormal(0.0, 0.15)    # some lots wear faster than others
            knee = rng.uniform(*s.knee_range)

            true_age = first_age_reaching(s.replace_at_rise, knee, rate_factor, s)
            n_cycles = true_age + int(rng.integers(*s.overrun_range))
            ages = np.arange(1, n_cycles + 1)
            d = degradation(ages, knee, rate_factor, s)

            # Timestamps: each batch starts on its own date; one cycle every ~6-10 hours.
            start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 120)) + 200 * (b - 1))
            gaps_h = rng.uniform(6, 10, size=n_cycles)
            end_times = start + pd.to_timedelta(np.cumsum(gaps_h), unit="h")

            # A shared run-level noise term makes the two pressures (and the two flows) move together,
            # as they would in a real run; each signal also gets its own small independent noise.
            run_noise = rng.normal(0, s.noise_cv, size=n_cycles)
            values = {}
            for col in PRESSURE_COLS:
                own = rng.normal(0, s.noise_cv * 0.5, size=n_cycles)
                values[col] = base[col] * pressure_offset * (1 + d) * (1 + run_noise + own)
            for col in FLOW_COLS:
                own = rng.normal(0, s.noise_cv * 0.5, size=n_cycles)
                values[col] = base[col] * flow_offset * (1 - s.flow_sensitivity * d) * (1 - 0.5 * run_noise + own)

            for i in range(n_cycles):
                cycle_rows.append({
                    "gel_batch": gel_batch,
                    "chromatography_type": chrom_type,
                    "resin_age": int(ages[i]),
                    "cycle_end_time": end_times[i],
                    **{c: values[c][i] for c in PROCESS_COLS},
                    "true_degradation": d[i] - d[0],
                    "injected_issue": "",
                })
            batch_rows.append({
                "gel_batch": gel_batch,
                "chromatography_type": chrom_type,
                "knee_age": round(knee, 1),
                "true_replacement_age": true_age,
                "last_age_in_data": n_cycles,
            })

    df = pd.DataFrame(cycle_rows)
    df = _inject_missing(df, rng, s)
    df = _inject_outliers(df, rng, s)

    df = df.sort_values(["cycle_end_time", "gel_batch"], kind="mergesort").reset_index(drop=True)
    df[PROCESS_COLS] = df[PROCESS_COLS].round(4)
    df["true_degradation"] = df["true_degradation"].round(5)
    df["cycle_end_time"] = df["cycle_end_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    input_cols = ["gel_batch", "chromatography_type", "resin_age", "cycle_end_time", *PROCESS_COLS]
    truth_cols = ["gel_batch", "chromatography_type", "resin_age", "true_degradation", "injected_issue"]
    return df[input_cols], pd.DataFrame(batch_rows), df[truth_cols]


def _inject_missing(df: pd.DataFrame, rng: np.random.Generator, s: SimSettings) -> pd.DataFrame:
    """Interrupted runs: ~3% of cycles lose one or more features. Values are set to NaN, never guessed."""
    n_missing = int(round(s.missing_rate * len(df)))
    idx = rng.choice(len(df), size=n_missing, replace=False)
    patterns = [
        (["max_pressure_displacement", "flow_median_displacement"], "interrupted_before_displacement"),
        (["elution_flow_median"], "elution_flow_sensor_dropout"),
        (["pt_80_percentile"], "pressure_logging_gap"),
        (PROCESS_COLS[1:], "interrupted_early"),  # only 2 of 5 features left -> "insufficient"
    ]
    choice = rng.choice(len(patterns), size=n_missing, p=[0.55, 0.2, 0.15, 0.10])
    for row, k in zip(idx, choice):
        cols, label = patterns[k]
        df.loc[row, cols] = np.nan
        df.loc[row, "injected_issue"] = label
    return df


def _inject_outliers(df: pd.DataFrame, rng: np.random.Generator, s: SimSettings) -> pd.DataFrame:
    """A few sensor spikes: a pressure jumps 1.5-2.5x, or a flow drops to 30-50% for one cycle."""
    for chrom_type in TYPE_BASELINES:
        candidates = df.index[(df["chromatography_type"] == chrom_type) & (df["injected_issue"] == "")]
        rows = rng.choice(candidates, size=s.n_outliers_per_type, replace=False)
        for row in rows:
            if rng.random() < 0.6:
                col = str(rng.choice(PRESSURE_COLS))
                df.loc[row, col] *= rng.uniform(1.5, 2.5)
            else:
                col = str(rng.choice(FLOW_COLS))
                df.loc[row, col] *= rng.uniform(0.3, 0.5)
            df.loc[row, "injected_issue"] = f"outlier_{col}"
    return df


def write(out_dir: Path, seed: int = 42) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cycles, batches, truth = simulate(seed)
    cycles.to_csv(out_dir / "cycles.csv", index=False)
    batches.to_csv(out_dir / "true_replacement.csv", index=False)
    truth.to_csv(out_dir / "cycles_truth.csv", index=False)
    print(f"Wrote {len(cycles)} cycles for {cycles['gel_batch'].nunique()} gel batches to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic chromatography cycle data.")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write(args.out, args.seed)


if __name__ == "__main__":
    main()
