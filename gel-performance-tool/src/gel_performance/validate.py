"""Step 6 - Compare the recommendations with the "true" replacement point built into the simulator.

This module is the only place that reads the simulator's hidden truth (true_replacement.csv).
The scoring tool itself never sees it.

Definitions (judgement calls, documented in the README):
  True class of a cycle, from cycles_to_go = true_replacement_age - resin_age:
      cycles_to_go <= REPLACE_HORIZON (25)               -> REPLACE SOON  (includes cycles past the true point)
      REPLACE_HORIZON < cycles_to_go <= WATCH_HORIZON (75) -> WATCH
      otherwise                                           -> OK
  Batch split (fixed, by batch number): odd-numbered batches calibrate the thresholds,
  even-numbered batches are held out and used only to report agreement.
  Calibration target: balanced accuracy (mean recall over the three classes), so the many
  OK cycles cannot dominate.

The smoothed score does not depend on the thresholds, so candidate thresholds can be tried
directly on the run output without re-running the pipeline.

Usage:
    python -m gel_performance.validate --scores output/cycle_scores.csv \\
        --truth data/synthetic/true_replacement.csv --config config/config.yaml [--calibrate]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .scoring import NO_SCORE, OK, REPLACE, WATCH

REPLACE_HORIZON = 25
WATCH_HORIZON = 75
CLASSES = [OK, WATCH, REPLACE]
TOO_EARLY = 100  # a first REPLACE SOON more than this many cycles early counts as "too early"


@dataclass(frozen=True)
class Thresholds:
    ok_min: float
    watch_min: float


def batch_role(gel_batch: str) -> str:
    """'calibration' for odd-numbered batches (GB-A01, GB-A03 ...), 'holdout' for even ones."""
    return "calibration" if int(gel_batch[-2:]) % 2 == 1 else "holdout"


def true_class(cycles_to_go: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([cycles_to_go <= REPLACE_HORIZON, cycles_to_go <= WATCH_HORIZON], [REPLACE, WATCH], OK),
        index=cycles_to_go.index,
    )


def classify(smoothed: pd.Series, t: Thresholds) -> pd.Series:
    out = pd.Series(np.select([smoothed >= t.ok_min, smoothed >= t.watch_min], [OK, WATCH], REPLACE),
                    index=smoothed.index)
    return out.where(smoothed.notna(), NO_SCORE)


def attach_truth(scores: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    df = scores.merge(truth[["gel_batch", "true_replacement_age"]], on="gel_batch", how="left")
    if df["true_replacement_age"].isna().any():
        raise ValueError("Some gel batches have no true replacement point")
    df["cycles_to_go"] = df["true_replacement_age"] - df["resin_age"]
    df["true_class"] = true_class(df["cycles_to_go"])
    df["role"] = df["gel_batch"].map(batch_role)
    return df


def balanced_accuracy(true: pd.Series, pred: pd.Series) -> float:
    recalls = [(pred[true == c] == c).mean() for c in CLASSES if (true == c).any()]
    return float(np.mean(recalls))


def calibrate(df: pd.DataFrame, step: float = 1.0) -> Thresholds:
    """Grid-search (watch_min < ok_min) maximising balanced accuracy. df: ONE type, calibration batches."""
    d = df[df["smoothed_score"].notna()]
    best, best_t = -1.0, None
    for watch in np.arange(step, 100, step):
        for ok in np.arange(watch + step, 100 + step / 2, step):
            ba = balanced_accuracy(d["true_class"], classify(d["smoothed_score"], Thresholds(ok, watch)))
            if ba > best + 1e-12:  # first best in a fixed order -> deterministic
                best, best_t = ba, Thresholds(float(ok), float(watch))
    return best_t


def evaluate(df: pd.DataFrame, t: Thresholds) -> dict:
    """Cycle-level and batch-level agreement for cycles in df, using thresholds t."""
    scored = df[df["smoothed_score"].notna()].copy()
    scored["pred"] = classify(scored["smoothed_score"], t)
    confusion = pd.crosstab(
        pd.Categorical(scored["true_class"], CLASSES), pd.Categorical(scored["pred"], CLASSES),
        rownames=["true"], colnames=["predicted"], dropna=False,
    )
    batches = []
    for b, g in scored.sort_values("resin_age").groupby("gel_batch"):
        first = g.loc[g["pred"] == REPLACE, "resin_age"]
        true_age = int(g["true_replacement_age"].iloc[0])
        first_age = int(first.iloc[0]) if len(first) else None
        lead = None if first_age is None else true_age - first_age
        batches.append({
            "gel_batch": b, "true_replacement_age": true_age, "first_replace_soon_age": first_age,
            "warning_cycles_before_true_point": lead,
            "outcome": ("missed" if first_age is None else "late" if lead < 0
                        else "too early" if lead > TOO_EARLY else "in time"),
        })
    return {
        "n_cycles": len(scored),
        "n_no_score": int(df["smoothed_score"].isna().sum()),
        "agreement": float((scored["pred"] == scored["true_class"]).mean()),
        "balanced_accuracy": balanced_accuracy(scored["true_class"], scored["pred"]),
        "confusion": confusion,
        "batches": pd.DataFrame(batches),
    }


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Small markdown table writer (avoids an extra dependency)."""
    if index:
        df = df.reset_index()
    cells = [[str(c) for c in df.columns]] + [["" if pd.isna(v) else str(v) for v in r] for r in df.to_numpy()]
    rows = ["| " + " | ".join(r) + " |" for r in cells]
    rows.insert(1, "|" + "---|" * len(df.columns))
    return "\n".join(rows)


def report(scores_path: Path, truth_path: Path, config_path: Path, do_calibrate: bool,
           baseline: Thresholds = Thresholds(70, 40)) -> str:
    cfg = load_config(config_path)
    df = attach_truth(pd.read_csv(scores_path), pd.read_csv(truth_path))
    lines = [f"# Validation report (synthetic data, config_version {cfg.config_version})", ""]
    for type_name, tc in cfg.types.items():
        d = df[df["chromatography_type"] == type_name]
        current = Thresholds(tc.ok_min, tc.watch_min)
        lines.append(f"## {type_name}")
        if do_calibrate:
            t = calibrate(d[d["role"] == "calibration"])
            lines.append(f"Calibrated on {sorted(d.loc[d['role'] == 'calibration', 'gel_batch'].unique())}: "
                         f"ok_min = {t.ok_min:.0f}, watch_min = {t.watch_min:.0f}")
            lines.append("")
        hold = d[d["role"] == "holdout"]
        for label, t in [("config thresholds", current), ("placeholder 70 / 40", baseline)]:
            r = evaluate(hold, t)
            lines += [
                f"### Held-out batches {sorted(hold['gel_batch'].unique())} - {label} "
                f"(ok_min {t.ok_min:.0f}, watch_min {t.watch_min:.0f})",
                f"- cycles compared: {r['n_cycles']} (no score: {r['n_no_score']})",
                f"- cycle-level agreement: {r['agreement']:.1%}",
                f"- balanced accuracy: {r['balanced_accuracy']:.1%}",
                "", "Confusion (rows = true class, columns = predicted):", "",
                md_table(r["confusion"].rename_axis("true \\ predicted"), index=True), "",
                md_table(r["batches"].astype(object)), "",
            ]
            if label == "config thresholds" and current == baseline:
                break
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate recommendations against the simulator's truth.")
    parser.add_argument("--scores", type=Path, default=Path("output/cycle_scores.csv"))
    parser.add_argument("--truth", type=Path, default=Path("data/synthetic/true_replacement.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--calibrate", action="store_true", help="also grid-search thresholds on calibration batches")
    parser.add_argument("--out", type=Path, help="optional markdown file for the report")
    args = parser.parse_args()
    text = report(args.scores, args.truth, args.config, args.calibrate)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
