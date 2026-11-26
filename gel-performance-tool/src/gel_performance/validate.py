"""Compare the GO / NO-GO decisions with the "true" replacement point built into the simulator.

Only this module and train.py read the simulator's hidden truth. Scoring never sees it.

Truth per cycle: "replace" if the true replacement point is <= replace_horizon (config, 25) cycles away
or already passed. Results are reported separately for:
  * calibration batches (odd numbers) - the tree was TRAINED on these, so results are optimistic;
  * held-out batches (even numbers)   - never seen by the tree: the honest result.

Usage:
    python -m gel_performance.validate --out docs/validation_report.md
    python -m gel_performance.validate --compare-scopes --out docs/scope_comparison.md
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import load_cycles
from .scoring import NO_GO
from .train import ROOT, label_replace, train

TOO_EARLY = 100  # a first NO-GO more than this many cycles before the true point counts as "too early"


def evaluate(df: pd.DataFrame) -> dict:
    """df: cycle table joined with truth (label_replace). Binary comparison NO-GO vs true 'replace'."""
    scored = df[df["decision"] != "NO SCORE"]
    truth = scored["label_replace"] == 1
    pred = scored["decision"] == NO_GO
    tp, fn = int((truth & pred).sum()), int((truth & ~pred).sum())
    fp, tn = int((~truth & pred).sum()), int((~truth & ~pred).sum())
    confusion = pd.DataFrame({"predicted GO": [tn, fn], "predicted NO-GO": [fp, tp]},
                             index=["true: keep", "true: replace"])
    batches = []
    for b, g in scored.sort_values("resin_age").groupby("gel_batch"):
        true_age = int(g["true_replacement_age"].iloc[0])
        first = g.loc[g["decision"] == NO_GO, "resin_age"]
        first_age = int(first.iloc[0]) if len(first) else None
        lead = None if first_age is None else true_age - first_age
        batches.append({
            "gel_batch": b, "true_replacement_age": true_age, "first_no_go_age": first_age,
            "warning_cycles_before_true_point": lead,
            "outcome": ("missed" if first_age is None else "late" if lead < 0
                        else "too early" if lead > TOO_EARLY else "in time"),
        })
    recall_replace = tp / (tp + fn) if tp + fn else np.nan
    recall_keep = tn / (tn + fp) if tn + fp else np.nan
    return {
        "n_cycles": len(scored), "n_no_score": int(len(df) - len(scored)),
        "agreement": (tp + tn) / len(scored) if len(scored) else np.nan,
        "balanced_accuracy": np.nanmean([recall_replace, recall_keep]),
        "replace_caught": recall_replace,
        "no_go_correct": tp / (tp + fp) if tp + fp else np.nan,
        "confusion": confusion, "batches": pd.DataFrame(batches),
    }


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Small markdown table writer (avoids an extra dependency)."""
    if index:
        df = df.reset_index()
    cells = [[str(c) for c in df.columns]] + [["" if pd.isna(v) else str(v) for v in r] for r in df.to_numpy()]
    rows = ["| " + " | ".join(r) + " |" for r in cells]
    rows.insert(1, "|" + "---|" * len(df.columns))
    return "\n".join(rows)


def _section(title: str, r: dict) -> list[str]:
    return [
        f"### {title}",
        f"- cycles compared: {r['n_cycles']} (no score: {r['n_no_score']})",
        f"- cycle-level agreement: {r['agreement']:.1%}",
        f"- balanced accuracy: {r['balanced_accuracy']:.1%}",
        f"- true 'replace' cycles caught by NO-GO: {r['replace_caught']:.1%}",
        f"- NO-GO decisions that were right: {r['no_go_correct']:.1%}",
        "", md_table(r["confusion"].astype(object).rename_axis("truth"), index=True), "",
        md_table(r["batches"].astype(object)), "",
    ]


def report(scores_path: Path, truth_path: Path, config_path: Path) -> str:
    cfg = load_config(config_path)
    df = label_replace(pd.read_csv(scores_path), pd.read_csv(truth_path), cfg.model.replace_horizon)
    lines = [f"# Validation report (synthetic data, config_version {cfg.config_version})", "",
             f"Truth: 'replace' = true replacement point <= {cfg.model.replace_horizon} cycles away or passed.", ""]
    for type_name in cfg.types:
        d = df[df["chromatography_type"] == type_name]
        lines.append(f"## {type_name}")
        lines += _section("Held-out batches (never seen by the tree) - the honest result",
                          evaluate(d[d["role"] == "holdout"]))
        lines += _section("Calibration batches (the tree was trained on these - optimistic)",
                          evaluate(d[d["role"] == "calibration"]))
    return "\n".join(lines)


def compare_scopes(input_path: Path, truth_path: Path, config_path: Path) -> str:
    """Train and evaluate the whole method once with a pooled and once with a per-batch baseline."""
    from .run import build_tables

    base = load_config(config_path)
    truth = pd.read_csv(truth_path)
    cycles = load_cycles(input_path, known_types=set(base.types))
    summary, detail = [], []
    for scope in ["pooled", "per_batch"]:
        cfg = replace(base, types={k: replace(t, scope=scope) for k, t in base.types.items()})
        model = train(cycles, truth, cfg)
        table = build_tables(cycles, cfg, model)["cycle_scores.csv"]
        df = label_replace(table, truth, cfg.model.replace_horizon)
        for type_name in cfg.types:
            d = df[(df["chromatography_type"] == type_name) & (df["role"] == "holdout")]
            r = evaluate(d)
            summary.append({
                "type": type_name, "scope": scope,
                "agreement": f"{r['agreement']:.1%}", "balanced accuracy": f"{r['balanced_accuracy']:.1%}",
                "batches in time": f"{(r['batches']['outcome'] == 'in time').sum()} of {len(r['batches'])}",
            })
            detail.append(r["batches"].assign(scope=scope))
    detail = pd.concat(detail)
    lead = detail.pivot_table(index="gel_batch", columns="scope", values="warning_cycles_before_true_point")
    outcome = detail.pivot_table(index="gel_batch", columns="scope", values="outcome", aggfunc="first")
    per_batch = pd.DataFrame({
        "true replacement": detail.groupby("gel_batch")["true_replacement_age"].first(),
        "warning, pooled": lead["pooled"].astype("Int64").astype(str) + " (" + outcome["pooled"] + ")",
        "warning, per_batch": lead["per_batch"].astype("Int64").astype(str) + " (" + outcome["per_batch"] + ")",
    })
    return "\n".join([
        "# Reference scope comparison (synthetic data, held-out batches only)", "",
        "For each scope the tree is re-trained on the calibration batches and evaluated on the held-out ones.", "",
        md_table(pd.DataFrame(summary)), "",
        "Warning = cycles between the first NO-GO and the true replacement point (negative = late).", "",
        md_table(per_batch.rename_axis("gel_batch"), index=True), "",
    ])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate GO / NO-GO decisions against the simulator's truth.")
    parser.add_argument("--scores", type=Path, default=ROOT / "output/cycle_scores.csv")
    parser.add_argument("--truth", type=Path, default=ROOT / "data/synthetic/true_replacement.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument("--input", type=Path, default=ROOT / "data/synthetic/cycles.csv")
    parser.add_argument("--compare-scopes", action="store_true",
                        help="re-train and re-score with pooled and per_batch baselines and compare them")
    parser.add_argument("--out", type=Path, help="optional markdown file for the report")
    args = parser.parse_args(argv)
    if args.compare_scopes:
        text = compare_scopes(args.input, args.truth, args.config)
    else:
        text = report(args.scores, args.truth, args.config)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
