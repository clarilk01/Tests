"""Train the GO / NO-GO decision tree on the CALIBRATION batches and save it as readable JSON.

    python -m gel_performance.train --input data/synthetic/cycles.csv \\
        --truth data/synthetic/true_replacement.csv --config config/config.yaml

What it learns from:
  * inputs  = the three standardised features (gel age, rolling elution flow, rolling elution pressure)
  * label   = 1 ("replace") if the simulator's true replacement point is <= replace_horizon cycles
              away or already passed, otherwise 0. The truth only exists because the data is synthetic.
  * batches = odd-numbered gel batches only (GB-A01, A03, A05, B01, B03, B05). The even-numbered batches
              are held out so validate.py can report honest results on batches the tree never saw.

Writes models/gonogo_tree.json (the tree) and docs/tree_rules.txt (the same tree as if/else text).
Training is deterministic (fixed random_state), so re-training on the same data gives the same file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import MODEL_FEATURES, Config, load_config
from .data import load_cycles
from .features import standardise
from .model import rules_text, save_model, tree_to_dict

ROOT = Path(__file__).resolve().parents[2]


def batch_role(gel_batch: str) -> str:
    """'calibration' for odd-numbered batches (GB-A01, GB-A03 ...), 'holdout' for even ones."""
    return "calibration" if int(gel_batch[-2:]) % 2 == 1 else "holdout"


def label_replace(df: pd.DataFrame, truth: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Attach the true replacement point, cycles_to_go, the 0/1 label and the batch role."""
    out = df.merge(truth[["gel_batch", "true_replacement_age"]], on="gel_batch", how="left")
    if out["true_replacement_age"].isna().any():
        raise ValueError("Some gel batches have no true replacement point")
    out["cycles_to_go"] = out["true_replacement_age"] - out["resin_age"]
    out["label_replace"] = (out["cycles_to_go"] <= horizon).astype(int)
    out["role"] = out["gel_batch"].map(batch_role)
    return out


def train(cycles: pd.DataFrame, truth: pd.DataFrame, cfg: Config) -> dict:
    from sklearn import __version__ as sk_version
    from sklearn.tree import DecisionTreeClassifier

    features, _ = standardise(cycles, cfg)
    df = label_replace(features, truth, cfg.model.replace_horizon)
    train_df = df[(df["role"] == "calibration")].dropna(subset=MODEL_FEATURES)

    clf = DecisionTreeClassifier(
        max_depth=cfg.model.max_depth, min_samples_leaf=cfg.model.min_samples_leaf, random_state=0,
    )
    clf.fit(train_df[MODEL_FEATURES].to_numpy(), train_df["label_replace"].to_numpy())
    return {
        "model_version": f"tree-{cfg.config_version}",
        "trained_with_config_version": cfg.config_version,
        "features": MODEL_FEATURES,
        "label": f"replace if true replacement point <= {cfg.model.replace_horizon} cycles away",
        "trained_on_batches": sorted(train_df["gel_batch"].unique().tolist()),
        "n_training_cycles": int(len(train_df)),
        "n_training_replace": int(train_df["label_replace"].sum()),
        "max_depth": cfg.model.max_depth,
        "min_samples_leaf": cfg.model.min_samples_leaf,
        "sklearn_version": sk_version,
        "nodes": tree_to_dict(clf, MODEL_FEATURES),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the GO / NO-GO decision tree.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/synthetic/cycles.csv")
    parser.add_argument("--truth", type=Path, default=ROOT / "data/synthetic/true_replacement.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument("--rules-out", type=Path, default=ROOT / "docs/tree_rules.txt")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cycles = load_cycles(args.input, known_types=set(cfg.types))
    model = train(cycles, pd.read_csv(args.truth), cfg)
    model_path = (args.config.parent.parent / cfg.model.file).resolve()
    save_model(model, model_path)
    text = rules_text(model)
    args.rules_out.parent.mkdir(parents=True, exist_ok=True)
    args.rules_out.write_text(
        f"# GO / NO-GO decision tree ({model['model_version']})\n"
        f"# Trained on {model['trained_on_batches']} ({model['n_training_cycles']} cycles, "
        f"{model['n_training_replace']} labelled 'replace'). Synthetic data.\n\n{text}\n"
    )
    print(f"Saved {model_path}\n\n{text}")


if __name__ == "__main__":
    main()
