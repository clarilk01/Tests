"""The decision tree as a plain, readable JSON file, and a tiny predictor for it.

Training (train.py) uses scikit-learn once and saves the tree here as JSON. Scoring only needs this
file and this module - no machine-learning library - and anyone can open the JSON and read the rules.

JSON layout:
    {"model_version": ..., "features": [...], "nodes": [
        {"id": 0, "feature": "z_rm_pt_80_percentile", "threshold": 2.1, "left": 1, "right": 2, "n": 2300},
        {"id": 1, "leaf": true, "score": 0.02, "n": 1500, "n_replace": 30},
        ...]}
Rule at a split: value <= threshold -> go left, otherwise go right (scikit-learn's convention).
A leaf's score = share of training cycles in that leaf that were "replace" cases (0..1).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {
    "z_resin_age": "gel age",
    "z_rm_elution_flow_median": "elution flow (rolling)",
    "z_rm_pt_80_percentile": "elution pressure (rolling)",
}


class ModelError(ValueError):
    """Raised when the model file is missing or does not match the expected features."""


def tree_to_dict(clf, features: list[str]) -> list[dict]:
    """Convert a fitted sklearn DecisionTreeClassifier (classes 0/1) into a list of plain nodes."""
    t = clf.tree_
    replace_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else None
    nodes = []
    for i in range(t.node_count):
        n = int(t.n_node_samples[i])
        if t.children_left[i] == -1:  # leaf
            frac = t.value[i][0]
            frac = frac / frac.sum()
            score = float(frac[replace_idx]) if replace_idx is not None else 0.0
            nodes.append({"id": i, "leaf": True, "score": round(score, 4), "n": n,
                          "n_replace": int(round(score * n))})
        else:
            nodes.append({"id": i, "feature": features[t.feature[i]], "threshold": round(float(t.threshold[i]), 4),
                          "left": int(t.children_left[i]), "right": int(t.children_right[i]), "n": n})
    return nodes


def save_model(model: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2) + "\n")


def load_model(path: str | Path, expected_features: list[str]) -> dict:
    path = Path(path)
    if not path.exists():
        raise ModelError(f"Model file not found: {path}. Train it first: python -m gel_performance.train")
    model = json.loads(path.read_text())
    if model.get("features") != expected_features:
        raise ModelError(f"Model features {model.get('features')} do not match the expected {expected_features}")
    return model


def predict(model: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Walk every row down the tree. Returns score (0..1), leaf id and the readable rule path.
    Rows with any missing feature get score NaN (they are never guessed)."""
    nodes = {n["id"]: n for n in model["nodes"]}
    feats = model["features"]
    scores, leaves, paths = [], [], []
    for values in X[feats].to_numpy(dtype=float):
        if np.isnan(values).any():
            scores.append(np.nan), leaves.append(-1), paths.append("")
            continue
        row = dict(zip(feats, values))
        node, steps = nodes[0], []
        while not node.get("leaf"):
            f, thr = node["feature"], node["threshold"]
            if row[f] <= thr:
                steps.append(f"{LABELS.get(f, f)} z {row[f]:.2f} <= {thr:.2f}")
                node = nodes[node["left"]]
            else:
                steps.append(f"{LABELS.get(f, f)} z {row[f]:.2f} > {thr:.2f}")
                node = nodes[node["right"]]
        scores.append(node["score"]), leaves.append(node["id"]), paths.append(" AND ".join(steps))
    return pd.DataFrame({"score": scores, "leaf": leaves, "rule_path": paths}, index=X.index)


def rules_text(model: dict) -> str:
    """The whole tree as indented if/else text (for the README and the docs folder)."""
    nodes = {n["id"]: n for n in model["nodes"]}
    lines = []

    def walk(i: int, depth: int) -> None:
        n, pad = nodes[i], "    " * depth
        if n.get("leaf"):
            lines.append(f"{pad}-> score {n['score']:.2f}  ({n['n_replace']} of {n['n']} training cycles were 'replace')")
            return
        name = LABELS.get(n["feature"], n["feature"])
        lines.append(f"{pad}if {name} z <= {n['threshold']:.2f}:")
        walk(n["left"], depth + 1)
        lines.append(f"{pad}else:  # {name} z > {n['threshold']:.2f}")
        walk(n["right"], depth + 1)

    walk(0, 0)
    return "\n".join(lines)
