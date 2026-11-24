"""Back-testing on resin lots the model never saw."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .decision import Economics, Limits, recommend
from .features import HORIZONS, build_features
from .model import YieldForecaster


def split_lots(history: pd.DataFrame, test_frac: float = 0.25, seed: int = 0):
    lots = history["resin_lot"].unique()
    rng = np.random.default_rng(seed)
    test = set(rng.choice(lots, max(1, int(round(len(lots) * test_frac))), replace=False))
    is_test = history["resin_lot"].isin(test)
    return history[~is_test].copy(), history[is_test].copy()


def backtest(model: YieldForecaster, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast every batch of the test lots; compare with naive baselines.

    Returns (per-row predictions, summary table per horizon).
    """
    feats = build_features(test, model.use_yield_history)
    preds = model.predict_features(feats)
    rows = pd.concat([feats[["resin_lot", "cycle", "yield"] + [f"yield_h{h}" for h in HORIZONS]], preds], axis=1)
    summary = []
    for h in HORIZONS:
        r = rows.dropna(subset=[f"yield_h{h}"])
        actual = r[f"yield_h{h}"]
        summary.append({
            "horizon": f"+{h} batch",
            "MAE model (%)": (r[f"h{h}_med"] - actual).abs().mean(),
            "MAE last yield (%)": (r["yield"] - actual).abs().mean(),
            "80 % interval coverage": ((actual >= r[f"h{h}_lo"]) & (actual <= r[f"h{h}_hi"])).mean(),
            "n": len(r),
        })
    return rows, pd.DataFrame(summary)


def decision_backtest(
    model: YieldForecaster,
    test: pd.DataFrame,
    limits: Limits | None = None,
    econ: Economics | None = None,
    min_cycles: int = 10,
) -> pd.DataFrame:
    """Replay each test lot batch by batch: when would the rule have said "change"?

    ``first_low_yield`` is the first batch whose yield *and* the next one were
    below the minimum (the event we want to anticipate).
    """
    limits = limits or Limits()
    out = []
    for lot, g in test.groupby("resin_lot"):
        g = g.sort_values("cycle").reset_index(drop=True)
        low = (g["yield"] < limits.min_yield) & (g["yield"].shift(-1) < limits.min_yield)
        first_low = int(g.loc[low, "cycle"].iloc[0]) if low.any() else None
        change_at = None
        for i in range(min_cycles, len(g)):
            hist = g.iloc[: i + 1]
            rec = recommend(hist, model.forecast(hist), limits, econ)
            if rec["status"] in ("CHANGE_NOW", "CHANGE_AFTER_NEXT"):
                change_at = int(hist["cycle"].iloc[-1]) + (0 if rec["status"] == "CHANGE_NOW" else 1)
                break
        out.append({
            "resin_lot": lot,
            "cycles_run": len(g),
            "recommended_change_after": change_at,
            "first_low_yield_batch": first_low,
            "low_yield_batches_avoided": None if change_at is None else int((g["cycle"] > change_at).mul(g["yield"] < limits.min_yield).sum()),
        })
    return pd.DataFrame(out)
