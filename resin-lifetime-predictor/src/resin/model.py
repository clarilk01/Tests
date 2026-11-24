"""Forecast the step yield of the next two batches from the KPI history.

One gradient-boosting model per horizon (next batch, batch after) and per
quantile (10 %, 50 %, 90 %), so every forecast comes with an 80 % interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from .features import HORIZONS, build_features, feature_names

QUANTILES = {"lo": 0.10, "med": 0.50, "hi": 0.90}


@dataclass
class YieldForecaster:
    use_yield_history: bool = True
    random_state: int = 0
    calibrate: bool = True  # widen the intervals so they really cover 80 %
    models: dict = field(default_factory=dict, init=False)
    widen: dict = field(default_factory=dict, init=False)

    @property
    def features(self) -> list[str]:
        return feature_names(self.use_yield_history)

    def fit(self, history: pd.DataFrame) -> "YieldForecaster":
        """Train on ``history``.

        With ``calibrate`` the 80 % interval is first measured on resin lots held
        out of training (split-conformal) and widened by the amount needed to
        reach 80 % coverage on lots the model has not seen.
        """
        self.widen = {h: 0.0 for h in HORIZONS}
        lots = history["resin_lot"].unique()
        if self.calibrate and len(lots) >= 8:
            rng = np.random.default_rng(self.random_state)
            cal = rng.choice(lots, max(2, len(lots) // 4), replace=False)
            is_cal = history["resin_lot"].isin(cal)
            self._fit(history[~is_cal])
            feats = build_features(history[is_cal], self.use_yield_history)
            preds = self.predict_features(feats)
            for h in HORIZONS:
                ok = feats[f"yield_h{h}"].notna()
                y = feats.loc[ok, f"yield_h{h}"]
                score = np.maximum(preds.loc[ok, f"h{h}_lo"] - y, y - preds.loc[ok, f"h{h}_hi"])
                n = len(score)
                level = min(1.0, np.ceil((n + 1) * 0.8) / n)
                self.widen[h] = max(0.0, float(np.quantile(score, level)))
        return self._fit(history)

    def _fit(self, history: pd.DataFrame) -> "YieldForecaster":
        data = build_features(history, self.use_yield_history)
        for h in HORIZONS:
            rows = data.dropna(subset=[f"yield_h{h}"])
            X, y = rows[self.features], rows[f"yield_h{h}"] - self._anchor(rows)
            for name, q in QUANTILES.items():
                m = HistGradientBoostingRegressor(
                    loss="quantile", quantile=q, max_iter=300, learning_rate=0.05,
                    max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                    random_state=self.random_state,
                )
                self.models[(h, name)] = m.fit(X, y)
        return self

    def _anchor(self, feats: pd.DataFrame) -> pd.Series:
        """Level the forecast is made relative to.

        With yield history the trees learn the *change* from the recent mean
        yield, which extrapolates to columns older than any seen in training.
        """
        if self.use_yield_history:
            return feats["yield_mean3"]
        return pd.Series(0.0, index=feats.index)

    def predict_features(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Forecast for every row of an already-featurised frame."""
        out = {}
        X = feats[self.features]
        for h in HORIZONS:
            preds = np.column_stack([self.models[(h, n)].predict(X) for n in QUANTILES])
            preds += self._anchor(feats).to_numpy()[:, None]
            preds.sort(axis=1)  # keep lo <= med <= hi
            w = self.widen.get(h, 0.0)
            preds[:, 0] -= w
            preds[:, 2] += w
            for i, n in enumerate(QUANTILES):
                out[f"h{h}_{n}"] = preds[:, i]
        return pd.DataFrame(out, index=feats.index)

    def forecast(self, lot: pd.DataFrame) -> pd.DataFrame:
        """Yield forecast for the two batches after the last one recorded on ``lot``.

        Returns one row per horizon: ``cycle, horizon, lo, med, hi``.
        """
        feats = build_features(lot, self.use_yield_history)
        last = feats.iloc[[-1]]
        p = self.predict_features(last).iloc[0]
        c = int(last["cycle"].iloc[0])
        return pd.DataFrame(
            [{"cycle": c + h, "horizon": h, **{n: p[f"h{h}_{n}"] for n in QUANTILES}}
             for h in HORIZONS]
        )

    def importance(self, history: pd.DataFrame, horizon: int = 1, seed: int = 0) -> pd.Series:
        """Permutation importance (MAE increase) of each feature for the median model."""
        data = build_features(history, self.use_yield_history).dropna(subset=[f"yield_h{horizon}"])
        r = permutation_importance(
            self.models[(horizon, "med")], data[self.features], data[f"yield_h{horizon}"],
            scoring="neg_mean_absolute_error", n_repeats=5, random_state=seed,
        )
        return pd.Series(r.importances_mean, index=self.features).sort_values(ascending=False)
