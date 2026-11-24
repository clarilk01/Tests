"""Decide when to change the resin.

Three independent signals, the most urgent one wins:

1. **Quality limits** – the yield forecast for the next two batches against the
   minimum acceptable step yield.
2. **Equipment / process limits** – the KPIs (and their linear trend over the next
   two batches) against their action limits: maximum back-pressure, minimum
   elution flow, maximum pool volume, minimum tail gradient.
3. **Economics** – classic replacement rule: change the resin once the yield
   lost on the *next* batch (versus a fresh column) costs more than the average
   cost per batch of the lot so far (resin + all yield lost until now). Past
   that point every extra batch raises the cost per batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import BASELINE_CYCLES, SLOPE_WINDOW
from .simulator import KPI_INFO

STATUS_ORDER = ["OK", "PLAN", "CHANGE_AFTER_NEXT", "CHANGE_NOW"]
STATUS_TEXT = {
    "OK": "Resin OK – no change needed in the next 2 batches",
    "PLAN": "Plan a resin change – risk of falling out of limits within 2 batches",
    "CHANGE_AFTER_NEXT": "Change the resin after the next batch",
    "CHANGE_NOW": "Change the resin now – before the next batch",
}


@dataclass
class Limits:
    min_yield: float = 80.0          # %
    max_pressure: float = 45.0       # bar
    min_flow: float = 80.0           # L/h
    max_pool_volume: float = 3.3     # CV
    min_tail_gradient: float = 450.0  # mAU/CV

    def checks(self) -> list[tuple[str, str, float]]:
        """(kpi, direction, limit); direction 'max' means value must stay below."""
        return [
            ("elution_pressure", "max", self.max_pressure),
            ("elution_flow", "min", self.min_flow),
            ("pool_volume", "max", self.max_pool_volume),
            ("tail_gradient", "min", self.min_tail_gradient),
        ]


@dataclass
class Economics:
    resin_cost: float = 400_000.0   # $ per change: resin, packing, qualification, downtime
    batch_value: float = 100_000.0  # $ value of a batch at 100 % step yield


def _breach(value: float, direction: str, limit: float) -> bool:
    return value > limit if direction == "max" else value < limit


def project_kpis(lot: pd.DataFrame, n_ahead: int = 2, window: int = SLOPE_WINDOW) -> pd.DataFrame:
    """Linear projection of every KPI over the next ``n_ahead`` batches."""
    recent = lot.sort_values("cycle").tail(window)
    x = recent["cycle"].to_numpy(float)
    last = x[-1]
    rows = []
    for h in range(1, n_ahead + 1):
        row = {"cycle": int(last) + h}
        for k, d, _ in Limits().checks():
            y = recent[k].to_numpy(float)
            if len(y) >= 3:
                slope, icpt = np.polyfit(x, y, 1)
                # Anchor to the smoothed current level, then add the trend.
                row[k] = y[-3:].mean() + slope * (last + h - x[-3:].mean())
            else:
                row[k] = y[-1]
        rows.append(row)
    return pd.DataFrame(rows)


def economics_curve(lot: pd.DataFrame, econ: Economics, reference_yield: float | None = None) -> pd.DataFrame:
    """Per-batch yield loss and running average cost per batch for one lot."""
    lot = lot.sort_values("cycle").reset_index(drop=True)
    ref = reference_yield if reference_yield is not None else lot["yield"].iloc[:BASELINE_CYCLES].mean()
    loss = (ref - lot["yield"]).clip(lower=0) / 100.0 * econ.batch_value
    n = np.arange(1, len(lot) + 1)
    return pd.DataFrame({
        "cycle": lot["cycle"],
        "loss": loss,
        "avg_cost": (econ.resin_cost + loss.cumsum()) / n,
    })


def optimal_lifetime(history: pd.DataFrame, econ: Economics) -> tuple[int, pd.DataFrame]:
    """Replacement cycle that minimises cost per batch for the *typical* lot in history."""
    curve = history.groupby("cycle")["yield"].median()
    counts = history.groupby("cycle").size()
    curve = curve[counts >= max(3, counts.max() // 5)]  # ignore the thinly populated tail
    ref = curve.iloc[:BASELINE_CYCLES].mean()
    loss = (ref - curve).clip(lower=0) / 100 * econ.batch_value
    avg = (econ.resin_cost + loss.cumsum()) / np.arange(1, len(curve) + 1)
    df = pd.DataFrame({"cycle": curve.index, "median_yield": curve.values,
                       "loss": loss.values, "avg_cost": avg.values})
    return int(df.loc[df["avg_cost"].idxmin(), "cycle"]), df


def remaining_batches(lot: pd.DataFrame, min_yield: float, window: int = 30, cap: int = 200) -> int | None:
    """Rough number of batches until the yield trend crosses ``min_yield``.

    Quadratic fit to the last ``window`` batches, extrapolated. ``None`` when the
    trend is flat or improving (nothing to extrapolate).
    """
    recent = lot.sort_values("cycle").tail(window)
    if len(recent) < 10:
        return None
    x = recent["cycle"].to_numpy(float)
    y = recent["yield"].to_numpy(float)
    coef = np.polyfit(x - x[-1], y, 2 if len(recent) >= 15 else 1)
    poly = np.poly1d(coef)
    for h in range(0, cap + 1):
        if poly(h) < min_yield:
            return h
    return None


def recommend(
    lot: pd.DataFrame,
    forecast: pd.DataFrame,
    limits: Limits | None = None,
    econ: Economics | None = None,
) -> dict:
    """Combine forecast, KPI limits and economics into a resin-change recommendation."""
    limits, econ = limits or Limits(), econ or Economics()
    lot = lot.sort_values("cycle").reset_index(drop=True)
    last = lot.iloc[-1]
    reasons: list[tuple[str, str]] = []  # (status, text)

    def add(status: str, text: str) -> None:
        reasons.append((status, text))

    f1 = forecast.set_index("horizon").loc[1]
    f2 = forecast.set_index("horizon").loc[2]
    m = limits.min_yield

    # 1. yield forecast
    if f1["med"] < m:
        add("CHANGE_NOW", f"Next batch (#{int(f1.cycle)}) is forecast at {f1['med']:.1f} % yield, below the {m:.0f} % minimum.")
    elif f2["med"] < m:
        add("CHANGE_AFTER_NEXT", f"Batch #{int(f2.cycle)} is forecast at {f2['med']:.1f} % yield, below the {m:.0f} % minimum.")
    elif f1["lo"] < m:
        add("CHANGE_AFTER_NEXT", f"Next batch has a >10 % risk of yield below {m:.0f} % (80 % interval down to {f1['lo']:.1f} %).")
    elif f2["lo"] < m:
        add("PLAN", f"Batch #{int(f2.cycle)} has a >10 % risk of yield below {m:.0f} % (80 % interval down to {f2['lo']:.1f} %).")

    # 2. KPI limits: now (smoothed over 3 batches to ignore a single noisy point) and projected
    proj = project_kpis(lot)
    smooth = lot.tail(3)
    for k, d, lim in limits.checks():
        label, unit = KPI_INFO[k]["label"], KPI_INFO[k]["unit"]
        word = "above the maximum" if d == "max" else "below the minimum"
        now = smooth[k].mean()
        if _breach(now, d, lim):
            add("CHANGE_NOW", f"{label} is {now:.1f} {unit} (last 3 batches), {word} of {lim:g} {unit}.")
        elif _breach(proj[k].iloc[0], d, lim):
            add("CHANGE_AFTER_NEXT", f"{label} is trending to {proj[k].iloc[0]:.1f} {unit} next batch, {word} of {lim:g} {unit}.")
        elif _breach(proj[k].iloc[1], d, lim):
            add("PLAN", f"{label} is trending to {proj[k].iloc[1]:.1f} {unit} in 2 batches, {word} of {lim:g} {unit}.")

    # 3. economics: marginal loss of the next batch(es) vs. average cost per batch so far
    curve = economics_curve(lot, econ)
    ref = lot["yield"].iloc[:BASELINE_CYCLES].mean()
    avg_now = curve["avg_cost"].iloc[-1]
    loss1 = max(ref - f1["med"], 0) / 100 * econ.batch_value
    loss2 = max(ref - f2["med"], 0) / 100 * econ.batch_value
    avg_after1 = (avg_now * len(curve) + loss1) / (len(curve) + 1)
    if loss1 > avg_now:
        add("CHANGE_NOW", f"Economics: the next batch would lose ${loss1:,.0f} of product, more than the "
                          f"${avg_now:,.0f} average cost per batch of this resin so far – a fresh column is cheaper.")
    elif loss2 > avg_after1:
        add("CHANGE_AFTER_NEXT", f"Economics: from batch #{int(f2.cycle)} the yield loss (${loss2:,.0f}) exceeds the "
                                 f"average cost per batch (${avg_after1:,.0f}).")

    status = max([r[0] for r in reasons], key=STATUS_ORDER.index, default="OK")
    return {
        "status": status,
        "headline": STATUS_TEXT[status],
        "reasons": sorted(reasons, key=lambda r: -STATUS_ORDER.index(r[0])),
        "last_cycle": int(last["cycle"]),
        "kpi_projection": proj,
        "economics": curve,
        "reference_yield": ref,
        "remaining_batches": remaining_batches(lot, m),
    }
