"""Synthetic reverse-phase chromatography history.

Each *resin lot* (one column packing) is used for many purification cycles
(one cycle = one batch) until it is replaced. Two kinds of wear build up
cycle after cycle, and the four monitored KPIs reveal them:

* **fouling**: lipids, precipitated protein and fines plug the bed. Back-pressure
  rises, so the skid runs a lower elution flow. Cleaning-in-place (CIP) removes
  only part of it.
* **ligand loss**: the C4/C8 ligand hydrolyses at low pH and the silica/polymer
  matrix ages. Peaks broaden (bigger pool volume, shallower peak tail) and
  product binds less selectively, so step yield drops faster and faster.

Lots differ in packing quality and lifetime, which the operator cannot see
directly, so the KPIs carry information beyond the cycle count.

KPI definitions (one value per batch):

=================  ======  ===========================================================
elution_flow       L/h     mean flow during the elution step
elution_pressure   bar     mean column back-pressure during the elution step
pool_volume        CV      volume of the collected product pool, in column volumes
tail_gradient      mAU/CV  steepness of the trailing edge of the elution peak
                           (absolute UV slope); a fresh column has a sharp tail, a
                           worn one a shallow, dragging tail, so this falls with wear
yield              %       step yield (product in pool / product loaded)
=================  ======  ===========================================================
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

KPIS = ["elution_flow", "elution_pressure", "pool_volume", "tail_gradient"]
COLUMNS = ["resin_lot", "cycle"] + KPIS + ["yield"]

KPI_INFO = {
    "elution_flow": {"label": "Elution flow", "unit": "L/h"},
    "elution_pressure": {"label": "Elution pressure", "unit": "bar"},
    "pool_volume": {"label": "Pool volume", "unit": "CV"},
    "tail_gradient": {"label": "Tail gradient", "unit": "mAU/CV"},
    "yield": {"label": "Step yield", "unit": "%"},
}

# Fresh-column reference values.
FLOW0, PRESSURE0, POOL0, TAIL0, YIELD0 = 120.0, 18.0, 2.4, 850.0, 92.0


def _lot_params(rng: np.random.Generator) -> dict:
    """Hidden per-lot properties (packing quality, ligand stability...)."""
    return dict(
        life=rng.normal(130.0, 22.0),         # cycles until ligand loss reaches 1
        shape=rng.uniform(2.2, 3.2),          # how sharply wear accelerates
        foul_rate=rng.lognormal(np.log(0.010), 0.30),
        cip_eff=rng.uniform(0.55, 0.80),      # fraction of fouling CIP removes
        pack=rng.normal(1.0, 0.04),           # packing quality (1 = nominal)
        yield_offset=rng.normal(0.0, 0.8),
    )


def simulate_lot(
    lot: str,
    n_cycles: int,
    rng: np.random.Generator | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """Simulate ``n_cycles`` batches on one resin lot."""
    rng = rng or np.random.default_rng()
    p = params or _lot_params(rng)
    rows = []
    fouling = 0.0
    for c in range(1, n_cycles + 1):
        # Fouling accumulates; CIP after each cycle removes part of the new deposit.
        fouling += p["foul_rate"] * (1.0 - p["cip_eff"]) * rng.lognormal(0.0, 0.5) * 1.5
        if rng.random() < 0.015:  # occasional dirty feed
            fouling += rng.uniform(0.03, 0.08)
        ligand = (c / max(p["life"], 40.0)) ** p["shape"]

        pressure = PRESSURE0 / p["pack"] * (1.0 + 1.4 * fouling + 0.15 * ligand)
        flow = FLOW0 * p["pack"] / (1.0 + 0.5 * fouling)
        pool = POOL0 * (1.0 + 0.35 * ligand + 0.10 * fouling) / p["pack"]
        tail = TAIL0 * p["pack"] / (1.0 + 0.9 * ligand + 0.2 * fouling)
        yld = (
            YIELD0
            + p["yield_offset"]
            - 9.0 * ligand
            - 6.0 * ligand**2
            - 4.0 * fouling
        )
        rows.append(
            dict(
                resin_lot=lot,
                cycle=c,
                elution_flow=flow * rng.normal(1.0, 0.012),
                elution_pressure=pressure * rng.normal(1.0, 0.015),
                pool_volume=pool * rng.normal(1.0, 0.02),
                tail_gradient=tail * rng.normal(1.0, 0.03),
                **{"yield": float(np.clip(yld + rng.normal(0.0, 0.7), 0.0, 100.0))},
            )
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def generate_history(n_lots: int = 30, seed: int = 0) -> pd.DataFrame:
    """Historical batches from ``n_lots`` resin lots, each replaced at some point.

    Historical replacement was done by the calendar or by gut feel, so lots
    were retired anywhere from well before to well after their best point.
    """
    rng = np.random.default_rng(seed)
    lots = []
    for i in range(n_lots):
        p = _lot_params(rng)
        n = int(np.clip(rng.normal(p["life"] * 1.05, 20.0), 50, 220))
        lots.append(simulate_lot(f"LOT-{i + 1:03d}", n, rng, p))
    return pd.concat(lots, ignore_index=True)


def generate_current_lot(n_cycles: int = 62, seed: int = 123) -> pd.DataFrame:
    """A lot still in use: the column whose next batches we want to forecast."""
    rng = np.random.default_rng(seed)
    return simulate_lot("CURRENT", n_cycles, rng)


if __name__ == "__main__":
    out = pathlib.Path(__file__).resolve().parents[2] / "data" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    generate_history().to_csv(out / "resin_history.csv", index=False)
    generate_current_lot().to_csv(out / "current_lot.csv", index=False)
    print(f"Wrote {out / 'resin_history.csv'} and {out / 'current_lot.csv'}")
