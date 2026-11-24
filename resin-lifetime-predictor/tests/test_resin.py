import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from resin import (  # noqa: E402
    Economics,
    Limits,
    YieldForecaster,
    build_features,
    generate_current_lot,
    generate_history,
    optimal_lifetime,
    recommend,
)
from resin.decision import project_kpis  # noqa: E402
from resin.validation import backtest, split_lots  # noqa: E402


@pytest.fixture(scope="module")
def data():
    h = generate_history(n_lots=24, seed=1)
    tr, te = split_lots(h)
    return YieldForecaster().fit(tr), tr, te


def test_features_use_only_the_past():
    lot = generate_current_lot(40)
    full = build_features(lot)
    part = build_features(lot.head(20))
    cols = [c for c in part.columns if not c.startswith("yield_h")]
    pd.testing.assert_frame_equal(full[cols].head(20), part[cols])


def test_forecast_shape_and_order(data):
    model, _, _ = data
    f = model.forecast(generate_current_lot(50))
    assert f["cycle"].tolist() == [51, 52]
    assert ((f["lo"] <= f["med"]) & (f["med"] <= f["hi"])).all()


def test_model_beats_naive_and_covers(data):
    model, _, te = data
    _, s = backtest(model, te)
    assert (s["MAE model (%)"] < s["MAE last yield (%)"]).all()
    assert (s["80 % interval coverage"] > 0.7).all()


def test_fresh_column_ok_worn_column_change(data):
    model, _, _ = data
    lot = generate_current_lot(200, seed=7)
    fresh = lot.head(20)
    assert recommend(fresh, model.forecast(fresh))["status"] == "OK"
    worn = lot.head(160)
    assert recommend(worn, model.forecast(worn))["status"] == "CHANGE_NOW"


def test_pressure_limit_triggers_change(data):
    model, _, _ = data
    lot = generate_current_lot(20)
    rec = recommend(lot, model.forecast(lot), Limits(max_pressure=10.0))
    assert rec["status"] == "CHANGE_NOW"
    assert any("pressure" in t for _, t in rec["reasons"])


def test_kpi_projection_follows_trend():
    lot = pd.DataFrame({"resin_lot": "A", "cycle": range(1, 11), "elution_flow": 100.0,
                        "elution_pressure": [20 + i for i in range(10)], "pool_volume": 2.5,
                        "tail_gradient": 800.0, "yield": 90.0})
    p = project_kpis(lot)
    assert p["elution_pressure"].tolist() == pytest.approx([30.0, 31.0])


def test_cheaper_resin_means_shorter_life(data):
    _, tr, _ = data
    cheap, _ = optimal_lifetime(tr, Economics(resin_cost=100_000))
    dear, _ = optimal_lifetime(tr, Economics(resin_cost=1_000_000))
    assert cheap < dear
