"""Resin Lifetime Predictor — Streamlit dashboard.

Run from the project folder:

    streamlit run dashboard/resin_dashboard.py
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from resin import (  # noqa: E402
    KPI_INFO,
    KPIS,
    Economics,
    Limits,
    YieldForecaster,
    generate_current_lot,
    generate_history,
    optimal_lifetime,
    recommend,
)
from resin.simulator import COLUMNS  # noqa: E402
from resin.validation import backtest, decision_backtest, split_lots  # noqa: E402

# Reference data-viz palette: one hue per measure, neutral ink for context.
BLUE, BLUE_BAND = "#2a78d6", "rgba(42, 120, 214, 0.15)"
ORANGE = "#eb6834"
MUTED, FAINT = "#8a8984", "rgba(138, 137, 132, 0.25)"
CRITICAL = "#d03b3b"

st.set_page_config(page_title="Resin Lifetime Predictor", page_icon="🧪", layout="wide")


# --------------------------------------------------------------------------- data & model
@st.cache_data(show_spinner="Generating historical resin lots…")
def load_synthetic(n_lots: int, seed: int) -> pd.DataFrame:
    return generate_history(n_lots=n_lots, seed=seed)


@st.cache_resource(show_spinner="Training yield forecaster…")
def train(history: pd.DataFrame, use_yield_history: bool) -> YieldForecaster:
    return YieldForecaster(use_yield_history=use_yield_history).fit(history)


@st.cache_data(show_spinner="Back-testing on held-out resin lots…")
def run_validation(history: pd.DataFrame, use_yield_history: bool, limits: Limits, econ: Economics):
    tr, te = split_lots(history)
    m = YieldForecaster(use_yield_history=use_yield_history).fit(tr)
    rows, summary = backtest(m, te)
    return rows, summary, decision_backtest(m, te, limits, econ), m.importance(tr)


def check_columns(df: pd.DataFrame, name: str) -> pd.DataFrame | None:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        st.error(f"{name} is missing columns: {', '.join(missing)}")
        return None
    return df[COLUMNS].dropna().sort_values(["resin_lot", "cycle"]).reset_index(drop=True)


def base_layout(fig: go.Figure, height: int = 360, **kw) -> go.Figure:
    kw.setdefault("hovermode", "x unified")
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **kw,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(138,137,132,0.2)")
    return fig


# --------------------------------------------------------------------------- sidebar
st.sidebar.header("Data")
src = st.sidebar.radio("Historical batches", ["Synthetic demo data", "Upload CSV"], horizontal=True)
if src == "Upload CSV":
    up = st.sidebar.file_uploader("History CSV", type="csv", key="hist")
    if up is None:
        st.info("Upload a CSV with one row per batch and the columns "
                f"`{', '.join(COLUMNS)}`, or switch to the synthetic demo data.")
        st.stop()
    history = check_columns(pd.read_csv(up), "History CSV")
    if history is None:
        st.stop()
else:
    c1, c2 = st.sidebar.columns(2)
    n_lots = c1.number_input("Resin lots", 10, 80, 30, step=5)
    seed = c2.number_input("Seed", 0, 999, 0)
    history = load_synthetic(int(n_lots), int(seed))

st.sidebar.header("Current column")
cur_src = st.sidebar.radio("Batches on the resin in use",
                           ["Demo column", "Replay a historical lot", "Upload CSV"])
if cur_src == "Upload CSV":
    up = st.sidebar.file_uploader("Current column CSV (same columns)", type="csv", key="cur")
    if up is None:
        st.info("Upload the batches run so far on the current resin.")
        st.stop()
    current = check_columns(pd.read_csv(up), "Current column CSV")
    if current is None:
        st.stop()
elif cur_src == "Replay a historical lot":
    lot = st.sidebar.selectbox("Resin lot", sorted(history["resin_lot"].unique()))
    current = history[history["resin_lot"] == lot].reset_index(drop=True)
    st.sidebar.caption("The model is still trained on this lot. Use the Model validation tab for an honest test.")
else:
    current = generate_current_lot(160)

max_cycle = int(current["cycle"].max())
default = min(max_cycle, 62) if cur_src == "Demo column" else max_cycle
upto = st.sidebar.slider("Batches run so far", min(10, max_cycle), max_cycle, default,
                         help="Replay the column as it was after this batch.")
future = current[current["cycle"] > upto]
current = current[current["cycle"] <= upto].reset_index(drop=True)

with st.sidebar.expander("Action limits", expanded=True):
    limits = Limits(
        min_yield=st.number_input("Minimum step yield (%)", 0.0, 100.0, 80.0, 0.5),
        max_pressure=st.number_input("Max elution pressure (bar)", 0.0, 400.0, 45.0, 1.0),
        min_flow=st.number_input("Min elution flow (L/h)", 0.0, 1000.0, 80.0, 1.0),
        max_pool_volume=st.number_input("Max pool volume (CV)", 0.0, 20.0, 3.3, 0.1),
        min_tail_gradient=st.number_input("Min tail gradient (mAU/CV)", 0.0, 5000.0, 450.0, 10.0),
    )
with st.sidebar.expander("Economics"):
    econ = Economics(
        resin_cost=st.number_input("Cost of a resin change ($)", 0.0, 1e8, 400_000.0, 10_000.0,
                                   help="Resin, packing, qualification and downtime."),
        batch_value=st.number_input("Batch value at 100 % yield ($)", 0.0, 1e8, 100_000.0, 5_000.0),
    )
use_yield = st.sidebar.toggle("Also use past yields as input", True,
                              help="Off = forecast from cycle number and the 4 KPIs only.")

model = train(history, use_yield)
forecast = model.forecast(current)
rec = recommend(current, forecast, limits, econ)
opt_cycle, typical = optimal_lifetime(history, econ)

# --------------------------------------------------------------------------- header
st.title("🧪 Resin Lifetime Predictor")
st.caption("Reverse-phase chromatography: forecast the step yield of the next 2 batches from "
           "elution flow, elution pressure, pool volume and tail gradient, and decide when to change the resin.")

banner = {"OK": st.success, "PLAN": st.warning, "CHANGE_AFTER_NEXT": st.warning, "CHANGE_NOW": st.error}
icon = {"OK": "✅", "PLAN": "🟡", "CHANGE_AFTER_NEXT": "🟠", "CHANGE_NOW": "🛑"}
banner[rec["status"]](f"**{rec['headline']}** (after batch #{rec['last_cycle']})", icon=icon[rec["status"]])
for status, text in rec["reasons"]:
    st.markdown(f"- {icon[status]} {text}")

f = forecast.set_index("horizon")
k1, k2, k3, k4 = st.columns(4)
last_y = current["yield"].iloc[-1]
k1.metric(f"Yield batch #{int(f.loc[1, 'cycle'])}", f"{f.loc[1, 'med']:.1f} %",
          f"{f.loc[1, 'med'] - last_y:+.1f} vs last batch", delta_color="normal",
          help=f"80 % interval {f.loc[1, 'lo']:.1f}–{f.loc[1, 'hi']:.1f} %")
k1.caption(f"80 % interval {f.loc[1, 'lo']:.1f}–{f.loc[1, 'hi']:.1f} %")
k2.metric(f"Yield batch #{int(f.loc[2, 'cycle'])}", f"{f.loc[2, 'med']:.1f} %",
          f"{f.loc[2, 'med'] - last_y:+.1f} vs last batch", delta_color="normal")
k2.caption(f"80 % interval {f.loc[2, 'lo']:.1f}–{f.loc[2, 'hi']:.1f} %")
rb = rec["remaining_batches"]
k3.metric(f"Batches until yield < {limits.min_yield:g} %", "—" if rb is None else f"≈ {rb}")
k3.caption("Rough: extrapolated yield trend")
k4.metric("Cost-optimal resin life (typical lot)", f"{opt_cycle} batches")
k4.caption(f"This column: batch #{rec['last_cycle']}")

tab_fc, tab_econ, tab_hist, tab_val = st.tabs(["Forecast & KPIs", "Economics", "Lot history", "Model validation"])

# --------------------------------------------------------------------------- forecast tab
with tab_fc:
    fig = go.Figure()
    fig.add_scatter(x=current["cycle"], y=current["yield"], mode="lines", name="Measured yield",
                    line=dict(color=BLUE, width=2))
    fig.add_scatter(x=forecast["cycle"], y=forecast["med"], mode="markers", name="Forecast (median, 80 % interval)",
                    marker=dict(color=ORANGE, size=10, line=dict(color="white", width=2)),
                    error_y=dict(type="data", symmetric=False, array=forecast["hi"] - forecast["med"],
                                 arrayminus=forecast["med"] - forecast["lo"], color=ORANGE, thickness=2))
    if len(future):
        nxt = future.head(2)
        fig.add_scatter(x=nxt["cycle"], y=nxt["yield"], mode="markers", name="What actually happened",
                        marker=dict(color=MUTED, size=9, symbol="x"))
    fig.add_hline(y=limits.min_yield, line=dict(color=CRITICAL, dash="dash", width=1.5),
                  annotation_text=f"min {limits.min_yield:g} %", annotation_position="bottom left")
    fig.update_yaxes(title="Step yield (%)")
    fig.update_xaxes(title="Batch (cycle on this resin)")
    st.subheader("Step yield: history and next 2 batches")
    st.plotly_chart(base_layout(fig, 380), width="stretch")

    st.subheader("KPIs against their action limits")
    proj = rec["kpi_projection"]
    lim = {k: v for k, _, v in limits.checks()}
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{KPI_INFO[k]['label']} ({KPI_INFO[k]['unit']})" for k in KPIS],
                        horizontal_spacing=0.08, vertical_spacing=0.15)
    for i, k in enumerate(KPIS):
        r, c = i // 2 + 1, i % 2 + 1
        fig.add_scatter(x=current["cycle"], y=current[k], mode="lines", line=dict(color=BLUE, width=2),
                        name="Measured", showlegend=i == 0, legendgroup="m", row=r, col=c)
        fig.add_scatter(x=[current["cycle"].iloc[-1]] + proj["cycle"].tolist(),
                        y=[current[k].tail(3).mean()] + proj[k].tolist(), mode="lines+markers",
                        line=dict(color=ORANGE, width=2, dash="dot"), marker=dict(size=8),
                        name="Trend, next 2 batches", showlegend=i == 0, legendgroup="p", row=r, col=c)
        fig.add_hline(y=lim[k], line=dict(color=CRITICAL, dash="dash", width=1.5), row=r, col=c)
    fig = base_layout(fig, 580, hovermode="closest")
    fig.update_layout(margin=dict(t=80), legend=dict(y=1.08))
    st.plotly_chart(fig, width="stretch")
    st.caption("Red dashed lines: action limits (pressure and pool volume are maxima, flow and tail gradient minima). "
               "A shallow tail gradient means the peak drags: a sign of ligand loss.")

    with st.expander("Data for the current column"):
        st.dataframe(current.iloc[::-1], width="stretch", hide_index=True)

# --------------------------------------------------------------------------- economics tab
with tab_econ:
    st.markdown(
        "**Replacement rule.** Every batch on an ageing resin loses product compared with a fresh column. "
        "The resin cost is spread over all the batches it runs. Change the resin when the **loss of the next batch** "
        "is bigger than the **average cost per batch so far**: from then on each extra batch makes the column more expensive."
    )
    ec = rec["economics"]
    ref = rec["reference_yield"]
    fc_loss = [(max(ref - v, 0) / 100 * econ.batch_value) for v in forecast["med"]]
    fig = go.Figure()
    fig.add_scatter(x=ec["cycle"], y=ec["avg_cost"], name="Average cost per batch (resin + losses)",
                    line=dict(color=BLUE, width=2))
    fig.add_scatter(x=ec["cycle"], y=ec["loss"].rolling(3, min_periods=1).mean(),
                    name="Yield loss per batch (3-batch mean)", line=dict(color=ORANGE, width=2))
    fig.add_scatter(x=forecast["cycle"], y=fc_loss, mode="markers", name="Forecast loss, next 2 batches",
                    marker=dict(color=ORANGE, size=10, symbol="diamond", line=dict(color="white", width=2)))
    top = max(ec["avg_cost"].iloc[min(9, len(ec) - 1)], ec["loss"].max(), max(fc_loss))
    fig.update_yaxes(title="$ per batch", tickformat="$,.0f", range=[0, top * 1.15])
    fig.update_xaxes(title="Batch")
    st.plotly_chart(base_layout(fig, 400), width="stretch")
    st.caption("The first batches carry most of the resin cost and are cut off the top of the chart.")

    st.subheader("Typical lot in the history")
    fig = go.Figure()
    fig.add_scatter(x=typical["cycle"], y=typical["avg_cost"], name="Average cost per batch", line=dict(color=BLUE, width=2))
    fig.add_scatter(x=typical["cycle"], y=typical["loss"], name="Yield loss per batch", line=dict(color=ORANGE, width=2))
    fig.add_vline(x=opt_cycle, line=dict(color=MUTED, dash="dash"),
                  annotation_text=f"cost optimum: {opt_cycle} batches", annotation_position="top")
    fig.update_yaxes(title="$ per batch", tickformat="$,.0f",
                     range=[0, max(typical["avg_cost"].iloc[9], typical["loss"].max()) * 1.15])
    fig.update_xaxes(title="Batch")
    st.plotly_chart(base_layout(fig, 360), width="stretch")
    lifetimes = history.groupby("resin_lot")["cycle"].max()
    st.caption(f"Historical lots were run {int(lifetimes.median())} batches (median) before replacement; "
               f"the cost optimum for the median yield curve is {opt_cycle} batches. "
               f"The per-column rule above adapts this to how fast the current resin is ageing.")

# --------------------------------------------------------------------------- history tab
with tab_hist:
    sig = st.selectbox("Signal", ["yield"] + KPIS, format_func=lambda k: f"{KPI_INFO[k]['label']} ({KPI_INFO[k]['unit']})")
    fig = go.Figure()
    for i, (lot, g) in enumerate(history.groupby("resin_lot")):
        fig.add_scatter(x=g["cycle"], y=g[sig], mode="lines", line=dict(color=FAINT, width=1),
                        name="Historical lots", legendgroup="h", showlegend=i == 0,
                        hovertemplate=f"{lot}<br>batch %{{x}}<br>%{{y:.1f}}<extra></extra>")
    fig.add_scatter(x=current["cycle"], y=current[sig], mode="lines", line=dict(color=BLUE, width=2.5), name="Current column")
    fig.update_xaxes(title="Batch")
    fig.update_yaxes(title=f"{KPI_INFO[sig]['label']} ({KPI_INFO[sig]['unit']})")
    st.plotly_chart(base_layout(fig, 460, hovermode="closest"), width="stretch")
    st.dataframe(
        history.groupby("resin_lot").agg(batches=("cycle", "max"), first_yield=("yield", "first"),
                                         last_yield=("yield", "last"), max_pressure=("elution_pressure", "max"))
        .round(1), width="stretch")

# --------------------------------------------------------------------------- validation tab
with tab_val:
    st.markdown("A quarter of the resin lots is held out; the model is trained on the rest and then forecasts "
                "every batch of the held-out lots. Baseline: assume the next yield equals the last one.")
    rows, summary, dec, imp = run_validation(history, use_yield, limits, econ)
    st.dataframe(summary.style.format({"MAE model (%)": "{:.2f}", "MAE last yield (%)": "{:.2f}",
                                       "80 % interval coverage": "{:.0%}"}), hide_index=True, width="stretch")
    c1, c2 = st.columns([3, 2])
    with c1:
        lot = st.selectbox("Held-out lot", sorted(rows["resin_lot"].unique()))
        g = rows[rows["resin_lot"] == lot]
        h = st.radio("Horizon", [1, 2], format_func=lambda h: f"+{h} batch", horizontal=True)
        x = g["cycle"] + h
        fig = go.Figure()
        fig.add_scatter(x=pd.concat([x, x[::-1]]), y=pd.concat([g[f"h{h}_hi"], g[f"h{h}_lo"][::-1]]), fill="toself",
                        fillcolor=BLUE_BAND, line=dict(width=0), name="80 % interval", hoverinfo="skip")
        fig.add_scatter(x=x, y=g[f"h{h}_med"], name="Forecast", line=dict(color=BLUE, width=2))
        fig.add_scatter(x=g["cycle"], y=g["yield"], name="Actual", mode="markers", marker=dict(color=MUTED, size=5))
        fig.add_hline(y=limits.min_yield, line=dict(color=CRITICAL, dash="dash", width=1.5))
        fig.update_xaxes(title="Batch")
        fig.update_yaxes(title="Step yield (%)")
        st.plotly_chart(base_layout(fig, 380), width="stretch")
    with c2:
        top = imp.head(10)[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=BLUE))
        fig.update_xaxes(title="MAE increase when shuffled (% yield)")
        st.markdown("**What drives the +1 batch forecast**")
        st.plotly_chart(base_layout(fig, 420, hovermode="closest"), width="stretch")
    st.markdown("**Decision replay on held-out lots**: the batch after which the rule first said "
                "*change now / after next batch*, compared with the first time yield fell below the minimum "
                "for 2 batches in a row.")
    st.dataframe(dec, hide_index=True, width="stretch")
