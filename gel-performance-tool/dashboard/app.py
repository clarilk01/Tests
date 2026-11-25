"""Streamlit dashboard for the GO / NO-GO gel replacement tool (synthetic data).

The dashboard only READS the files written by the scoring run; it never recomputes anything,
so what you see is exactly what the job produced.

    python -m gel_performance.run --input data/synthetic/cycles.csv --output output --config config/config.yaml
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works even without `pip install -e .`

from gel_performance.config import load_config  # noqa: E402

OUTPUT_DIR = Path(os.environ.get("GEL_OUTPUT_DIR", ROOT / "output"))
CONFIG_PATH = ROOT / "config" / "config.yaml"

# Colours: one highlight hue for the rolling means / score, neutral grey for single cycles,
# reserved status colours for GO / NO-GO (always shown with an icon and a word, never colour alone).
LINE = "#2a78d6"
DOTS = "rgba(120,120,115,0.45)"
GREEN, RED = "#0ca30c", "#d03b3b"

st.set_page_config(page_title="Gel GO / NO-GO", page_icon="🧪", layout="wide")


@st.cache_data
def load_outputs(folder: str, stamp: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the run outputs. `stamp` (file modification time) refreshes the cache after a new run."""
    cycles = pd.read_csv(Path(folder) / "cycle_scores.csv", parse_dates=["cycle_end_time"])
    summary = pd.read_csv(Path(folder) / "batch_summary.csv")
    return cycles, summary


scores_file = OUTPUT_DIR / "cycle_scores.csv"
if not scores_file.exists():
    st.error(f"No results found in {OUTPUT_DIR}. Run the scoring job first:")
    st.code("python -m gel_performance.run --input data/synthetic/cycles.csv --output output "
            "--config config/config.yaml", language="bash")
    st.stop()

cycles, summary = load_outputs(str(OUTPUT_DIR), scores_file.stat().st_mtime)
cfg = load_config(CONFIG_PATH)

# ---------------------------------------------------------------- sidebar
qp_batch = st.query_params.get("batch")
types = sorted(cycles["chromatography_type"].unique())
default_type = (cycles.loc[cycles["gel_batch"] == qp_batch, "chromatography_type"].iloc[0]
                if qp_batch in set(cycles["gel_batch"]) else types[0])
with st.sidebar:
    st.header("Select")
    ctype = st.selectbox("Chromatography type", types, index=types.index(default_type))
    batches = sorted(cycles.loc[cycles["chromatography_type"] == ctype, "gel_batch"].unique())
    batch = st.selectbox("Gel batch", batches, index=batches.index(qp_batch) if qp_batch in batches else 0)
    rows = cycles[cycles["gel_batch"] == batch].sort_values("cycle_end_time")
    t_min, t_max = rows["cycle_end_time"].min().date(), rows["cycle_end_time"].max().date()
    period = st.slider("Period", min_value=t_min, max_value=t_max, value=(t_min, t_max), format="YYYY-MM-DD")
    st.caption(f"Last data: {cycles['cycle_end_time'].max():%Y-%m-%d %H:%M}. "
               f"Config {cycles['config_version'].iloc[0]}, model {cycles['model_version'].iloc[0]}.")

view = rows[(rows["cycle_end_time"].dt.date >= period[0]) & (rows["cycle_end_time"].dt.date <= period[1])]
latest = rows[rows["decision"] != "NO SCORE"].iloc[-1]
tc = cfg.types[ctype]

st.title("Gel replacement: GO / NO-GO")
st.caption("⚠️ Synthetic data only - demonstrates the method, not real-world performance.")

# ---------------------------------------------------------------- side panel: score + GO / NO-GO
side, main = st.columns([1, 3.2])
with side:
    nogo = latest["decision"] == "NO-GO"
    colour, word, icon = (RED, "NO-GO", "🛑") if nogo else (GREEN, "GO", "✅")
    st.markdown(
        f"""
        <div style="text-align:center;margin-top:8px">
          <div style="font-size:0.9rem;opacity:0.75">REPLACEMENT SCORE</div>
          <div style="margin:8px auto;width:120px;height:120px;border-radius:50%;
                      border:4px solid {LINE};display:flex;align-items:center;justify-content:center;
                      font-size:2.3rem;font-weight:700">{latest['score']:.2f}</div>
          <div style="font-size:0.8rem;opacity:0.7">0 = keep · 1 = replace</div>
          <div style="margin:18px auto 6px;width:150px;height:150px;border-radius:50%;background:{colour};
                      color:white;display:flex;flex-direction:column;align-items:center;justify-content:center">
            <div style="font-size:2rem">{icon}</div>
            <div style="font-size:1.6rem;font-weight:700">{word}</div>
            <div style="font-size:0.8rem">{'replace the gel' if nogo else 'keep using'}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric("Gel age (cycles)", int(latest["resin_age"]))
    st.metric("Rolling elution pressure", f"{latest['rm_pt_80_percentile']:.2f} bar",
              delta=f"{latest['rm_pt_80_percentile'] - tc.pressure_limit_bar:+.2f} vs limit {tc.pressure_limit_bar:.2f}",
              delta_color="inverse")
    st.caption(f"Latest cycle: {latest['cycle_end_time']:%Y-%m-%d %H:%M}")

# ---------------------------------------------------------------- main chart: flow, pressure, score vs time
with main:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.38, 0.38, 0.24],
        subplot_titles=("Elution flow (L/h)", "Elution pressure PT_80 (bar)", "Replacement score (0–1)"),
    )
    hover_extra = ("<br>gel age %{customdata[0]} cycles<br><b>score %{customdata[1]:.2f}</b>"
                   "<br>%{customdata[2]}<extra></extra>")
    custom = view[["resin_age", "score", "decision"]].to_numpy()

    for r, raw, rm, unit in [(1, "elution_flow_median", "rm_elution_flow_median", "L/h"),
                             (2, "pt_80_percentile", "rm_pt_80_percentile", "bar")]:
        fig.add_trace(go.Scatter(
            x=view["cycle_end_time"], y=view[raw], mode="markers", name="single cycle",
            legendgroup="cycle", showlegend=(r == 1), marker=dict(size=5, color=DOTS), customdata=custom,
            hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>cycle value %{{y:.2f}} {unit}" + hover_extra,
        ), row=r, col=1)
        fig.add_trace(go.Scatter(
            x=view["cycle_end_time"], y=view[rm], mode="lines", name=f"rolling mean ({cfg.rolling_window} cycles)",
            legendgroup="rm", showlegend=(r == 1), line=dict(color=LINE, width=2), customdata=custom,
            hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>rolling mean %{{y:.2f}} {unit}" + hover_extra,
        ), row=r, col=1)

    fig.add_hline(y=tc.pressure_limit_bar, row=2, col=1, line=dict(color=RED, width=1.5, dash="dash"),
                  annotation_text=f"limit {tc.pressure_limit_bar:.2f} bar", annotation_position="top left",
                  annotation_font_size=11)
    fig.add_trace(go.Scatter(
        x=view["cycle_end_time"], y=view["score"], mode="lines", line_shape="hv", name="replacement score",
        showlegend=False, line=dict(color=LINE, width=2), customdata=custom,
        hovertemplate="%{x|%Y-%m-%d %H:%M}" + hover_extra,
    ), row=3, col=1)
    fig.add_hline(y=cfg.decision_threshold, row=3, col=1, line=dict(color="gray", width=1, dash="dot"),
                  annotation_text=f"high ≥ {cfg.decision_threshold:.1f}", annotation_position="top left",
                  annotation_font_size=11)

    # Shade the NO-GO period(s) on all three panels
    is_nogo = (view["decision"] == "NO-GO").astype(int)
    starts = view.loc[is_nogo.diff().fillna(is_nogo).eq(1), "cycle_end_time"]
    ends = view.loc[is_nogo.diff(-1).fillna(is_nogo).eq(1), "cycle_end_time"]
    for s, e in zip(starts, ends):
        fig.add_vrect(x0=s, x1=e, fillcolor=RED, opacity=0.08, line_width=0)

    # y-ranges from the normal data (1st-99th percentile), so a single sensor spike does not squash the chart
    for r, raw, rm, extra in [(1, "elution_flow_median", "rm_elution_flow_median", None),
                              (2, "pt_80_percentile", "rm_pt_80_percentile", tc.pressure_limit_bar)]:
        vals = pd.concat([view[raw].quantile([0.01, 0.99]), view[rm]]).dropna()
        lo, hi = vals.min(), vals.max()
        if extra is not None:
            lo, hi = min(lo, extra), max(hi, extra)
        pad = 0.08 * (hi - lo or 1)
        fig.update_yaxes(range=[lo - pad, hi + pad], row=r, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], row=3, col=1)
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=40, b=10), hovermode="closest",
                      legend=dict(orientation="h", y=1.08, x=0))
    st.plotly_chart(fig, width="stretch", theme="streamlit")
    st.caption("Hover over any point to see the replacement score at that time. Single-cycle spikes outside "
               "the normal range are off-chart (see the table below). "
               "Red shading = NO-GO period (rolling pressure ≥ limit AND score ≥ "
               f"{cfg.decision_threshold:.1f}).")

st.info(latest["reason"])

# ---------------------------------------------------------------- tables
st.subheader("Current status per gel batch")
table = summary.assign(status=summary["decision"].map({"GO": "✅ GO", "NO-GO": "🛑 NO-GO"}))
st.dataframe(table[["gel_batch", "chromatography_type", "latest_resin_age", "latest_cycle_end", "status", "score",
                    "rm_pt_80_percentile", "pressure_limit_bar", "first_no_go_age", "cycles", "incomplete_cycles"]],
             hide_index=True, width="stretch", height=36 * (len(table) + 1) + 3)
with st.expander(f"All cycles of {batch} (table view)"):
    st.dataframe(rows[["cycle_end_time", "resin_age", "decision", "score", "rm_elution_flow_median",
                       "rm_pt_80_percentile", "flag", "reason"]], hide_index=True, width="stretch")
