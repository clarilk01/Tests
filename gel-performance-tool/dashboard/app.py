"""Step 5 - Streamlit dashboard for the Gel Performance Tool (synthetic data).

The dashboard only READS the files written by the scheduled run; it never recomputes scores,
so what you see is exactly what the job produced.

    python -m gel_performance.run --input data/synthetic/cycles.csv --output output --config config/config.yaml
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works even without `pip install -e .`

from gel_performance.config import FEATURES, load_config  # noqa: E402
from gel_performance.report import SHORT  # noqa: E402

OUTPUT_DIR = ROOT / "output"
CONFIG_PATH = ROOT / "config" / "config.yaml"

# Colours (validated reference palette): one highlight hue, neutral context, reserved status colours.
HIGHLIGHT = "#2a78d6"          # selected batch / points lost
CONTEXT = "rgba(140,140,135,0.45)"
GAINED = "#8c8c87"             # points gained (feature better than reference)
STATUS = {"OK": "#0ca30c", "WATCH": "#fab219", "REPLACE SOON": "#d03b3b"}
ICON = {"OK": "✅", "WATCH": "⚠️", "REPLACE SOON": "🛑", "NO SCORE": "❔"}

st.set_page_config(page_title="Gel Performance Tool", page_icon="🧪", layout="wide")


@st.cache_data
def load_outputs(folder: str, stamp: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the run outputs. `stamp` (file modification time) refreshes the cache after a new run."""
    cycles = pd.read_csv(Path(folder) / "cycle_scores.csv")
    summary = pd.read_csv(Path(folder) / "batch_summary.csv")
    return cycles, summary


scores_file = OUTPUT_DIR / "cycle_scores.csv"
if not scores_file.exists():
    st.error("No results found. Run the scoring job first:")
    st.code("python -m gel_performance.run --input data/synthetic/cycles.csv --output output "
            "--config config/config.yaml", language="bash")
    st.stop()

cycles, summary = load_outputs(str(OUTPUT_DIR), scores_file.stat().st_mtime)
cfg = load_config(CONFIG_PATH)

# ---------------------------------------------------------------- header + sidebar filters
st.title("Gel Performance Tool")
st.caption("⚠️ Synthetic data only - demonstrates the method, not real-world performance. "
           f"Results produced with config_version {cycles['config_version'].iloc[0]}.")

# Optional link parameters, e.g. ?batch=GB-A01&age=250 (handy for sharing a specific cycle)
qp = st.query_params
qp_batch = qp.get("batch")
types = sorted(cycles["chromatography_type"].unique())
qp_type = cycles.loc[cycles["gel_batch"] == qp_batch, "chromatography_type"].iloc[0] \
    if qp_batch in set(cycles["gel_batch"]) else types[0]

with st.sidebar:
    st.header("Select")
    ctype = st.selectbox("Chromatography type", types, index=types.index(qp_type))
    batches = sorted(cycles.loc[cycles["chromatography_type"] == ctype, "gel_batch"].unique())
    batch = st.selectbox("Gel batch", batches, index=batches.index(qp_batch) if qp_batch in batches else 0)
    batch_rows = cycles[cycles["gel_batch"] == batch].sort_values("resin_age")
    ages = batch_rows["resin_age"].tolist()
    qp_age = int(qp["age"]) if qp.get("age", "").isdigit() and int(qp["age"]) in ages else ages[-1]
    age = st.select_slider("Cycle (resin_age)", options=ages, value=qp_age)
    st.caption("Defaults to the latest cycle of the batch.")

tc = cfg.types[ctype]
row = batch_rows[batch_rows["resin_age"] == age].iloc[0]
rec = row["recommendation"]

# ---------------------------------------------------------------- recommendation card
c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
c1.markdown(f"### {ICON.get(rec, '')} {rec}")
c1.caption(f"{batch} · cycle {age} · {row['cycle_end_time']}")
c2.metric("Smoothed score", "–" if pd.isna(row["smoothed_score"]) else f"{row['smoothed_score']:.0f} / 100",
          help=f"Rolling median of the last {cfg.smoothing_window} scores. Drives the recommendation.")
c3.metric("This cycle's score", "–" if pd.isna(row["score"]) else f"{row['score']:.0f} / 100")
c4.metric("Data flag", row["flag"], help="complete / incomplete (re-normalised weights) / insufficient (no score)")
st.info(row["reason"])

# ---------------------------------------------------------------- chart 1: score vs resin_age
left, right = st.columns([3, 2])
with left:
    st.subheader(f"Smoothed score vs resin age - {ctype}")
    fig = go.Figure()
    # Status bands from the config thresholds (faint, labelled - colour is never the only cue)
    for lo, hi, label in [(tc.ok_min, 100, "OK"), (tc.watch_min, tc.ok_min, "WATCH"), (0, tc.watch_min, "REPLACE SOON")]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=STATUS[label], opacity=0.08, line_width=0)
        fig.add_annotation(x=1.01, xref="paper", xanchor="left", y=(lo + hi) / 2, yref="y",
                           text=f"{ICON[label]} {label}", showarrow=False, font_size=12)
    for b in batches:
        d = cycles[(cycles["gel_batch"] == b) & cycles["smoothed_score"].notna()].sort_values("resin_age")
        selected = b == batch
        first_other = not selected and b == next(x for x in batches if x != batch)
        fig.add_trace(go.Scatter(
            x=d["resin_age"], y=d["smoothed_score"], mode="lines",
            name=b if selected else "other batches", legendgroup="sel" if selected else "other",
            showlegend=selected or first_other,
            line=dict(color=HIGHLIGHT if selected else CONTEXT, width=2.5 if selected else 1.5),
            customdata=d[["recommendation"]],
            hovertemplate=f"<b>{b}</b><br>cycle %{{x}}<br>smoothed score %{{y:.1f}}<br>%{{customdata[0]}}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[age], y=[row["smoothed_score"]], mode="markers", name="selected cycle",
        marker=dict(size=11, color=HIGHLIGHT, line=dict(color="white", width=2)),
        hovertemplate="selected cycle %{x}<br>smoothed score %{y:.1f}<extra></extra>",
    ))
    # Put the selected batch on top
    fig.data = tuple(sorted(fig.data, key=lambda t: t.name in (batch, "selected cycle")))
    fig.update_layout(
        height=430, margin=dict(l=10, r=120, t=10, b=10), hovermode="closest",
        xaxis_title="resin_age (cycles)", yaxis_title="smoothed score", yaxis_range=[-2, 102],
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.caption(f"Highlighted: {batch}. Grey: other {ctype} batches for context. "
               f"Bands: OK ≥ {tc.ok_min:.0f}, WATCH ≥ {tc.watch_min:.0f}, below = REPLACE SOON.")

# ---------------------------------------------------------------- chart 2: contributions
with right:
    st.subheader(f"What drives cycle {age}")
    contrib = pd.DataFrame({
        "feature": [SHORT[f] for f in FEATURES],
        "points": [row[f"pts_{f}"] for f in FEATURES],
        "weight": [row[f"w_{f}"] for f in FEATURES],
        "z": [row[f"z_{f}"] for f in FEATURES],
    })
    missing = contrib["points"].isna()
    shown = contrib[~missing].sort_values("points")
    bar = go.Figure(go.Bar(
        x=shown["points"], y=shown["feature"], orientation="h",
        marker=dict(color=[HIGHLIGHT if p > 0 else GAINED for p in shown["points"]]),
        text=[f"{p:+.1f}" for p in -shown["points"]], textposition="outside",
        customdata=shown[["z", "weight"]],
        hovertemplate="<b>%{y}</b><br>points lost %{x:.1f}<br>oriented z %{customdata[0]:.2f}"
                      "<br>weight used %{customdata[1]:.2f}<extra></extra>",
    ))
    span = shown["points"].abs().max() if len(shown) else 1
    bar.update_xaxes(range=[min(0, shown["points"].min()) - 0.25 * span, max(0, shown["points"].max()) + 0.25 * span])
    bar.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="score points lost (blue) / gained (grey)")
    st.plotly_chart(bar, use_container_width=True, theme="streamlit")
    if missing.any():
        st.warning("Missing this cycle (not filled in, weight 0): "
                   + ", ".join(contrib.loc[missing, "feature"]))
    if pd.notna(row["score"]):
        st.caption(f"100 − {contrib['points'].sum():.1f} points lost = {row['score']:.1f} "
                   "(capped to 0–100). Higher z = worse than fresh resin.")

# ---------------------------------------------------------------- tables
st.subheader("Current status per gel batch")
view = summary.assign(status=summary["recommendation"].map(lambda r: f"{ICON.get(r, '')} {r}"))
st.dataframe(
    view[["gel_batch", "chromatography_type", "latest_resin_age", "smoothed_score", "status",
          "first_replace_soon_age", "cycles", "incomplete_cycles"]],
    hide_index=True, use_container_width=True, height=36 * (len(view) + 1) + 3,
)
with st.expander(f"All cycles of {batch} (table view)"):
    st.dataframe(batch_rows[["resin_age", "cycle_end_time", "score", "smoothed_score", "recommendation",
                             "flag", "missing_features", "reason"]],
                 hide_index=True, use_container_width=True)
