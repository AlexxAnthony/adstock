import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import COLOURS, NAMES, PLOT_LAYOUT, CHANNELS, footer, gbp, load, page_setup
from adstock.planner import current_weekly_spend

page_setup(
    "Response curves",
    "Each curve shows the weekly revenue a channel produces at a steady weekly spend. "
    "The curves flatten because of **diminishing returns**: the slope at today's spend "
    "(the *marginal ROI*) tells you what the **next** pound would earn, which is what "
    "matters for budget decisions, not the average ROI.",
)
meta, post, df, fitted = load()
cur = current_weekly_spend(df)

i = st.selectbox("Channel", range(len(CHANNELS)), format_func=lambda k: NAMES[k])
c = CHANNELS[i]
max_x = max(cur[i] * 4, df[c].max())
grid = np.linspace(0, max_x, 120)
spend_grid = np.zeros((len(grid), len(CHANNELS)))
spend_grid[:, i] = grid
resp = post.weekly_response(spend_grid)[:, :, i]  # (D, G)
mean, lo, hi = resp.mean(0), np.percentile(resp, 5, 0), np.percentile(resp, 95, 0)

now = np.zeros((1, len(CHANNELS))); now[0, i] = cur[i]
bump = now.copy(); bump[0, i] += 1000
r_now = post.weekly_response(now)[:, 0, i]
m_roi = (post.weekly_response(bump)[:, 0, i] - r_now) / 1000

col1, col2, col3 = st.columns(3)
col1.metric("Current weekly spend (last 52 wks avg)", gbp(cur[i]))
col2.metric("Average ROI at that spend", f"£{(r_now / cur[i]).mean():.2f}")
col3.metric("Marginal ROI (next £1k)", f"£{m_roi.mean():.2f}",
            help=f"90% interval £{np.percentile(m_roi, 5):.2f} to £{np.percentile(m_roi, 95):.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=grid, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=grid, y=lo, line=dict(width=0), fill="tonexty", name="90% interval",
                         fillcolor="rgba(127,127,127,0.2)", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=grid, y=mean, name="Expected weekly revenue", line=dict(color=COLOURS[c], width=3)))
fig.add_trace(go.Scatter(x=grid, y=grid, name="Break-even (revenue = spend)",
                         line=dict(color="grey", dash="dash")))
fig.add_vline(x=cur[i], line_dash="dot", annotation_text="current")
fig.update_layout(height=430, xaxis_title="Weekly spend (£)", yaxis_title="Weekly revenue (£)",
                  title=f"{NAMES[i]}: diminishing returns", **PLOT_LAYOUT)
st.plotly_chart(fig, width="stretch")

st.subheader("All channels, marginal ROI at current spend")
bumped = np.tile(cur, (len(CHANNELS), 1)) + np.eye(len(CHANNELS)) * 1000
base = post.weekly_response(cur)
mr = np.stack([(post.weekly_response(bumped[k])[:, k] - base[:, k]) / 1000 for k in range(len(CHANNELS))], 1)
order = np.argsort(mr.mean(0))
fig = go.Figure(go.Bar(x=mr.mean(0)[order], y=[NAMES[k] for k in order], orientation="h",
                       marker_color=[COLOURS[CHANNELS[k]] for k in order]))
fig.add_vline(x=1, line_dash="dash", line_color="grey")
fig.update_layout(height=280, xaxis_title="Revenue from the next £1", **PLOT_LAYOUT)
st.plotly_chart(fig, width="stretch")
st.caption("Money should flow from channels with low marginal ROI to those with high marginal ROI "
           "until they even out. That is exactly what the Budget optimiser does.")
footer()
