import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import COLOURS, NAMES, PLOT_LAYOUT, CHANNELS, footer, gbp, load, page_setup
from adstock.transforms import adstock_weights, half_life

page_setup(
    "Channel performance",
    "Return on investment (ROI) for each channel over the full four years, with the "
    "uncertainty the data leaves us. A wide bar means *we can't be sure yet*, which "
    "is a reason to run a test before moving big money.",
)
meta, post, df, fitted = load()
spend = df[CHANNELS].to_numpy()
contrib = post.media_contrib(spend).sum(1)
roi = contrib / spend.sum(0)

# ---- ROI with intervals ----------------------------------------------------
mean, lo, hi = roi.mean(0), np.percentile(roi, 5, 0), np.percentile(roi, 95, 0)
order = np.argsort(mean)
fig = go.Figure(go.Bar(
    x=mean[order], y=[NAMES[i] for i in order], orientation="h",
    marker_color=[COLOURS[CHANNELS[i]] for i in order],
    error_x=dict(type="data", symmetric=False, array=(hi - mean)[order], arrayminus=(mean - lo)[order]),
    hovertemplate="%{y}: £%{x:.2f} per £1<extra></extra>"))
fig.add_vline(x=1, line_dash="dash", line_color="grey", annotation_text="break-even")
fig.update_layout(height=320, title="Revenue per £1 spent (mean and 90% interval)",
                  xaxis_title="ROI (£ revenue per £1)", **PLOT_LAYOUT)
st.plotly_chart(fig, width="stretch")

# ---- spend share vs contribution share ------------------------------------
c1, c2 = st.columns(2)
spend_share = spend.sum(0) / spend.sum()
contrib_share = contrib.mean(0) / contrib.mean(0).sum()
fig = go.Figure()
fig.add_trace(go.Bar(x=NAMES, y=spend_share, name="Share of spend", marker_color="#bfbfbf"))
fig.add_trace(go.Bar(x=NAMES, y=contrib_share, name="Share of media revenue",
                     marker_color=[COLOURS[c] for c in CHANNELS]))
fig.update_layout(height=340, barmode="group", yaxis_tickformat=".0%",
                  title="Is each channel pulling its weight?", **PLOT_LAYOUT)
c1.plotly_chart(fig, width="stretch")

# ---- carryover ------------------------------------------------------------
w = adstock_weights(post.draws["alpha"].mean(0))  # (C, L)
fig = go.Figure()
for i, c in enumerate(CHANNELS):
    fig.add_trace(go.Scatter(x=np.arange(w.shape[1]), y=w[i], name=NAMES[i], mode="lines+markers",
                             line=dict(color=COLOURS[c])))
fig.update_layout(height=340, title="Carryover: how £1 of spend is felt over the following weeks",
                  xaxis_title="Weeks after spend", yaxis_tickformat=".0%", **PLOT_LAYOUT)
c2.plotly_chart(fig, width="stretch")

# ---- table --------------------------------------------------------------
hl = half_life(post.draws["alpha"])
table = pd.DataFrame({
    "Channel": NAMES,
    "Spend": [gbp(s) for s in spend.sum(0)],
    "Revenue driven": [gbp(c) for c in contrib.mean(0)],
    "ROI (mean)": [f"£{m:.2f}" for m in mean],
    "ROI 90% interval": [f"£{a:.2f} - £{b:.2f}" for a, b in zip(lo, hi)],
    "P(ROI > 1)": [f"{p:.0%}" for p in (roi > 1).mean(0)],
    "Carryover half-life (weeks)": [f"{np.median(h):.1f}" for h in hl.T],
})
st.dataframe(table, hide_index=True, width="stretch")
st.caption("P(ROI > 1) is the posterior probability the channel returns more revenue than it costs. "
           "ROI here is revenue, not profit - multiply by gross margin to judge profitability.")
footer()
