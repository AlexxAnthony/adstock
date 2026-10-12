import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import (COLOURS, NAMES, PLOT_LAYOUT, CHANNELS, footer, gbp, load, page_setup)
from adstock.planner import current_weekly_spend, optimise

page_setup(
    "Adstock - Marketing Mix Model & Budget Optimiser",
    "Which marketing channels actually drive revenue, how long does each one keep "
    "working after the spend stops, and how should next quarter's budget be split? "
    "This dashboard answers those questions with a **Bayesian marketing mix model (MMM)** "
    "that uses aggregate weekly data only - no cookies or user-level tracking.",
)
meta, post, df, fitted = load()

spend = df[CHANNELS].to_numpy()
contrib = post.media_contrib(spend).sum(1)              # (D, C) total over 4 years
roi = contrib / spend.sum(0)
media_total = contrib.sum(1)
blended = media_total / spend.sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue (4 years)", gbp(df["revenue"].sum()))
c2.metric("Media spend", gbp(spend.sum()))
c3.metric("Revenue driven by media", f"{media_total.mean() / df['revenue'].sum():.0%}",
          help="Share of revenue the model attributes to paid media; the rest is baseline "
               "demand (market conditions, seasonality, newsletter, trend).")
c4.metric("Blended media ROI", f"£{blended.mean():.2f} per £1",
          help=f"90% interval £{np.percentile(blended, 5):.2f} to £{np.percentile(blended, 95):.2f}")

# ---- headline findings, generated from the posterior -------------------------
order = np.argsort(-roi.mean(0))
best, worst = order[0], order[-1]
safe = int(np.argmax(np.percentile(roi, 5, axis=0)))  # highest pessimistic-case ROI
share_spend = spend.sum(0) / spend.sum()
cur = current_weekly_spend(df)
alloc = optimise(post, cur.sum(), cur)
uplift = (alloc.revenue_draws - alloc.baseline_draws) * 13
p_up = (uplift > 0).mean()

st.subheader("What the model says")
st.markdown(f"""
- **{NAMES[worst]} takes {share_spend[worst]:.0%} of the budget but returns about £{roi[:, worst].mean():.2f}
  per £1** (90% interval £{np.percentile(roi[:, worst], 5):.2f}-£{np.percentile(roi[:, worst], 95):.2f}),
  the weakest of the five channels.
- **{NAMES[safe]} is the most dependable earner**: £{roi[:, safe].mean():.2f} per £1, and even the
  pessimistic end of its interval is £{np.percentile(roi[:, safe], 5):.2f}.
  {NAMES[best]} has the highest average return (£{roi[:, best].mean():.2f}), but the model is far less
  certain about it (£{np.percentile(roi[:, best], 5):.2f}-£{np.percentile(roi[:, best], 95):.2f}).
- **Re-allocating the same budget** within ±50% guard-rails is worth about
  **{gbp(uplift.mean())} extra revenue per quarter**
  ({p_up:.0%} probability it beats the current mix). See *Budget optimiser*.
""")

# ---- fit chart --------------------------------------------------------------
st.subheader("Model fit: actual vs modelled revenue")
fig = go.Figure()
fig.add_trace(go.Scatter(x=fitted["date"], y=fitted["fitted_hi"], line=dict(width=0),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fitted["date"], y=fitted["fitted_lo"], line=dict(width=0), fill="tonexty",
                         fillcolor="rgba(31,78,121,0.15)", name="90% interval", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fitted["date"], y=fitted["actual"], name="Actual",
                         line=dict(color="black", width=1.5)))
fig.add_trace(go.Scatter(x=fitted["date"], y=fitted["fitted"], name="Model",
                         line=dict(color=COLOURS["tv_S"], width=1.5, dash="dot")))
fig.update_layout(height=360, yaxis_title="Weekly revenue (£)", **PLOT_LAYOUT)
st.plotly_chart(fig, width="stretch")
st.caption(f"In-sample R² {meta['in_sample']['r2']:.2f}, MAPE {meta['in_sample']['mape']:.1f}%. "
           f"On 26 weeks the model never saw, MAPE was {meta['holdout']['mmm']['mape']:.1f}% "
           "(see *Model validation*).")

# ---- decomposition ----------------------------------------------------------
st.subheader("Where weekly revenue comes from")
fig = go.Figure()
fig.add_trace(go.Scatter(x=fitted["date"], y=fitted["baseline"], name="Baseline (non-media)",
                         stackgroup="one", line=dict(width=0), fillcolor=COLOURS["baseline"]))
for c, n in zip(CHANNELS, NAMES):
    fig.add_trace(go.Scatter(x=fitted["date"], y=fitted[c], name=n, stackgroup="one",
                             line=dict(width=0), fillcolor=COLOURS[c]))
fig.update_layout(height=380, yaxis_title="Weekly revenue (£)", **PLOT_LAYOUT)
st.plotly_chart(fig, width="stretch")
st.caption("Baseline covers competitor/market sales, the newsletter, seasonality, trend and two "
           "one-off events. Media contributions include carryover from earlier weeks' spend.")
footer()
