import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import COLOURS, NAMES, PLOT_LAYOUT, CHANNELS, footer, gbp, interval, load, page_setup
from adstock.planner import HORIZON, current_weekly_spend, optimise

page_setup(
    "Budget optimiser",
    "Given a quarterly budget, find the channel split that maximises expected revenue. "
    "Guard-rails stop the optimiser from cutting or scaling any channel more than you allow, "
    "because the model is only trustworthy near spend levels it has actually seen.",
)
meta, post, df, fitted = load()
cur = current_weekly_spend(df)
cur_q = cur.sum() * HORIZON

c1, c2 = st.columns([1, 1])
budget_q = c1.number_input("Quarterly media budget (£)", min_value=50_000, max_value=5_000_000,
                           value=int(round(cur_q, -3)), step=10_000,
                           help=f"Current run-rate is {gbp(cur_q)} per quarter.")
lo_pct, hi_pct = c2.slider("Guard-rails: each channel may move between", 0, 300, (50, 150), step=10,
                           format="%d%%", help="Percentage of the channel's current share of budget.")


@st.cache_data(show_spinner=False)
def solve(total_weekly, lo, hi):
    return optimise(post, total_weekly, cur, lo / 100, hi / 100)


a = solve(budget_q / HORIZON, lo_pct, hi_pct)
gain = (a.revenue_draws - a.baseline_draws) * HORIZON

m1, m2, m3 = st.columns(3)
m1.metric("Media revenue, current mix", gbp(a.baseline_draws.mean() * HORIZON),
          help=interval(a.baseline_draws * HORIZON))
m2.metric("Media revenue, optimised mix", gbp(a.revenue_draws.mean() * HORIZON),
          help=interval(a.revenue_draws * HORIZON))
m3.metric("Gain from re-allocating", gbp(gain.mean()),
          delta=f"{(gain > 0).mean():.0%} probability of improvement", delta_color="off",
          help=interval(gain))
if not a.success:
    st.warning("The optimiser did not fully converge for these settings - treat the split as indicative.")

fig = go.Figure()
fig.add_trace(go.Bar(x=NAMES, y=a.baseline_weekly * HORIZON, name="Current mix", marker_color="#bfbfbf"))
fig.add_trace(go.Bar(x=NAMES, y=a.weekly * HORIZON, name="Optimised", marker_color=[COLOURS[c] for c in CHANNELS]))
fig.update_layout(barmode="group", height=360, yaxis_title="Quarterly spend (£)",
                  title="Recommended quarterly split", **PLOT_LAYOUT)
st.plotly_chart(fig, width="stretch")

bound_lo = a.baseline_weekly * lo_pct / 100
bound_hi = a.baseline_weekly * hi_pct / 100
status = np.where(np.isclose(a.weekly, bound_hi, rtol=1e-3), "at upper guard-rail",
                  np.where(np.isclose(a.weekly, bound_lo, rtol=1e-3), "at lower guard-rail", "balanced"))
st.dataframe(pd.DataFrame({
    "Channel": NAMES,
    "Current (quarter)": [gbp(x) for x in a.baseline_weekly * HORIZON],
    "Optimised (quarter)": [gbp(x) for x in a.weekly * HORIZON],
    "Change": [f"{(o / c - 1):+.0%}" for o, c in zip(a.weekly, a.baseline_weekly)],
    "Marginal ROI at optimum": [f"£{m:.2f}" for m in a.marginal_roi],
    "Constraint": status,
}), hide_index=True, width="stretch")
st.caption("Channels marked *at upper guard-rail* would take even more money if allowed: they are "
           "the best candidates for a geo-holdout or lift test to confirm the effect before scaling.")

# ---- how big should the budget be? ----------------------------------------
st.subheader("How much should we spend in total?")


@st.cache_data(show_spinner=False)
def budget_curve(lo, hi):
    levels = np.linspace(0.25, 2.5, 19) * cur.sum()
    rows = []
    for b in levels:
        r = optimise(post, b, cur, lo / 100, hi / 100)
        rows.append((b * HORIZON, r.revenue_draws.mean() * HORIZON,
                     *np.percentile(r.revenue_draws * HORIZON, [5, 95])))
    return pd.DataFrame(rows, columns=["budget", "revenue", "lo", "hi"])


bc = budget_curve(lo_pct, hi_pct)
bc["marginal"] = bc["revenue"].diff() / bc["budget"].diff()
fig = go.Figure()
fig.add_trace(go.Scatter(x=bc["budget"], y=bc["hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=bc["budget"], y=bc["lo"], mode="lines", fill="tonexty", line=dict(width=0),
                         fillcolor="rgba(31,78,121,0.15)", name="90% interval", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=bc["budget"], y=bc["revenue"], name="Media revenue at optimal split",
                         line=dict(color="#1f4e79", width=3)))
fig.add_vline(x=cur_q, line_dash="dot", annotation_text="current budget")
fig.update_layout(height=380, xaxis_title="Quarterly budget (£)", yaxis_title="Quarterly media revenue (£)",
                  **PLOT_LAYOUT)
st.plotly_chart(fig, width="stretch")
near = bc.iloc[(bc["budget"] - cur_q).abs().argsort().iloc[0]]
st.caption(f"Around the current budget, each extra £1 spent (optimally split) returns roughly "
           f"£{near['marginal']:.2f} in revenue. Whether that is worth it depends on gross margin: "
           "with a 40% margin, £1 of spend needs £2.50 of revenue to break even.")

with st.expander("How the optimiser works"):
    st.markdown("""
For each posterior draw the model gives a saturating response curve per channel. The optimiser
(SciPy SLSQP) maximises the **average across all draws** of total weekly media revenue, subject to
the budget adding up and each channel staying inside its guard-rails. The comparison mix is the
current split scaled to the same budget, at a steady weekly spend, so the gain shown comes only
from *re-allocation*, not from spending more or from smoothing out flighted campaigns.
""")
footer()
