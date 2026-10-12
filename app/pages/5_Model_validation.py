import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import ART, NAMES, PLOT_LAYOUT, footer, load, page_setup

page_setup(
    "Model validation",
    "A marketing mix model is only useful if it is right about *why* revenue moves, not just "
    "*how much*. This page shows three checks: an out-of-sample forecast test, sampler "
    "diagnostics, and a parameter-recovery test where the true answer is known.",
)
meta, post, df, fitted = load()
h = meta["holdout"]

# ---- 1. holdout -------------------------------------------------------------
st.subheader("1. Forecasting 26 weeks the model never saw")
ho = pd.read_csv(ART / "holdout.csv", parse_dates=["date"])
tr = pd.read_csv(ART / "holdout_train.csv", parse_dates=["date"]).tail(52)
fig = go.Figure()
fig.add_trace(go.Scatter(x=tr["date"], y=tr["actual"], name="Training data", line=dict(color="#999")))
fig.add_trace(go.Scatter(x=ho["date"], y=ho["mmm_hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=ho["date"], y=ho["mmm_lo"], fill="tonexty", line=dict(width=0),
                         fillcolor="rgba(31,78,121,0.15)", name="MMM 90% interval", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=ho["date"], y=ho["actual"], name="Actual (holdout)", line=dict(color="black", width=2)))
fig.add_trace(go.Scatter(x=ho["date"], y=ho["mmm"], name="MMM forecast", line=dict(color="#1f4e79", width=2)))
fig.add_trace(go.Scatter(x=ho["date"], y=ho["seasonal_naive"], name="Same week last year",
                         line=dict(color="#c55a11", dash="dot")))
fig.add_vline(x=ho["date"].iloc[0], line_dash="dash", line_color="grey")
fig.update_layout(height=380, yaxis_title="Weekly revenue (£)", **PLOT_LAYOUT)
st.plotly_chart(fig, width="stretch")

st.dataframe(pd.DataFrame({
    "Model": ["Bayesian MMM", "Regression without media", "Same week last year (naive)"],
    "Holdout MAPE": [f"{h['mmm']['mape']:.1f}%", f"{h['no_media_regression']['mape']:.1f}%",
                     f"{h['seasonal_naive']['mape']:.1f}%"],
    "Holdout R²": [f"{h['mmm']['r2']:.3f}", f"{h['no_media_regression']['r2']:.3f}",
                   f"{h['seasonal_naive']['r2']:.3f}"],
}), hide_index=True, width="stretch")
st.markdown(f"""
- The MMM beats the naive baseline by a wide margin. **{h['mmm']['coverage_90']:.0f}%** of holdout weeks
  fell inside its 90% interval (ideal ≈ 90%; higher means the intervals are on the cautious side).
- **Honest caveat:** a regression with *no media at all* forecasts almost as well, because
  competitor/market sales explain most week-to-week revenue in this dataset. Forecast accuracy
  therefore cannot prove the media effects are right - which is why check 3 exists.
- The holdout model used competitor sales for the holdout weeks, as you would when re-scoring a
  past period. A true forward forecast has to estimate them (see *Scenario planner*).
""")

# ---- 2. diagnostics ----------------------------------------------------------
st.subheader("2. Did the sampler converge?")
d = meta["diagnostics"]
c1, c2, c3 = st.columns(3)
c1.metric("Max R-hat", f"{d['max_r_hat']:.3f}", help="Should be below 1.01: chains agree.")
c2.metric("Min effective sample size", f"{d['min_ess_bulk']:.0f}", help="Should be above ~400.")
c3.metric("Divergent transitions", d["divergences"], help="Should be zero or close to it.")
st.caption("4 chains × 1,000 draws with the NUTS sampler (target_accept 0.95).")

# ---- 3. recovery ------------------------------------------------------------
st.subheader("3. Parameter recovery: can it find a known truth?")
rec_path = ART / "recovery.json"
if rec_path.exists():
    rec = json.loads(rec_path.read_text())
    true = np.array(rec["true_roi"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=NAMES, y=true, mode="markers", name="True ROI",
                             marker=dict(symbol="x", size=14, color="black")))
    for j, run in enumerate(rec["runs"]):
        est, lo, hi = map(np.array, (run["est"], run["lo"], run["hi"]))
        fig.add_trace(go.Scatter(x=[f"{n}" for n in NAMES], y=est, mode="markers",
                                 name=f"Recovered (simulation {j + 1})", offsetgroup=j,
                                 error_y=dict(type="data", symmetric=False, array=hi - est, arrayminus=est - lo)))
    fig.update_layout(height=380, yaxis_title="ROI (£ per £1)", scattermode="group", **PLOT_LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"""
Revenue was simulated three times from a known set of parameters using the real spend pattern,
then the model was refitted from scratch each time. **The true ROI fell inside the 90% interval in
{rec['coverage_rate']:.0f}% of channel-simulation cases.** Channels with lots of spend variation are
pinned down tightly; channels that rarely change spend come back with wide intervals - the model
says *I don't know* rather than inventing precision.
""")
else:
    st.info("Run `python scripts/recovery_test.py` to generate this check.")

# ---- limitations --------------------------------------------------------------
st.subheader("Limitations")
st.markdown("""
- **Correlation, not experiment.** An MMM is observational. High-stakes shifts (anything at a
  guard-rail) should be confirmed with a geo-holdout or conversion-lift test, and those results can
  be fed back in as priors.
- **Simulated data.** The Robyn dataset is synthetic, so findings illustrate the method rather than a
  real brand's results.
- **National, weekly, five channels.** No geo-level or creative-level effects; Facebook and search
  have wide intervals because their spend varies little.
- **Static effects.** ROI is assumed constant over four years; a time-varying model would catch
  channels that wear out.
""")
footer()
