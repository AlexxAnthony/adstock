"""Shared loading, formatting and styling for every dashboard page."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adstock.data import CHANNEL_NAMES, CHANNELS, load_raw  # noqa: E402
from adstock.posterior import Posterior  # noqa: E402

ART = ROOT / "artifacts"
GITHUB = "https://github.com/AlexxAnthony/adstock"

COLOURS = {
    "tv_S": "#1f4e79", "ooh_S": "#c55a11", "print_S": "#7f6000",
    "facebook_S": "#2e75b6", "search_S": "#548235", "baseline": "#bfbfbf",
}
NAMES = [CHANNEL_NAMES[c] for c in CHANNELS]


@st.cache_resource
def load():
    meta = json.loads((ART / "meta.json").read_text())
    draws = dict(np.load(ART / "posterior_draws.npz"))
    post = Posterior(draws, np.array(meta["media_scale"]), np.array(meta["ctrl_scale"]),
                     meta["y_scale"], meta["n_weeks"])
    df = load_raw()
    fitted = pd.read_csv(ART / "fitted.csv", parse_dates=["date"])
    return meta, post, df, fitted


def gbp(x: float, dp: int = 1) -> str:
    x = float(x)
    if abs(x) >= 1e6:
        return f"£{x / 1e6:.{dp}f}m"
    if abs(x) >= 1e3:
        return f"£{x / 1e3:.{dp if abs(x) < 1e4 else 0}f}k"
    return f"£{x:.0f}"


def interval(a: np.ndarray, fmt=gbp) -> str:
    lo, hi = np.percentile(a, [5, 95])
    return f"90% interval {fmt(lo)} to {fmt(hi)}"


def page_setup(title: str, intro: str | None = None):
    st.set_page_config(page_title=f"{title} | Adstock MMM", page_icon="📈", layout="wide")
    st.title(title)
    if intro:
        st.markdown(intro)


def footer():
    st.divider()
    st.caption(
        "Data: Meta Robyn simulated weekly dataset (208 weeks, Nov 2015 - Nov 2019), "
        "shown in £ for readability. Bayesian model fitted in PyMC; every number with a "
        f"range is computed across 1,000 posterior draws. Code: [{GITHUB}]({GITHUB})"
    )


PLOT_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.28),
                   hovermode="x unified")
