"""Budget planning: scenario forecasts and the budget optimiser.

Both work across every posterior draw, so each answer carries a credible
interval rather than a single point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import CHANNELS, CONTROLS
from .posterior import Posterior
from .transforms import L_MAX

HORIZON = 13  # one quarter of weekly planning


def current_weekly_spend(df: pd.DataFrame, weeks: int = 52) -> np.ndarray:
    """Average weekly spend per channel over the last `weeks` weeks."""
    return df[CHANNELS].tail(weeks).mean().to_numpy()


# ---------------------------------------------------------------- forecasting
def future_frame(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Next `horizon` weeks. Controls we cannot know in advance (competitor
    sales, newsletter volume) are assumed equal to the same weeks last year."""
    last = df["date"].iloc[-1]
    dates = pd.date_range(last + pd.Timedelta(weeks=1), periods=horizon, freq="7D")
    same_weeks_last_year = df.iloc[len(df) - 52: len(df) - 52 + horizon]
    fut = pd.DataFrame({"date": dates})
    for c in CONTROLS:
        fut[c] = same_weeks_last_year[c].to_numpy()
    fut["event"] = 0
    return fut


def forecast(post: Posterior, df: pd.DataFrame, weekly_plan: np.ndarray,
             horizon: int = HORIZON, rng_seed: int = 0) -> dict:
    """Forecast revenue for the next `horizon` weeks if each channel spends
    `weekly_plan[c]` per week. History is prepended so carryover from recent
    campaigns flows into the forecast window."""
    fut = future_frame(df, horizon)
    hist_spend = df[CHANNELS].to_numpy()[-L_MAX:]
    plan = np.tile(np.asarray(weekly_plan, float), (horizon, 1))
    contrib = post.media_contrib(np.vstack([hist_spend, plan]))[:, L_MAX:, :]  # (D, H, C)
    t_index = np.arange(len(df), len(df) + horizon)
    base = post.baseline(fut["date"], t_index, fut[CONTROLS].to_numpy(), fut["event"].to_numpy())
    mu = base + contrib.sum(2)
    rng = np.random.default_rng(rng_seed)
    pred = mu + rng.normal(0, 1, mu.shape) * post.draws["sigma"][:, None] * post.y_scale

    # incremental revenue from the plan's own spend: compare with zero future spend
    zero = post.media_contrib(np.vstack([hist_spend, np.zeros_like(plan)]))[:, L_MAX:, :]
    incremental = (contrib - zero).sum((1, 2))  # (D,)
    return {"dates": fut["date"], "mu": mu, "pred": pred, "media": contrib,
            "total_revenue": pred.sum(1), "incremental": incremental,
            "spend": plan.sum()}


# ---------------------------------------------------------------- optimiser
@dataclass
class Allocation:
    weekly: np.ndarray            # (C,) optimal weekly spend
    revenue_draws: np.ndarray     # (D,) weekly media revenue at optimum
    baseline_weekly: np.ndarray   # (C,) comparison mix at the same total
    baseline_draws: np.ndarray    # (D,)
    marginal_roi: np.ndarray      # (C,) extra revenue per extra pound at optimum
    success: bool


def expected_response(post: Posterior, weekly: np.ndarray) -> float:
    return float(post.weekly_response(weekly).sum(-1).mean())


def marginal_roi(post: Posterior, weekly: np.ndarray, step: float = 100.0) -> np.ndarray:
    base = post.weekly_response(weekly).mean(0)
    out = []
    for c in range(len(weekly)):
        bumped = weekly.copy(); bumped[c] += step
        out.append((post.weekly_response(bumped).mean(0)[c] - base[c]) / step)
    return np.array(out)


def optimise(post: Posterior, total_weekly: float, reference: np.ndarray,
             lower: float = 0.5, upper: float = 1.5) -> Allocation:
    """Maximise expected weekly media revenue for a fixed weekly budget.

    Each channel is held within [lower, upper] x its reference share of the
    budget - the same guard-rails marketing teams use so a model never tells
    them to switch a channel off overnight.
    reference: current weekly spend mix (its total is rescaled to the budget).
    """
    ref = reference / reference.sum() * total_weekly
    bounds = [(lower * r, upper * r) for r in ref]
    scale = total_weekly  # optimise in budget-share units for numerical stability

    def neg(x):
        return -expected_response(post, x * scale) / scale

    res = minimize(neg, ref / scale, method="SLSQP", bounds=[(lo / scale, hi / scale) for lo, hi in bounds],
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-12})
    best = res.x * scale
    return Allocation(
        weekly=best,
        revenue_draws=post.weekly_response(best).sum(-1),
        baseline_weekly=ref,
        baseline_draws=post.weekly_response(ref).sum(-1),
        marginal_roi=marginal_roi(post, best),
        success=bool(res.success),
    )
