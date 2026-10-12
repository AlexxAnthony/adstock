"""Fit the MMM twice and save everything the dashboard needs.

1. Holdout fit: train on the first 182 weeks, score the last 26 weeks it never
   saw, and compare against simple baselines.
2. Full fit: train on all 208 weeks; these draws drive the dashboard.

Run:  python scripts/fit_model.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adstock.data import CHANNELS, CONTROLS, load_raw, prepare  # noqa: E402
from adstock.model import extract_draws, fit  # noqa: E402
from adstock.posterior import Posterior  # noqa: E402

ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
HOLDOUT_WEEKS = 26


def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def diagnostics(idata) -> dict:
    s = az.summary(idata, var_names=["intercept", "trend_coef", "fourier_coef", "ctrl_coef",
                                     "event_coef", "alpha", "lam", "beta", "sigma"])
    return {"max_r_hat": float(s["r_hat"].max()),
            "min_ess_bulk": float(s["ess_bulk"].min()),
            "divergences": int(idata.sample_stats["diverging"].sum())}


def posterior_obj(draws, p) -> Posterior:
    return Posterior(draws, p.media_scale, p.ctrl_scale, p.y_scale, len(p.df))


def predict(post: Posterior, df: pd.DataFrame, rng) -> np.ndarray:
    """Posterior predictive revenue (D, T) in pounds, including noise."""
    base = post.baseline(df["date"], np.arange(len(df)), df[CONTROLS].to_numpy(),
                         df["event"].to_numpy())
    mu = base + post.media_contrib(df[CHANNELS].to_numpy()).sum(axis=2)
    noise = rng.normal(0, 1, mu.shape) * post.draws["sigma"][:, None] * post.y_scale
    return mu, mu + noise


def main():
    rng = np.random.default_rng(7)
    df = load_raw()
    T = len(df)
    train = slice(0, T - HOLDOUT_WEEKS)
    test_idx = np.arange(T - HOLDOUT_WEEKS, T)

    # ---------------- 1. holdout fit ----------------
    t0 = time.time()
    p_h = prepare(df, fit_rows=train)
    _, idata_h = fit(p_h, obs_idx=np.arange(T - HOLDOUT_WEEKS))
    print(f"holdout fit {time.time() - t0:.0f}s", diagnostics(idata_h))
    post_h = posterior_obj(extract_draws(idata_h), p_h)

    # consistency check: numpy forward model must reproduce PyMC's mu
    mu_np, pred = predict(post_h, df, rng)
    mu_pm = idata_h.posterior["mu"].mean(("chain", "draw")).values * p_h.y_scale
    gap = np.max(np.abs(mu_np.mean(0) - mu_pm) / mu_pm)
    print(f"numpy vs PyMC max relative gap: {gap:.4%}")
    assert gap < 0.01, "numpy forward model disagrees with PyMC"

    y = df["revenue"].to_numpy()
    y_test = y[test_idx]
    mmm_point = mu_np.mean(0)[test_idx]
    lo, hi = np.percentile(pred[:, test_idx], [5, 95], axis=0)

    # baseline A: same week last year
    seasonal_naive = y[test_idx - 52]
    # baseline B: OLS on controls + trend + seasonality, no media at all
    Xb = np.column_stack([np.ones(T), p_h.trend, p_h.fourier, p_h.X_ctrl, p_h.event])
    coef, *_ = np.linalg.lstsq(Xb[train], y[train], rcond=None)
    no_media = Xb[test_idx] @ coef

    holdout = {
        "weeks": HOLDOUT_WEEKS,
        "mmm": {"mape": mape(y_test, mmm_point), "r2": r2(y_test, mmm_point),
                "coverage_90": float(np.mean((y_test >= lo) & (y_test <= hi)) * 100)},
        "seasonal_naive": {"mape": mape(y_test, seasonal_naive), "r2": r2(y_test, seasonal_naive)},
        "no_media_regression": {"mape": mape(y_test, no_media), "r2": r2(y_test, no_media)},
        "diagnostics": diagnostics(idata_h),
    }
    pd.DataFrame({"date": df["date"].iloc[test_idx], "actual": y_test, "mmm": mmm_point,
                  "mmm_lo": lo, "mmm_hi": hi, "seasonal_naive": seasonal_naive,
                  "no_media": no_media}).to_csv(ART / "holdout.csv", index=False)
    # in-sample fit for the training period too (shown greyed out)
    in_lo, in_hi = np.percentile(pred[:, train], [5, 95], axis=0)
    pd.DataFrame({"date": df["date"].iloc[train], "actual": y[train],
                  "mmm": mu_np.mean(0)[train], "mmm_lo": in_lo, "mmm_hi": in_hi}
                 ).to_csv(ART / "holdout_train.csv", index=False)
    print(json.dumps(holdout, indent=2))

    # ---------------- 2. full fit ----------------
    t0 = time.time()
    p = prepare(df)
    _, idata = fit(p)
    diag = diagnostics(idata)
    print(f"full fit {time.time() - t0:.0f}s", diag)
    draws = extract_draws(idata)
    np.savez_compressed(ART / "posterior_draws.npz", **draws)

    post = posterior_obj(draws, p)
    mu, pred = predict(post, df, rng)
    contrib = post.media_contrib(df[CHANNELS].to_numpy())  # (D, T, C)
    fitted = pd.DataFrame({"date": df["date"], "actual": y, "fitted": mu.mean(0),
                           "fitted_lo": np.percentile(pred, 5, 0),
                           "fitted_hi": np.percentile(pred, 95, 0),
                           "baseline": (mu - contrib.sum(2)).mean(0)})
    for i, c in enumerate(CHANNELS):
        fitted[c] = contrib[:, :, i].mean(0)
    fitted.to_csv(ART / "fitted.csv", index=False)

    meta = {
        "channels": CHANNELS, "controls": CONTROLS,
        "media_scale": p.media_scale.tolist(), "ctrl_scale": p.ctrl_scale.tolist(),
        "y_scale": p.y_scale, "n_weeks": T,
        "date_start": str(df["date"].min().date()), "date_end": str(df["date"].max().date()),
        "in_sample": {"mape": mape(y, mu.mean(0)), "r2": r2(y, mu.mean(0))},
        "holdout": holdout, "diagnostics": diag,
    }
    (ART / "meta.json").write_text(json.dumps(meta, indent=2))
    print("saved artifacts", json.dumps(meta["in_sample"]))


if __name__ == "__main__":
    main()
