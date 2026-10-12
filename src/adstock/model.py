"""Bayesian marketing mix model written directly in PyMC.

revenue_t = intercept
          + trend_coef * t
          + seasonality_t (yearly Fourier terms)
          + controls_t (competitor sales, newsletter) + event_t
          + sum_c beta_c * saturation_c(adstock_c(spend_c))_t
          + noise

All quantities are on the scaled revenue axis (revenue / max revenue).
"""
from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .data import Prepared
from .transforms import L_MAX

PARAMS = ["intercept", "trend_coef", "fourier_coef", "ctrl_coef", "event_coef",
          "alpha", "lam", "beta", "sigma"]


def lagged_stack(x: np.ndarray, l_max: int = L_MAX) -> np.ndarray:
    T = x.shape[0]
    out = np.zeros((l_max,) + x.shape)
    for lag in range(l_max):
        out[lag, lag:] = x[: T - lag]
    return out


def build_model(p: Prepared, obs_idx: np.ndarray | None = None) -> pm.Model:
    """obs_idx: rows whose revenue enters the likelihood (defaults to all).
    Spend from every row is still used so carryover into the holdout is right."""
    T, C = p.X_media.shape
    obs_idx = np.arange(T) if obs_idx is None else obs_idx
    lagged = lagged_stack(p.X_media)  # (L, T, C) - fixed data

    coords = {"channel": list(range(C)), "control": list(range(p.X_ctrl.shape[1])),
              "fourier": list(range(p.fourier.shape[1]))}
    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", 0.2, 0.3)
        trend_coef = pm.Normal("trend_coef", 0, 0.2)
        fourier_coef = pm.Normal("fourier_coef", 0, 0.1, dims="fourier")
        ctrl_coef = pm.Normal("ctrl_coef", 0, 0.5, dims="control")
        event_coef = pm.Normal("event_coef", 0, 0.2)

        alpha = pm.Beta("alpha", 1, 3, dims="channel")           # carryover rate
        lam = pm.Gamma("lam", alpha=3, beta=1, dims="channel")    # saturation speed
        beta = pm.HalfNormal("beta", 0.3, dims="channel")         # max weekly effect
        sigma = pm.HalfNormal("sigma", 0.1)

        lags = pt.arange(L_MAX)
        w = alpha[:, None] ** lags[None, :]                       # (C, L)
        w = w / w.sum(axis=1, keepdims=True)
        adstocked = (pt.as_tensor(lagged) * w.T[:, None, :]).sum(axis=0)  # (T, C)
        e = pt.exp(-lam * adstocked)
        saturated = (1 - e) / (1 + e)
        media = pm.Deterministic("media_contrib", saturated * beta)  # (T, C)

        mu = (intercept + trend_coef * p.trend + pt.dot(p.fourier, fourier_coef)
              + pt.dot(p.X_ctrl, ctrl_coef) + event_coef * p.event + media.sum(axis=1))
        pm.Deterministic("mu", mu)
        pm.Normal("y", mu=mu[obs_idx], sigma=sigma, observed=p.y[obs_idx])
    return model


def fit(p: Prepared, obs_idx=None, draws=1000, tune=1500, chains=4, seed=42):
    model = build_model(p, obs_idx)
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=2,
                          target_accept=0.95, random_seed=seed, progressbar=False)
    return model, idata


def extract_draws(idata, n: int = 1000, seed: int = 0) -> dict[str, np.ndarray]:
    """Flatten chains and keep a random subset of draws as plain numpy arrays."""
    post = idata.posterior
    if hasattr(post, "to_dataset"):  # newer ArviZ returns a DataTree
        post = post.to_dataset()
    post = post.stack(sample=("chain", "draw"))
    total = post.sizes["sample"]
    idx = np.random.default_rng(seed).choice(total, size=min(n, total), replace=False)
    out = {}
    for name in PARAMS:
        arr = post[name].values
        out[name] = (arr[..., idx].T if arr.ndim > 1 else arr[idx]).astype(float)
    return out
