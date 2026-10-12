"""Numpy forward model driven by saved posterior draws.

The dashboard uses this instead of PyMC, so it starts instantly and every
slider move is recomputed across all draws (giving honest uncertainty bands).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import fourier_features
from .transforms import geometric_adstock_np, logistic_saturation_np

import pandas as pd


@dataclass
class Posterior:
    draws: dict[str, np.ndarray]
    media_scale: np.ndarray
    ctrl_scale: np.ndarray
    y_scale: float
    n_weeks_train: int  # used to scale the trend term

    @property
    def n_draws(self) -> int:
        return len(self.draws["sigma"])

    # ---- core pieces (all return pounds) ------------------------------------
    def media_contrib(self, spend: np.ndarray) -> np.ndarray:
        """spend: (T, C) in pounds -> (D, T, C) revenue contribution in pounds."""
        x = spend / self.media_scale
        ad = geometric_adstock_np(x, self.draws["alpha"])              # (D, T, C)
        sat = logistic_saturation_np(ad, self.draws["lam"][:, None, :])
        return sat * self.draws["beta"][:, None, :] * self.y_scale

    def baseline(self, dates: pd.Series, t_index: np.ndarray, controls: np.ndarray,
                 event: np.ndarray) -> np.ndarray:
        """Non-media revenue (D, T) in pounds."""
        d = self.draws
        trend = t_index / (self.n_weeks_train - 1)
        f = fourier_features(pd.Series(pd.to_datetime(dates)))
        xc = controls / self.ctrl_scale
        out = (d["intercept"][:, None] + d["trend_coef"][:, None] * trend[None, :]
               + d["fourier_coef"] @ f.T + d["ctrl_coef"] @ xc.T
               + d["event_coef"][:, None] * event[None, :])
        return out * self.y_scale

    # ---- steady-state response curves --------------------------------------
    def weekly_response(self, weekly_spend: np.ndarray) -> np.ndarray:
        """Revenue per week (D, ..., C) if each channel spent `weekly_spend`
        (..., C) every week indefinitely. Normalised adstock means steady-state
        adstocked spend equals spend, so only saturation matters."""
        x = np.asarray(weekly_spend, dtype=float) / self.media_scale
        lam = self.draws["lam"].reshape((self.n_draws,) + (1,) * (x.ndim - 1) + (-1,))
        beta = self.draws["beta"].reshape(lam.shape)
        return logistic_saturation_np(x[None, ...], lam) * beta * self.y_scale


def summarise(a: np.ndarray, axis: int = 0, lo: float = 5, hi: float = 95):
    """Mean and 90% credible interval along the draw axis."""
    return a.mean(axis=axis), np.percentile(a, lo, axis=axis), np.percentile(a, hi, axis=axis)
