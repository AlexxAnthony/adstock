"""Media transforms shared by the PyMC model and the numpy dashboard code.

Two ideas sit at the heart of a marketing mix model:

* Adstock (carryover): an ad seen this week still nudges sales next week.
  Geometric adstock spreads each week's spend over the following L_MAX weeks
  with weights alpha^0, alpha^1, ... normalised to sum to 1, so total
  "effective spend" equals actual spend - it is only shifted in time.

* Saturation (diminishing returns): the 10th thousand pounds buys less than the
  first. A logistic curve maps scaled spend x >= 0 to [0, 1):
      s(x) = (1 - exp(-lam * x)) / (1 + exp(-lam * x))
  Larger lam means the channel saturates sooner.
"""
from __future__ import annotations

import numpy as np

L_MAX = 8  # weeks of carryover modelled


def adstock_weights(alpha, l_max: int = L_MAX) -> np.ndarray:
    """Normalised geometric weights. alpha may be scalar or array (..., C)."""
    alpha = np.asarray(alpha, dtype=float)
    lags = np.arange(l_max)
    w = alpha[..., None] ** lags  # (..., l_max)
    return w / w.sum(axis=-1, keepdims=True)


def geometric_adstock_np(x: np.ndarray, alpha, l_max: int = L_MAX) -> np.ndarray:
    """Apply geometric adstock along the time axis.

    x: (T, C) spend.  alpha: (C,) or (D, C) for D posterior draws.
    Returns (T, C) or (D, T, C).
    """
    x = np.asarray(x, dtype=float)
    w = adstock_weights(alpha, l_max)  # (C, L) or (D, C, L)
    T = x.shape[0]
    # lagged[l] = x shifted down by l weeks (zeros before the series starts)
    lagged = np.zeros((l_max,) + x.shape)
    for lag in range(l_max):
        lagged[lag, lag:] = x[: T - lag]
    # sum over lags: out[t, c] = sum_l w[c, l] * x[t - l, c]
    if w.ndim == 2:
        return np.einsum("ltc,cl->tc", lagged, w)
    return np.einsum("ltc,dcl->dtc", lagged, w)


def logistic_saturation_np(x, lam):
    x = np.asarray(x, dtype=float)
    e = np.exp(-lam * x)
    return (1 - e) / (1 + e)


def half_life(alpha) -> np.ndarray:
    """Weeks until a geometric effect halves: alpha^h = 0.5."""
    alpha = np.clip(np.asarray(alpha, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(0.5) / np.log(alpha)
