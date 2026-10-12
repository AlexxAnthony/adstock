"""Load and prepare the Robyn simulated weekly dataset for modelling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "dt_simulated_weekly.csv"

CHANNELS = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
CHANNEL_NAMES = {
    "tv_S": "TV",
    "ooh_S": "Out-of-home",
    "print_S": "Print",
    "facebook_S": "Facebook",
    "search_S": "Paid search",
}
CONTROLS = ["competitor_sales_B", "newsletter"]
N_FOURIER = 2  # yearly seasonality harmonics


@dataclass
class Prepared:
    df: pd.DataFrame          # tidy weekly data
    X_media: np.ndarray       # (T, C) spend scaled by channel max
    X_ctrl: np.ndarray        # (T, K) controls scaled by max
    fourier: np.ndarray       # (T, 2*N_FOURIER)
    event: np.ndarray         # (T,) 1 in event weeks
    trend: np.ndarray         # (T,) 0..1
    y: np.ndarray             # (T,) revenue / y_scale
    media_scale: np.ndarray   # (C,)
    ctrl_scale: np.ndarray    # (K,)
    y_scale: float


def load_raw(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["DATE"]).rename(columns={"DATE": "date"})
    df["event"] = (df["events"] != "na").astype(int)
    return df.sort_values("date").reset_index(drop=True)


def fourier_features(dates: pd.Series, n: int = N_FOURIER) -> np.ndarray:
    day = dates.dt.dayofyear.to_numpy() / 365.25
    cols = []
    for k in range(1, n + 1):
        cols += [np.sin(2 * np.pi * k * day), np.cos(2 * np.pi * k * day)]
    return np.column_stack(cols)


def prepare(df: pd.DataFrame, fit_rows: slice | None = None) -> Prepared:
    """Scale features. Scales are taken from `fit_rows` only so a holdout test
    never sees information from the weeks it is scored on."""
    fit = df.iloc[fit_rows] if fit_rows is not None else df
    media_scale = fit[CHANNELS].max().to_numpy()
    ctrl_scale = fit[CONTROLS].max().to_numpy()
    y_scale = float(fit["revenue"].max())
    t = np.arange(len(df)) / (len(df) - 1)
    return Prepared(
        df=df,
        X_media=df[CHANNELS].to_numpy() / media_scale,
        X_ctrl=df[CONTROLS].to_numpy() / ctrl_scale,
        fourier=fourier_features(df["date"]),
        event=df["event"].to_numpy().astype(float),
        trend=t,
        y=df["revenue"].to_numpy() / y_scale,
        media_scale=media_scale,
        ctrl_scale=ctrl_scale,
        y_scale=y_scale,
    )
