"""Parameter-recovery test: can the model find the truth when we know it?

Forecast accuracy alone does not prove an MMM attributes revenue correctly -
a model can predict well while crediting the wrong channel. So we:
  1. take the fitted model's posterior means as a known "true" world,
  2. simulate revenue from it using the real spend and controls plus noise,
  3. refit the model on the simulated revenue,
  4. check whether each channel's true ROI lands inside the 90% interval.

Run after fit_model.py:  python scripts/recovery_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adstock.data import CHANNELS, CONTROLS, load_raw, prepare  # noqa: E402
from adstock.model import extract_draws, fit  # noqa: E402
from adstock.posterior import Posterior  # noqa: E402

ART = ROOT / "artifacts"
SEEDS = [11, 22, 33]


def roi_draws(post: Posterior, df) -> np.ndarray:
    spend = df[CHANNELS].to_numpy()
    return post.media_contrib(spend).sum(1) / spend.sum(0)  # (D, C)


def main():
    df = load_raw()
    p = prepare(df)
    meta = json.loads((ART / "meta.json").read_text())
    fitted = dict(np.load(ART / "posterior_draws.npz"))
    truth = {k: v.mean(0, keepdims=True) for k, v in fitted.items()}  # one "draw"
    true_post = Posterior(truth, p.media_scale, p.ctrl_scale, p.y_scale, len(df))
    true_roi = roi_draws(true_post, df)[0]

    base = true_post.baseline(df["date"], np.arange(len(df)), df[CONTROLS].to_numpy(),
                              df["event"].to_numpy())[0]
    mu = base + true_post.media_contrib(df[CHANNELS].to_numpy())[0].sum(1)

    runs = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sim = df.copy()
        sim["revenue"] = mu + rng.normal(0, truth["sigma"][0] * p.y_scale, len(df))
        ps = prepare(sim)
        _, idata = fit(ps, draws=500, tune=1000, seed=seed)
        post = Posterior(extract_draws(idata), ps.media_scale, ps.ctrl_scale, ps.y_scale, len(sim))
        r = roi_draws(post, sim)
        lo, hi = np.percentile(r, [5, 95], axis=0)
        runs.append({"seed": seed, "est": r.mean(0).tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                     "covered": ((true_roi >= lo) & (true_roi <= hi)).tolist()})
        print(seed, np.round(r.mean(0), 2), "covered:", runs[-1]["covered"])

    covered = np.array([run["covered"] for run in runs])
    out = {"channels": CHANNELS, "true_roi": true_roi.tolist(), "runs": runs,
           "coverage_rate": float(covered.mean() * 100)}
    (ART / "recovery.json").write_text(json.dumps(out, indent=2))
    print("true ROI", np.round(true_roi, 2), "coverage", out["coverage_rate"])
    meta["recovery_coverage"] = out["coverage_rate"]
    (ART / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
