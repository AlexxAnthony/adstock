# Adstock - Marketing Mix Model & Budget Optimiser

A Bayesian **marketing mix model (MMM)** built from scratch in PyMC, with a Streamlit dashboard
for channel ROI, response curves, next-quarter forecasting and budget optimisation.

**Live dashboard:** https://adstock-mmm.streamlit.app| **Code:** https://github.com/AlexxAnthony/adstock

## The question

> "If I have a fixed budget next quarter, how should I split it across TV, out-of-home, print,
> Facebook and paid search, and what revenue will that produce?"

MMM answers this from aggregate weekly spend and sales, with no cookies or user-level tracking.
That's why it has come back into use as third-party tracking has declined.

## Key findings

Data: Meta Robyn's simulated weekly dataset, 208 weeks, Nov 2015 - Nov 2019.

| Finding | Evidence |
|---|---|
| Out-of-home takes 62% of the budget but returns about **£0.69 per £1** | 90% interval £0.10-£1.47; only a 22% probability it breaks even |
| TV is the most dependable channel at **£5.75 per £1** | Tightest interval (£4.13-£7.55) |
| Facebook and search have high average ROI but **wide uncertainty** | Their spend varies little, so the data can't pin them down. Test before scaling |
| Re-allocating the **same budget** within ±50% guard-rails adds about **£587k revenue per quarter** | 97% posterior probability of beating the current mix |

## How the model works

```
revenue_t = baseline_t + Σ_channels β_c · saturation_c( adstock_c( spend_c ) )_t + noise
baseline_t = intercept + trend + yearly seasonality (Fourier) + competitor sales + newsletter + events
```

- **Adstock (carryover):** geometric decay over 8 weeks, with the decay rate learned per channel
- **Saturation (diminishing returns):** logistic curve, with the shape learned per channel
- **Bayesian inference:** NUTS sampler, 4 chains × 1,000 draws. Every output in the dashboard is computed
  across 1,000 posterior draws, so each number comes with a credible interval
- **Budget optimiser:** SciPy SLSQP maximises expected media revenue across all draws, subject to
  the budget total and per-channel guard-rails

The dashboard doesn't run PyMC. It reruns the model in numpy from the saved posterior draws. The fit
script asserts that the numpy version matches PyMC to within 1%, so the app starts instantly and sliders
update live.

## Validation

| Check | Result |
|---|---|
| 26-week out-of-sample forecast | MAPE **4.4%** vs 22.3% for "same week last year" |
| Interval calibration | 100% of holdout weeks inside the 90% interval (slightly conservative) |
| Convergence | max R-hat 1.005, min ESS 1,237, 1 divergence in 4,000 draws |
| **Parameter recovery** (simulate from known parameters, refit, check) | True ROI inside the 90% interval in **93%** of 15 channel-simulation cases |

**Caveat:** a regression with no media terms forecasts almost as well (4.7% MAPE), because
competitor/market sales drive most week-to-week revenue in this dataset. So forecast accuracy alone
can't prove the attribution is right. That's why the parameter-recovery test is included.

## Limitations

- MMM is observational. Moves at a guard-rail should be confirmed with a geo-holdout or lift test
- Simulated data illustrates the method, not a real brand
- National level, weekly, constant effects over time. No geo, creative or time-varying ROI
- Forward forecasts assume competitor sales and newsletter volume repeat last year's same weeks

## Project structure

```
src/adstock/
  data.py          load, scale, Fourier seasonality
  transforms.py    adstock + saturation (numpy, unit-tested)
  model.py         PyMC model definition and fitting
  posterior.py     numpy forward model driven by posterior draws
  planner.py       scenario forecasts + budget optimiser
scripts/
  fit_model.py     holdout fit, baselines, full fit -> artifacts/
  recovery_test.py parameter-recovery test -> artifacts/recovery.json
app/               Streamlit dashboard (Home + 5 pages)
artifacts/         saved posterior draws, fitted values, metrics
tests/             unit tests for the transforms
```

## Run it

```bash
pip install -r requirements.txt
streamlit run app/Home.py

# to refit the model (about 3 minutes on a laptop)
pip install -r requirements-model.txt
python scripts/fit_model.py
python scripts/recovery_test.py
pytest tests
```

## Roadmap

- Cross-check against PyMC-Marketing / Google Meridian
- Add lift-test results as ROI priors (calibration)
- Time-varying media effects
- A/B testing and incrementality tab (CUPED, uplift modelling)
