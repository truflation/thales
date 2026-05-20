# Phase 1.1 Archetype — Commodity Pass-through Recovery

**Date:** 2026-04-25
**Module:** `src/thales/models/archetypes/commodity.py`
**Tests:** `tests/test_commodity_archetype.py` (9/9 passing)
**Demo:** `scripts/demo_commodity_recovery.py`
**DGP:** `src/thales/synthetic/commodity_passthrough.py`

## Headline result

**The TVP-Commodity Kalman+RTS smoother recovers the latent
time-varying pass-through coefficient to within 4% MAE on a 2000-obs
synthetic DGP** — a Pearson correlation of **0.999** with the true latent
β path. Static OLS, which collapses the drift to a single estimate, has
MAE 0.80 in the same setting (a 95% disadvantage).

This is **gate-1 evidence** for the architecture. The Phase 1
state-space estimation core demonstrably works on synthetic data with
known ground truth, exactly the discipline `docs/planning/02-evaluation.md`
prescribes.

## What's modeled

```
Observation:  y_t = α + β_t · x_t + ε_t,    ε_t ~ N(0, σ²)
State:        β_t = β_{t-1} + η^β_t,         η^β_t ~ N(0, σ_β²)
```

with `y = log(retail)`, `x = log(commodity)`. Linear-Gaussian state
space — exact closed-form Kalman filter forward pass plus the RTS
backward smoother. Hyperparameters `(α, σ, σ_β)` are fit by ML via
`scipy.optimize.minimize` (Nelder-Mead in log-parameterization to keep
variances positive).

## What's NOT modeled (intentional Phase 1.1 scope)

- **Stochastic volatility on ε_t** — Phase 1.2. Will need MCMC (NumPyro)
  because the state space is no longer linear-Gaussian.
- **Cointegration / VECM error-correction** between commodity and retail
  in levels (rather than flow-through in this single-equation form) —
  Phase 1.2.
- **Hierarchical / regional pass-through** (gas-by-PADD, etc.) — Phase
  1.5 (housing flagship).

The module's job is to **prove the TVP estimation core works in
isolation**, before the SV and VECM extensions are layered on top.

## Recovery metrics (default DGP, seed=42, T=2000)

```
true β:  range [0.281, 0.881]  mean=0.5867   ← significant drift

α̂ = -0.1375
σ̂_ε = 0.0345
σ̂_β = 0.0079        ← matches the true 0.008 within 1%

Pearson(smoothed, true)  = 0.9986
Pearson(filtered, true)  = 0.9982   ← virtually equal — for slow drift,
                                       smoothing is barely needed
MAE smoothed             = 0.0409
MAE filtered             = 0.0408
MAE static OLS           = 0.7991   ← OLS is broken on TVP DGP

TVP improvement vs OLS   = +94.9%
```

## Test coverage

9 recovery tests in `tests/test_commodity_archetype.py`:

1. **Mean recovery** — time-average smoothed β within 0.05 of truth
2. **Path correlation** — Pearson(smoothed, true) > 0.7
3. **Path MAE** — < 0.07 averaged over the post-burn-in path
4. **TVP > OLS** — TVP MAE strictly less than static OLS MAE
5. **σ_ε recovery** — within 50% of true noise SD
6. **σ_β recovery** — within factor of 3 of true drift SD
7. **Determinism** — same data → same fit (smoke check)
8. **Short-series rejection** — informative ValueError on n < 50
9. **Mismatched-input rejection** — informative ValueError

All pass. Full repo test suite is **67/67 green**.

## Why static OLS fails so badly

The synthetic DGP has:
- `log_commodity` as a random walk with drift (unit root)
- `β_t` as a bounded random walk (also unit-root-ish, in [0,1])
- `log_retail = α + β_t · log_commodity + ε_t`

When OLS regresses `log_retail` on `log_commodity`, both LHS and RHS
contain unit-root trends. OLS fits whatever single slope minimizes MSE
across the entire sample — but β IS NOT CONSTANT, and the
constant-coefficient fit is dominated by the trend levels (not the
pass-through dynamics). Result: a meaningless single number that bears
no relation to either the time-mean β or any individual β_t. This is
the standard Granger-Newbold spurious-regression result, and the TVP
model is exactly the cure.

## What this enables for Phase 1.1 production

1. The estimation core is validated. The next steps are:
   - Layer SV onto ε_t (PyMC / NumPyro, MCMC)
   - Add the VECM cointegration layer
   - Fit on **real** Truflation Utilities data + EIA gasoline / Henry Hub
2. Compose archetype 1's output through the CBDF layer once it lands.
3. Add this archetype's recovery test to the CI gate so future model
   changes can't regress it.

## Files

- `commodity_recovery_seed42_T2000.csv` — true β + filtered β + smoothed β
  + observables for one realization. Plot in any tool for visual sanity.

## Next archetypes (in difficulty order, per `01-architecture.md`)

- **1.2 BSTS discretionary** — dual-seasonal Bayesian structural time series
- **1.3 UC-SV-MS sticky services** — Kim-Nelson filter + regime switching
- **1.4 VECM tradables** — proper error-correction with tariff dummies
- **1.5 Hierarchical housing SSM** — flagship, hardest, A100 GPU territory
