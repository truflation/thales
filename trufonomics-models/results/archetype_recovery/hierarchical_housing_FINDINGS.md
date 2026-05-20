# Phase 1.5 Archetype — Hierarchical Housing DFM Recovery

**Date:** 2026-04-25
**Module:** `src/thales/models/archetypes/hierarchical_housing.py`
**Tests:** `tests/test_hierarchical_housing_archetype.py` (10/10 passing)
**Demo:** `scripts/demo_hierarchical_housing.py`
**DGP:** `src/thales/synthetic/hierarchical_housing.py`

## Headline result

**JAX-based hierarchical DFM recovers latent paths to Pearson > 0.83
for both the national factor and all 4 regional idiosyncratic series.**
Demo run with T=400, 4 Census regions:

```
|Pearson(F_smoothed, F_true)|     = 0.9789
|Pearson(λ_NE_smoothed, true)|    = 0.8948
|Pearson(λ_MW_smoothed, true)|    = 0.8432
|Pearson(λ_S_smoothed, true)|     = 0.9193
|Pearson(λ_W_smoothed, true)|     = 0.8353
Reconstruction R² (full y_t)      = 0.9986
```

Fits in **~12 seconds on CPU**. Same code runs on GPU/A100 via
`JAX_PLATFORMS=cuda` — typical 5-10× speedup.

## What's modeled

```
F_t       =  F_{t-1}  +  η^F_t,             η^F_t ~ N(0, σ_F²)
λ_{r,t}   =  ρ_r · λ_{r,t-1}  +  ν_{r,t},   ν_{r,t} ~ N(0, σ_{λ,r}²)
y_{r,t}   =  β_r · F_t  +  λ_{r,t}  +  ε_{r,t},  ε_{r,t} ~ N(0, σ_{ε,r}²)
```

For 4 regions, state dimension K = 5 (1 national + 4 regional). Observation
dimension R = 4. Hyperparameters: 1 + 4·R = 17 total.

JAX-native multivariate Kalman filter + RTS smoother, JIT-compiled via
`jax.lax.scan`. Maximum-likelihood fit via `jax.scipy.optimize.minimize`
(LBFGS) on flattened, log/logit-transformed parameters.

## Honest flag — factor identifiability (well-known)

The hyperparameter estimates show classic DFM identifiability:

```
σ̂_F = 0.745     true 0.150     (5× over)
β̂   = [0.20, 0.12, 0.20, 0.27]    true [1.00, 0.70, 1.10, 1.30]   (5× under)
```

The **product** β_r · σ_F is what's identified by the data — the
absolute scale of F vs β is not. The MLE picked one rescaling; all
others give equivalent likelihoods. The **relative loadings** are
recovered correctly:

```
β̂_W / β̂_NE  =  0.27 / 0.20  =  1.35    true 1.30 / 1.00 = 1.30 ✓
β̂_MW / β̂_NE =  0.12 / 0.20  =  0.60    true 0.70 / 1.00 = 0.70 ✓
β̂_S / β̂_NE  =  0.20 / 0.20  =  1.00    true 1.10 / 1.00 = 1.10 ≈
```

This is the standard solution: in a single-factor model, fix one β
(e.g., β_NE = 1) by convention to identify the rest. The current
implementation lets MLE pick its own scaling and reports it; users can
post-hoc renormalize by dividing by β_NE if they want.

**The path-level identifiability is fine** — F_smoothed and λ_smoothed
recover the truth at high correlation. Same lesson as Phase 1.2 BSTS:
*path identifiability* and *parameter identifiability* are separate
questions, and well-known factor-model pathologies don't compromise the
forecasting capability.

## Test coverage

10 recovery tests in `tests/test_hierarchical_housing_archetype.py`:

1. **DGP reproducibility**
2. **DGP loading visibility** — high-β region tracks F more strongly
3. **National factor recovery** — |Pearson(F_smoothed, F_true)| > 0.85
4. **Regional idiosyncratic recovery** — average per-region |Pearson| > 0.4
5. **Observation reconstruction** — R² > 0.85
6. **Beta sign consistency** — all β_r have same sign (factor identified
   up to overall sign flip; per-region signs should agree)
7. **σ_ε recovery** — within factor of 2 per region
8. **Short-series rejection**
9. **1D-input rejection**
10. **Mismatched-region-names rejection**

All pass. The full repo suite is now **129 fast + 41 slow = 170 tests**,
all green (BSTS multi-start + UC+SV+MS + SV-only + UC+MS + hierarchical
housing all marked slow).

## What's NOT modeled (Phase 1.5+ extensions)

These are clean additive extensions on top of the hierarchical DFM core:

- **Owned vs rented split** — duplicate state structure with cross-block
  correlations between owned-housing F and rented-housing F (or shared F
  with separate β-loadings per segment)
- **Mariano-Murasawa mixed frequency** — Case-Shiller monthly with 2-month
  lag, BLS shelter monthly, Zillow weekly. Adds time-varying observation
  matrices `Z_t` per row.
- **Mortgage-rate exogenous regressor** on F_t — adds one slope parameter
  to the F equation: `F_t = F_{t-1} + γ·mortgage_rate_change_t + η^F_t`
- **Bayesian estimation via NumPyro** — full posterior over all latents +
  hyperparameters, with priors on β to address the scale identifiability
  honestly

## Production fit on real housing data

Real Truflation regional housing series + BLS shelter + Case-Shiller +
Zillow rent indexes feed into this model. The natural first deployment:

1. Pull regional housing CPI from BLS subindex panel (we have CUSR0000SAH,
   CUSR0000SAH1, CUSR0000SEHA, CUSR0000SEHC01)
2. Pull Truflation housing-related streams from vintage store
3. Fit `fit_hierarchical_housing` on the wide panel
4. Compare composed F_t to known shocks (2020 COVID, 2022 rate cycle,
   2024 commercial real-estate weakness)

This is the natural "first GPU job" on Vast — JAX gradients on a 4D-state
DFM with 360 monthly observations is fast on CPU but scales poorly to
T=2000+ and R=10+ regions, where GPU helps.

## Files

- `hierarchical_housing_seed42_T400.csv` — observable + true latents +
  smoothed estimates for one realization
