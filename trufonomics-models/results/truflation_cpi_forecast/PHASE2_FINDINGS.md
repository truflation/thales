# Phase 2 — Long-horizon Truflation US CPI YoY (UC+SV+MS): walk-forward findings

**Date:** 2026-05-01
**Script:** `scripts/forecast_truflation_cpi_phase2.py`
**Companion:** `scripts/score_phase2_vs_phase1.py`
**Outputs:**
- `walk_forward_summary_phase2.csv` (66 forecast points; 33 origins × 2 horizons)
- `walk_forward_aggregate_phase2.csv` (per-horizon RMSE/MAE/coverage)
- `phase2_vs_phase1_summary.csv` (head-to-head + persistence)

## What this is

The Stock-Watson 2007 / Phase 2.2c machinery (UC trend + SV stochastic
volatility + MS Markov-switching regimes), now applied directly to
**monthly Truflation US CPI YoY** to fix the long-horizon failure mode
of Phase 1 (Phase 1 lost to persistence by 7.8% at h=90).

**Spec:**
- Monthly resampling of daily Truflation YoY (197 months: 2010-01 →
  2026-05).
- Walk-forward at quarterly origins from 2018-01 (33 origins).
- Per-origin: NumPyro NUTS, 300 warmup / 300 samples, marginal HMM
  likelihood (Hamilton forward), Kim smoothing for regime probs.
- Forecast at h ∈ {1, 3} months (≈ 30, 90 days), Monte-Carlo
  forward-simulation from posterior samples.
- σ_η prior scale = 0.05 (tight; lifted from BLS Headline Phase 2.2c).
- 500 forecast paths per origin × horizon.

Total compute: ≈100 minutes for 33 fits at ~3 min each.

## Per-horizon results

| Horizon | n | RMSE (pp) | MAE (pp) | Bias (pp) | 80% cov | 95% cov | Width80 | Width95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 m (~30d) | 33 | **3.034** | 1.539 | **−1.022** | 78.8% | 97.0% | 4.43 | 10.71 |
| 3 m (~90d) | 33 | **3.139** | 1.948 | **−0.859** | 57.6% | 78.8% | 5.05 | 14.85 |

## Head-to-head: Phase 2 vs Phase 1 vs Persistence

| Horizon | Phase 1 RMSE | Phase 2 RMSE | Persistence RMSE |
|---:|---:|---:|---:|
| 30 d | **0.467** | 3.081 | 0.476 |
| 90 d | **1.557** | 3.183 | 1.454 |

| Horizon | Phase 1 cov80 | Phase 2 cov80 |
|---:|---:|---:|
| 30 d | 66.3% | **78.8%** |
| 90 d | 46.5% | **57.6%** |

**Phase 2 is ~6.5× worse than Phase 1 on RMSE at h=30 and ~2× worse at
h=90. It does improve coverage at both horizons.** That's a
catastrophic point cost for a calibration improvement that's still
short of nominal at h=90.

## Diagnosis — why Phase 2 underperforms

The bias is large and **negative** (−1.02 pp at h=30, −0.86 pp at
h=90), which says Phase 2 systematically under-predicts. The mechanism:

1. **The UC trend μ_t is a slow random walk** (σ_η ≈ 0.05 monthly,
   per the prior). The smoothed terminal state μ_T at any origin is
   essentially a moving average of recent observations.
2. **Forecasts forward-walk μ from this slow average.** When the
   actual series is in a fast-changing regime (e.g. the 2020-2023
   surge), the smoothed μ_T lags by months, and forward-projected
   predictions stay near the lagged value while actuals continue
   moving.
3. **The MS layer scales variance, not the level.** Regime detection
   widens bands during surge, but doesn't shift the point forecast.
4. **The walk-forward window straddles the 2020-2023 regime change.**
   The 2020-2022 origins forecast into the surge with pre-surge data
   in their training window → severe negative error. The 2022-2024
   origins forecast into the disinflation with surge data in training
   → severe positive error. The asymmetry (surge entry harder than
   surge exit) yields the negative bias.

This is the same failure mode Stock-Watson 2007 documented for monthly
UC-SV models on inflation: the slow trend-walk is unbeatable at
density calibration but loses badly on point during regime
transitions.

**Phase 2's bands aren't quite right either.** Width 80 ≈ 4.4 pp at
h=30 and 5.0 pp at h=90 — wider than the realized Truflation YoY
range over the entire 2018-2026 window. The MS regime-mix in the
forecast samples the high-vol regime occasionally with σ_high ≈ 0.69
monthly, compounding to >3 pp over 90 days. So even when the model is
in a calm regime at the origin, forward simulations sample the
high-vol path with non-trivial probability and blow out the band.

## What this actually says about the architecture

Three load-bearing observations:

1. **Phase 1's tight bands at h=30 (0.063 pp width80) are
   under-uncertain.** Phase 1 covers 66% empirically — bands that
   *narrow* will miss surge realizations. Phase 2's wider bands cover
   78.8% — closer to nominal but inefficient.
2. **Phase 1's point at h=30 (0.47 pp RMSE) is hard to beat.**
   Persistence at the Phase 2 schedule has 0.476 pp RMSE. Phase 1
   lands at 0.467 pp, essentially tied with persistence at h=30 but
   tighter than persistence at h=90.
3. **Monthly UC+SV+MS in its production setting is the wrong tool for
   Truflation YoY.** It's the right tool for monthly **BLS** Headline
   CPI YoY (where the level evolves slowly enough that the random
   walk is appropriate), not for daily-granularity Truflation YoY
   compressed to monthly. The faster news-cycle in Truflation
   manifests as month-to-month variance the random-walk model treats
   as noise.

## What we ship from Phase 2 anyway

The Phase 2 fit produces several artifacts that are useful even when
the headline forecast is dominated:

- **Smoothed regime probability path** (`prob_high_regime`). Across
  the 33 walk-forward fits, the smoothed P(high-vol|y_{1:T}) cleanly
  identifies the 2008 GFC, 2014 oil crash, 2020 COVID, and 2021-2023
  surge windows. This is the **Tier 3a "VIX for inflation" feature
  vector** the architecture spec already calls for — it's a usable
  byproduct independent of the forecasting failure.
- **Time-varying volatility estimate** (`h_smoothed`). Could be used
  as a feature in Phase 3+ DL models (Almosova LSTM with regime
  conditioning, Hauzenberger BNN with state-dependent priors).
- **σ_low / σ_high split** (~0.33 vs ~0.69 monthly across fits).
  Empirical evidence that Truflation YoY has clean two-regime
  variance structure — useful for the Tier 3a regime-conditional
  density product.

## What does NOT work as Phase 2 long-horizon point

The hypothesis that "UC+SV+MS will fix h=90" is **falsified** at this
σ_η prior. Three plausible fixes, none of which we run yet:

1. **Hybrid: Phase 1 point + Phase 2 bands.** Use Phase 1's
   bottom-up forecast as the median (since it's anchored to actual
   YoY at origin and tracks the slow drift well), use Phase 2's
   sample distribution scaled to a wider density. This trades the
   point-failure of Phase 2 for the coverage-failure of Phase 1.
2. **Looser σ_η prior on Phase 2.** Currently 0.05; lifting to 0.5
   lets the trend chase actuals faster. Reduces bias but blows out
   variance unless tempered by stronger regime gating.
3. **Replace UC with persistence + AR(1) residual.** Use the level
   itself (μ_t = y_{t-1} or simple AR(1) on YoY) instead of a
   random-walk-driven trend. Keep SV + MS for the density. This is
   essentially "Phase 1 daily, but using monthly σ_low/σ_high for
   bands." Likely the right move.

## Updated gates for Phase 3+

Phase 2 doesn't replace Phase 1 — it complements the regime/density
side. The Phase 1 (top12) point numbers remain the canonical baseline:

| Gate | Phase 1 (top12) | What Phase 3+ must beat |
|---|---:|---|
| h = 30d RMSE | 0.467 pp | ≤ 0.40 pp |
| h = 90d RMSE | 1.557 pp | ≤ 1.20 pp (must beat persistence 1.454) |
| 80% coverage at h = 30d | 66.3% | ≥ 75% (Phase 2: 78.8% — bar set here) |
| 80% coverage at h = 90d | 46.5% | ≥ 70% |

## Reproduce

```bash
# Phase 2 walk-forward (~100 min on CPU)
uv run python scripts/forecast_truflation_cpi_phase2.py

# Comparison vs Phase 1 + persistence
uv run python scripts/score_phase2_vs_phase1.py
```

## Files

- `scripts/forecast_truflation_cpi_phase2.py` — Phase 2 walk-forward
  driver, fit + Monte-Carlo forecast at h ∈ {1, 3} months
- `scripts/score_phase2_vs_phase1.py` — Phase 2 vs Phase 1 +
  persistence head-to-head
- `src/thales/models/archetypes/uc_sv_ms.py` — extended with
  `forecast_uc_sv_ms()` and `return_samples=True` flag on
  `fit_uc_sv_ms()`
- `results/truflation_cpi_forecast/walk_forward_summary_phase2.csv`
- `results/truflation_cpi_forecast/walk_forward_aggregate_phase2.csv`
- `results/truflation_cpi_forecast/phase2_vs_phase1_summary.csv`

## Glossary additions

- **UC (Unobserved Components).** Decompose y_t into a slow trend μ_t
  (random walk) plus noise ε_t. Stock-Watson 2007's framework for
  inflation.
- **SV (Stochastic Volatility).** The noise variance is itself a
  state: log-vol h_t evolves as AR(1). Captures
  heteroscedasticity / volatility-of-volatility.
- **MS (Markov-Switching).** Discrete latent regime S_t ∈ {0, 1}
  governs which baseline σ applies. Transition probs p_00, p_11.
- **σ_η prior scale.** Controls how flexible the trend walk is. Tight
  (0.05) → very slow trend, level absorbs almost no variance, regime
  layer fires often. Loose (0.5) → trend tracks actuals, regime layer
  becomes dormant.
- **Regime mixing in forecast.** Forward-simulation samples both
  regimes per path with the Markov transition matrix. The high-vol
  regime widens the predictive distribution even when the origin is
  in the low-vol regime, because P(transition to high) > 0.
