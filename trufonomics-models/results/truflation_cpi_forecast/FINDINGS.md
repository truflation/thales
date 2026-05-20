# Phase 1 / 1.5 — Bottom-up Truflation US CPI YoY forecaster: walk-forward findings

**Date:** 2026-05-01
**Script:** `scripts/forecast_truflation_cpi_bottomup.py`
**Companion:** `scripts/score_phase1_vs_persistence.py`

**Outputs (per-phase, label = top12 | leaves58):**
- `walk_forward_summary_<label>.csv` (510 forecast points; 102 origins × 5 horizons)
- `walk_forward_aggregate_<label>.csv` (per-horizon RMSE/MAE/coverage)
- `persistence_comparison_<label>.csv` (vs persistence baseline + DM tests)

## What this is

End-to-end **Truflation-only** multi-horizon CPI YoY forecaster.
Trained on Truflation per-component data only, predicts Truflation US
CPI YoY only — no BLS, no FRED, no PCE on either side.

Two compositional granularities tested at the same Phase-1-level
forecaster (per-component AR(1) on log-returns, bootstrap residuals):

| Phase | Crosswalk level | n components | Weight sum |
|---|---|---:|---:|
| 1 | `top12` — 12 top-level Truflation categories | 12 | 100.000% |
| 1.5 | `leaves58` — leaf-set of the 80 ingested CPI streams | 58 | 100.000% |

Both run identical machinery: per-component AR(1) on log-returns,
bootstrap residuals from the trailing 30-day calibration window, M2
composition (weighted sum of rebased levels), anchor-offset correction
to actual Truflation YoY at origin, sample-median point, sample
quantile bands.

## Per-horizon results

**Phase 1 — top12 composition (n=102 origins, 2018-01-01 → 2026-05-01):**

| Horizon | n | RMSE (pp) | MAE (pp) | Bias (pp) | 80% cov | 95% cov | Width80 | Width95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 102 | 0.091 | 0.038 | −0.017 | 71.6% | 93.1% | 0.063 | 0.324 |
| 7 | 102 | 0.183 | 0.131 | −0.036 | 69.6% | 77.5% | 0.465 | 0.672 |
| 14 | 101 | 0.305 | 0.230 | +0.008 | 62.4% | 74.3% | 0.606 | 0.986 |
| 30 | 101 | 0.467 | 0.361 | −0.006 | 66.3% | 74.3% | 0.984 | 1.499 |
| 90 | 99 | 1.557 | 1.171 | +0.033 | 46.5% | 59.6% | 1.754 | 2.651 |

**Phase 1.5 — leaves58 composition (same window):**

| Horizon | n | RMSE (pp) | MAE (pp) | Bias (pp) | 80% cov | 95% cov | Width80 | Width95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 102 | 0.091 | 0.038 | −0.017 | 69.6% | 94.1% | 0.064 | 0.327 |
| 7 | 102 | 0.184 | 0.131 | −0.033 | 69.6% | 75.5% | 0.465 | 0.673 |
| 14 | 101 | 0.309 | 0.232 | +0.018 | 60.4% | 73.3% | 0.607 | 0.989 |
| 30 | 101 | 0.472 | 0.365 | +0.021 | 66.3% | 73.3% | 0.997 | 1.532 |
| 90 | 99 | 1.590 | 1.193 | +0.173 | 40.4% | 60.6% | 1.837 | 2.823 |

## Phase 1 vs Phase 1.5 head-to-head

RMSE (pp) at each horizon, plus relative change:

| Horizon | Phase 1 RMSE | Phase 1.5 RMSE | Δ Phase 1.5 vs Phase 1 |
|---:|---:|---:|---:|
| 1 | 0.0908 | 0.0905 | −0.3% |
| 7 | 0.1825 | 0.1837 | +0.7% |
| 14 | 0.3045 | 0.3086 | +1.3% |
| 30 | 0.4673 | 0.4717 | +0.9% |
| 90 | 1.5574 | 1.5901 | +2.1% |

**Phase 1.5 is uniformly weaker.** Going from 12 → 58 streams under a
**naive per-component forecaster** (independent AR(1) on log-returns)
adds noise faster than signal at every horizon. The damage compounds
with horizon: the h=90 RMSE is 2.1% larger.

This is the expected behaviour when composing many noisy independent
forecasters: each component's forecast variance shows up additively in
the composed forecast, but the cross-component correlation structure
that constrains real Truflation movement is **not modelled**. The
12-stream aggregation hides this by aggregating over the unmodelled
correlations *before* forecasting; the 58-leaf decomposition exposes
them.

## Vs naive persistence baseline (both phases)

Persistence at horizon h: predict YoY[origin] for YoY[origin + h]. Same
test set, same actuals. Diebold-Mariano with Newey-West HAC lag = 3.

| Horizon | Phase 1 RMSE red. | Phase 1.5 RMSE red. | Persistence RMSE |
|---:|---:|---:|---:|
| 1 | 0.0% | 0.0% | 0.091 |
| 7 | +0.5% | −0.1% | 0.183 |
| 14 | +0.1% | −1.3% | 0.305 |
| **30** | **+21.6%** | **+20.8%** | **0.595** |
| 90 | −7.8% | −10.1% | 1.444 |

**Read:**

- **h ∈ {1, 7, 14}:** persistence is unbeatable for both phases.
  Truflation YoY moves too slowly day-over-day.
- **h = 30:** the bottom-up architecture's value lives here. Both
  phases beat persistence by ~21% (DM stat ~1.24, p ~0.21 — n=101 too
  small for significance). Phase 1 edges Phase 1.5 by ~0.8 pp of RMSE
  reduction.
- **h = 90:** both phases lose to persistence. Phase 1.5 loses worse
  (−10.1% vs −7.8%). AR(1) drift compounds badly when actual YoY
  mean-reverts.

## What this tells us about the architecture

**The composition layer is not the bottleneck at Phase 1.x.** Going
from 12 to 58 components under independent AR(1) forecasters does not
help — not at h=1 (persistence wins), not at h=30 (where the
forecaster does add value, both phases get the same ~21% reduction),
not at h=90 (both lose to persistence, leaves58 worse). The component
forecaster is the bottleneck.

**Three improvements that target the bottleneck:**

1. **Cross-component dependence.** Replace independent AR(1)s with a
   joint dynamic factor model that captures cross-component
   correlations during composition. Per `architecture` (CLAUDE.md):
   five archetype models (commodity pass-through, rate-sensitive
   durables, sticky services, import-exposed tradables, discretionary
   demand-cycle) one per category-generating process, each
   state-space, then composed via CBDF (O'Keeffe & Petrova 2025) which
   respects the accounting identity.
2. **Long-horizon trend filter.** AR(1) drift overshoots persistence
   at h=90. UC+SV+MS trend extraction (Stock-Watson 2007 / Phase 2.2c
   already shipped on BLS Headline) extracts a slow-moving trend that
   shrinks the drift contribution at long horizons.
3. **Wider density at h ≥ 30.** 80% bands cover 47-66% empirically.
   Bootstrap residuals from a single AR(1) under-estimate
   regime-shift variance. Rolling-conformal residuals computed on
   *YoY-level errors* (not propagated component-level errors) would
   widen the bands directly to match empirical miscoverage.

## What works, what doesn't (across both phases)

**Works:**
- Anchor-offset correction. Mean error ≤ 0.05 pp for top12 and
  ≤ 0.18 pp for leaves58 at every horizon.
- M2 composition. Validated 0.000pp median residual on top12 in
  `composition_check.py`. Composition arithmetic is not the loss
  source.
- Sample-median point. Removes Jensen bias on long-horizon
  log-return → level transforms.
- Computational cost. Phase 1 ~60s, Phase 1.5 ~6 min on a laptop.

**Doesn't work yet:**
- **Component forecaster.** Independent AR(1) on log-returns is too
  weak to translate finer-grained data into better aggregate
  forecasts.
- **Long-horizon AR(1) drift.** Compounds badly past h=30.
- **Density calibration at h ≥ 30.** Bootstrap residuals
  underestimate true uncertainty.

## What this enables (gates Phase 2+ must clear)

The Phase 1 **top12** numbers are the canonical baseline:

| Gate | Phase 1 (top12) | Required of Phase 2+ |
|---|---:|---|
| h = 1 RMSE | 0.091 pp | ≤ 0.090 pp |
| h = 30 RMSE | 0.467 pp | ≤ 0.40 pp; DM beat over persistence p < 0.05 |
| h = 90 RMSE | 1.557 pp | ≤ 1.20 pp (must beat persistence 1.444) |
| 80% coverage at h = 30 | 66.3% | ≥ 75% |
| 80% coverage at h = 90 | 46.5% | ≥ 70% |

**Phase 1.5 is parked.** The leaf-58 panel and the matching weights
loader stay in the codebase (via the `--crosswalk-level leaves58`
flag) for re-use in Phase 5 (HRNN), Phase 3+ (DL component-aware
encoders), and any cross-component research. They are not a
production track on their own.

## Reproduce

```bash
# Phase 1 (top12) — walk-forward
uv run python scripts/forecast_truflation_cpi_bottomup.py \
    --crosswalk-level top12

# Phase 1.5 (leaves58) — walk-forward
uv run python scripts/forecast_truflation_cpi_bottomup.py \
    --crosswalk-level leaves58

# Persistence comparison + DM tests
uv run python scripts/score_phase1_vs_persistence.py --label top12
uv run python scripts/score_phase1_vs_persistence.py --label leaves58

# Single-origin live forecast
uv run python scripts/forecast_truflation_cpi_bottomup.py \
    --crosswalk-level top12 --origin 2026-05-01
```

## Files

- `scripts/forecast_truflation_cpi_bottomup.py` — main forecaster +
  walk-forward driver; supports `--crosswalk-level top12 | leaves58`
- `scripts/score_phase1_vs_persistence.py` — persistence baseline + DM
  tests; supports `--label top12 | leaves58`
- `results/truflation_cpi_forecast/walk_forward_summary_<label>.csv` —
  per-(origin, horizon) forecasts + actuals + errors + bands
- `results/truflation_cpi_forecast/walk_forward_aggregate_<label>.csv` —
  per-horizon RMSE/MAE/coverage
- `results/truflation_cpi_forecast/persistence_comparison_<label>.csv` —
  vs persistence headline + DM stats

## Glossary (stats terms)

- **Walk-forward / rolling-origin eval.** At each origin t, train on
  data up to t only, predict t+h, advance origin, repeat. Honest OOS;
  no peek-ahead.
- **AR(1) on log-returns.** `r_t = α + φ · r_{t-1} + ε_t` on daily
  log-changes. Forecasted log-returns compose into a level path.
- **M2 composition.** Composite level = Σ_c w_c · rebased_c(t). YoY
  computed on the composite (Method 2 in `composition_check.py`).
- **Bootstrap residual density.** Resample empirical residuals (with
  replacement) onto the AR(1) point path to produce sample paths.
- **Sample median as point.** Median of the sample paths used as the
  point forecast — invariant under monotone transforms (level → YoY),
  no Jensen bias.
- **Anchor offset.** Constant additive correction so forecast at origin
  = actual at origin within rounding.
- **Coverage / sharpness / width.** Fraction inside the band, mean
  width of the band — both must be reported (wide bands trivially
  cover everything).
- **Diebold-Mariano (DM).** Test of equal MSE between two forecasts.
  Statistic > 0 ⇒ model A has larger loss (B wins). Newey-West HAC
  lag = 3 controls for serial correlation in the loss differential.
- **Persistence baseline.** ŷ_{t+h} = y_t. Hard to beat on
  slow-moving series at short horizons (Stock-Watson 2007).
- **CBDF — Component-Based Dynamic Factor model.** O'Keeffe & Petrova
  2025 (NY Fed SR 1152). Composes per-component forecasters into a
  headline forecast that respects the accounting identity, with
  density. The Phase 2+ replacement for the M2 weighted sum used here.
- **UC + SV + MS.** Unobserved-Components model (slow-moving trend) +
  Stochastic Volatility (time-varying noise) + Markov Switching
  (regimes). Stock-Watson 2007 on CPI YoY level. Phase 2 trend filter.
- **Leaf set.** In a hierarchical taxonomy, the subset of nodes that
  have no descendant also in the subset. Used to avoid double-counting
  (e.g. don't include both "Food" and "Food at home" — pick one
  level).
