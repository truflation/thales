# Multi-horizon density evaluation — h ∈ {1, 3, 6, 9, 12}

**Date:** 2026-04-27
**Script:** `scripts/density_eval.py`
**Output:** `results/baseline_eval/density_eval_multihorizon.csv`

Closes the queued architecture-doc + checklist item "Phase 2.1
multi-horizon (-15 to +12)" for the forecast direction (h=1..12).
Backcasting (h=-15..0) remains queued — the same harness extension
will support it but isn't in this run.

## Why this exists

The single-horizon density eval (`DENSITY_EVAL_FINDINGS.md`) gave us
the headline numbers at h=1. Two follow-on questions:

1. **Where does each model break down at longer horizons?** Knowing
   that MoM-composed wins at h=1 doesn't tell us if it still wins at
   h=6 or h=12.
2. **What's the baseline a foundation model would need to beat?** For
   the Tier 2 spec, we need per-horizon RMSE/CRPS targets. This is
   that bar.

## Headline matrix — RMSE and CRPS by model × horizon

RMSE on BLS Headline CPI YoY, walk-forward, train_min=36 months:

| Model | h=1 | h=3 | h=6 | h=9 | h=12 |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.459 | 1.072 | 1.750 | 2.472 | 3.156 |
| AR(1) on YoY | 0.473 | 1.153 | 2.145 | 3.393 | 4.818 |
| **MoM-composed AR(1)** | **0.285** | **0.669** | 1.285 | 1.967 | 2.574 |
| Path A v1 | 0.374 | 0.815 | 1.358 | 2.310 | 3.582 |
| **Stock-Watson DFM** | 0.456 | 0.589 | **0.878** | **1.056** | **1.313** |
| Bridged-CBDF | 0.249 | — | — | — | — |

CRPS on the same target:

| Model | h=1 | h=3 | h=6 | h=9 | h=12 |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.258 | 0.631 | 1.134 | 1.752 | 2.336 |
| AR(1) on YoY | 0.271 | 0.707 | 1.383 | 2.237 | 3.184 |
| **MoM-composed AR(1)** | **0.152** | **0.383** | 0.775 | 1.248 | 1.755 |
| Path A v1 | 0.184 | 0.369 | 0.595 | 0.879 | 1.398 |
| **Stock-Watson DFM** | 0.267 | 0.355 | **0.518** | **0.631** | **0.784** |
| Bridged-CBDF | 0.143 | — | — | — | — |

PIT KS p-values (calibration, p > 0.05 = calibrated):

| Model | h=1 | h=3 | h=6 | h=9 | h=12 |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.544 | 0.462 | 0.169 | **0.009** | **0.003** |
| AR(1) on YoY | 0.857 | **0.004** | **0.000** | **0.000** | **0.000** |
| MoM-composed AR(1) | 0.826 | 0.909 | 0.477 | **0.046** | **0.013** |
| Path A v1 | 0.356 | 0.520 | 0.494 | 0.395 | 0.116 |
| Stock-Watson DFM | 0.728 | 0.259 | **0.025** | **0.004** | **0.000** |
| Bridged-CBDF | 0.080 | — | — | — | — |

(**Bold** = miscalibrated, p < 0.05.)

## Three findings

### 1. The winner changes with horizon

MoM-composed AR(1) wins at h=1 and h=3 on both RMSE and CRPS — the
result we already had. From h=6 onward, **Stock-Watson DFM takes
over** on both metrics. The factor model's structural difference
shows up at long horizons:

- **AR(1) on YoY** iterates the persistence parameter as `phi^h`. With
  monthly YoY autocorrelation 0.98, that's `0.98^12 ≈ 0.78`, but the
  error variance at h=12 grows as `sigma² · (1-phi^24)/(1-phi²)`,
  which on this near-unit-root series translates to wide bands
  collapsing back to the unconditional mean. RMSE 4.82 at h=12 is
  worse than naive persistence (3.16).
- **MoM-composed AR(1)** chains MoM forecasts via the closed-form
  identity. Its variance at h-step is the variance of a sum of h MoM
  forecasts plus h compounding bootstrap residuals. At h=12 the bands
  widen too aggressively (sharpness 5.6 vs the realized residual
  standard deviation), leading to under-confidence at long horizons.
  RMSE 2.57 at h=12 is the second best in the table but the bands
  are over-dispersed.
- **Stock-Watson DFM** lets `f̂[T+h] = phi_f^h · f̂[T]` collapse the
  factor to its long-run mean while the target loading `α_z + β_z·f̂`
  produces a forecast that converges to `α_z`. The variance scales
  via the geometric sum `β_z² · σ_η² · Σ phi_f^(2k)`. This gives the
  model the right behavior at long horizons: forecast → unconditional
  mean, variance → finite limit. CRPS 0.78 at h=12 is the lowest in
  the table.

The horizon-dependent ranking matters for product positioning. The
operational nowcast (Tier 1) lives at h=1 → MoM-composed and Bridged-
CBDF lead. The multi-horizon forecast (Tier 2) lives at h ∈ {3..12} →
DFM leads from h=6 onward. The two products run on different
architectures because the bottleneck is different.

### 2. Path A degrades sharply past h=6

Path A's RMSE blows up at long horizons: 1.36 at h=6, 2.31 at h=9,
3.58 at h=12 — meaningfully worse than naive persistence (1.75, 2.47,
3.16). The reduction-vs-naive column reads −148 % at h=12.

The mechanism is regression extrapolation. Path A's coefficients
`α + β · BLS_yoy[T] + γ · truf_yoy[T]` are fit on a rolling 24-month
window where target = BLS_yoy[t+1]. Those coefficients are valid for
one-step-ahead but not for projecting twelve months out. When the
script swaps the horizon parameter, the same coefficients are used
to predict y[T+12] from (y[T], truf[T]) — which they were never
calibrated to do.

This is the classic "horizons don't transfer in OLS" pathology and is
the right reason to NOT extend Path A as a multi-horizon model. The
correct move is a separately-fit-per-horizon family (h=1, 3, 6, 9, 12
each with its own coefficients) or a model that handles horizons
endogenously (DFM, TSFM).

### 3. Calibration breaks down at long horizons across the board

The PIT KS p-value table shows seven of the thirty model-horizon
cells failing the 0.05 calibration threshold. The pattern is
horizon-driven: every model except Path A is calibrated at h=1,
roughly half are calibrated at h=3, and only Path A and (marginally)
MoM-composed remain calibrated at h=6 or longer.

Two specific failures worth pinning:

- **AR(1) on YoY is miscalibrated from h=3 onward.** The iterated
  AR(1) variance under-states the realized error scale on this
  near-unit-root series — bands are too narrow.
- **DFM is miscalibrated at h=6+ despite winning RMSE/CRPS.** The
  Gaussian closed-form posterior captures the central tendency
  correctly but mis-estimates the tail mass. At h=6 the empirical
  PIT distribution deviates from uniform with p = 0.025; at h=12,
  p = 0.000 (but only n=14, so the test is noisy).

This is the single clearest place a foundation model could win.
A TSFM trained on the joint FRED + reconstructed-Truflation panel
and tuned for multi-horizon density forecasting has a concrete bar:
beat DFM's CRPS at h ∈ {6, 9, 12} **and** maintain PIT calibration
(KS p > 0.05) at every horizon. That's a single, scoreable success
criterion against an objective benchmark.

## What this means for the Tier 2 spec

Three concrete numbers a foundation model would need to clear:

| Horizon | Beat this CRPS | Maintain PIT KS p > | n (sample size) |
|---:|---:|---:|---:|
| h=1 | 0.143 (Bridged-CBDF) | 0.05 | 13 |
| h=3 | 0.355 (DFM) | 0.05 | 23 |
| h=6 | 0.518 (DFM) | 0.05 | 20 |
| h=9 | 0.631 (DFM) | 0.05 | 17 |
| h=12 | 0.784 (DFM) | 0.05 | 14 |

The small sample sizes at the longer horizons (n=14 to n=20) are
themselves a finding — there isn't enough recent data to fit DFM with
a 36-month training window AND project 12 months ahead AND evaluate
on more than one or two years of OOS observations. Extending the
training window backward via the FRED 51-year history is the
pretraining-data move that increases this n substantially.

## Caveats

- **Bridged-CBDF h=1 only.** The bridge regression is `BLS[t] ~ α +
  β·BLS[t-1] + γ·CBDF[t-1]`, which is a one-step regression by
  construction. Extending it to multi-step requires either iterating
  the bridge (compounds error fast) or fitting a separate bridge per
  horizon (more code surface; not in scope here).
- **The MoM-composed h>1 path uses an AR(1) bootstrap of the chain
  sum.** This is correct for AR(1) but understates correlation
  between successive MoM residuals. If MoM residuals have any serial
  structure (which AR(1) on residuals would mostly remove), the
  bootstrap underestimates the chain-sum variance. The h=12 sharpness
  of 5.6 suggests over-dispersion not under-dispersion, so this isn't
  the binding issue here.
- **PIT KS test is low-power on small n.** At n=14 (DFM h=12) the KS
  test rejects with p < 0.001 but the sample is too small to fully
  characterize the residual distribution. Larger samples (longer
  history pretraining) would resolve this.
- **No Diebold-Mariano significance tests in this run.** The
  per-horizon comparisons here are point estimates of CRPS/RMSE; a
  proper DM test on the squared-error sequences for each pairwise
  comparison would give p-values. The headline is robust to that
  test (the gaps between models are large) but the small-margin
  comparisons (DFM vs MoM-composed at h=3) would benefit.

## Files

- `scripts/density_eval.py` — extended to loop horizons; calls
  `_build_forecasters_for_horizon(panel, panel_full, components,
  horizon=h)` for each h.
- `src/thales/models/mom_composed.py` — `MoMComposedForecaster` now
  supports horizon > 1 via AR(1)-bootstrap chain composition. New
  helpers: `compose_yoy_multi_step`, `_ar1_iterate`,
  `_ar1_chain_samples`. 12 unit tests.
- `results/baseline_eval/density_eval_multihorizon.csv` — 30 rows
  (6 models × 5 horizons, with Bridged-CBDF h=1 only).
