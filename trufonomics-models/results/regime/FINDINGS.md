# Phase 2.2 — Regime Model on Real BLS CPI: Architectural Finding

**Date:** 2026-04-25
**Script:** `scripts/regime_on_headline_cpi.py`
**Module:** `src/thales/models/archetypes/uc_sv_ms.py`
**Output:** `results/regime/regime_on_bls_headline_cpi.csv`

## Headline finding

**The full UC + SV + MS model is over-parameterized for monthly BLS CPI
YoY. Applied to real data, the level walk absorbs ALL of the variance
that should drive regime detection — including the dramatic 2021-2022
inflation surge.**

This is a real architectural finding, not a bug. The synthetic recovery
on Phase 1.3 worked because the synthetic data was generated with a
genuine slow-walking level + regime jumps + SV. Real monthly CPI YoY is
already differenced — there is no genuine slow level walk for the UC
component to identify. With three latent processes (level, log-vol,
regime) competing to explain the same observed variance, MCMC picks the
mode where the most flexible component (the level) wins.

## Diagnostic

```
Year   YoY     level_smoothed   log_vol      P(high)
2014   1.62    1.61             -1.63        0.0022
2015   0.12    0.12             -1.93        0.0022
2018   2.44    2.44             -2.67        0.0022
2020   1.26    1.26             -2.57        0.0022
2021   4.68    4.68             -2.69        0.0022   ← surge starts
2022   8.00    8.00             -2.90        0.0022   ← peak inflation
2023   4.15    4.15             -2.91        0.0022
2024   2.95    2.95             -2.96        0.0022
```

The level state μ_t tracks YoY essentially exactly (correlation > 0.999).
The log-volatility h_t is stable around -2.5 to -3 (very low variance).
The high-vol regime probability P(S_t = 1) never rises above 0.002.

The model's interpretation: "There has only ever been one regime
(low-vol). The variation you see is a slowly-walking level."

This is technically correct under the model. It is also useless for
regime detection.

## Tightening the prior didn't help

Re-fit with `sigma_eta_prior_scale=0.05` (HalfNormal scale 10× tighter).
Posterior `σ̂_eta` = 0.31 — the data likelihood overrode the prior. The
level still absorbs the surge.

Why: σ_eta = 0.31 means the level can walk by 0.31 per month. With T=182
months, the level can move by 0.31 × √182 ≈ 4.2 over the sample. That's
enough to cover the ~9pp YoY range. To prevent this, σ_eta would need to
be < 0.01, which is essentially "fix the level" — defeating the UC
layer's purpose entirely.

## What the right architecture is

The choice depends on the target:

| Target          | Use                                            |
|-----------------|------------------------------------------------|
| **CPI level** (CPIAUCSL index value, trends up monotonically) | UC + SV + MS — the level walk is genuine, captures secular drift |
| **CPI YoY** (already differenced, mean-reverting around ~2-3%) | **MS + SV only** — drop the UC layer; YoY doesn't need level walk |
| **CPI MoM** (already double-differenced, very volatile) | Pure SV — no regime structure usually identifiable at monthly frequency |

For monthly CPI YoY specifically, **MS + SV without UC** is the right
spec. We ship the components separately:

- `fit_hamilton_2state` (MS only, pure numpy)  — Phase 1.3 part 1
- `fit_sv` (SV only, NumPyro)                  — Phase 1.3 part 2
- Combining MS + SV (without UC) is ~50 LoC of additive code

The full UC+SV+MS we built is correct for the synthetic recovery test
(where data has genuine level walk) and would work on CPI levels. It's
mismatched to the YoY-target frame.

## Lesson for the wider methodology

**Recovery on synthetic ≠ correct architecture for the production
target.** The Phase 1.3 synthetic recovery passed because we generated
data with the matching DGP. Applying that model to real-world data
where the DGP doesn't match (no genuine level walk in YoY) exposes the
mismatch.

This generalizes: every archetype should be re-validated on the actual
target series before claiming production readiness. Synthetic recovery
proves the *estimation core works*. Real-data application proves
*you've chosen the right model class for the data*. Both gates matter.

## Action items

1. **Phase 2.2b — MS+SV combined model** (no UC layer) — 50-100 LoC,
   1-2 hours. Apply to real BLS Headline CPI YoY, validate against
   known shocks.
2. **Phase 2.2c — UC+SV+MS on CPI LEVEL** (where level walk IS real) —
   apply existing model to CPIAUCSL index value, see if regime
   classification on level matches our intuition (it should).
3. Update Phase 2.2 production guidance to flag the UC-vs-YoY mismatch
   and prescribe MS+SV (or pure SV) for already-differenced targets.

## Outputs preserved

- `regime_on_bls_headline_cpi.csv` — full smoothed paths + diagnostic table
  for the failed-fit run. Useful as a counter-example.
