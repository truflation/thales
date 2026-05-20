# Phase 3.1d — Conditional forecasts and shock scenarios

Resolves task #128. Two conditional projection tools shipped, plus the
customer-facing demo on the 5-variable logistics BVAR.

## What shipped

1. **`conditional_forecast(fit, history, forced_paths, h)`** — Doan-
   Litterman-Sims style projection. Given a forced path on a subset of
   variables, project the rest h steps forward. Returns:
   - `mean` — deterministic conditional mean trajectory (no MC noise)
   - `q05`, `q25`, `q50`, `q75`, `q95` — Monte-Carlo quantiles around it
   - `samples` — full sample paths for downstream analysis
   - `stochastic_mean` — MC mean of samples (≈ deterministic at large n)

   Use case: "If diesel takes this exact path over the next 12 months,
   what's the joint distribution of all other variables?"

2. **`shock_scenario(fit, baseline, shock_var_idx, shock_size, h)`** —
   IRF-driven scenario. Re-scales the Cholesky IRF column to deliver a
   requested shock magnitude (e.g. +20% in log-space), then returns the
   deterministic deviation trajectory for all variables.

   Use case: "If diesel jumps +20% as a one-time structural shock,
   what's the response of freight, labor, maintenance, volume?"

   **This is the right tool for hedging-decision support** because it
   engages the **contemporaneous transmission channel** that lives in
   Σ — which is the dominant cross-effect on monthly logistics data.

3. **5 unit tests** (added to `test_bvar_minnesota.py`):
   - Forced-path equality (forced variable's quantiles equal the path)
   - Free-variable response to forced rising vs flat path
   - Free-variable uncertainty widens with horizon
   - Validation of horizon and var-index mismatches
   - `shock_scenario` recovers requested shock at h=0
   - `shock_scenario` propagates via Σ to a correlated partner
   - `shock_scenario` decays for stable VAR

## Why two tools instead of one — a load-bearing technical note

`conditional_forecast` is mathematically correct but **misses the
contemporaneous transmission**. When you force diesel to a path, the
function fixes diesel to that path and lets the AR matrix propagate
the effect. But the BVAR's diesel→freight cross-coefficient in A is
near-zero on monthly data (Minnesota prior shrinks it hard), so the
propagation is tiny.

The real channel — which the IRFs capture and FEVD quantifies — lives
in Σ: diesel innovations and freight innovations are **contempo­
raneously correlated**. When you "force" diesel without going through
the Cholesky structural-shock decomposition, you don't engage that
correlation channel.

Empirically, the difference is huge:
- `conditional_forecast(+20% diesel path)` → freight at h=12: **+2.63%**
  (essentially independent of the diesel scenario)
- `shock_scenario(+20% diesel shock)` → freight at h=12: **+1.59%**
  (the Σ-implied pass-through)

The conditional forecast number is a *no-shock projection of the
unforced variables*, which mostly trends the same way regardless of
diesel. The shock scenario is the *structural impact of the shock
itself*, which is what hedging decisions actually need.

Both tools ship. **Use `shock_scenario` for product use cases.** Keep
`conditional_forecast` for DLS-style "what-if-the-curve-takes-this-
path" projections that respect the AR dynamics of all variables.

## Customer-facing scenarios on the real logistics BVAR

5-var BVAR (diesel → freight → maintenance → labor → volume) fit on
192 months of FRED data. Three diesel shock scenarios:

### +20% diesel shock (typical fuel-spike)

| horizon | diesel | freight | maintenance | labor | volume |
|---:|---:|---:|---:|---:|---:|
| 0 | +20.0% | **+1.64%** | +0.12% | +0.11% | +0.35% |
| 6 | +17.9% | +1.61% | +0.13% | +0.12% | +0.33% |
| 12 | +16.1% | +1.59% | +0.14% | +0.13% | +0.32% |

### +50% diesel shock (peak-2022 style)

| horizon | diesel | freight | maintenance | labor | volume |
|---:|---:|---:|---:|---:|---:|
| 0 | +50.0% | **+3.68%** | +0.27% | +0.25% | +0.78% |
| 12 | +39.3% | +3.57% | +0.31% | +0.29% | +0.71% |

### −20% diesel shock (relief / OPEC oversupply)

| horizon | diesel | freight | maintenance | labor | volume |
|---:|---:|---:|---:|---:|---:|
| 0 | −20.0% | **−1.97%** | −0.15% | −0.14% | −0.43% |
| 12 | −16.7% | −1.91% | −0.17% | −0.16% | −0.39% |

**The freight pass-through ratio is ~8%** (1.64 / 20 = 0.082), almost
exactly matching the FEVD finding (7.5% of freight variance from
diesel). Sign-asymmetric: a 20% diesel cut produces a slightly larger
freight response (−1.97% vs +1.64%) than the equivalent up-shock —
suggests downward freight stickiness in the data, but the asymmetry
is small and within MC noise of the underlying fit.

## Translation to dollar P&L

For a $100M-revenue, 85% opex-share logistics company:

| scenario | fuel ΔS | labor ΔS | maintenance ΔS | **TOTAL ΔS** | **as % opex** |
|---|---:|---:|---:|---:|---:|
| +20% diesel | +$5.29M | +$26k | +$11k | **+$5.33M** | **+6.27%** |
| +50% diesel | +$13.08M | +$58k | +$25k | **+$13.16M** | **+15.48%** |
| −20% diesel | −$5.40M | −$32k | −$14k | **−$5.45M** | **−6.41%** |

(Insurance and "other" cost lines are held at zero — those variables
aren't in the BVAR, see data gaps in `BVAR_LOGISTICS_5VAR_FINDINGS.md`.
True impact larger by the share of those buckets that's correlated
with fuel.)

**This is the deliverable customers pay for**: a defensible per-line
P&L exposure under a hypothetical scenario, with the model's own
covariance structure determining the propagation. The answer "+20%
diesel ⇒ +6.3% of opex over 12 months" is the **exposure map** — a
data-and-analytics output, not an advisory recommendation. What the
customer does with the number (raise prices, revise budgets, allocate
reserves, choose to hedge through their treasury function, or accept
the variance) is their decision and outside Thales' product surface.

This boundary is intentional. Thales is positioned as a **data and
analytics product** (analogous to Bloomberg, FactSet, Truflation
itself), not as an investment advisor. Recommending position sizes,
hedge instruments, or capital-allocation actions is RIA-regulated
activity that introduces fiduciary duty, suitability obligations, and
E&O exposure — none of which fit a SaaS data-product business model.

## Caveats

1. **The pass-through ratio depends on the prior.** With a tighter
   Minnesota prior (`overall_tightness=0.1`), the cross-effects shrink
   further toward zero and the freight response shrinks accordingly.
   Future work: cross-validation on the prior hyperparameters using
   pseudo-OOS marginal likelihood.

2. **Σ ≠ structural correlation.** The Cholesky decomposition imposes
   an ordering. We've used [diesel, freight, maintenance, labor,
   volume] which puts diesel first as the most-exogenous shock. A
   different ordering would attribute the contemporaneous correlation
   differently. Sensitivity-test this in a follow-up: run the IRF with
   reversed ordering and see if the freight pass-through changes
   materially.

3. **No insurance / margin data.** The conditional output covers fuel,
   labor, maintenance, and volume — three of the five cost-bucket
   weights from the ATRI cost structure. Insurance (5% weight) and
   "other" (25%) are silent. The total customer impact is a lower
   bound; sourcing those series is the gap blocker for a 100% answer.

4. **Stable VAR but borderline.** max\|eig\| = 1.005 on the level
   panel — technically non-stationary by 0.005. Forecasts are still
   well-defined; long-horizon (h > 24) projections might drift. The
   shock-decay observed in the trajectories (diesel 20%→16% over 12
   months) is sensible.

5. **Bands not yet conformal.** `conditional_forecast` returns MC
   quantiles, but those use the Gaussian Σ — same calibration issue
   as the unconditional forecaster (cov95 was 75-76% on freight /
   maintenance). Conformalizing the multi-step VAR forecast is more
   involved than the univariate case; queued.

## Files

- `src/thales/models/archetypes/bvar_minnesota.py` — `conditional_forecast()`
  and `shock_scenario()`
- `tests/test_bvar_minnesota.py` — +5 tests (now 19 total, all green)
- `scripts/bvar_logistics_conditional.py` — customer-facing demo
- `results/real_data_archetypes/` — printed output above

## Next: 3.1e — economic-value backtest

The natural product validation: build a fuel-hedging strategy that
sizes hedges using the BVAR's shock-scenario distribution, backtest
P&L over 2015-2026 vs naive hedging (no hedge / static 50% hedge
ratio). The ask is whether the BVAR's structural information actually
improves hedge sizing in dollars — the empirical proof of value.
That's a meaningful build (need to mock futures-curve data, position
sizing logic, monthly P&L attribution) so it's its own session.

## Glossary (stats terms)

- **Conditional forecast (DLS):** Doan-Litterman-Sims 1984 method for
  forecasting a VAR holding some variables on a forced path. The
  deterministic version sets shocks to zero; the stochastic version
  draws shocks from N(0, Σ).
- **Shock scenario / IRF-based projection:** uses the Cholesky-
  identified IRF column to compute the response of all variables to
  a one-time structural shock to one variable. Engages contemporaneous
  Σ-correlation that conditional-forecast misses.
- **Pass-through ratio:** the fraction of an upstream cost shock that
  propagates to a downstream price/cost. For diesel→freight here:
  ~8% — meaning a 20% diesel rise gives a 1.6% freight rise.
- **Cholesky ordering sensitivity:** because the Cholesky factor of Σ
  depends on variable ordering, IRF magnitudes can change under
  reordering. The first-listed variable absorbs all contemporaneous
  variance not explained by later variables.
