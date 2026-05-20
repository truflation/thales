# Phase 1.75 — Hybrid (Phase 1 point + Phase 2 density): walk-forward findings

**Date:** 2026-05-01
**Script:** `scripts/forecast_truflation_cpi_hybrid.py`

## What this is

A no-extra-fits hybrid: take Phase 2's quantile bands and **translate**
them to be centered on Phase 1's anchor-corrected point. Quantile
bands shift uniformly under translation, so::

    hybrid_point = phase1_point
    hybrid_band  = phase2_band + (phase1_point - phase2_point)

Equivalent to recentering Phase 2's sample distribution on Phase 1's
better point estimate, preserving the regime-aware width.

Evaluated at the 33 Phase 2 origins (quarterly month-ends, 2018-01-31
→ 2026-01-31), h ∈ {30, 90} days.

## Per-horizon results

| Horizon | n | RMSE (pp) | MAE (pp) | Bias (pp) | 80% cov | 95% cov | Width80 | Width95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 d | 33 | 0.688 | 0.490 | −0.267 | 69.7% | 81.8% | 4.43 | 10.71 |
| 90 d | 33 | 1.869 | 1.401 | −0.553 | 51.5% | 63.6% | 5.05 | 14.85 |

## Head-to-head (same 33 origins, Phase 2 schedule)

| Horizon | Phase 1 RMSE | Phase 2 RMSE | Hybrid RMSE | Phase 2 cov80 | Hybrid cov80 |
|---:|---:|---:|---:|---:|---:|
| 30 d | 0.688 | 3.034 | **0.688** | 78.8% | 69.7% |
| 90 d | 1.869 | 3.139 | **1.869** | 57.6% | 51.5% |

(Phase 1 RMSE here is higher than its 102-origin baseline of 0.467 pp /
1.557 pp because the 33-origin Phase 2 schedule weights toward
surge-transition origins.)

## What the hybrid fixes — and what it doesn't

**Fixes (vs Phase 2):**
- Point RMSE collapses from 3.03 → 0.69 at h=30 (4.4× better) and
  3.14 → 1.87 at h=90 (1.7× better). The −1 pp Phase 2 bias is
  eliminated — hybrid bias drops to −0.27 pp at h=30.

**Does not fix:**
- Bands are still 4.43 pp wide at h=30 — Phase 2's regime-mixing
  inflated them when sampling all paths through the Markov chain,
  even when the origin is in a calm regime. The hybrid inherits this
  width; the recentering doesn't shrink it.
- Coverage at h=30 (69.7%) is *worse* than Phase 2 alone (78.8%)
  because Phase 2's wide off-center bands "rescued" surge-period
  realizations by accident. Recentering on the better Phase 1 point
  removes that accidental coverage without compensating it
  elsewhere.

## What this actually says about the architecture

The hybrid surfaces a structural insight: **Phase 1 and Phase 2 are
solving different problems and can't be merged by translation alone.**

- **Phase 1's bands are too narrow** because they propagate only
  component-level AR(1) residual uncertainty, ignoring (a) cross-
  component covariance and (b) regime-shift variance.
- **Phase 2's bands are too wide** because they propagate full Markov-
  chain regime variance even at calm origins. The forward simulation
  samples high-vol paths with non-trivial probability regardless of
  P(high|origin).

**The right architecture for bands** is regime-conditional:
- When P(high-vol|origin) is low (Phase 2's smoothed terminal state),
  use Phase 1's tight bands scaled by a constant.
- When P(high-vol|origin) is high, use Phase 2's wide bands.
- The smoothed regime probability from Phase 2 — already saved as a
  per-origin column in the Phase 2 walk-forward CSV — is the natural
  gating signal.

This is the "VIX for inflation" Tier 3a regime indicator the
architecture spec already calls for, used here as a *band-width
modulator* rather than a standalone product.

## Verdict

**Phase 1.75 hybrid as a deliverable: not shippable.** RMSE matches
Phase 1; bands are inherited-from-Phase-2 wide without the
calibration justification. We've gained no point accuracy and lost
some Phase 2 coverage.

**Phase 1.75 as a diagnostic: useful.** It localizes the architectural
gap to "bands should be regime-conditional, not propagated from
either model in isolation." That's a Phase 3 or Tier 3a engineering
item, not a Phase 1.x fix.

## Next move

Continue to **Phase 3 — Almosova LSTM** per the architecture spec.
Phase 1's bottom-up baseline (RMSE 0.467 pp at h=30 over 102
origins) is the production CPI track for now. Phase 3's job: a deep
encoder over the same 12-/58-stream Truflation panel that captures
the cross-component covariance Phase 1 ignores, and produces a
density that's tight in calm regimes, wide in volatile ones.

## Reproduce

```bash
# Hybrid (no new fits — reads Phase 1 + Phase 2 outputs)
uv run python scripts/forecast_truflation_cpi_hybrid.py
```

## Files

- `scripts/forecast_truflation_cpi_hybrid.py` — hybrid driver (calls
  Phase 1 forecaster at Phase 2 origins, translates Phase 2 bands)
- `results/truflation_cpi_forecast/walk_forward_summary_hybrid.csv`
- `results/truflation_cpi_forecast/walk_forward_aggregate_hybrid.csv`
