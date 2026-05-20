# Gate-2 (first cut) — Composed Thales bridged to BLS

**Date:** 2026-04-25
**Script:** `scripts/gate2_thales_vs_clevfed.py`
**Frame:** +1 month forecast
**Window:** 2023-01-31 → 2026-02-28 (37 origins)
**Target:** BLS Headline CPI YoY

## Headline result — honest

**Thales-bridged loses to BLS persistence at the +1 month frame** when
the inner is per-component persistence:

| Model | RMSE | MAE | 80% cov | 95% cov | Direction |
|-------|-----:|----:|--------:|--------:|----------:|
| `thales_bridged_v1` (composed persistence + bridge) | 0.620 | 0.519 | 91.9% | 100.0% | 54.1% |
| `bls_persistence_v1` (the floor) | 0.361 | 0.254 | 89.2% | 94.6% | 54.1% |
| Δ RMSE vs persistence | **−71.55%** worse | | | | |

This is an expected and informative null result. **It tells us
exactly what the bridge does and doesn't add when stacked on a
per-component-persistence inner.**

## Why this happens — structural

1. **Per-component persistence ≈ direct Truflation persistence by
   accounting identity** (the gate-2-lite finding from earlier today).
   Composing N per-component persistence forecasts gives a Truflation
   forecast that's structurally ~equal to direct Truflation persistence.
2. **The bridge converts Truflation persistence → BLS prediction** via
   `BLS_yoy ≈ α + β · truf_yoy`. With β = 0.39 (recovered) and α = 1.96
   (capturing the level gap), the bridge is mostly accounting for
   structural offset, not adding leading-indicator information.
3. **Truflation_yoy as a leading indicator of BLS_yoy at +1m has
   weak signal**. At end of month T, Truflation_yoy[T] tells you
   roughly the same thing as BLS_yoy[T-1] (the last published print).
   So the bridge effectively substitutes BLS persistence with
   Truflation persistence + linear noise — strictly worse.

## What this DOES NOT mean

- It doesn't mean Path A v1 was wrong. Path A v1 was evaluated in the
  **same-month nowcast** frame: at day 25 of month T, predict BLS for
  month T (BLS publishes mid-T+1). Truflation's daily-updating signal
  has 13-25 days of LEAD on the BLS release within the same month —
  that's where the +42% MSE reduction came from.
- It doesn't mean composition + bridge are useless. They're correctly
  passing data through; the limitation is in **the inner forecasters
  being persistence rather than the actual archetypes**.
- It doesn't mean Thales is failing. The architecture is sound; this
  test exercises one corner of it (the +1m forecast with persistence
  inputs) and documents the floor.

## What this DOES mean

The product value of Thales (Tier 1) lives in two distinct frames:

### Frame A — same-month nowcast (Path A's frame)

```
At day d of month T, predict BLS_CPI_yoy[T] (which BLS will publish
~mid-T+1, ~13 days from now).
```

Inputs at day d:
- BLS_yoy[T-1] (last published, ~30 days old)
- Truflation_yoy[d] (current, daily-aggregated)
- 12 Truflation per-component readings at day d
- FRED covariates current

This is where Truflation HAS lead value. Our archetypes + composition +
bridge SHOULD beat persistence in this frame. **Gate-2 needs to be
re-run in the same-month nowcast frame.**

### Frame B — multi-horizon density forecast (+1, +3, +6, +12)

At horizons ≥ 1m, persistence is genuinely hard to beat on the *level*
(Stock-Watson 2007). The product value is:
- Density (bands), not just point — bands ARE a real product even if
  point matches persistence
- Scenario conditionals (Tier 2) — give the user "what if" answers
- Per-component attribution — explains why the forecast moved
- Regime-conditional widening (Tier 3a integrated) — wider bands in
  high-vol regimes

## What we need to do

1. **Reframe gate-2 to the same-month nowcast** — this is a non-trivial
   harness change because our `walk_forward` is set up for h ≥ 1.
   Either extend the harness (cleaner) or write a one-shot
   same-month-eval script.
2. **Replace per-component persistence with archetype forecasters**
   (BSTS-LL on Recreation, commodity TVP on Utilities, pure MS on
   Health, VECM on Clothing). Each archetype should add information
   beyond persistence on its own component. THEN compose + bridge and
   re-eval.
3. **Score density** (CRPS, PIT, sharpness) not just point RMSE — Tier
   1 product is density first, point second.

## Current bridge performance (for reference, +1m frame)

The bridge itself converged cleanly:

```
Bridge at last origin (Feb 2026):
  α = +1.9594  (level adjustment for BLS-Truflation methodology gap)
  β = +0.3946  (slope, modest)
  residual SD = 0.2455 pp
  window = 24 months sliding
```

Bridge correctly identifies:
- BLS runs ~2 percentage points above Truflation YoY in absolute level
  (consistent with our Clothing finding)
- Slope < 1 means a 1pp move in Truflation translates to ~0.4pp in BLS
  — Truflation is more volatile than BLS

## Status

- ✅ Bridge built, 7/7 tests passing
- ✅ Gate-2 plumbing end-to-end (composed → bridged → harness → scoreboard)
- ⏳ Gate-2 in the SAME-MONTH NOWCAST frame (the right frame)
- ⏳ Replace persistence inners with archetype forecasters
- ⏳ Density-aware scoring (CRPS, PIT)

## Verdict

**The integration plumbing works.** The +1m result is consistent with
known macro literature (Stock-Watson). The right next move is reframing
to same-month nowcast where Truflation has lead value, and swapping
inner forecasters from persistence to actual archetypes.

This is genuine progress — we now have an end-to-end pipeline from
12 Truflation components → composed forecast → BLS prediction →
scored against persistence in a queryable scoring DB. Even though
this specific run loses, the next layer (archetype inners + same-month
frame) is the proper Tier 1 product test.
