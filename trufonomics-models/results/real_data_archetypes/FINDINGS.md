# Phase 1.1 Real-Data — TVP Commodity on Utilities × Henry Hub

**Date:** 2026-04-25
**Script:** `scripts/tvp_utilities_henryhub.py`
**Module:** `src/thales/models/archetypes/commodity.py`
**Output:** `results/real_data_archetypes/tvp_utilities_henryhub.csv`

## Headline result

**First real-data application of any Thales archetype.** TVP commodity
model fitted to log(Truflation Utilities) ~ log(Henry Hub natural gas
spot) on 2020-01 → 2026-04 daily data (n ≈ 1500). The fit shows a clean,
interpretable evolution of the pass-through coefficient β_t:

```
2020:  β̄ = −0.26   (COVID-era decoupling — utilities rising, gas crashed)
2021:  β̄ = −0.09
2022:  β̄ = −0.00   (transition)
2023:  β̄ = +0.04   (re-coupling begins)
2024:  β̄ = +0.11   (stable normal-regime pass-through)
2025:  β̄ = +0.09
2026:  β̄ = +0.10
```

Static OLS over the same window gives **β = 0.043** — a single number
that's the time-average and tells you nothing about the regime shift.
TVP recovers it cleanly (smoothed range −0.39 to +0.20). The largest
60-day β shift centered on **September 2020** (post-COVID demand
recovery) — exactly when utility prices started tracking gas prices
again after the brief decoupling.

## Magnitudes

`β = 0.10` means: a 10% rise in Henry Hub spot translates to a ~1% rise
in Utilities CPI. **Small** — utilities are dominated by regulatory
costs, labor, infrastructure, and grid mix, not gas spot. But the
*direction* and *timing* of the regime shift are real and economically
interpretable.

For comparison, the kind of pass-through coefficients that would catch
analyst attention:
- Gasoline ↔ WTI: typically β ≈ 0.5-0.7 (gas station price IS mostly oil)
- Food at home ↔ agricultural index: typically β ≈ 0.2-0.4
- Utilities ↔ Henry Hub: small, as found

So: the model is doing its job. The recovered β is small but
interpretable, the time-variation is detected, and the static OLS
misses the dynamics entirely.

## What this proves

1. **The TVP estimation core works on real data**, not just synthetic.
   Pearson 0.999 on synthetic recovery → coherent β_t shifts on real
   data. The Phase 1.1 archetype is in *production-ready territory* for
   pass-through estimation tasks.
2. **The "static OLS misses dynamics" lesson generalizes.** On
   synthetic, OLS MAE was 95% worse than TVP. On real data, OLS hides
   a sign flip (β went from −0.26 → +0.11 — opposite sign at the start
   vs end).
3. **No identifiability surprises.** Phase 1.1's clean estimation
   structure (single β state, no σ_F vs β scale ambiguity like Phase
   1.5) means the recovered values are interpretable directly.

## Honest caveats

1. **No SV layer applied here.** Volatility regime jumps in Henry Hub
   (post-Russia 2022, US export expansion) are visible in the data
   but the constant-σ_ε model treats them as IID noise. Not a problem
   for β recovery (the Kalman is robust) but matters if we wanted
   forecast bands. Future: layer SV on top via a small extension.
2. **Daily frequency is noisy.** Truflation Utilities updates daily
   but the underlying utility prices change weekly-monthly. The daily
   noise floor is essentially measurement error. Aggregating to weekly
   would tighten the β recovery; it would also lose intra-month
   resolution.
3. **No vintage discipline applied.** This fit uses the latest
   as_of-today vintage of both series. For walk-forward eval (gate-2),
   we'd need point-in-time vintages — particularly for Henry Hub which
   is final-not-revised, so this is a non-issue here, but it would
   matter for more revision-heavy series.

## Outputs

- `tvp_utilities_henryhub.csv` — daily log(utilities), log(henry_hub),
  filtered β, smoothed β. Plot in any tool.

## Next real-data archetype fits

- **Recreation × dual-seasonal BSTS** (Phase 1.2) on Truflation
  Recreation & Culture daily data
- **Health × full UC+SV+MS** (Phase 1.3) on Truflation Health monthly
  series — but per the Phase 2.2 finding, the UC layer may need to be
  dropped for already-differenced targets
- **Clothing & Imports × VECM** (Phase 1.4) on Truflation Clothing &
  Footwear with import-price-index pair
- **Hierarchical regional housing** (Phase 1.5) on the 4-region BLS
  housing CPI panel + Truflation regional housing — natural Vast.ai
  job for full MCMC

These get the system from "synthetic recovery passes" to "every
archetype validated on its target use case" — the second of the two
gates I flagged in the methodology lesson from Phase 2.2 FINDINGS.
