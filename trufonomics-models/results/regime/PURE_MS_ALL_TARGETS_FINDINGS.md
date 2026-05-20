# Pure MS regime detector — applied to all 4 official inflation YoY targets

**Date:** 2026-04-25
**Script:** `scripts/regime_pure_ms_all_targets.py`
**Model:** `fit_hamilton_2state` (pure Markov-switching variance, Phase 1.3 part 1)
**Outputs:** `results/regime/regime_pure_ms_{cpi,core_cpi,pce,core_pce}_yoy.csv`

## Headline result

The pure MS regime detector applied to all 4 official inflation YoY
measures (BLS Headline CPI, BLS Core CPI, BEA Headline PCE, BEA Core
PCE) cleanly identifies coherent regime windows. **Highest-confidence
cross-target signal: the post-COVID inflation surge.**

```
                                   CPI    Core CPI    PCE    Core PCE
Oil price collapse 2014-15        0.92      0.00     0.96      0.00
COVID-19 onset 2020 (6 mo)        0.02      0.00     0.01      0.00
Post-COVID surge 2021-23          0.81      1.00     0.93      1.00
Disinflation 2024                 0.00      1.00     0.01      1.00
```

Numbers are mean P(high-vol regime) across each window.

## Three economically meaningful patterns

### 1. Energy shocks fire only on Headline measures (as they should)

The 2014-2015 oil price collapse appears as a 13-month high-vol regime
in Headline CPI (P=0.92) and Headline PCE (P=0.96), but **doesn't fire
at all** in Core CPI (P=0.00) or Core PCE (P=0.00). This is exactly
correct — Core measures exclude food and energy, so an oil shock
shouldn't move them.

The model is correctly distinguishing energy-driven volatility from
underlying inflation volatility — without us telling it to.

### 2. Core measures are STILL in high-vol regime today (60 months and counting)

```
Core CPI:   2021-03 → 2026-03  (60 months, peak P=1.00)
Core PCE:   2021-03 → 2026-02  (60 months, peak P=1.00)
```

While Headline measures returned to low-vol (CPI by mid-2023, PCE by
end-2023), the Core measures are STILL in high-vol regime through the
most recent published data. This is the most interesting finding from
this run. **Core inflation has not normalized to its pre-COVID variance
regime** even though the levels have come down (Core CPI was 6.6% peak,
now ~3.0%).

Reading: the underlying *volatility* of core inflation moves remains
elevated. Month-to-month changes are larger than they used to be even
though the LEVEL of inflation has settled. This is consistent with what
Fed governors are saying about "stickiness in services inflation" —
they're not just talking about levels.

### 3. COVID onset (2020) was not a regime change

All four measures show P(high) < 0.05 across 2020-03 → 2020-08. CPI
YoY did dip briefly negative (2015-style oil-driven), but the 6-month
window was too short to overcome the Hamilton filter's prior on regime
persistence (p_00 ≈ 0.99). The model correctly says "this was a brief
disturbance, not a regime change."

In contrast, the 2021-2023 surge was a sustained 22-30 month elevation
across measures — the filter commits to a regime change because the
evidence is overwhelming.

## Per-target windows

```
BLS Headline CPI:  σ_low=0.79, σ_high=3.94, p_00=0.99, p_11=0.94
  • 2014-12 → 2015-12  (13 months, peak P=1.000)  oil collapse
  • 2021-04 → 2023-05  (26 months, peak P=1.000)  post-COVID surge

BLS Core CPI:      σ_low=0.27, σ_high=2.53, p_00=0.99, p_11=0.99
  • 2011-01 → 2011-04   (4 months, peak P=0.993)  post-GFC residual
  • 2021-03 → 2026-03  (60 months, peak P=1.000)  STILL HIGH

BEA Headline PCE:  σ_low=0.64, σ_high=3.05, p_00=0.99, p_11=0.95
  • 2014-12 → 2016-02  (15 months, peak P=1.000)  oil collapse
  • 2021-04 → 2023-10  (31 months, peak P=1.000)  post-COVID surge

BEA Core PCE:      σ_low=0.24, σ_high=2.45, p_00=1.00, p_11=0.99
  • 2021-03 → 2026-02  (60 months, peak P=1.000)  STILL HIGH
```

## Cross-target coherence as a robustness check

The fact that all 4 measures agree (within their structural definitions)
on the post-COVID surge is **strong validation** that pure MS is
detecting genuine economic regime changes rather than statistical
artifacts. Independent regression on 4 different transformations of
similar underlying data converging on the same regime windows is the
right kind of signal.

## What this enables

1. **The "VIX for inflation" product has its core working.** Pure MS on
   monthly YoY identifies regime changes coherently across measures.
2. **Sticky vs flexible decomposition (Bils-Klenow)** is a natural
   next step: apply pure MS to the Atlanta Fed sticky-price CPI vs
   their flex-price counterpart. Expect sticky to STILL be in high-vol
   regime (consistent with Core findings); flex to have returned to low.
3. **Component-level regime detection** for the 12 Truflation top-level
   categories — see if Health (sticky services) is in different regime
   from Food at home (flexible).

## Outputs

```
results/regime/regime_pure_ms_cpi_yoy.csv
results/regime/regime_pure_ms_core_cpi_yoy.csv
results/regime/regime_pure_ms_pce_yoy.csv
results/regime/regime_pure_ms_core_pce_yoy.csv
```

Each contains date, YoY value, P(high-vol regime). Plot in any tool to
visualize the regime probability time series.
