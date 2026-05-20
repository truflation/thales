# Tier 1 — Density nowcast of BLS CPI / BEA PCE

## One-line definition

**A daily-updating density forecast of the next BLS CPI and BEA PCE
official inflation releases, with point estimates, calibrated 80/95%
prediction bands, and per-component attribution.**

## What subscribers receive

For each of the 4 official inflation series (Headline CPI, Core CPI,
Headline PCE, Core PCE):

- **Point forecast** of the next monthly print (e.g., "BLS Headline CPI
  YoY for May 2026: 2.84%")
- **80% and 95% bands** (e.g., 80% band [2.65%, 3.03%])
- **Per-component contribution** to the forecast (Utilities +0.08pp,
  Food +0.05pp, Health +0.03pp, …) — explains *why* the forecast moved
- **Cleveland Fed comparator** alongside (so subscribers can see
  divergence)
- **Updated daily** as new Truflation observations land

## Inputs

- Truflation daily component panel (12 top-level + 80 sub-streams)
- BLS CPI subindex panel (23 series, monthly)
- BEA PCE price indexes (PCEPI, PCEPILFE, monthly)
- Cleveland Fed nowcast (comparator, daily)
- FRED macro covariates (rates, oil, gas, FX, employment, ~47 series)
- All ingested with vintage discipline via ALFRED

## How it works

```
12 per-category Truflation streams
        ↓
   (5 Phase 1 archetypes — commodity TVP for utilities/transport,
    BSTS for recreation/food-away, UC-SV-MS for health/education,
    VECM for clothing/imports, hierarchical for housing)
        ↓
   Per-category density forecast (point + bands + samples)
        ↓
   (CBDF composition with real BLS/Truflation weights, multivariate
    Gaussian residual covariance for cross-component dependence)
        ↓
   Composed headline density forecast
        ↓
   (Bridge to BLS CPI / BEA PCE via cross-walk + Path A-style
    linear adjustment for known composition drift)
        ↓
   Tier 1 published forecast
```

Density bands come from the CBDF Monte Carlo at the composition layer,
NOT from per-component bands inflated independently.

## Downstream tasks

- **Trading desks**: size positions on BLS CPI surprise (forecast vs
  consensus). Density bands tell you 80% prediction interval — direct
  input to options pricing and TIPS-vs-nominal trade sizing.
- **Asset managers**: long-duration bond positioning, inflation-linked
  ETF allocation. Density informs hedge-ratio decisions.
- **Economics research**: cite alongside Cleveland Fed in research
  notes; provides density that Cleveland Fed doesn't (their nowcast is
  point-only).
- **News outlets**: per-component attribution feeds "what's driving
  today's CPI move?" explanatory journalism.
- **Internal Truflation product team**: density forecast as a feature
  for downstream regime products.

## Comparison benchmarks (the "what we're trying to beat")

| Comparator | RMSE on Headline CPI (h=0) | Notes |
|------------|---------------------------:|-------|
| Last-release persistence | 0.382 | Floor; nobody beats this on h=1+ on level |
| AR(1) | 0.398 | Worse than persistence (unit-root issue) |
| Path A v1 (kairos, +1m frame) | 0.360 | Direct prior-art comparator |
| **Cleveland Fed (h=0 native frame)** | **0.173** | The bar; +54.8% reduction vs last-release |
| Thales target | < 0.17 | Aspirational |

(Numbers from `results/baseline_eval/FINDINGS.md` and
`results/baseline_eval/clevfed_native_FINDINGS.md`.)

## What's built today vs not

**Built ✅**
- Foundation: vintage store, ALFRED, harness, scoring DB
- All 5 archetype models with synthetic recovery
- CBDF composition layer (2.1a + 2.1b)
- End-to-end demo on real Truflation data (composition residual
  median -0.014pp)
- First archetype on real data: commodity TVP × Utilities × Henry Hub

**Not yet ⏳**
- Per-archetype real-data fits for the other 4 archetypes
- Walk-forward eval against Cleveland Fed on BLS CPI / BEA PCE
- Production deployment (daily-updating endpoint)

## Pricing / distribution model (TBD)

Likely tiered API access — institutional subscribers get full density +
component attribution + historical archive; cheaper tier gets just the
point forecast + 80% band, current snapshot only. To be specified by
Truflation business team.

## Status: **Architecture in place, gate-2 evaluation pending.**

Once gate-2 (synthetic recovery → real-data fit per archetype →
composed eval against Cleveland Fed) passes, this product can ship.
