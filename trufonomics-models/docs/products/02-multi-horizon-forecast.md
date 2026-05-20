# Tier 2 — Multi-horizon forecast with scenario conditionals

## One-line definition

**Daily-updating density forecasts of BLS CPI / BEA PCE at horizons
from −1 month (real-time backcast) through +12 months, with the ability
to condition on user-specified scenarios (e.g., "what if oil rises 20%
over 6 months?").**

## What subscribers receive

For each official target × each horizon:

- **Density forecast at h ∈ {−1, 0, +1, +3, +6, +9, +12} months**
- **Scenario conditionals**: same forecast but conditioned on
  user-supplied future paths for key drivers (oil, mortgage rates,
  wages, FX, etc.)
- **Forecast updates daily** as new Truflation data lands
- **Path-coherent fan charts** — h+3 forecast respects the h+1 forecast
  (no inconsistent multi-horizon paths)

## Why density + scenarios > point forecast at long horizons

Beyond +1 month, point forecasts are nearly useless — Stock-Watson
2007 says nobody beats persistence on CPI YoY at +1 to +6 months on
RMSE alone. The product value at long horizons is:

1. **Density** — how wide is the band, not where the point is
2. **Scenario conditionals** — "what's my CFO budget if oil stays
   here vs runs to $120?" — this is the question CFOs actually ask

Tier 1 sells the surprise; Tier 2 sells the planning.

## Inputs

Same as Tier 1, plus:

- User-supplied scenario paths for one or more drivers
- Cross-driver correlation structure (estimated from historical data)
  for honest scenario forecast — when oil moves, gas moves; when
  mortgage rates move, housing rents follow with lag

## How it works

The same per-category state-space archetypes that power Tier 1 already
support multi-step forecasting natively (Kalman filter forward
iteration; for the SV/MS layer, Monte Carlo from the posterior).
Scenario conditionals come from substituting the conditioning variable
into the forward iteration:

```
Standard +12m forecast:
    Use estimated drift / regime / SV to project archetype states
    forward 12 steps; aggregate via CBDF.

Scenario-conditional +12m forecast (e.g. "oil at $120"):
    For commodity-passthrough archetypes, replace the projected
    commodity series with the scenario path. For all others, project
    normally. Aggregate via CBDF.
```

The CBDF composition layer ensures the scenario propagates through to
headline density consistently with cross-component correlations.

## Downstream tasks

- **CFOs / treasurers**: annual budget scenarios; multi-year capex
  planning. Subscribers feed in their commodity exposure and get a
  forecast distribution under various scenarios.
- **Lenders**: price floating-rate loans against +12m inflation
  density. Risk-management.
- **Commodity buyers**: hedge fuel, food, metals exposure. Use density
  + scenario to size hedge ratios.
- **Policy researchers**: counterfactual analysis ("what if Fed cuts
  rates 50bp?"). Conditioning on scenarios lets researchers test
  stories.
- **Macro hedge funds**: medium-term thematic positioning (next 6-12
  months) needs density, not just point.

## Comparison benchmarks

- **SPF** (Survey of Professional Forecasters) — quarterly, point + IQR
- **Blue Chip** — paid; consensus mean
- **Atlanta Fed GDPNow** — quarterly GDP nowcast (different target but
  similar methodology lessons)
- **Bloomberg consensus** — survey aggregation

Thales differentiates by:
- **Daily updating** vs monthly/quarterly survey
- **Density** with proper bands vs point + IQR
- **Scenario conditionals** that traditional surveys can't deliver
- **Per-component attribution** at every horizon

## Status: **Phase 2.3 / not started**

The Tier 1 architecture supports this natively — multi-step Kalman
projection and scenario substitution are 1-2 weeks of additional work
once Tier 1 is shipped. The current archetypes already produce all the
state needed.

The harder work is scenario *interface design*: which variables can
subscribers condition on, how do they specify paths, what's the
default cross-driver correlation. That's product work, not modeling
work.

## What needs to land before Tier 2 ships

1. Tier 1 production (archetypes fitted on real data, composed eval
   passes vs Cleveland Fed)
2. Multi-horizon harness extension — `walk_forward` currently supports
   horizon parameter but the bands at h>1 need extra MC iteration
3. Scenario-conditional API — design the input format for user scenarios
4. Cross-driver correlation model — for scenario coherence

All four are clear, scoped work. None require new Phase 1 archetype
research.
