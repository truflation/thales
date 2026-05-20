# 07 — Product Positioning

Three products, same model stack, three distinct buyer personas. Getting this positioning right is commercially critical because buyers lump these together and undervalue the model layer.

## The three things

| | Real-time index | Nowcast | Forecast |
| --- | --- | --- | --- |
| **Example** | "Today Truflation US CPI YoY is 1.84%" | "March 2026 BLS CPI will print at 3.31% ± 0.07%" | "June 2026 BLS CPI will be 3.15% ± 0.25% under baseline rates path" |
| **Reference period** | Current (today) | Current/recent but unreleased | Future |
| **Model required?** | No (aggregation) | Yes (state-space nowcast) | Yes (state-space forward projection) |
| **What it measures** | Current reality | Expected official release | Expected future release |
| **Uncertainty** | Not quantified | Calibrated density | Calibrated density, wider |
| **Horizon** | 0 (settled) | h ≈ 0 (not yet released) | h ≥ +1 month |

## The distinction is horizon, not model

The nowcast-vs-forecast distinction is not a modeling distinction. It's a horizon distinction. The same state-space model produces all horizons — horizon 0 = nowcast, horizon +6 = 6-month forecast. The math is identical; uncertainty expands with horizon.

This matters because some shops make the mistake of having "nowcast models" and "forecast models" as separate stacks. We don't. One stack, multiple horizons, three product packagings.

## Why a nowcast differs from the real-time index

The Truflation real-time index today = aggregation. No model. No prediction of BLS. Leads BLS by ~45 days mechanically because it measures the same reality sooner without smoothing.

The nowcast adds four things the current product structurally cannot do:

1. **Explicit prediction of the official print with confidence bands.** Clients currently eyeball the Truflation-to-BLS gap. A nowcast delivers "April BLS CPI = 2.81% ± 0.09% at 92% confidence." That's a different product — forecast of the government number, not measurement of reality.

2. **Learned methodological wedge.** BLS smooths, imputes, uses OER, samples monthly. Truflation doesn't. The wedge is predictable and regime-dependent. CBDF learns it per component and translates real-time reading into BLS equivalent.

3. **Component-level attribution with uncertainty.** When index moves 12bps, current product shows which components moved. Nowcast tells you which components drove the *forecast change* for next month's official print, with shock decomposition.

4. **Density forecasts, not point estimates.** Options desks, risk managers, TIPS traders need full distribution — P(CPI > 3.0%), surprise probabilities, fan charts.

Framing: Truflation today = real-time measurement. Nowcast = model-based prediction of official release built on top. Bloomberg analogy: spot FX vs forward curves. Same relationship.

## Why a forecast differs from a nowcast

Different horizon, dramatically different buyer.

**Nowcasts** are easy to validate. Release comes in weeks, score, repeat. Tight feedback loop generates credibility fast. Buyer: near-term decision tied to specific release — inflation-swap traders, TIPS desks, macro funds around CPI day, reporters, central bank watchers. Sales cycle: short.

**Forecasts** are harder to validate — wait months to score six-month forecasts. But buyer is much larger: CFOs, procurement, lenders, real estate operators, restaurants, insurance underwriters. They don't care about April's print; they care about where inflation goes over their planning horizon. Sales cycle: longer, which is exactly why pre-registered methodology and track record matter.

Nowcasts → sold to financial markets.
Forecasts → sold to operating businesses.
Both from same model stack, priced differently.

## How to demonstrate forecast ability

The evaluation harness handles this natively at longer horizons. Make it *visible and credible* to non-nowcast buyers:

1. **Multi-horizon performance tables.** RMSE, CRPS, calibration at horizons 0, +1, +3, +6, +12. Compare against random walk, AR(1), SPF at matching horizons, TIPS-implied breakevens.

2. **Fan charts.** Bank of England-style, expanding bands over 12+ months. Most effective visualization for non-technical buyers. Uncertainty widening visibly builds trust.

3. **Conditional forecasts.** State-space models crush ML blackboxes here. Take exogenous paths (rates, WTI, FX, tariff) and produce scenario-conditional forecasts: "If Fed holds flat, here's inflation; if they cut 100bps, here's the alternative." CFOs and policy analysts need this; pure time-series models can't give it. Sell explicitly as scenario planning.

4. **Forecast revision tracking.** Log every forecast ever made and its revision history. Chart: "Here's how our June 2026 CPI forecast evolved January through June as new data arrived, vs realization." Rare and impressive.

5. **Economic value backtests.** Don't just show RMSE — show decision value. "A restaurant chain using our 6-month food inflation forecast to time menu-price changes would have captured X bps of margin over 2022–2025." That closes a CFO.

6. **Rolling-window re-evaluation.** Every quarter, re-run evaluation and publish. Live track record beats any claim. Public URL, like Cleveland Fed.

## The "45-day lead" claim — careful framing

Currently Truflation's "45-day lead" is a real-time index property, not a nowcast or forecast claim. It says: "reality Truflation measures today shows up in BLS ~45 days later." True, but easy to misread as forecasting.

When we launch model products, distinguish clearly:

- Index **leads** BLS by ~45 days — *measurement-frequency advantage*
- Nowcast **predicts** BLS prints with quantified accuracy — *modeling advantage on top of measurement advantage*
- Forecast **predicts** future BLS prints conditional on scenarios — *model extrapolation*

Three products, three value propositions, one data + model stack.

## The practical test

A clean self-check: can you point the model at June 2026 (future) and ask for a forecast? Yes → forecast model. At March 2026 (past but unreleased) and ask for a nowcast? Yes → nowcast model. At February 2026 (past and released) and produce a backcast matching within error? Yes → retrospective validation.

All three from the same state-space machinery. Three products because buyer brains work that way, not because three model stacks are needed.

## Buyer personas and pricing

| Persona | Product tier | What they pay for |
| --- | --- | --- |
| CPI-swap trader | Nowcast API, real-time | Low-latency density forecast of next print |
| TIPS desk | Regime model + nowcast | Regime probability + density forecast |
| Macro hedge fund | Everything, research channel | Alpha signal + white-glove analyst access |
| Central bank watcher / reporter | Public dashboard freemium | Credibility of track record + shareable chart |
| CFO (mid-market) | Forecast + transmission | Category-specific forward inflation tied to cost structure |
| Procurement (enterprise) | Forecast + scenario API | Input-cost forecasts, multi-horizon, per-SKU mapping |
| Lender / credit | Forecast + macro | Household cash-flow stress projection |
| Real estate operator | Housing archetype + forecast | Rent and owned-cost forecasts, metro-level |

Each persona has different tolerance for latency, different density granularity needs, different horizon interest. Price accordingly.

## Competitive framing

Not competing with:
- BLS / BEA (they're the reference we predict)
- Truflation the real-time index (that's the data asset, upstream of us)

Competing with:
- Cleveland Fed inflation nowcast (public, free, monthly — we beat on frequency, density, category detail)
- SPF (quarterly, point forecasts only)
- Bank research inflation forecasts (proprietary, opinionated, slow)
- Bloomberg consensus (aggregation, not a model)
- Internal CFO forecasts (usually one-number with no bands)

The winning pitch: "Fed-grade nowcast, updated daily, with calibrated density, component attribution, and scenario conditionals, on data that leads official by 45 days by construction."

No one else ships all four at once.
