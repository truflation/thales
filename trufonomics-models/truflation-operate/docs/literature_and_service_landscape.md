# Truflation Operate — Literature review and service landscape

A focused review of (1) the academic literature on cost transmission and pass-through that grounds what the BVAR is modelling, (2) the commercial landscape of products operators already buy adjacent to this space, and (3) the concrete service patterns Truflation can attach to its data to move from "here's an index" to "here's an actionable decision-support product."

The framing is the user's: *"we can't just show an index and tell them to do what they want there has to be a service attached to them."* The literature and the commercial landscape both confirm the same thing — operators don't pay for indices, they pay for **briefings, alerts, scenarios, and recommendations** built on top of indices.

---

## 1. Academic anchors — what pass-through and transmission literature actually says

The intellectual foundation for the BVAR is the **pass-through / pricing-to-market** literature. Two things matter for product design:

1. Pass-through is **incomplete and heterogeneous.** A 10% input shock does not flow through to a 10% output cost change.
2. The pass-through coefficient is **structurally explained** by market power and product differentiation, not by macro forecasting accuracy.

### Empirical magnitudes

| Result | Source |
|---|---|
| US import-price pass-through is roughly **50% on average**, varying widely by industry | Goldberg & Knetter 1997 |
| Recent decades show secular decline in aggregate pass-through; importers absorb more of FX shocks in margin | Federal Reserve IFDP 833 (Marazzi et al. 2005) |
| Pass-through is endogenous to importing economy and import structure | ECB WP 951 (Bussière et al. 2008) |
| Market structure (firm market share, product differentiation) dominates the explanation of incomplete pass-through | Atkeson & Burstein 2008, AER |
| Decomposition of incomplete pass-through into non-traded costs and markup adjustment | Hellerstein 2005, FRBNY |
| Retail cost pass-through varies systematically with market structure | Hong & Li 2014, AEA |

### Implication for our product

The right operator-facing answer is **not "diesel will be X next month"** — that's a forecast problem the literature itself says is unreliable. The right answer is **"here is the pass-through coefficient from diesel to your landed cost, given your market structure, and here is the distribution of landed-cost outcomes under shock scenarios."**

That maps directly to what the existing `bvar_minnesota.py` produces: **IRFs** (impulse-response functions = exactly the structural pass-through coefficients), **FEVD** (which input shocks explain what share of the operator's cost variance), and **`shock_scenario`/`conditional_forecast`** (the scenario console). The model layer is already aligned with what the literature says is the right answer.

---

## 2. Commercial landscape — what operators already buy

Five categories, with the dominant players and what they actually sell.

### A. FX hedging platforms (transactional + advisory)

| Player | What they sell | Pricing model | Target |
|---|---|---|---|
| **Convera** (ex-Western Union Business Solutions) | Forward contracts, NDFs, FX options, FX swaps + advisory | Tailored per customer, fee on transaction | Medium and large |
| **Kantox** | Dynamic Hedging — automated exposure capture + hedge execution | Subscription + transaction fees | Mid-market enterprise |
| **Airwallex** | Multi-currency accounts, FX, AI agents for autonomous finance ops | Interchange++ or blended | SMB to enterprise |
| **Cushion** | Newer SMB-focused FX risk product | Subscription | SMB |

**Pattern:** every one of them bundles **data + advisory + execution rails**. They make money on transactions, not on the data itself. The advisory is the customer-acquisition and retention wedge.

### B. Commodity hedging advisory (white-glove + software)

| Player | What they sell |
|---|---|
| **StoneX** | 100-year-old commodity advisory; identify total commodity exposure, develop hedge strategies, execute |
| **Mobius Risk Group** | Energy + ag risk consulting |
| **AEGIS Hedging** | Energy producer specialty (oil & gas E&P) |
| **Chatham Financial** | Cross-asset (rates + FX + commodity) risk advisory, hedge accounting |
| **Breakthrough Fuel** | Logistics-fleet fuel hedging specialist |
| **Derivative Path** | Platform combining market data + hedging workflows + advisory |
| **Oahu Capital** | CTA-licensed energy/ag/FX hedge program |
| **World Bank Treasury** | Commodity Price Risk Management Advisory (for sovereigns + DFIs) |

**Pattern:** white-glove human consulting layered on top of platform. Customer pays for **expert interpretation** — not raw indices.

### C. Cost intelligence / supplier risk

| Player | What they sell |
|---|---|
| **Sphera Supplier 360** | AI supplier view linking risk to business impact; ESG + carbon + regulatory tracking |
| **Resilinc EventWatch** | 24/7 incident monitoring, multi-tier supplier mapping, predictive scoring on geopolitical / financial / weather events |
| **JAGGAER** | RFx automation + what-if risk scenario modelling |
| **Spendflo** | SaaS spend management + vendor negotiations |

**Pattern:** continuous monitoring + alerts + recommended actions. Retention is high because the alerts become operational triggers in the customer's workflow.

### D. Tariff scenario tooling (the very hot 2025-26 wedge)

Tariff rates fluctuated dramatically across 2025 (2.4% Jan → 28% Apr → 16.8% Nov), driving an explosion of "tariff simulator" products:

| Player | What they sell |
|---|---|
| **Flexport Tariff Simulator** | Landed cost estimation under tariff scenarios (free tier) |
| **project44 Tariff Simulator** | Logistics-platform-integrated tariff impact (free) |
| **KPMG Tariff Modeler** | Consulting-led scenario planning |
| **Kearney PERLab Tariff Simulator** | Consulting-led |
| **Harmonize.ai Tariff Impact Simulator** | Per-product breakdown via catalog upload |
| **Suplari** | AI procurement agent simulating tariff scenarios + alternative sourcing |
| **ITC Trade Map** | Free public scenario download |
| **OEC Tariff Simulator** | Free public global-flow impact |

**Pattern:** "upload your catalog → see your tariff exposure → simulate scenarios → get sourcing recommendations." Nearly all of these were launched or expanded in 2025 in response to policy volatility.

### E. Logistics / supply chain visibility

- **project44**, **FourKites**: real-time shipment visibility, recently expanded into tariff scenario tooling
- **Flexport**: freight forwarder with embedded analytics + tariff sim

**Pattern:** start with operational data (shipments in motion) and layer pricing/cost intelligence on top.

---

## 3. Where Truflation's data is genuinely differentiated

What Truflation **has** that the players above largely don't:

| Asset | Strategic value |
|---|---|
| **Daily updates** vs monthly BLS/BEA with lag | Operators get cost-direction signal weeks earlier than from official statistics. This is the data lead the rest of the doc 08 stack already articulates. |
| **101 component streams** (80 CPI + 21 PCE) | Vertical-specific basket construction is possible at fine grain; competitors operate at aggregate or single-commodity level. |
| **Independent / non-bank-affiliated** | Buyer doesn't have to share confidential exposure data with a counterparty bank that might trade against them. |
| **Already in the macro indicator conversation** | DeFi oracle, public chart, journalist citations — the brand is "the real-time inflation source," not "another fintech." |

What Truflation **does not have** (and where it would compete poorly trying):

| Gap | Don't compete on this |
|---|---|
| Hedge execution rails | Convera/Kantox/Airwallex own this. Better to **partner**. |
| Supplier-level granular visibility | Resilinc/Sphera own this. |
| HS-code-level tariff classification | Flexport/Harmonize own this. |
| White-glove human consulting | StoneX/KPMG own this. |

**Implication.** Truflation's competitive position is the **decision-support layer in the middle** — better than free public indices (OEC, ITC), more accessible than enterprise consulting (KPMG, StoneX), data-richer than the bank-affiliated hedging platforms (because Truflation is independent), and high-frequency in a way nothing else in the operator-facing space currently is.

---

## 4. Service patterns we can attach

Five service patterns, ranked by attachment depth (low effort → high). Each maps to literature evidence that operators value it, and each plays to existing Truflation+BVAR strengths rather than gaps.

### Pattern 1 — Cost-of-doing-business **briefing** (low effort)

A periodic (weekly or monthly) digest tailored to the operator's specific cost basket. *"This week, EUR/USD moved −1.3%, diesel was flat, container freight rose 4%. For your basket (60% vehicle cost / 12% freight / 6% diesel), expected landed-cost change is +0.5% over the next 30 days, with 80% band [+0.1%, +0.9%]."*

- **Why it's sticky:** weekly cadence creates muscle memory. The operator opens it, sees their numbers, builds the habit. Same retention mechanic as Stratechery / The Information for tech executives, applied to operator costs.
- **What BVAR provides:** IRF + FEVD compute the basket-weighted expected change; conditional forecast gives the band.
- **Effort to build:** ~1 week of templating + a Jinja PDF/HTML render.

### Pattern 2 — **Threshold alerts** on operator-relevant components (low effort)

*"Diesel up 5% WoW," "EUR/USD broke 1.10," "Container freight to Asia-US +12% over 30d."* The alert thresholds are weighted by the operator's cost-share so the noise filter is calibrated to **what matters for that operator**, not generic noise.

- **Why it's sticky:** Azure Cost Management and AWS Cost Anomaly Detection both report high stickiness with this exact pattern. Once an operator's ops team starts using the alerts as workflow triggers, switching cost is operational.
- **What BVAR provides:** the cost-share weights + the per-input volatility distribution to calibrate thresholds (don't fire alerts on noise inside the normal range).
- **Effort to build:** ~1 week (notification service + per-operator threshold config + email/Slack/webhook delivery).

### Pattern 3 — Interactive **scenario console** (medium effort)

Operator types in hypothetical shocks ("EUR/USD to 1.05, diesel +20%, duty +10%") and the BVAR propagates and returns the joint distribution of landed cost. This is the **KPMG Tariff Modeler + Flexport Tariff Simulator equivalent**, but powered by Truflation's daily data and transmission VAR.

- **Why it's sticky:** the tariff-simulator boom of 2025 is direct evidence that operators will pay for this UI. project44, Flexport, Kearney, Harmonize all launched simulators in 2025.
- **What BVAR provides:** `shock_scenario` and `conditional_forecast` already exist in `bvar_minnesota.py` (Phase 3.1d). They just need a UI wrapper.
- **Effort to build:** ~2-3 weeks for a usable web UI; ~1 week for a CLI/API version. The compute is instant; the work is presentation.

### Pattern 4 — **Hedge sizing recommendation** engine (medium effort, partnership)

Use BVAR exposure quantification to recommend hedge ratios per input. *"Given your $30M annual fuel exposure and the current 30-day cost-shock distribution, the optimal-Sharpe hedge ratio is 38% (range 28-48%). At today's DBO ETF price, that's 12,400 contracts."*

- **Why it's sticky:** Hedge sizing is the actual operator-facing decision. Chatham, Mobius, Breakthrough all sell exactly this. Truflation differentiates by having the daily transmission model that those advisors don't.
- **Honest caveat:** the Phase 3.1e fuel hedge backtest showed simple rolling OLS was tied with the BVAR on hedge sizing. So this should be sold as "data-driven hedge sizing recommendations" not "BVAR beats every alternative." Combine with a Convera/Kantox **execution partnership** so Truflation doesn't take FX or commodity transaction risk.
- **Effort to build:** ~2 weeks for the recommendation logic; the partnership / execution rails are a separate business question.

### Pattern 5 — **Pass-through advisor** for outbound pricing (higher effort, highest value)

Given input cost moves and operator's estimated market-structure parameters (market share, product differentiation; estimable via Atkeson & Burstein 2008 framework), recommend the **output price adjustment** that defends margin. *"Your input cost basket moved +1.8%. Given your estimated market-share-derived pass-through coefficient of 0.62, recommended menu/list-price increase to defend margin is +1.1%."*

- **Why it's sticky:** this is the **only** service in the landscape that closes the loop from input cost intelligence to output pricing decision. Even StoneX/KPMG mostly stop at input-side hedging.
- **What's required:** the academic Atkeson-Burstein-Hellerstein pass-through estimation, which Truflation can do with public market-structure data. Per-operator calibration is more invasive (need their pricing history).
- **Effort to build:** ~4-6 weeks. This is the moonshot service — most differentiated, hardest to execute, biggest moat.

---

## 5. Recommended product architecture

**Three tiers, mirroring the SaaS DaaS pricing benchmarks above** ($18-32k SMB / ~$65k mid / $100-250k enterprise, from McKinsey + Cognism research):

| Tier | Price (suggested) | Includes |
|---|---|---|
| **Operate Standard** | $200-500/month | Pattern 1 (briefing) + Pattern 2 (alerts) on standard vertical baskets |
| **Operate Pro** | $1,500-2,500/month | + Pattern 3 (scenario console) + custom basket + Pattern 4 (hedge sizing rec) |
| **Operate Enterprise** | $5,000-10,000/month | + API access + ERP integration + Pattern 5 (pass-through advisor) + 4 advisory hours/mo |

For comparison: Convera and Kantox both bundle data+advisory+execution but make their money on transaction fees; Truflation's tier here is pure SaaS, no transaction-rev. That's a clean positioning.

**Bundling rationale:** Patterns 1 and 2 are easy to ship and create the daily-touch habit. Pattern 3 is the wow-factor that justifies Pro pricing. Pattern 5 is the moat at Enterprise that nothing else in the landscape offers.

---

## 6. What to build first (concrete, in order)

Given everything above:

1. **Pattern 1 + 2 packaged for the two existing clients.** Take the auto-importer and textile-importer cost structures already in `docs/cost_structures.md`. Build a weekly briefing template that consumes the BVAR's IRF + FEVD + the latest week's component moves, plus a threshold-alert config that fires on operator-weighted MoM moves. Two weeks of work, ships a real product.
2. **Pattern 3 — scenario console as a Streamlit/Gradio app.** Wires `shock_scenario` and `conditional_forecast` into a single-page UI. Operator slides FX, diesel, freight, duty shocks; sees landed cost distribution update live. Three weeks. Internal demo first, then external.
3. **Pattern 5 pilot — pass-through advisor.** Estimate Atkeson-Burstein pass-through coefficient from public industry data for each client's industry. Pair with their input-cost forecast distribution from Pattern 3 to produce an output-price-defense recommendation. Six weeks.

Patterns 1-3 are buildable in this repo on top of the existing BVAR. Pattern 5 needs additional public-data ingest (market share / margin data) but no new modeling architecture.

---

## 7. Honest framing for go-to-market

When this product is pitched, the framing should be:

> "Truflation runs the daily transmission model on your specific cost basket. You get a weekly briefing, threshold alerts on what matters to you, an instant scenario console for any what-if, and a quarterly review of your input exposures and recommended actions. We don't take a cut of any hedge you execute — partner with your existing FX or commodity broker. We're the **decision-support layer** between your operational data and your treasury/procurement decisions."

That positioning is defensibly distinct from every player surveyed above, plays to Truflation's actual data advantage, and is honest about what the BVAR does well (transmission, scenario) and what it does not (point forecasting of FX/diesel/freight — which we explicitly do not promise).

---

## 8. References

### Academic — cost pass-through and transmission

- Goldberg, P. K., & Knetter, M. M. (1997). *Goods Prices and Exchange Rates: What Have We Learned?* Journal of Economic Literature, 35(3), 1243-1272.
- Goldberg, L. S., & Tille, C. (2009). *Macroeconomic Interdependence and the International Role of the Dollar.* Journal of Monetary Economics.
- Goldberg, P. K., & Hellerstein, R. (2008). *A Structural Approach to Explaining Incomplete Exchange-Rate Pass-Through and Pricing-to-Market.* American Economic Review, 98(2), 423-429. https://www.aeaweb.org/articles?id=10.1257/aer.98.2.423
- Hellerstein, R. (2005). *A Decomposition of the Sources of Incomplete Cross-Border Transmission.* Federal Reserve Bank of New York Staff Reports. https://www.newyorkfed.org/medialibrary/media/research/economists/goldberg/passthrough093004.pdf
- Atkeson, A., & Burstein, A. (2008). *Pricing-to-Market, Trade Costs, and International Relative Prices.* American Economic Review.
- Nakamura, E., & Zerom, D. (2009). *Accounting for Incomplete Pass-Through.* NBER Working Paper 15255. https://www.nber.org/system/files/working_papers/w15255/w15255.pdf
- Hong, G. H., & Li, N. (2014). *Market Structure and Cost Pass-Through in Retail.* AEA Conference Paper. https://www.aeaweb.org/conference/2014/retrieve.php?pdfid=147
- Marazzi, M. et al. (2005). *Exchange Rate Pass-Through to U.S. Import Prices: Some New Evidence.* Federal Reserve IFDP 833. https://www.federalreserve.gov/pubs/ifdp/2005/833/ifdp833.htm
- Bussière, M. et al. (2008). *Exchange Rate Pass-Through in the Global Economy.* ECB Working Paper 951. https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp951.pdf

### Bayesian VAR — methodological foundation already cited in the codebase

- Crump, R. K. et al. *A Large Bayesian VAR of the United States Economy.* Federal Reserve Bank of New York Staff Reports SR 976. https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr976.pdf

### Commercial — FX and commodity hedging services

- Convera: https://convera.com/products/forward-contracts/
- Kantox profile: https://pitchbook.com/profiles/company/55278-19
- Airwallex on managing FX risk: https://www.airwallex.com/us/blog/managing-foreign-exchange-risk-volatility
- Chatham Financial: https://www.chathamfinancial.com/insights/7-ways-to-maximize-fx-and-commodity-hedging-impact-while-minimizing-costs
- StoneX: https://www.stonex.com/en/risk-management/commodities/consultancy/
- Mobius Risk Group: https://www.mobiusriskgroup.com/seo-resources/introduction-to-commodity-hedging-advisory-a-comprehensive-guide
- Breakthrough Fuel: https://www.breakthroughfuel.com/blog/fuel-hedging-strategies-for-market-stability/
- Derivative Path: https://derivativepath.com/platform/commodity-hedging/
- World Bank Treasury Commodity Risk Advisory: https://treasury.worldbank.org/en/about/unit/treasury/client-services/commodity-price-risk-management-advisory

### Commercial — supplier risk + tariff simulation

- Sphera Supplier 360: https://sphera.com/solutions/supply-chain-risk-management/supplier-intelligence-solution/
- Resilinc EventWatch: comparison table at https://slashdot.org/software/comparison/Resilinc-vs-Sphera-Supply-Chain-Risk-Management/
- KPMG Tariff Modeler: https://kpmg.com/us/en/capabilities-services/tax-services/international-tax-trade-and-transfer-pricing/trade-customs/kpmg-tariff-modeler.html
- Flexport Tariff Simulator: announced via https://www.freightwaves.com/news/firms-launch-tools-to-help-shippers-measure-tariff-costs
- project44 Tariff Simulator: https://www.project44.com/tariff-simulator/
- Kearney PERLab: https://www.kearney.com/web/product-excellence-and-renewal-lab/what-we-do/tariff-simulator
- Harmonize.ai Tariff Impact Simulator: https://harmonize-trade.com/blog/tariff-impact-simulator
- Suplari: https://suplari.com/blog/tariff-management-software-best-tools-to-plan-and-respond-to-tariffs-in-2025
- ITC Trade Map: https://www.intracen.org/resources/data-and-analysis/trade-statistics
- OEC: https://oec.world/en

### Pricing benchmarks for data-as-a-service / B2B advisory

- McKinsey on B2B pricing in the AI era: https://www.mckinsey.com/capabilities/growth-marketing-and-sales/our-insights/b2b-pricing-navigating-the-next-phase-of-the-ai-revolution
- McKinsey on B2B customer stickiness: https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/winning-b2b-customers-in-technology-and-telecommunications
- SaaS pricing forecast 2026: https://medium.com/@aymane.bt/the-future-of-saas-pricing-in-2026-an-expert-guide-for-founders-and-leaders-a8d996892876
- Cognism on DaaS pricing models: https://www.cognism.com/data-as-a-service
