# 08 — User-facing products

The five product families built on the foundational Truflation model
stack. Each section: what it is, how it surfaces, who buys it, how
it's priced, and how it relates to the others.

**Important framing.** The end product across all five surfaces is
**Truflation indices** (US CPI, US PCE, and component breakdowns),
not BLS CPI or BEA PCE. Truflation is the daily real-time inflation
signal; everything we ship — nowcasts, forecasts, regimes, vertical
cost projections — is computed against, anchored to, and reported in
Truflation values. BLS / BEA / FRED are training-time covariates and
external sanity checks, never the prediction target.

## Contents

- [Product taxonomy](#product-taxonomy)
- [Truflation Live](#1-truflation-live)
- [Truflation Nowcast](#2-truflation-nowcast)
- [Truflation Forecast](#3-truflation-forecast)
- [Truflation Regime](#4-truflation-regime)
- [Truflation Operate](#5-truflation-operate)
- [Cross-cutting elements](#cross-cutting-elements)
- [Example end-to-end user journey](#example-end-to-end-user-journey)
- [Sequencing](#sequencing)
- [What unifies all five](#what-unifies-all-five)

---

## Product taxonomy

| # | Product | Core output | Primary buyer | Latency tolerance |
|---|---|---|---|---|
| 1 | Truflation Live | Real-time index value | DeFi, traders, journalists | Real-time |
| 2 | Truflation Nowcast | Density forecast of Truflation CPI/PCE at h ∈ {1, 7, 14, 30, 90} days | Inflation-swap traders, TIPS desks, macro funds | Daily, sometimes intraday |
| 3 | Truflation Forecast | Multi-horizon (3, 6, 12, 24m) forecasts + scenario console | CFOs, treasurers, procurement, lenders, real estate | Weekly to monthly |
| 4 | Truflation Regime | Regime probabilities + trend/cycle decomposition on Truflation | Macro hedge funds, quant funds, FX macro | Daily |
| 5 | Truflation Operate | Industry-vertical cost forecasts + scenarios | Restaurant / logistics / retail / healthcare operators | Weekly |

All five products consume the same foundational model stack from
[`00-thales-overview.md`](./00-thales-overview.md). The model layer
is unchanged across products; what changes is latency, granularity,
abstraction, and packaging.

---

## 1. Truflation Live

The existing real-time index. Context for the rest. The data
foundation that everything else builds on.

**Output:** Daily Truflation index value, category breakdowns,
component drill-downs.

**Surfaces:**

- Web dashboard (current truflation.com)
- API endpoint (Feed API + Truf Network on-chain streams)
- On-chain oracle for DeFi

**Buyer:** DeFi protocols, traders watching real-time inflation,
journalists, analysts.

**Pricing:** Freemium for raw values, paid for granular component
access and API rate limits.

---

## 2. Truflation Nowcast

The flagship new product. Institutional-finance offering.

### What it is

A daily-updating density forecast of **Truflation US CPI YoY (and
PCE)** at multiple horizons — h ∈ {1, 7, 14, 30, 90} days — with
calibrated bands and component attribution. Built on the bottom-up
Truflation-only stack: per-component AR(1) composition for short
horizons (h ≤ 30d) and the Almosova LSTM for longer horizons (h =
90d). Anchor-corrected to today's published Truflation YoY so the
h=1 forecast lands exactly on the live print.

The validated frontier numbers (102 walk-forward origins, 2018-2026):

| Horizon | RMSE (pp) | 80% coverage | vs persistence |
|---:|---:|---:|---:|
| 1 d | 0.090 | 89.2% | tied |
| 7 d | 0.183 | 74.5% | tied |
| 14 d | 0.292 | 75.3% | +4.7% |
| 30 d | 0.467 | 66.3% | +21.6% |
| 90 d | 1.384 | 60.6% | +4.3% |

These numbers are what the dashboard, API, and email briefing surface
to the user.

### Surfaces

**Web dashboard.** Single page at `nowcast.truflation.com`. Top:
Truflation US CPI YoY at h=30d as a big number with credible
interval. Example:

> Truflation US CPI YoY in 30 days: **2.18%**
> 80% CI: 1.69%–2.67% · 95% CI: 1.45%–2.91%
> Origin: today, 2026-05-01 · Latest published Truflation YoY: 1.72%
> Daily change: +0.04 pp

Below: cards for each horizon (1d, 7d, 14d, 30d, 90d) with point and
density. Then a **fan chart** panel — multi-horizon forecast
trajectory with expanding bands, the entire {1, 7, 14, 30, 90}-day
arc on one chart. Then **component attribution** — which of the 12
top-level Truflation components is driving today's change ("Shelter
+0.02 pp, Gasoline +0.03 pp, Food at home −0.01 pp"). Then a
**comparator panel**: persistence baseline + Phase 1 (bottom-up) +
Phase 3 (LSTM) shown side-by-side, plus the actual Truflation print
when it lands at each target date so users can see the model's track
record live.

**API.** REST and WebSocket endpoints. JSON schema returns:

- Point estimate per horizon
- Full quantile array (10/25/50/75/90/97.5)
- Predictive samples (200 per horizon, for clients running their own
  scoring)
- Component contributions
- Model version (which phase the point came from at each horizon)
- Anchor offset (the constant correction applied at origin)
- Timestamp

WebSocket push channel for real-time updates as input streams update
on Truf Network.

**Daily email / Slack briefing.** Example:

> Truflation US CPI YoY today: 1.72%. 30-day forecast: 2.18% ± 0.49 pp
> (80%). 90-day forecast: 1.62% ± 0.97 pp (80%). Largest movers
> today: gasoline (+5 bps from WTI), shelter (−2 bps from rent
> moderation). Persistence baseline at 30d: 1.72%. Our edge over
> persistence: +21.6% RMSE on rolling 102-origin walk-forward.

**Bloomberg / Refinitiv terminal feed.** Eventually. Premium
institutional distribution.

### Buyer

- Inflation-swap traders (1Y, 5Y, 10Y inflation swaps)
- TIPS / breakeven traders
- Macro hedge funds positioning around inflation regime turns
- Rates desks at sell-side banks
- Sovereign macro and central-bank watchers
- Financial journalists

### Pricing

| Tier | Price | What's included |
|---|---|---|
| Free public dashboard | $0 | Today's nowcast, ~30-day history, single chart. Top-of-funnel and credibility. |
| Pro API | $1–5K/month | Full density, all horizons, full history, component attribution, WebSocket |
| Institutional | $25–100K/year | White-glove SLA, dedicated support, custom analyst access, redistribution rights |

### Why it beats the alternatives

The closest comparable thing in the market is the [Cleveland Fed
Inflation Nowcasting Model](https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting),
which is **point-only, monthly cadence, lagged updates, and
predicts BLS CPI**. Different target (BLS vs Truflation), different
cadence (monthly vs daily), different output type (point vs density).
There is no incumbent forecasting Truflation specifically — the
nowcast is the first density-aware, multi-horizon product in that
space, and it inherits Truflation's daily real-time data lead by
construction.

For users who care about BLS CPI specifically (TIPS, inflation
swaps), the Truflation nowcast is *complementary* — Truflation has
historically led BLS by 41-75 days at the optimal lead shift, so a
forecast of where Truflation will be in 30-90 days is a useful
forward indicator for where BLS will print.

---

## 3. Truflation Forecast

Same model stack as Nowcast, longer horizons, packaged for planning.

### What it is

Multi-horizon Truflation US CPI YoY forecasts at 3, 6, 12, 24 months
with scenario conditionals. Same density properties as the nowcast,
just at longer horizons. Conditioning on user-specified paths for
exogenous variables.

Long-horizon forecasts use the UC + SV + MS trend filter (Stock-
Watson 2007 architecture) on monthly Truflation YoY, ensembled with
the Almosova LSTM where appropriate. Density bands include
regime-switching variance.

### Surfaces

**Scenario console.** The headline UI. User picks horizon and a path
for exogenous variables: Fed funds, mortgage rates, WTI, FX, tariff
regime. Model produces conditional Truflation forecast distribution.
Side-by-side scenario comparison cards: "Baseline vs Aggressive Cuts
vs No Cuts." Fan chart per scenario. Consultative — used by humans
making plans.

**Excel / Google Sheets connector.** For corporate users who live in
spreadsheets. Pull forecast and density into a workbook, refresh
daily. CFOs and procurement teams will not learn an API; they will
use a spreadsheet refresh button.

**Quarterly outlook PDF.** Like Goldman's "Top of Mind" or BlackRock
Investment Institute outlooks. Branded, designed, with headline
Truflation forecast, scenario fans, regime context, category-by-
category commentary. Drives lead-gen, gets shared, builds reputation
around the Truflation data.

**Webhook + alerts.** "Tell me when the 6-month Truflation CPI
forecast crosses 3% / falls below 2.5% / when regime probability of
persistent inflation crosses 70%."

### Buyer

- CFOs (mid-market and enterprise) doing annual planning
- Treasurers managing cash and short-duration positioning
- Procurement teams negotiating multi-quarter contracts
- Lenders pricing variable-rate products
- Real estate operators projecting rent growth
- Asset managers with longer-horizon mandates than CPI traders

### Pricing

| Tier | Price | What's included |
|---|---|---|
| Self-serve SaaS | $500–2K/month per seat | Scenario console + spreadsheet connector |
| Enterprise | $25–100K/year | Multiple seats, custom scenarios, account team |
| Reports-only | $5–15K/year | Quarterly outlook + monthly briefing, no live tool |

### Why it differs from Nowcast in user behavior

Nowcast users check daily, often hourly. Forecast users check weekly
or monthly. Different latency tolerance, different UI density,
different buying motion. Selling Forecast to an inflation-swap
trader is wrong; selling Nowcast to a CFO is wrong. Same machinery,
different products.

---

## 4. Truflation Regime

Standalone product from the UC + SV + MS regime layer. Premium,
specialized, lower volume but higher unit price.

### What it is

A daily probability feed for Truflation regime states (persistent vs
transitory, demand-driven vs supply-driven, stable vs shock), plus a
daily latent trend Truflation series stripped of noise, plus a
sticky/flexible decomposition. Powered by the regime layer of the
foundational stack, fitted on monthly Truflation YoY.

The Phase 2 walk-forward established that the smoothed P(high-vol)
series cleanly identifies the 2008 GFC, 2014 oil crash, 2020 COVID,
and 2021-2023 surge windows on Truflation data — usable as a
standalone signal independent of any forecasting performance.

### Surfaces

**Regime dashboard.** Signature view: stacked area chart showing
regime probabilities over time, current regime highlighted. Sticky-
trend and flex-trend lines overlaid. Recent regime transitions
called out:

> Currently 78% probability of **persistent regime**, transitioned 23
> days ago from transitory.
> Sticky trend: 3.2% · Flex trend: 1.8% · Spread: 1.4σ above 5-year mean.
> Driving categories: Health (+), Education (+), Communications (−)

**"Inflation VIX" headline number.** A single index value, 0–100,
capturing forward Truflation uncertainty conditional on regime. The
marketing artifact. Easy to understand, easy to chart, easy to
subscribe to. Can become the reference number macro traders cite.

**Regime alert API.** "Notify me when regime probability of persistent
inflation crosses 70% / when sticky trend exceeds 3% / when sticky-
flex spread widens beyond 1.5σ." Webhooks into trading systems.

**Backtested signal library.** For sophisticated quant clients:
downloadable history of every regime probability for every date
back to 2010, with realized regime classifications retrospectively
assigned. Lets clients backtest their own strategies on the signal.

### Buyer

- Macro hedge funds (this is their bread and butter)
- TIPS / inflation-derivative desks
- FX macro funds (regime affects DXY, USD/EM)
- CTA / systematic funds incorporating regime as a feature
- Sophisticated family offices and prop shops

### Pricing

| Tier | Price | What's included |
|---|---|---|
| Premium add-on | $25–50K/year on top of Nowcast Pro | Regime dashboard + alerts |
| Standalone signal | $50–200K/year | Full signal feed for quant funds, backtesting library |
| Bespoke | $500K+ | Custom model parameterization for top-tier funds |

### Why it's a separate product, not a feature

Different buyer (quant macro), different sales motion (signal/alpha
pitch, not infrastructure), different evaluation criteria (Sharpe
ratio of signal-driven strategy, not nowcast accuracy). Bundling
dilutes both.

---

## 5. Truflation Operate

The Main Street wedge. The thing nobody else has and the largest
TAM by far.

### What it is

Industry-specific cost forecasting plus scenario tools, powered by
Bayesian transmission VARs that map Truflation component movements
to industry-vertical cost lines. Each vertical is its own product
packaging:

- **Operate / Restaurants** — food + labor + occupancy forecasting
- **Operate / Logistics** — fuel + labor + maintenance forecasting + hedging
- **Operate / Retail** — tradables COGS + rent + utilities forecasting
- **Operate / Healthcare** — labor + supplies + insurance forecasting
- **Operate / Real Estate** — rate-sensitive cap rate and rent growth
- (More verticals as model coverage expands)

The driving signal is **Truflation per-component daily streams**
(101 of them — 80 CPI sub-components + 21 PCE), not BLS or BEA
aggregates. This is the data lead — Truflation operators get
forward cost projections days to weeks before the same information
arrives via official statistics.

### Surfaces (general pattern, customized per vertical)

**Vertical landing page.** `truflation.com/operate/restaurants`,
`/logistics`, etc. Each tailored to the buyer's mental model.
Restaurant operators don't think in CPI components; they think in
food cost, labor cost, occupancy cost. The UI mirrors this.

**Cost-structure onboarding.** User enters their cost structure once:
"Our COGS is 32% food, 28% labor, 8% rent, 4% utilities, 3% credit
card fees, 25% other." Or pick from common templates ("QSR average,"
"casual dining average," "fine dining average"). System uses this to
weight transmission VAR outputs over Truflation components.

**Forward cost dashboard.** The headline view, vertical-specific.
Restaurant example:

> Your blended COGS inflation forecast over the next 6 months: **3.2% ± 0.8%**
> Driven by: labor (+4.1%), dairy (+5.3%), partially offset by produce (−1.1%)
> If menu prices stay flat: margin compression of ~85 bps over 6 months
> Recommended action: review menu pricing in May to maintain target margin

Cards per cost line. Margin projection. Specific recommendations.

**Scenario / what-if console.** "If WTI rises to $95, what happens to
my logistics cost? If wages outpace expectation by 1%, what happens
to my margin? If rent inflation moderates, what does my breakeven
look like?" Same Bayesian VAR machinery, packaged in business
language.

**Pricing recommendation engine (advanced tier).** "Based on your
cost forecast and historical price-elasticity in your category, we
recommend menu price increases of 2.5% in May to maintain target
margin." Opinionated. Crosses into prescriptive territory. Exactly
what operators want.

**Hedging recommendations (logistics-specific).** "Given the 6-month
diesel forecast, hedging 60% of expected consumption at current
futures levels reduces your worst-case COGS by 4.2%." Calls out
specific futures contracts, sizes, expirations. Operators rarely
have this analysis in-house.

**Quarterly business review report.** PDF or email, branded,
summarizing vertical's cost outlook, regime context, recommended
actions. The artifact a CFO presents to their CEO or board.

### Buyer

- Restaurant operators (chains, franchise groups, independent
  multi-units)
- Trucking and logistics CFOs
- Retail CFOs and merchandising teams
- Hospital and healthcare system CFOs
- Real estate operators and REITs
- Manufacturing CFOs (input cost forecasting)
- Insurance underwriters (claims-cost forecasting)

### Pricing

| Tier | Price | What's included |
|---|---|---|
| Self-serve SaaS (SMB) | $200–800/month per location or business | Cost dashboard, scenario console |
| Mid-market | $1K–5K/month | Multi-location, integrations, account support |
| Enterprise | $50K–200K/year | Custom verticals, dedicated analyst, ERP API integration |

### Why it's the biggest opportunity

Financial markets are big-revenue-per-account, low-account-count.
Operate is the inverse — small revenue per account, enormous account
count. Restaurants alone are 700K+ in the US. Even at low
single-digit penetration with $300/month average, the math is large.
And it's a market with no real incumbent — IHS Markit, Kalepa, etc.
don't operate at this granularity or daily resolution, and none of
them have a proprietary daily price source equivalent to Truflation.

### UI principle for Operate

Never expose CPI components, archetype names, or model jargon.
Translate everything into the buyer's vocabulary:

- "Food at home inflation" → "your grocery costs"
- "PCE shelter" → "your occupancy costs"
- "Bayesian VAR impulse response" → "if diesel jumps 10%, here's
  what happens to your shipping costs over the next 12 months"
- "CRPS" → "forecast accuracy score"
- "Regime probability" → "likelihood of cost pressure persisting"

The model machinery is identical to what serves the institutional
finance products. The wrapper is completely different.

---

## Cross-cutting elements

These appear across multiple products. Specify once, reuse.

### Universal API
All products consumable programmatically. JSON schema consistent
across Nowcast, Forecast, Regime, Operate. SDKs in Python and
JavaScript. Webhooks for alerts. Mandatory for institutional
credibility.

### Embeddable widgets
Scriptable embed for journalists, blogs, advisor sites. "Add the
live Truflation inflation widget to your site." Small, branded,
free, viral.

### Mobile
iOS first eventually. Nowcast and Regime translate cleanly to mobile
cards; Operate less so. Push notifications for alerts.

### Slack and Microsoft Teams bot
`/inflation forecast 6m` or `/inflation regime` commands.
Particularly powerful for Operate — embed forecast queries in the
operations channel where decisions actually get made.

### ChatGPT / Claude plugin or MCP server
Once stable. Lets users ask "what's the 6-month inflation forecast
for restaurant labor?" in their AI assistant of choice and get
authoritative Truflation-backed answers. 2026–2027 distribution
play.

### On-chain oracle
For DeFi. The nowcast as a smart contract data feed. Builds on
Truflation's existing oracle work but extends to forward-looking
data. Niche but high-margin and fits the existing infrastructure.

### Public live track record
Dated, unrevised, publicly visible URL with rolling evaluation
results vs persistence baseline + the model frontier. Cleveland Fed
does this for BLS CPI; we do it for Truflation CPI. Compounds
credibility over 18+ months.

### Research and credibility layer
- Working papers (target IJF, JBES, Journal of Forecasting)
- Conference presentations (NBER, SED, central bank conferences)
- Quarterly evaluation reports (public)
- Annual research review

This isn't a product per se, but it's what makes the products
credible to institutional buyers.

---

## Example end-to-end user journey

Persona: regional restaurant chain CFO, 40 locations.

1. **Discovery.** She sees an article quoting Truflation CPI on
   release day. Lands on `truflation.com/operate/restaurants`.

2. **Onboarding (3 minutes).** Enters her cost structure or picks
   "casual dining mid-market" template. Dashboard populates.

3. **First value (free tier).** Sees a 6-month forward cost outlook:
   3.4% blended COGS inflation expected, driven by labor and dairy.
   Headline number free, scenario console gated.

4. **Conversion.** Pays $1.5K/month for scenario console + pricing
   recommendations.

5. **Habit formation.** Uses it weekly to inform menu pricing
   decisions, quarterly to brief her board. Quarterly board deck
   pulls a Truflation report directly.

6. **Network expansion.** Forwards email briefing to her ops VP and
   finance team. Slack alerts go to CFO channel when regime
   transitions trigger.

7. **Treasury upsell.** Her treasurer separately picks up a
   Truflation Forecast subscription for cash management. Now two
   products, same account.

8. **Enterprise upgrade.** Parent company subscribes to Nowcast Pro
   and Regime to inform working capital decisions across all
   subsidiaries.

Net: one customer relationship, $20–50K/year in MRR across product
tiers, high retention because all four products feed her job.

Multiply by industries × geographies × scale.

---

## Sequencing

Don't build all five at once. Right sequence:

| Phase | Product | Months | Why this order |
|---|---|---|---|
| 1 | Truflation Nowcast | 6–9 | Highest credibility output. Drives institutional partnerships and press. First density-aware multi-horizon Truflation forecaster sets the public standard. |
| 2 | Truflation Regime | 9–12 | Same model machinery, separate packaging. Quant fund customers fund this with high unit prices. Validates the regime layer commercially. |
| 3 | Truflation Forecast | 12–15 | Corporate planning product. Lower price, higher volume than Regime. Built on already-shipped model stack. |
| 4 | Truflation Operate (vertical 1) | 15–18 | Logistics or restaurants first. Validates transmission VAR approach commercially. |
| 5 | Operate (additional verticals) | 18+ | Once vertical 1 proven, each additional vertical is months not quarters because foundation is stable. |

The first product gets you institutional credibility. The last
product gets you scale revenue. Both come from the same foundation,
which is why "foundational" is the right word for the model layer.

---

## What unifies all five

Every product is the same machinery served at different latency,
granularity, and abstraction:

| Product | Latency | Granularity | Abstraction |
|---|---|---|---|
| Live | Real-time | Index level | Raw measurement |
| Nowcast | Daily | Headline + components | Financial framing |
| Forecast | Weekly to monthly | Headline + scenarios | Planning framing |
| Regime | Daily | Latent regime state | Signal framing |
| Operate | Weekly | Cost-structure-mapped states | Operator framing |

Same data + model stack. Five product surfaces. One foundation,
many products. Truflation is the prediction target across all five
— that's the foundational property in action.
