# 01 — Architecture

Full model architecture for the Trufonomics foundational stack. Read `CLAUDE.md` first for the summary. This doc is the depth reference.

## Contents

- [Data hierarchy](#data-hierarchy)
- [The 12 categories and 2026 weights](#the-12-categories-and-2026-weights)
- [Five archetype models](#five-archetype-models)
- [Composition layer — CBDF](#composition-layer--cbdf)
- [Regime model — UC-SV-MS](#regime-model--uc-sv-ms)
- [Transmission layer — industry VARs](#transmission-layer--industry-vars)
- [Tech stack rationale](#tech-stack-rationale)
- [Sequencing](#sequencing)

---

## Data hierarchy

Truflation's data structure:

```
Index (US CPI)
├── Category (12)
│   ├── Subcategory
│   │   └── Component (daily price observations)
```

All components have known weights that roll up to subcategory → category → headline via weighted averaging. Weights are updated annually in February using prior-year household expenditure data. See [Truflation methodology](https://truflation.com/blog/everything-you-need-to-know-about-truflations-index-methodology) and [2026 weights announcement](https://blog.truflation.com/truflation-2026-cpi-weights/).

Key properties of the data asset:
- Daily frequency at component level
- 15M+ daily price points from 60+ providers
- Annual weight updates (more current than BLS's 24-month-old weights)
- Multi-country coverage expanding (US, UK, with [ONS CPI comparison](https://truflation.com/blog/truflation-ons-cpi-vs-cpih))
- Housing uses mortgage-rate-based owned component vs BLS's OER imputation

## The 12 categories and 2026 weights

| # | Category | 2026 weight | Archetype |
| --- | --- | --- | --- |
| 1 | Housing | 23.1% | Rate-sensitive durables |
| 2 | Transportation | 19.8% | Mixed (fuel = pass-through, vehicles = rate-sensitive) |
| 3 | Food & non-alcoholic beverages | 15.2% | Mixed (food-at-home = pass-through, food-away = discretionary) |
| 4 | Health | 8.8% | Sticky administered services |
| 5 | Housing durables & daily-use | 7.2% | Import-exposed tradables |
| 6 | Utilities | 6.0% | Commodity pass-through |
| 7 | Recreation & culture | 5.6% | Discretionary demand-cycle |
| 8 | Clothing & footwear | 3.8% | Import-exposed tradables |
| 9 | Communications | 3.2% | Sticky administered services |
| 10 | Education | 2.3% | Sticky administered services |
| 11 | Alcohol & tobacco | 1.8% | Sticky administered services |
| 12 | All Other | 2.9% | Discretionary demand-cycle |

Category-specific sub-splits (e.g. fuel portion of Transportation → commodity archetype, vehicle portion → rate-sensitive) happen at the subcategory level.

## Five archetype models

Each archetype is a state-space model matched to the generating process of that category type. One archetype model class is instantiated for each category it covers. All produce daily density forecasts.

### Archetype 1 — Commodity pass-through

**Covers:** Utilities, fuel portion of Transportation, food-at-home portion of Food.

**Model class:** Time-varying parameter VECM in state-space form with stochastic volatility (TVP-VECM-SV).

**Observation equation:**
```
log(price_t) = α + β_t · log(commodity_spot_t) + γ · inventory_state_t + ε_t
ε_t ~ N(0, exp(h_t))
```

**State equations:**
- `β_t = β_{t-1} + η_t^β`, η_t^β ~ N(0, Q_β), with β_t bounded to [0, 1]
- `h_t = ρ · h_{t-1} + η_t^h`, η_t^h ~ N(0, Q_h) — stochastic volatility
- Latent refinery/retail margin state for fuel (cointegration residual)

**Why TVP:** pass-through is 20–40% for gasoline and drifts with refinery capacity, retailer margins, tariff regime. Fixed β is a modeling error that shows up as massive miss during shocks.

**Estimation:** Kalman filter for linear components + particle filter or Gibbs sampler for SV. NumPyro for the Bayesian version.

**Covariates:** WTI spot, Henry Hub spot, Brent, refinery crack spreads, HDD/CDD from NOAA, grain futures for food.

**Forecast horizon:** days to weeks dominant. Refit weekly, filter daily.

---

### Archetype 2 — Rate-sensitive durables

**Covers:** Housing (owned + rented split), vehicle portion of Transportation, big-ticket Housing durables.

**Model class:** Hierarchical dynamic factor state-space model with separate measurement blocks for owned and rented, aggregated by tenure share.

**State equations:**
- National affordability latent factor A_t driven by mortgage rates, DTI, price-to-income
- National rate-environment factor R_t (exogenous observed: 30Y fixed, 10Y Treasury)
- Regional idiosyncratic factors for top 20–50 metros
- Separate latent "rental tightness" state from vacancy + absorption data

**Measurement (owned):**
```
Δlog(price_it) = λ_i^owned · A_t + δ_i · R_t + regional_i + ε_it^owned
```

**Measurement (rented):**
```
Δlog(rent_it) = λ_i^rented · A_t + η_i · tightness_t + regional_i + ε_it^rented
```

**State dynamics:** A_t follows VAR(1) with R_t as exogenous input; tightness_t is near-unit-root AR(1).

**Estimation:** Two-step. Doz-Giannone-Reichlin 2011 two-step (PCA for initial factor estimates, then EM on full SSM). Bayesian via Gibbs when fully tuned. Dynamax with JAX backend — state dimension blows up fast with regional panel.

**Design call:** mixed-frequency (daily rental feeds, weekly mortgage rates, monthly home price indexes) via Mariano-Murasawa cumulator.

**Covariates:** 30Y mortgage rate (FRED `MORTGAGE30US`), 10Y Treasury (`DGS10`), Zillow Rent Index, Case-Shiller, CoreLogic, Freddie Mac PMMS, vacancy rates by metro.

**Forecast horizon:** months to quarters. Flagship model.

---

### Archetype 3 — Sticky administered services

**Covers:** Health, Education, Communications, Alcohol & tobacco.

**Model class:** Unobserved components with stochastic volatility and Markov regime-switching for repricing events (UC-SV-MS).

**Observation:** `π_t = τ_t + c_t + ε_t`

**State equations:**
- τ_t (slow trend) — random walk with small innovation variance
- c_t (cycle) — `φ_{s_t} · c_{t-1} + η_t` where `s_t ∈ {stable, repricing, policy_break}`
- log σ_t² — stochastic volatility

**Why regime-switching matters:** health insurance reprices annually in Q4, education in fall, tobacco at tax changes. Without MS, the model averages over repricing and gives garbage during the window. With MS, repricing detected probabilistically.

**Estimation:** Kim-Nelson filter (Hamilton regime-switching embedded in Kalman filter). NumPyro hand-rolled cleaner than statsmodels MarkovSwitching extension.

**Covariates:** ECI, Atlanta Fed Wage Tracker for labor-cost component of services, administrative/tax calendar dummies.

**Forecast horizon:** multi-quarter. Slow signals.

---

### Archetype 4 — Import-exposed tradables

**Covers:** Clothing & footwear, most of Housing durables, imported Food.

**Model class:** Vector error-correction in state-space form with tariff regime as discrete state.

**Long-run relationship:**
```
log(price_t) = α + β · log(FX_t) + γ · log(shipping_cost_t) + δ · tariff_regime_t
```

**State equations:**
- Pass-through coefficient β_t (random walk, drifting)
- Cointegration residual — mean-reverting "pricing gap" with speed κ
- Tariff regime s_t ∈ {baseline, tariff_1, tariff_2, …} — observed step dummy

**Why this matters now:** tariff regime shifts in 2025–2026 are active. Model without explicit tariff regime treatment will spuriously attribute shocks to trend.

**Estimation:** statsmodels VECM for starting values, then Bayesian VECM with Minnesota-style priors in NumPyro. Pass-through β_t via Kalman smoother.

**Covariates:** DXY, bilateral FX rates (USDCNY, EURUSD, USDMXN, USDVND for specific supply chains), Baltic Dry Index, Shanghai Containerized Freight Index (SCFI), Drewry WCI, known tariff schedule.

**Forecast horizon:** one to two quarters.

---

### Archetype 5 — Discretionary demand-cycle

**Covers:** Recreation & culture, food-away-from-home portion of Food, All Other.

**Model class:** Harvey structural time series (BSTS-style) with exogenous covariates.

**Observation:** `y_t = μ_t + γ_t + ψ_t + β'x_t + ε_t`

Where:
- μ_t — local linear trend
- γ_t — trigonometric seasonal (**dual:** daily + annual, superposition)
- ψ_t — stochastic cycle
- x_t — covariates

**Key design call:** daily data requires BOTH daily-of-week seasonality AND annual seasonality. Single seasonal gets it wrong.

**Estimation:** statsmodels `UnobservedComponents` first pass, NumPyro for Bayesian version with priors on cycle parameters.

**Covariates:** UMich sentiment (`UMCSENT`), Conference Board consumer confidence, unemployment rate, quits rate, event calendars for concerts/travel/sports.

**Forecast horizon:** months. Easiest of the five.

---

## Composition layer — CBDF

**Reference:** [O'Keeffe & Petrova (2025), "Component-Based Dynamic Factor Nowcast Model", NY Fed Staff Report 1152](https://www.newyorkfed.org/research/staff_reports/sr1152). Also on [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5230835).

**Structure:**
```
headline_π_t = Σ_i w_i · category_π_{i,t}
```
Where each `category_π_{i,t}` is driven by its archetype model. Common factors F_t capture cross-category comovement not explained by archetype-specific dynamics. Accounting identity preserved by construction.

**Estimation:**
1. Fit each archetype model independently, extract category-level states
2. Extract common factors from archetype residuals via constrained DFM
3. Optional: re-estimate everything jointly via EM
4. Mariano-Murasawa aggregation for mixed frequencies

**Why this beats monolithic DFM:** categories have structurally different generating processes. Monolithic DFM assumes same dynamics for all series — which is what O'Keeffe-Petrova specifically improves on (15% RMSE, 20% log-score gain vs standard DFM on monthly data). With richer archetype dynamics on *daily* data, the gain should compound.

**Output:** daily density forecast of headline BLS CPI and BEA PCE, with component attribution, for horizons -15 to +12 months.

---

## Regime model — UC-SV-MS

Standalone product: regime probability estimates (persistent vs transitory, demand vs supply, stable vs shock), daily trend inflation state.

**Model class:** Unobserved components with stochastic volatility + Markov switching + sticky/flexible decomposition.

**Category-level component:** for each category, compute price-change frequency (Bils-Klenow / [Boston Fed methodology](https://www.bostonfed.org/publications/current-policy-perspectives/2025/frequency-of-price-changes-and-the-nature-of-inflation.aspx)). Classify as sticky vs flexible using median frequency as cutoff. This split is possible because we have component-level micro data that BLS protects as confidential.

**Aggregate:** sticky-sector inflation trend, flexible-sector trend, cross-sectional skewness across components.

**State vector:** national trend τ_t, sticky trend τ_t^s, flex trend τ_t^f, regime s_t ∈ {transitory, persistent, demand-driven, supply-driven, shock}, stochastic volatility h_t.

**Estimation:** Kim-Nelson filter for linear-Gaussian regime-switching SSM; particle filter for the nonlinear components. Bayesian implementation in NumPyro or CmdStanPy.

**Reference:** [BIS WP 713 (Mertens)](https://www.bis.org/publ/work713.pdf) for the UC-SV-MS methodology. [BSP working paper 2022-02](https://www.bsp.gov.ph/Media_And_Research/WPS/WPS202202.pdf) for state-space inflation nowcasting with high-frequency data.

**Products on top:**
- Daily trend inflation with credible bands (institutional subscription)
- Daily regime probability feed (hedge fund API)
- "VIX for inflation" — persistence-conditioned volatility of forward inflation
- Event-study dashboard: regime transition timing vs known episodes

---

## Transmission layer — industry VARs

For each industry vertical, a structural Bayesian VAR with Minnesota prior mapping category inflation → business P&L outcomes.

**Structure:**
```
Endogenous vector: [category_π_1, …, category_π_k, wage_π, industry_output, margin, volume]
Identification: Cholesky, category shocks ordered first (exogenous to individual businesses at short horizons)
Prior: Minnesota with tight own-lag, loose cross-variable
```

**Training data:** industry-level from BLS PPI by industry, Census monthly retail trade, Census transportation services, wage indexes by sector.

**Client plug-in:** individual business's cost structure is a weighting vector that collapses IRFs into their P&L sensitivity.

**Outputs:**
- **Impulse response functions** — "if diesel spikes 10%, your ton-mile cost rises 3.5% over 12 months"
- **Forecast error variance decomposition** — "40% of your margin risk is fuel, 30% wages, 10% rent, 20% other"
- **Conditional forecasts** — given Truflation's nowcast + forward curves, your 6-month margin path
- **Scenario analysis** — alternative Fed paths, commodity paths, tariff regimes

**Vertical priority (cost-structure magnitude × market size):**
1. Logistics / trucking (~35% fuel, 25% labor, sensitive to commodity and wage inflation)
2. Restaurants (~30% food, 30% labor, 8% rent)
3. Mid-market retail (tradables COGS + rent + utilities)
4. Healthcare operators (labor + supplies + utilities + insurance)
5. Real estate operators (rate-sensitive)

**Stack:** statsmodels VAR for point, `bvar` or custom NumPyro for Bayesian. Priors critical because industry sample sizes are modest.

---

## Tech stack rationale

| Component | Tool | Why |
| --- | --- | --- |
| Core language | Python 3.11+ | Ecosystem, scientific stack |
| Packaging | `uv` | Fast enough that iteration loop stays responsive |
| State-space models | [dynamax](https://github.com/probml/dynamax) | JAX-backed, GPU-accelerated, differentiable Kalman/particle filters, unified API for linear/switching/nonlinear SSMs |
| Bayesian inference | [NumPyro](https://num.pyro.ai/) primary, [PyMC 5](https://www.pymc.io/) secondary | JAX backend, NUTS sampler, composable |
| Reference implementations | statsmodels | VECM, UCM, MarkovSwitching — well-tested, use for baselines and sanity checks |
| Complex Bayesian SSMs | [CmdStanPy](https://mc-stan.org/cmdstanpy/) | Fallback when NumPyro's samplers struggle |
| Dataframes | polars primary, pandas for compatibility | Speed matters for vintage store queries |
| Vintage store | DuckDB | Embedded, columnar, SQL, works with Parquet files, zero ops overhead |
| GPU | JAX on CUDA | Hierarchical housing and MCMC-heavy fits benefit. Vast.ai for rental |
| Scoring | [`properscoring`](https://github.com/properscoring/properscoring) for CRPS, custom for PIT/DM/GW | Lightweight, correct |
| Dashboard | Streamlit | Fast iteration, SQL-backed, easy to host |
| Orchestration | Prefect or plain cron initially | Don't over-engineer until you need to |

---

## Sequencing

See [`03-checklist.md`](./03-checklist.md) for the actionable breakdown.

**Phase 0 (4 weeks, no component data needed):** scaffolding, vintage store, covariate ingest, synthetic DGPs, evaluation harness, three baseline nowcasts, Cleveland Fed comparator scrape.

**Phase 1 (weeks 5–16, needs component data):** five archetypes, starting with easiest (commodity + discretionary), then sticky services, then tradables, then housing (flagship).

**Phase 2 (weeks 17–22):** CBDF composition. Full headline nowcast with density. Regime model alongside.

**Phase 3 (weeks 23+):** first transmission VAR (logistics or restaurants). Multi-country replication. Additional verticals. Fed-grade paper.

---

## Methodology review — 2026-04-24 (post external code review + literature scan)

Outcome of the second-pass literature review against 2024-2026 work. The core architecture (archetype SSMs → CBDF composition → UC-SV-MS regime model → transmission VARs) is validated as current frontier. Two concrete additions and one explicit rejection are recorded here for posterity so future sessions don't re-litigate the decisions.

### Addition 1 — HRNN as an alternative archetype class

**What:** Hierarchical Recurrent Neural Networks for CPI component forecasting, per Benchimol, Kazinnik, Saadon — *"Forecasting CPI Inflation Components with Hierarchical Recurrent Neural Networks"*, International Journal of Forecasting 2022 (updated 2024); open-source implementation at https://github.com/AllonHammer/CPI_HRNN.

**Why it's worth including:** Uses the same Index → Category → Subcategory → Component hierarchy we have. Reported to "significantly outperform a vast array of well-known inflation prediction baselines" on disaggregated CPI forecasting. Different inductive bias from archetype SSMs — trades econometric interpretability for neural-net expressiveness.

**How it fits the stack:** Added as a **6th archetype class** available to any category where the structural SSM fit is weak. Its per-category output feeds the same CBDF composition layer as the other five archetypes. Does not replace any planned model. Benchmark HRNN alongside the archetype SSM in every per-category ablation; let the evidence decide which wins on which category.

**Concrete placement:** when Archetype 3 (Sticky administered services) and Archetype 5 (Discretionary demand-cycle) get built, add an HRNN variant as a parallel fit. These categories have the weakest mechanistic priors, so HRNN is most likely to outperform there.

### Rejection — zero-shot pretrained TSFMs as a flagship forecaster

**What was evaluated:** Chronos (Amazon, all sizes from tiny to large), TimesFM (Google, 8M-500M), Moirai (Salesforce), plus ten additional TSFMs surveyed in the literature.

**Evidence:** Chen & Kelly et al., *"Re(Visiting) Time Series Foundation Models in Finance"*, arXiv 2511.18578 (Nov 2025). Benchmarks TSFMs on daily excess returns across 94 countries, 1990-2023:

> "off-the-shelf pre-trained TSFMs perform weakly in zero-shot forecasting of daily excess returns, underperforming strong ensemble models such as CatBoost and LightGBM"

Specific numbers: Chronos-large **R² = −1.37%**, TimesFM-500M **R² = −2.80%**, vs CatBoost **R² = −0.10%**. The paper's own conclusion:

> "Finance-specific pre-training was essential — models trained from scratch on financial data substantially outperformed generic pretrained models."

The paper is on returns, not inflation, but the lesson transfers: generic pretrained TSFMs on macro data are unlikely to beat domain-tuned econometric methods zero-shot.

**Decision:** **Do not adopt** zero-shot Chronos / TimesFM / Moirai / etc. as the primary forecaster in Phase 0 or Phase 1 of the Thales stack. If TSFMs are revisited later, it must be via **fine-tuning on Truflation component data**, not zero-shot plug-in — and fine-tuning is treated as a separate research program, not a shortcut for the core stack.

### Validated as current SOTA (no changes — keep as-specified)

1. **CBDF composition** (O'Keeffe & Petrova 2025, NY Fed SR 1152) — still the frontier for component-based nowcasting. No 2024-2025 successor has emerged. The 15% RMSE / 20% density gain over standard DFM (their headline figure) is on GDP, but the architecture transfers to inflation with direct plug-in of the 80 Truflation component streams.

2. **UC-SV-MS regime detection** (Stock-Watson → Mertens multivariate extensions) — still the canonical approach in the 2024-2025 literature for trend/cycle/regime decomposition of inflation. Mertens' multi-series extensions handle our multi-category setting.

3. **Archetype-specific SSMs per category** — no 2024-2026 paper specifically challenges this decomposition. The generating-process matching of model class to category (commodity pass-through → TVP-VECM with SV; rate-sensitive → hierarchical DFM-SSM; etc.) is consistent with the econometric literature's handling of each category separately.

4. **Mixed-frequency factor models / MIDAS / MF-VAR** for daily-to-monthly — this is the canonical stack ECB, Cleveland Fed, and IMF papers use. CBDF is a richer variant.

### Orthogonal — potentially useful upstream, not a substitute

- **Project Spectrum (BIS 2024)** uses generative AI for **categorization** of scanner and web-scraped price data into official index categories (ECB Daily Price Dataset, billions of product-price observations). Not a forecaster. If Truflation's raw-price-ingestion pipeline ever needs help classifying new data sources, this framework is the reference. No impact on our forecasting architecture.

### Sources

- O'Keeffe & Petrova (2025). *Component-Based Dynamic Factor Nowcast Model.* NY Fed SR 1152. https://www.newyorkfed.org/research/staff_reports/sr1152
- Benchimol, Kazinnik, Saadon (2022). *Forecasting CPI Inflation Components with Hierarchical Recurrent Neural Networks.* IJF. https://www.sciencedirect.com/science/article/pii/S0169207022000607
- Chen & Kelly et al. (Nov 2025). *Re(Visiting) Time Series Foundation Models in Finance.* arXiv 2511.18578. https://arxiv.org/html/2511.18578v1
- IMF WP 2024/190. *Regime-Switching Factor Models and Nowcasting with Big Data.*
- BIS OTH 2024. *Project Spectrum.* https://www.bis.org/publ/othp109.htm
- Mertens, Nason (2020). *Inflation and professional forecast dynamics.* Quantitative Economics. BIS WP 713.
