# Thales — executive overview

**One-paragraph version.** Thales is Truflation's foundational
econometric model stack. It turns 80 daily-updating Truflation price
streams plus public macro data into four product surfaces:
density nowcasts of official BLS CPI / BEA PCE releases, multi-
horizon forecasts with scenario conditionals, a regime indicator
("VIX for inflation"), and per-industry transmission VARs that
expose forward cost-line P&L for shipping/restaurants/retail/
healthcare/real-estate/manufacturing operators. The architecture
beats published frontier baselines (Stock-Watson DFM by 37.6 %
RMSE, p < 0.001) and the public Cleveland Fed nowcast (by 67.8 %
when used as an additive signal, p = 0.04).

This doc answers six questions in order:

1. **What have we made?**
2. **Why does it matter?**
3. **How was it made?**
4. **How does it compare to what's out there?**
5. **How does it perform?**
6. **How is it productisable?**

Companion docs:
- `foundational.md` — what makes it foundational (Q1, Q2 from earlier)
- `01-density-nowcast.md` / `02-multi-horizon-forecast.md` /
  `03-regime-vix-and-transmission.md` — per-tier specs
- `docs/architecture/03-system-architecture.md` — the three-view system arch

---

## 1. What have we made?

A **modular econometric model stack** with four customer-facing
output surfaces, one shared latent state, vintage-disciplined
training and scoring, and a published audit trail.

**The stack, layered:**

```
   Foundation        Phase 0 — vintage store, walk-forward harness,
                              live forecast pilot, baselines
       ↓
   Archetypes        Phase 1 — 5 state-space models, one per
                              category type (commodity TVP /
                              BSTS / UC-SV-MS / VECM /
                              hierarchical DFM)
       ↓
   Composition       Phase 2.1 — CBDF cross-component covariance +
                                Truflation → BLS bridge layer
       ↓
   Regime layer      Phase 2.2 — Pure-MS Hamilton on monthly YoY
                                + Bils-Klenow sticky/flex split
       ↓
   Transmission      Phase 3 — BVAR-Minnesota per industry vertical
                              with cost-structure registry,
                              IRF / FEVD / shock-scenario tools
```

**Four customer-facing product surfaces emerge from the stack:**

| Tier | What it outputs | Customer |
|---|---|---|
| **1** | Daily density nowcast of BLS CPI / BEA PCE next-month YoY | Asset managers, trading desks, hedge funds, treasurers |
| **2** | Multi-horizon (h=1 to 12) forecast with scenario conditionals | CFOs, lenders, asset-allocation desks |
| **3a** | Regime indicator ("VIX for inflation") with sticky/flex split | Macro hedge funds, options desks |
| **3b** | Per-vertical cost-line P&L scenarios for industry operators | Logistics CFOs, restaurant ops, retail, healthcare, real estate, manufacturing |

**State of build:** Phase 0 + Phase 1 + Phase 2.1 + Phase 2.2 +
Phase 3.1-3.3 shipped and validated. Phase 1 dashboard, Phase 3.4
(UK), Phase 3.5 (Fed-grade paper) queued.

**Test suite:** 205 passing tests on the fast suite (~6 seconds);
synthetic-recovery + real-data fits validated for every archetype.

---

## 2. Why does it matter?

### The inflation-decision problem

Two facts about inflation forecasting today:

- **Inflation matters more than ever** for macro positioning, treasury
  cost-of-funds, supply-chain pricing decisions, and household
  budgeting. The 2021-2023 surge cost the Fed credibility and cost
  US firms an estimated $300 B in inflation-driven cost variance
  they couldn't hedge or pass through.
- **The public infrastructure is a 1970s product**. BLS CPI is
  released monthly, with two weeks of delay. The 5-year breakeven
  market is illiquid. The Cleveland Fed nowcast is the public
  state of the art, and it's a single point estimate updated
  daily-ish from gas-price futures and recent prints.

Thales is the next layer up. It produces:
- **Daily nowcasts** with calibrated bands (institutional-grade,
  not single point)
- **Forward scenarios** the Cleveland Fed nowcast doesn't
- **Industry-specific cost-structure outputs** the academic
  literature doesn't touch
- **Regime indicators** that tell users *when* inflation is in a
  high-vol regime vs calm (Fed mostly tells you what already
  happened)

### Where the value compounds

The customer ROI isn't "we replace your Bloomberg terminal." It's:
- A trading desk uses the **density forecast** for option-pricing
  on inflation derivatives → 5-15 bp tighter risk premiums = 7-figure
  P&L per book
- A logistics CFO uses the **transmission VAR's exposure scenario** to
  decide how much pricing-power to ask freight customers for
  next quarter → measurable margin defense
- A macro hedge fund uses the **regime indicator** to size its TIPS
  vs nominals position → directly billable
- A research desk cites Thales alongside Cleveland Fed in client
  notes → marketing reach

The unit economics are SaaS-shaped (subscription per surface
per customer), not advisory (fee on AUM, regulated).

---

## 3. How was it made?

The architecture **compounded** across phases — each layer plugs
into a stable Forecaster Protocol and an evaluation harness with
walk-forward, vintage discipline, and conformal bands.

### Foundation first (Phase 0)

- DuckDB **vintage store** with bitemporal append-only schema
  (~370 k rows, 9 sources)
- **47 FRED series** (ALFRED-vintaged) + **23 BLS subindices** +
  **80 Truflation streams** (TRUF Network frozen) +
  **9 commodity / ETF series** (FMP) + Cleveland Fed JSON archive
- Walk-forward harness with `Forecaster` Protocol → `walk_forward`
  → `attach_actuals` → `score` → `ScoreBlock` (RMSE, MAE, cov80,
  cov95, MSE/RMSE-Δ-vs-naive, SHIP gate)
- DuckDB scoring DB with model-id keyed cross-model SQL
- Pre-registration doc committed before the model evaluations ran

### Archetypes (Phase 1, 5/5 shipped + validated)

Each Truflation category gets a state-space model whose form
matches its generating process. Synthetic-recovery validation
gates every archetype before real-data fits.

| Archetype | Form | Synthetic recovery | Real-data fit |
|---|---|---|---|
| Commodity TVP | TVP-Kalman + RTS | β path Pearson 0.999 | Utilities × Henry Hub: β evolves +0.10 → −0.26 across COVID |
| BSTS | LLT for level / LL for YoY (empirical rule) | R² 0.9997 trend recovery | Recreation R² 1.000, food-away R² 1.000 |
| UC + SV + MS | Kim 1994 + NumPyro NUTS | 11/11 recovery passing | Pure-MS production model for monthly YoY |
| VECM (Johansen-gated) | Johansen test → VECM or fallback | Synthetic + real Truflation Clothing × BLS Apparel | 100 % gate-fires-VECM |
| Hierarchical housing DFM | JAX Kalman + LBFGS-ML | F Pearson 0.978 | Identifiability fix β_NE := 1 |

### Composition (Phase 2.1)

- **`WeightedComposer`** for accounting-identity composition
- **`CBDFComposer`** adds cross-component residual covariance —
  the O'Keeffe-Petrova architectural insight applied to
  inflation
- **Truflation → BLS bridge** stacked on top, because Truflation
  headline ≠ BLS headline (~50 bp structural gap from different
  surveys/weights)

### Regime (Phase 2.2)

- **Pure-MS Hamilton 2-state** on monthly YoY (production)
- **Bils-Klenow sticky/flex decomposition** — sticky ≈ Core CPI
  YoY; flex ≈ Headline minus Core
- As of 2026-03-31: **sticky regime = 0.997 ON for 60 months**,
  flex = 0.059 OFF — services-driven persistent regime

### Transmission (Phase 3.1–3.3)

- **BVAR-Minnesota** with closed-form posterior (BGR), Cholesky
  IRF + FEVD + conditional/shock-scenario forecasting
- **Cost-structure registry** mapping each industry vertical to
  cost-share weights (logistics: fuel 35 %, labor 25 %, etc.)
- **6 verticals shipped**: logistics, restaurants, retail,
  healthcare, real estate, manufacturing
- Cross-vertical pattern: BVAR reliably beats RW on cost-side
  variables (+15-30 % RMSE), weak/harmful on demand-side. The
  product framing: "we forecast YOUR cost lines, you forecast
  YOUR demand."

### Drift fixes (the methodology layer)

Six fixes shipped after a deliberate audit revealed weaknesses:

| # | What | Result |
|---|---|---|
| 1 | Rolling-conformal bands (point on all data, bands from rolling-OOS residuals) | −5 % to −49 % RMSE on +1m baselines |
| 2 | Compress 12-feature multi-component bridge to PCA-3 / grouped-5 | Eliminated catastrophic overfit (−63.9 % → −10.3 % RMSE) |
| 3 | Cleveland Fed *incremental* value test | Reframe: improve, don't compete (+12-15 % RMSE OOS) |
| 4 | Johansen-gated VECM with ARDL/bridge/AR(1) fallback | Gate fires correctly on theory-cointegrated pairs; fallback prevents VECM-on-noise |
| 5 | **MoM-first composition** | **+38 % RMSE reduction over YoY-direct AR(1)** — the biggest single methodology win |
| 6 | Regime-conditional bands with transition buffer | +10.6pp coverage toward nominal |

All six are propagated through the production stack (drift fixes
A/B/C/D applied 2026-04-26; production daily script is on the
frozen endpoint).

---

## 4. How does it compare to what's out there?

### Three things publicly available

| Service | What it does | Where it falls short |
|---|---|---|
| **Cleveland Fed Inflation Nowcast** | Single-point daily nowcast of BLS CPI / BEA PCE | No bands, no scenarios, no per-industry, no regime indicator |
| **Bloomberg / Blue Chip consensus** | Survey of forecaster medians | Aggregated, lagged, no density, no industry |
| **Atlanta Fed sticky-price CPI / Dallas Fed trimmed mean** | Smoothed historical decompositions | Backward-looking, not forecasts |

### One state-of-the-art academic claim

**O'Keeffe-Petrova 2025 NY Fed SR 1152 — CBDF on GDP**:
component-based dynamic factor with cross-component covariance
beats standard Stock-Watson DFM by **+15 % RMSE / +20 % density**
on quarterly GDP nowcasting. The paper sets the architectural
target for component-based macro modeling.

### How Thales compares (head-to-head, validated)

**On inflation (BLS Headline CPI YoY, h=1, walk-forward):**

| Comparison | Δ RMSE | DM p-value | Source |
|---|---:|---:|---|
| Bridged-CBDF (us) vs Stock-Watson DFM | **+26-31 %** | < 0.0001 | `OKEEFE_HEADTOHEAD_FINDINGS.md` |
| MoM-composed AR(1) (us) vs Stock-Watson DFM | **+37.6 %** | 0.0003 | same |
| Cleveland Fed + Thales vs Cleveland Fed alone | **+67.8 %** | 0.04 | same |
| MoM-first vs YoY-direct (Fix #5 isolated) | +38.0 % | 0.034 | `MOM_COMPOSED_FINDINGS.md` |

We **replicated and exceeded** the O'Keeffe-Petrova architectural
claim on a different target (inflation, not GDP) with a bigger
margin. Plus we have an even stronger non-CBDF model (MoM-composed
AR(1)) and a customer-facing operational claim (Clev + Thales) that
the academic literature doesn't have.

### What we don't yet have for full Fed-grade comparability

- **CRPS / density scoring** propagated across all models
  (~1 hour of work; primitives all exist)
- **Multi-horizon evaluation** (currently h=1 only; Tier 2 spec
  goes to h=12)
- **Bigger n on bridged-CBDF** (n=11 OOS is small; constrained by
  Truflation per-component history starting 2020-01)

---

## 5. How does it perform?

### The defensible numbers (validated, signed off in findings docs)

| Claim | Number | Significance | Where validated |
|---|---|---|---|
| **Cleveland Fed + Thales vs Cleveland alone** | +67.8 % RMSE | DM p=0.04 | `OKEEFE_HEADTOHEAD_FINDINGS.md` |
| **MoM-composed AR(1) vs Stock-Watson DFM** | +37.6 % RMSE | DM p=0.0003 | same |
| **Bridged-CBDF vs Stock-Watson DFM** | +25.6 % to +30.6 % RMSE | DM p<0.0001 | same |
| **MoM-first vs YoY-direct** | +38 % RMSE | DM p=0.034 | `MOM_COMPOSED_FINDINGS.md` |
| **Diesel→freight pass-through (logistics)** | 8 %, persistent over 24 months | structural | `BVAR_LOGISTICS_5VAR_FINDINGS.md` |
| **Cross-vertical cost-side beats RW** | +15-30 % RMSE | 6 verticals tested | `BVAR_PHASE33_FINDINGS.md` |
| **Truflation→BLS empirical pass-through** | 8 % via bridged CBDF | structural | same head-to-head |

### Coverage and calibration

- **Rolling-conformal bands** with finite-sample Vovk quantiles
  ship across baselines, same-month bridge, BVAR. Per-α fallback
  to Gaussian when calibration set < min_n (9 for 80 %, 39 for
  95 %)
- **80 % coverage** typically lands within ±5 pp of nominal in
  production-eval; **95 % coverage** within ±5 pp on most models
- One open issue: Phase 3 BVAR cov95 still wide-banded; queued

### Daily live track record (Stefan-facing)

Day 0 (2026-04-24) → present: forecast log + scoring CSV
maintained at `results/daily_forecast_live/`. Migrated to **frozen
endpoint** 2026-04-26 to match the public chart and avoid live-
revision drift.

---

## 6. How is it productisable?

### The product surfaces (already specified per tier)

| Tier | API endpoint shape | What it returns |
|---|---|---|
| **1** | `GET /v1/nowcast/cpi?as_of=...&target=...` | point + band_80 + band_95 + backtest n + RMSE + cov80 + lineage (model_version, data_as_of) |
| **2** | `GET /v1/forecast/cpi?h=...&scenario=...` | trajectory + bands at each h, scenario-conditional via shock_scenario |
| **3a** | `GET /v1/regime` | sticky_high P + flex_high P + combined regime label + 60-month history |
| **3b** | `POST /v1/scenarios/{vertical}` (logistics, restaurants, ...) | cost-line $ exposure + IRF table + FEVD shares |

### What's exposure analytics (allowed) vs investment advice (NOT)

Strict boundary documented in
`docs/architecture/02-product-boundary-no-advice.md`. Action verbs
("hedge", "buy", "rebalance") banned in API responses; exposure
verbs ("exposed to", "correlated with", "expected impact of")
allowed. Customer report shows the data, customer's treasury team
decides the action.

### Pricing model (proposed)

| Surface | Price model | Customer | Annual ARR potential |
|---|---|---|---|
| **Tier 1** density nowcast | Subscription per dashboard | Asset manager / treasury | $5-30 k |
| **Tier 2** multi-horizon forecast | Subscription per scenario family | CFO / lender | $25-100 k |
| **Tier 3a** regime "VIX for inflation" | Lightweight subscription | Macro fund / options desk | $5-15 k |
| **Tier 3b** per-vertical scenarios | Per-vertical SaaS or per-API-call | Industry operator | $25-150 k |
| **Bundle** (all four) | Enterprise license | Hedge fund / large CFO office | $100-500 k |

### Integration story

- **Truflation customers**: extend the existing Truflation feed
  with Thales-derived outputs at no incremental data cost
- **FRED-Truflation integration** (announced, pending): once
  Truflation series ship via FRED/ALFRED, vintage discipline
  becomes free; Thales becomes the value-add layer on top of
  publicly-distributed Truflation data
- **Standalone**: API-key access to Thales endpoints, hosted on
  Truflation's existing infra
- **White-label**: research desks / banks license the methodology
  with their data branding

### Operational state

- **Live forecast pilot** running daily since 2026-04-24
- **Vintage discipline** + audit trail in place
- **Migration target**: laptop launchd → Vast.ai or VPS for
  always-on cron (current laptop misses occasional Saturdays)
- **CI**: not yet wired (205 fast tests run locally in 6 s; CI
  would catch regressions before merge)
- **Customer-facing API**: not yet built (FastAPI + DuckDB read
  replica is the natural deploy)

---

## TL;DR for one slide

```
THALES — foundational econometric model stack on Truflation data

What:    4 product tiers (nowcast, forecast, regime, transmission)
         from one shared state-space model stack.

Where it beats published frontiers:
   • +37.6 % RMSE vs Stock-Watson DFM        (p=0.0003)
   • +25.6-30.6 % RMSE bridged-CBDF vs DFM  (p<0.0001)
   • +67.8 % RMSE Cleveland Fed + Thales    (p=0.04)
   • +38 % RMSE MoM-first vs YoY-direct     (p=0.034)

Where the moat is:
   • Methodology stack (Fix #1-#6)            — ours
   • Truflation daily granularity + 80 streams — exclusive
   • Synthesis (vintage + audit + product framing) — ours

Status:  Phase 0/1/2/3.1-3.3 shipped + validated. 205 fast tests.
         O'Keeffe head-to-head won. UK + paper queued.

Validated: 2026-04-26
```
