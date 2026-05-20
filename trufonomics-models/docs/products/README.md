# Thales Product Surfaces

Thales is a **foundational econometric model stack** built on top of
Truflation's daily proprietary inflation data. "Foundational" here means
*the same underlying architecture produces multiple downstream products*
— different output surfaces for different customer segments. A trading
desk consumes Tier 1; a CFO consumes Tier 2; a logistics operator
consumes Tier 3. All three are powered by the same per-category
state-space archetypes + CBDF composition + regime layer underneath.

This folder is the canonical answer to "what does Thales actually
output?" Each product page below has its own definition, inputs,
outputs, downstream tasks, and status.

## Three product tiers

| Tier | Product | Customer | Status |
|------|---------|----------|-------:|
| 1 | [Density nowcast of BLS CPI / BEA PCE](01-density-nowcast.md) | Institutional (banks, hedge funds, asset managers) | Architecture in place; gate-2 eval pending |
| 2 | [Multi-horizon forecast with scenario conditionals](02-multi-horizon-forecast.md) | Business planning (CFOs, treasurers, lenders) | Phase 2.3 / not started |
| 3 | [Regime indicator ("VIX for inflation") + per-industry transmission VARs](03-regime-vix-and-transmission.md) | Main-street operators (logistics, restaurants, retail) | Regime piece working on real BLS YoY; transmission VARs Phase 3 |

## Why "foundational"

A *foundational* model is one whose outputs are themselves inputs to
many downstream systems — analogous to a foundation model in ML, where
the same pre-trained representation is fine-tuned for translation,
classification, summarization, etc.

Thales is foundational in that sense:

```
Truflation daily index  +  ALFRED official-target vintages
            ↓
       (Phase 1 archetypes — TVP commodity, BSTS, UC-SV-MS, VECM,
        hierarchical housing — fitted per category)
            ↓
       (Phase 2 CBDF composition — accounting-identity-respecting
        weighted aggregation with cross-component covariance)
            ↓
       Internal latent state (level + log-vol + regime + per-component β)
            ↓
       ┌──────────────────┬──────────────────┬──────────────────┐
       ↓                  ↓                  ↓                  ↓
   Tier 1            Tier 2            Tier 3a            Tier 3b
   Density           Multi-horizon     Regime indicator   Transmission
   nowcast           forecast          (VIX for          VAR per
   of BLS / PCE      with scenarios    inflation)         industry
```

Same internal state, four expressed product surfaces. The state-space
representation makes this clean: `μ_t` feeds the level forecast; `h_t`
feeds the regime indicator; `β_r` feeds transmission VAR
identification; CBDF composition aggregates everything into the
headline density.

## How this differs from existing things

| Thing | What it is | How Thales differs |
|-------|------------|-------------------|
| **Truflation** (existing product) | Daily web-scraped/alt-data CPI index — a real-time inflation *reading* | Truflation is an INPUT to Thales. Thales forecasts what BLS will publish next, using Truflation as one feature among many. |
| **Path A v1** (kairos repo, locked) | Ridge stacker producing a point nowcast of BLS CPI | Path A is the v1 of Tier 1 — single model, point forecasts only. Thales rebuilds with proper Bayesian state-space architecture, density forecasts, per-category attribution. Path A's +42% MSE-reduction-vs-persistence is the floor Thales must beat. |
| **Cleveland Fed Inflation Nowcasting** (research benchmark) | Nowcast of current-month CPI/PCE published daily, free | Direct comparator for Tier 1. Their h=0 RMSE = 0.17 on Headline CPI is the institutional bar (+54.8% vs last-release). Thales must approach or beat this. |
| **SPF / Blue Chip / Bloomberg** (consensus forecasts) | Survey-based monthly/quarterly forecasts | Comparators for Tier 2 multi-horizon. Thales differs by being daily-updating + density. |

## Status snapshot — 2026-04-25

What's built today:

- Foundation: vintage store, ALFRED ingest, evaluation harness,
  scoring DB, baselines retargeted to BLS/PCE
- All 5 Phase 1 archetype state-space models with synthetic recovery
  validated; first one (commodity) validated on real Utilities data
- CBDF composition layer (Phase 2.1a + 2.1b) with cross-component
  covariance
- Regime detector (pure MS) working on all 4 official targets — Tier 3a
- Live Stefan-style day-ahead pilot ticking on Truflation's own LIVE
  YoY (separate from these three product tiers)

What's not yet:

- **Tier 1 production**: per-archetype real-data fits → composed
  density nowcast → walk-forward eval vs Cleveland Fed → first
  shippable headline number
- **Tier 2**: multi-horizon density forecasts with scenario conditionals
- **Tier 3b**: per-industry transmission VARs

## What downstream systems consume Thales

(Answers question 2 from `obsidian:Thales/my questions.md`.)

| Downstream | What it consumes | Use case |
|-----------|------------------|----------|
| Trading desks (banks, hedge funds) | Tier 1 density forecast + bands | Size positions on BLS CPI surprise; calibrate TIPS-vs-nominal allocation |
| Asset managers | Tier 1 + Tier 2 | Portfolio inflation hedge sizing; long-duration bond positioning |
| Economics research teams | Tier 1 + Tier 2 + Tier 3 regime | Compare to consensus; cite alongside Cleveland Fed; regime-stratified analyses |
| News / explanatory outlets | Tier 1 + per-component attribution | "What's driving today's headline inflation move?" |
| Corporate CFOs / treasurers | Tier 2 | Annual budget under inflation scenarios; multi-year capex planning |
| Commodity buyers | Tier 2 + Tier 3b | Hedge fuel / food / metals exposure; transmission to category-level price forecasts |
| Lenders | Tier 2 | Price floating-rate loans; credit-risk models |
| Logistics / restaurants / retail operators | Tier 3a + Tier 3b | "Should I raise prices now? When does my fuel exposure spike? What's my P&L if inflation regime stays high another year?" |
| Regime-API subscribers ("VIX for inflation") | Tier 3a alone | Lightweight: get an alert when regime probability crosses thresholds |

## Folder index

- [01-density-nowcast.md](01-density-nowcast.md) — Tier 1
- [02-multi-horizon-forecast.md](02-multi-horizon-forecast.md) — Tier 2
- [03-regime-vix-and-transmission.md](03-regime-vix-and-transmission.md) — Tier 3
- [foundational.md](foundational.md) — answers `obsidian:Thales/my questions.md` Q1 in detail
