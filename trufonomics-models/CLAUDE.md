# CLAUDE.md — Thales

Always-loaded brief for Claude Code sessions in this repo. Keep concise. For depth, read `docs/planning/`.

## What this is

**Thales** — Truflation's foundational econometric model stack. Three product tiers on top of Truflation's daily proprietary inflation data:

1. **Nowcasts** of official BLS CPI and BEA PCE releases with density forecasts
2. **Multi-horizon forecasts** with scenario conditionals for business planning
3. **Regime model ("VIX for inflation") + per-industry transmission VARs** for Main Street customers

Per-product specs and downstream-task mapping live in `docs/products/`. See `docs/products/README.md` for the canonical "what does Thales output?" reference.

The repo lives inside `kairos/` as a deliberately-isolated subproject. Kairos hosts the earlier linear-stacker nowcast (shipped, locked) and its forecast findings. Thales is the new Bayesian / state-space track. Do not cross-contaminate imports between the two. The earlier work is referenced only as **baseline numbers to beat** (Path A +42% vs persistence, Gated v2 +59%, and the forecast-track shrinkage interpretation at h ≥ 6).

## Architecture at a glance

**Five archetype models**, one per category generating process, each a state-space model:

1. Commodity pass-through — TVP-VECM with SV (Utilities, fuel, food-at-home)
2. Rate-sensitive durables — hierarchical DFM-SSM (Housing: owned + rented split)
3. Sticky administered services — UC-SV-MS with regime switching (Health, Education, Communications, Alcohol/Tobacco)
4. Import-exposed tradables — VECM with tariff regime dummy (Clothing, durables)
5. Discretionary demand-cycle — BSTS with dual seasonality (Recreation, food-away, Other)

**Composition layer:** Component-Based Dynamic Factor model (CBDF — O'Keeffe & Petrova 2025) respects the accounting identity, composes archetype outputs into headline nowcast with density.

**Regime layer:** UC-SV-MS on headline + sticky/flexible decomposition (Bils-Klenow methodology).

**Transmission layer:** Bayesian VAR with Minnesota priors per industry vertical.

## Evaluation philosophy (non-negotiable)

- **Vintage discipline.** Point-in-time data. No peeking at revisions. Vintage store is Tier 0.
- **Three evaluation tiers:** synthetic recovery → historical pseudo-real-time walk-forward → live production monitoring. All three always.
- **Regime-stratified always.** Full-sample averages hide everything. Report by regime (stable, COVID, surge, disinflation, tariff, shutdowns).
- **Density over point.** CRPS, PIT, calibration. Institutional buyers benchmark on density.
- **Pre-registration.** Commit comparators, windows, metrics, significance thresholds *before* running evaluation.
- **DM and GW tests mandatory.** Claimed improvements without significance tests are noise.
- **Public live track record.** Dated, unrevised, publicly visible. Credibility compounds over time.

## Tech stack (fixed)

- Python 3.12+, `uv` for packaging
- Phase 0 (current): `duckdb`, `pandas`, `polars`, `numpy`, `scipy`, `scikit-learn`, `fredapi`, `properscoring`, `pyarrow`, `requests`
- Phase 1+ (when models land): `jax`, `dynamax`, `numpyro`, `pymc`, `statsmodels`, `cmdstanpy`
- Vast.ai for GPU (RTX 3090/4090 dev, A100 for production MCMC) — not needed for Phase 0

Import paths: `from thales.vintage import VintageStore`, `from thales.evaluation.metrics import crps`, etc. All source lives under `src/thales/`.

## Current phase

**Phase 0 — Foundation.** Zero Truflation component data required. Focus: vintage store, covariate ingest (FRED, BLS, EIA, Cleveland Fed), synthetic DGPs, evaluation harness, three baseline nowcasts, comparator data. See `docs/planning/03-checklist.md` for the actionable list.

## Benchmarks to beat

- **Primary:** Cleveland Fed Inflation Nowcasting Model (https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting)
- **Secondary:** Survey of Professional Forecasters (Philadelphia Fed)
- **Tertiary:** Bloomberg / Blue Chip consensus, Dallas Fed trimmed mean PCE, Atlanta Fed sticky-price CPI
- **Research frontier:** O'Keeffe & Petrova 2025 CBDF (NY Fed SR 1152)
- **Internal prior:** kairos Path A nowcast (+42% MSE reduction vs persistence, shipped 2026-04-20). Thales Phase 1 archetype composition must beat this on the same aggregate same-month task to justify the architecture.

## Priorities for Claude Code sessions

1. Never skip vintage discipline. Every ingest is point-in-time. Every backtest walks forward. Ever.
2. Build evaluation harness *before* real models. Synthetic DGPs with known ground truth for every archetype.
3. Default to density forecasts. CRPS is the primary metric, not RMSE.
4. Commit pre-registration docs before running evaluations. Dated, timestamped.
5. Regime-stratify every results table. Averages lie.
6. Secrets belong in `.env`, never committed. `.env.example` tracks required variable names only.

## API keys (required in .env)

- `FRED_API_KEY` — https://fred.stlouisfed.org/docs/api/api_key.html
- `BLS_API_KEY` — https://data.bls.gov/registrationEngine/ (free, 500 queries/day registered)
- `EIA_API_KEY` — https://www.eia.gov/opendata/ (free, needed for granular energy)
- Truflation enterprise API key — when available for live ingest
