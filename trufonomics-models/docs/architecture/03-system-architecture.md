# Thales — system architecture

**Status:** v1, 2026-04-26. Reflects what's actually built and validated
through the O'Keeffe head-to-head plus all drift fixes.

This is the canonical "how does it all fit together" doc. Three views,
each useful for a different reader:

- §1 — the **modeling architecture** (the academic / model-design view)
- §2 — the **product surface** (what customers see)
- §3 — the **system / data flow** (the engineering view)

Companion docs:
- `01-live-vs-frozen-data.md` — vintage discipline for live deployment
- `02-product-boundary-no-advice.md` — the "exposure analytics not advice" rule
- `04-data-sources.md` — TBD; per-source ingest catalog

---

## §1 — Modeling architecture

### The three-tier product spine

```
                       ┌──────────────────────────────────────┐
                       │        BLS / BEA official targets    │
                       │   (CPIAUCSL, CPILFESL, PCEPI, …)     │
                       └─────────────┬────────────────────────┘
                                     │ scored
                                     │ point-in-time
                                     │ (frozen / vintage discipline)
                                     │
                                     ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                  Tier 1 — Density nowcasts                   │
        │                                                              │
        │   Inputs:  Truflation headline (Feed API frozen series)      │
        │            Truflation 12 components (TRUF Network frozen)    │
        │            FRED macro panel (47 series, ALFRED-vintaged)     │
        │            Cleveland Fed nowcast comparator                  │
        │                                                              │
        │   Models:  AR(1) on YoY, Persistence, PathA, MoM-composed    │
        │            AR(1), Bridged-CBDF, Same-month bridge family,    │
        │            Cleveland Fed + Thales linear ensemble            │
        │                                                              │
        │   Bands:   Rolling-conformal (finite-sample Vovk quantiles)  │
        │            with per-α fallback. Coverage validated.          │
        │                                                              │
        │   Frame:   h = 1 month (next BLS print) and h = 0 (same-     │
        │            month nowcast at end of month T)                  │
        └──────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────────────────────┐
        │             Tier 2 — Multi-horizon forecasts                 │
        │                       (planned; h=1 baseline shipped)        │
        │                                                              │
        │   Inputs:  Same as Tier 1, plus longer history               │
        │   Models:  Same primitive zoo, extended to h ∈ {-15..+12}    │
        │   Status:  h=1 baselines validated; multi-horizon queued     │
        └──────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Tier 3 — Regime indicator + per-vertical transmission VARs  │
        │                                                              │
        │   3a "VIX for inflation":  Pure MS Hamilton on YoY +         │
        │                            sticky/flex Bils-Klenow split     │
        │                            (services-driven regime currently │
        │                            ON; energy-driven regime OFF).    │
        │                                                              │
        │   3b transmission VARs:    BVAR-Minnesota, 6 verticals       │
        │                            (logistics, restaurants, retail,  │
        │                            healthcare, real-estate, mfg).    │
        │                            IRF + FEVD + shock-scenario tool. │
        │                            Cost-line $ exposure outputs.     │
        │                            Cost-structure registry.          │
        │                                                              │
        │   Surface: exposure analytics (NOT advice). See              │
        │            docs/architecture/02-product-boundary-no-advice.md│
        └──────────────────────────────────────────────────────────────┘
```

### The five archetypes (Phase 1 foundation)

Each Truflation category gets a state-space model whose form matches
its generating process. Synthetic-recovery validated in every case;
real-data fits documented per archetype in `results/archetype_recovery/`.

| Archetype | Categories | Form | Validated |
|---|---|---|---|
| Commodity TVP | Utilities, fuel | TVP-Kalman + RTS | β path Pearson 0.999 synthetic; Utilities × Henry Hub real-data β evolves +0.10 → −0.26 across COVID |
| BSTS | Recreation, food-away, Other | LLT for level / LL for YoY (empirical rule established) | R² 0.9997 trend recovery synthetic; real CPI Recreation R² 1.000 |
| UC + SV + MS | Health, Education, Communications, Alcohol | Kim 1994 + NumPyro NUTS | 11/11 recovery passing; Pure MS shipped as the production regime detector for monthly YoY |
| VECM (Johansen-gated) | Clothing, durables | Johansen trace test → VECM if rank≥1, else ARDL/bridge/AR(1) | Synthetic recovery + real-data Truflation Clothing × BLS Apparel CPI 100% gate-fires-VECM |
| Hierarchical housing DFM | Owned + rented dwellings | JAX Kalman + LBFGS-ML | F Pearson 0.978; identifiability fix β_NE := 1 |

### Composition layer (Phase 2.1)

Two composers:
- **`WeightedComposer`** — accounting-identity composition with Monte-Carlo bands
- **`CBDFComposer`** — adds cross-component residual covariance (O'Keeffe & Petrova 2025 NY Fed SR 1152). Captures fuel-shock-hits-utilities-and-transport-together kind of dependence.

**Important architectural finding from the head-to-head**:
direct CBDF on Truflation components **predicts Truflation headline**, not BLS headline. The two differ by a ~50 bp structural gap (different surveys, weights). For BLS targets, the proper architecture is **bridged CBDF**:

```
12 Truflation components → CBDF → Truflation-CBDF-nowcast
                                          ↓
                            α + β·BLS_lag + γ·CBDF_pred  (rolling OLS bridge)
                                          ↓
                                  BLS YoY forecast
```

Validated: Bridged-CBDF beats Stock-Watson DFM by +25.6 % / +30.6 % RMSE,
p < 0.0001 on n = 11 OOS months. See `OKEEFE_HEADTOHEAD_FINDINGS.md`.

### Regime layer (Phase 2.2)

- **Pure-MS Hamilton 2-state** on monthly YoY — the production regime
  detector (PUC + SV + MS over-parameterized for monthly YoY, MoM
  reframing reveals that regime is recoverable on MoM)
- **Bils-Klenow sticky/flex decomposition** — sticky ≈ Core CPI YoY,
  flex ≈ Headline − Core. Two parallel regime indicators.
- **As of 2026-03-31**: sticky=0.997 ON (60-month services-driven
  regime), flex=0.059 OFF — services persistence is the load-bearing
  signal.

### Transmission layer (Phase 3)

BVAR-Minnesota with closed-form posterior (Bańbura-Giannone-Reichlin).
Cholesky IRF + FEVD + conditional/shock-scenario forecasting.
Six verticals shipped:

| Vertical | k vars | Most exogenous | Top forecastable variable | Top dollar exposure |
|---|---:|---|---|---:|
| Logistics | 5 | Diesel | Maintenance (+19% RMSE Δ) | Fuel: $5.3M / +20% diesel |
| Restaurants | 6 | Food cogs | Rent (+31%) | Food: $939k / +20pp food MoM |
| Mid-market retail | 5 | Wholesale durables | Sales (+47%) | COGS: $976k |
| Healthcare operators | 4 | Pharma | Pharma (+27%) | Pharma: $262k |
| Real estate operators | 5 | Construction materials | Labor (+27%) | Maintenance: $963k |
| Manufacturing durables | 5 | Raw materials | Logistics (+27%) | Raw materials: $1.72M |

**Cross-vertical pattern (validated across 6 verticals):** BVAR
reliably beats RW on cost-side variables (+15-30%); weak/harmful on
demand/output-side. Product story: "we forecast YOUR cost lines, you
forecast YOUR demand."

### Live forecast committee (operational layer)

Three independent forecasters running daily on the **frozen** Truflation
endpoint:

1. **Persistence** (the floor)
2. **AR(1) rolling-conformal** (slight mean-reversion)
3. **Ridge stacker** (the production model from the kairos line)

Median committee output is what gets shared. SHIP rule: ±5 bp
agreement. As of 2026-04-26 close: all three within 1.1 bp of 1.798%
for 27 Apr.

---

## §2 — Product surface (what customers see)

### Tier 1 — Density nowcasts (committee output + bands)

```
{
  "as_of": "2026-04-26 16:00 ET",
  "data_as_of": "2026-04-26",          # last frozen-published value
  "model_version": "v2026-04-26",
  "target_date": "2026-04-27",
  "point": 1.798,
  "band_80": [1.785, 1.812],
  "band_95": [1.760, 1.835],
  "method": "3-model committee, frozen target",
  "backtest_n": 90,
  "backtest_rmse": 0.130,
  "backtest_cov80": 0.74,
  "scoring_target": "truflation_us_cpi_frozen_yoy"
}
```

### Tier 1.5 — BLS official-target nowcast

Same JSON shape, but the target is **BLS Headline CPI YoY** at the
next BLS release date. Two engines:
- **Direct**: MoM-composed AR(1) (best single-model on h=1)
- **Combined**: Cleveland Fed + Thales linear ensemble (best operational, +67.8% RMSE vs Cleveland alone)

### Tier 3a — Regime indicator ("VIX for inflation")

```
{
  "regime_sticky": 0.997,
  "regime_flex":   0.059,
  "regime_combined": "services-driven",
  "as_of": "2026-03-31",
  "explanation": "Sticky CPI in high-vol regime since 2021-03 (60 mo).
                  Flex (food + energy) in low-vol regime."
}
```

### Tier 3b — Per-industry exposure scenarios

```
POST /api/v1/scenarios/logistics
{
  "shock": {"variable": "diesel", "magnitude_pct": 0.20},
  "horizon_months": 12,
  "company_profile": {"revenue_usd": 100000000, "opex_share": 0.85}
}

→ Response:
{
  "scenario": "+20% diesel sustained",
  "horizon_months": 12,
  "cost_line_impact": {
    "fuel":        {"delta_usd": 5293281, "pct_of_opex": 0.062},
    "labor":       {"delta_usd": 25904, "pct_of_opex": 0.000},
    "maintenance": {"delta_usd": 11064, "pct_of_opex": 0.000}
  },
  "freight_rate_pass_through_pct": 1.64,
  "total_delta_usd": 5330249,
  "as_pct_opex": 0.0627,
  "model_version": "bvar_logistics_v1",
  "data_as_of": "2026-04-16",
  "framing": "exposure_analytics_not_advice"
}
```

The output is exposure mapping. The customer's treasury / pricing
team decides what to do with it.

---

## §3 — System and data flow

### Data tier

```
                     ┌─ Truflation Feed API (HTTPS, frozen YoY)
                     ├─ Truflation Network (TRUF SDK, 80 components,
                     │     10-day publication lag, requires private key)
                     ├─ FRED ALFRED (47 macro series, vintage-tracked)
                     ├─ BLS direct (23 CPI subindices via API v2)
                     ├─ Cleveland Fed (HTML scrape + JSON archive)
                     ├─ EIA (6 commodity series)
                     ├─ FMP (9 commodity futures + ETFs)
                     │     [DBO best diesel hedge, 0.60 monthly corr]
                     │
                     ▼
            ┌──────────────────────────────┐
            │     vintage_store DuckDB     │
            │  (~370k rows, 9 sources)     │
            │                              │
            │   Schema:                    │
            │   (series_id, ref_date,      │
            │    as_of_date, value,        │
            │    source, source_hash, ts)  │
            │   PK: (series_id, ref_date,  │
            │        as_of_date, source)   │
            │                              │
            │   Append-only.               │
            │   Conflicts on different     │
            │   value raise ValueError.    │
            └──────────────────────────────┘
                          │
                          ▼
           ┌──────────────────────────────────────────────┐
           │   Walk-forward harness (no peeking)          │
           │   src/thales/evaluation/harness.py           │
           │                                              │
           │   - Forecast / Forecaster Protocol           │
           │   - walk_forward (slices to [:origin])       │
           │   - attach_actuals (joins target to predictions) │
           │   - score → ScoreBlock (RMSE, MAE, cov80,    │
           │     cov95, dir hit, MSE/RMSE-Δ vs naive)     │
           │   - SHIP gate: cov80 ±7pp, cov95 ±4pp        │
           └──────────────────────────────────────────────┘
                          │
                          ▼
            ┌──────────────────────────────┐
            │     scoring_db (DuckDB)      │
            │   forecasts + scoring tables │
            │   model-id keyed             │
            │   cross-model SQL queryable  │
            └──────────────────────────────┘
```

### Operational stack

| Component | Where it runs today | Migration target |
|---|---|---|
| Daily Truflation forecast cron | Local laptop launchd at 09:00 ET | Vast.ai / VPS (laptop sleeps occasionally — see Saturday 04-25 missed run) |
| TRUF Network ingest | Subprocess via augustus venv-bridge Python (universal2 binary) | Linux container with native SDK |
| FRED / BLS / EIA / FMP ingest | One-shot via `uv run python -m thales.ingest.<source>` | Daily cron or service-triggered |
| Model training / backtests | `uv run pytest` (~6s fast suite, ~30s+ slow tests) | CI on every push (not yet set up) |
| GPU-bound work (UC+SV+MS NUTS, hierarchical housing JAX) | Vast.ai A100 (template ready) | Same |
| Tier-3 product API | Not yet built | FastAPI + DuckDB read replica |

### The frozen-vs-live discipline (Truflation specific)

Truflation publishes two parallel YoY series:

- **Live**: continually revising. Friday's 04-24 value was 1.7597 %; by Sunday it had revised up to 1.8246 % (+6.5 bp). For dates 1-2 weeks old, divergence vs frozen averages **~50 bp**.
- **Frozen**: pinned at first publication (22:30 UTC capture + 24-hour QC lock + 1-day publication lag). Never revised. Matches what the public chart shows.

**Production discipline**: every Thales-shipped output uses `truflation_us_cpi_frozen_yoy`. The committee, the production daily script, all backtests. Migrated 2026-04-26.

When FRED adds Truflation series (announced; rollout pending), this becomes simpler — ALFRED handles vintage tracking automatically, no Truflation-specific code path.

### Versioning and audit trail

Every customer-facing output stamps:

- `model_version` (e.g., `v2026-04-26-bvar-logistics-v1`)
- `data_as_of` (UTC timestamp of the latest input data)
- `as_of` (timestamp of the response generation)
- `scoring_target` (which series the model was trained against — frozen vs live, BLS vs Truflation)

Customers re-running the same scenario get identical numbers if the
inputs haven't changed; if they get different numbers, the diff is
attributable to one of (input change, model bump, code change), all
of which are logged.

---

## §4 — What's validated, what's queued

### Validated end-to-end

- Phase 0 foundation (vintage store, ingest, harness, baselines, pre-reg, live pilot)
- Phase 1 archetypes (5/5 with synthetic-recovery + real-data fits)
- Phase 2.1 composition (CBDF with cross-component covariance + bridged version)
- Phase 2.2 regime layer (Pure MS production model + sticky/flex decomposition)
- Phase 3.1-3.3 transmission VARs (6 verticals)
- O'Keeffe head-to-head (DFM baseline beaten by +37.6%, p=0.0003)
- Cleveland incremental value (+67.8% RMSE over Cleveland alone, p=0.04)
- Six "drift fixes" propagated through the stack
- **Bridged-CBDF wired as a Forecaster class** — `src/thales/models/composition/bridged_cbdf.py`, 6 unit tests; eliminates the inline-script-only dependency
- **CRPS / density-scoring plumbed through the harness** — `src/thales/evaluation/density.py` (samples helpers + DensityBlock), `Forecast.samples` flowing through `attach_actuals` / `score`, `ScoreBlock` extended with PIT KS p-value, density coverage, sharpness. See `results/baseline_eval/DENSITY_EVAL_FINDINGS.md`. Headline density numbers: MoM-composed AR(1) wins CRPS at 0.152 (vs DFM 0.263, +42 % CRPS), Bridged-CBDF posts the lowest CRPS overall (0.145) on the n=13 overlap, Stock-Watson DFM badly over-covers (96 % at 80 % nominal)

### Queued (genuinely not done)

- Phase 1 dashboard ship-outs (Streamlit)
- Phase 2.1 multi-horizon (-15 to +12)
- Phase 2.1 joint EM estimation
- Phase 2.2 TIPS economic-value backtest, ship VIX-for-inflation as billable product
- Phase 3.4 multi-country (UK)
- Phase 3.5 Fed-grade paper
- Streamlit dashboard / customer-facing API
- Migration of daily forecast cron from laptop to always-on host

### Architectural decisions still open (per `01-live-vs-frozen-data.md`)

1. Snapshot frequency: end-of-day vs hourly vs intra-day-trigger?
2. Model retrain cadence: nightly vs weekly vs quarterly?
3. Customer-facing model versioning: how do we communicate model bumps?

---

## §5 — Numbers worth remembering

| Claim | Number | Where it's validated |
|---|---|---|
| MoM-composed AR(1) vs Stock-Watson DFM | **+37.6 % RMSE, p=0.0003** | `OKEEFE_HEADTOHEAD_FINDINGS.md` |
| Bridged CBDF vs DFM | **+25.6-30.6 % RMSE, p<0.0001** | same |
| Cleveland Fed + Thales vs Cleveland Fed alone | **+67.8 % RMSE, p=0.04** | same |
| MoM-first vs YoY-direct (Fix #5) | **+38.0 % RMSE, p=0.034** | same + `MOM_COMPOSED_FINDINGS.md` |
| Diesel→freight pass-through (logistics) | **8 %, persistent over 24 months** | `BVAR_LOGISTICS_5VAR_FINDINGS.md` |
| Cross-vertical: cost-side beats RW; demand-side doesn't | **6 verticals tested** | `BVAR_PHASE33_FINDINGS.md` |
| Frozen-live divergence at age 1-2 weeks | **~50 bp, live revises upward** | this doc + `OKEEFE_HEADTOHEAD_FINDINGS.md` |
| Optimal hedge ratio: DBO for diesel | **0.30 (corr 0.60)** | `FUEL_HEDGE_BACKTEST_FINDINGS.md` |
| Theoretical hedge ceiling on retail diesel | **18.5 % σ-reduction (R²=0.34)** | same |
