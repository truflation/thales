# Truflation Operate

Product #5 from `docs/products/08-user-facing-products.md`. Industry-vertical **exposure quantification and scenario analysis** for operators with multi-input cost structures (importers, logistics, restaurants).

## Framing

The product leads with exposure and scenarios, not point forecasts. FX, diesel, and freight are honestly hard to forecast at monthly horizons (the underlying series are close to random walks); over-promising on accuracy is the fastest way to lose operator credibility. What we sell instead:

- **Where are you exposed?** Per-input variance decomposition for the operator's actual cost basket.
- **What happens if X moves?** Multi-shock joint scenarios with calibrated 80%/95% bands across all inputs.
- **How confident are the bands?** Empirically calibrated — see `docs/VIABILITY_FINDINGS.md`.

## Primary engine — Copula + AR(1)

After an empirical survey (BVAR with Minnesota prior, VECM, DCC-GARCH, foundation models like Chronos / TimesFM / Lag-Llama, local projections, gradient boosting), the **Copula + AR(1)** model is the engine. Sklar's decomposition:

- per-input AR(1) marginals — the strongest baseline at monthly grid (literature consensus, confirmed by walk-forward backtests)
- Gaussian copula on rank-uniformised residuals — adds the correct joint structure that per-input independence assumptions miss
- optional Student-t copula (df fit from data, typically 3-5) for stress / tail scenarios

Empirical record (135 OOS origins per vertical, h ∈ {1, 3, 6, 12}, both verticals — see `results/copula_bench_summary_*.json`):

| Method | CRPS vs naive_ar1 | Coverage gap from nominal 80% |
|---|---|---|
| **Copula + AR(1)** | **tied (within ±1.5%)** | **smaller gap on every cell** |
| BVAR (Minnesota) | −22% to −246% (worse) | larger gap (over-wide bands) |
| naive_ar1 independent | baseline | overconfident on most cells |

BVAR is retained as a research / explainability baseline only — useful for IRF and FEVD diagnostics, not as the production scenario engine.

## What this workspace contains

```
truflation-operate/
├── README.md                                 (this file)
├── docs/
│   ├── VIABILITY_FINDINGS.md                 honest empirical write-up
│   ├── literature_and_service_landscape.md   academic + commercial survey
│   └── cost_structures.md                    operator basket descriptions
├── ingest/
│   └── operate_fred_ingest.py                diesel + freight + 5 FX rates from FRED
├── scenarios/
│   ├── cost_baskets.py                       SINGLE SOURCE OF TRUTH for operator
│   │                                          baskets (cost_share + exposure weights)
│   ├── copula_landed_cost.py                 Copula + AR(1) fit + sample
│   ├── copula_scenario_console.py            operator-facing CLI (primary)
│   ├── landed_cost_forecast.py               point-forecast engine + naive baselines
│   ├── landed_cost_distribution.py           distributional sampler v1 (legacy)
│   ├── landed_cost_distribution_v2.py        v2 (BVAR-on-returns, regime-stratified)
│   ├── exposure_quantify.py                  scenario primitives (BVAR-based, archive)
│   └── scenario_console.py                   BVAR scenario CLI (archive)
├── verticals/
│   ├── import_export_auto.py                 Paris auto importer BVAR fit (archive baseline)
│   ├── import_export_textile.py              US textile importer BVAR fit (archive baseline)
│   ├── copula_benchmark.py                   Copula vs BVAR vs naive_ar1 head-to-head
│   ├── landed_cost_eval.py                   point-forecast benchmark
│   ├── landed_cost_distribution_eval.py      distributional v1 benchmark
│   ├── landed_cost_v2_eval.py                v2 benchmark with regime split
│   └── conditional_shock_test.py             cross-input forecast test
└── results/                                  CSVs + JSONs (all reproducible)
```

## Quickstart

```bash
# 1. Ensure data is fresh (one-time)
uv run python -m thales.ingest.fred_alfred --targets
uv run python -m thales.ingest.bls
uv run python -m thales.ingest.truf_network --streams \
    food_and_non_alcoholic_beverages housing transport utilities health \
    household_durables_and_daily_use_items alcohol_and_tobacco \
    clothing_and_footwear education communications recreation_and_culture other
uv run python truflation-operate/ingest/operate_fred_ingest.py

# 2. Operator scenario — auto importer, multi-shock
uv run python truflation-operate/scenarios/copula_scenario_console.py \
    --vertical auto \
    --shock log_fx_eurusd:-0.05 \
    --shock log_diesel:0.10 \
    --horizon 12

# 3. Operator scenario — textile, stress with t-copula tails
uv run python truflation-operate/scenarios/copula_scenario_console.py \
    --vertical textile \
    --family t \
    --shock log_fx_cnyusd:0.08 \
    --shock log_freight:0.30 \
    --shock log_diesel:0.15 \
    --horizon 12

# 4. Reproduce the head-to-head benchmark (the supporting evidence)
uv run python truflation-operate/verticals/copula_benchmark.py
```

## Cost baskets — single source of truth

All operator cost weights live in `scenarios/cost_baskets.py`. Two distinct quantities per basket:

- **`cost_share`** — operating-cost share (the accountant's view). Sums to 1.0.
- **`landed_cost_exposure_weight`** — derived from the cost basket; the vector applied to modelled-variable log-deviations. For foreign-currency cost lines, both the foreign-currency price variable AND the FX variable get weight equal to that line's cost share, so the exposure vector can sum above 1.0 by design (correct double-pass).

The benchmarks, scenario console, and findings doc all import from `cost_baskets.py`. If shares change, bump the basket's `version` field.

## Honest scope

This workspace is the **model + scenario engine layer**. UI surfaces (`truflation.com/operate/<vertical>`, alerts, briefings, embedded console) are downstream products; they consume this engine but are not built here.

## What's next

1. **Conditional copula scenario** — current `copula_scenario_console.py` imposes shocks by overwriting first-step paths; the proper primitive resamples non-shocked variables conditional on the shocked quantile. Also support persistent locked paths (tariff at +X% for N months, freight repriced for 12m, etc.).
2. **Data-source upgrades** — Freightos FBX / Drewry WCI for real ocean-freight indices; EC Weekly Oil Bulletin for EU diesel; USITC HTS / TARIC for HS-level duty resolution.
3. **Weekly briefing generator** — per-operator weekly PDF/email with current moves + exposure decomposition + bands.

See `docs/VIABILITY_FINDINGS.md` for the honest empirical record and what each next step would deliver.
