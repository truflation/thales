# Truflation Operate

Product #5 from `docs/products/08-user-facing-products.md`. Industry-vertical exposure quantification and scenario analysis built on the Bayesian transmission VAR infrastructure in `src/thales/models/archetypes/bvar_minnesota.py`.

## Framing

The product leads with **exposure quantification and scenario analysis**, not point forecasting. Diesel, freight, and FX are honestly hard to forecast at monthly horizons (the underlying series are close to random walks plus shocks), and trying to over-promise on those is the fastest way to lose credibility with operators.

What the BVAR does well is the **transmission** — Σ-driven cross-effects, IRF, FEVD, conditional projections. That answers the question operators actually have: *"How exposed am I, and what happens to my landed cost across a range of input scenarios?"* — not *"What is diesel doing next quarter?"*

This workspace is built around that reframe:

- **Inputs:** EIA-direct retail diesel (not PPI proxy), FRED freight indices, FRED FX rates, Truflation per-component daily streams.
- **Models:** existing `BVARMinnesota` (no rebuild) extended with vertical-specific variable sets.
- **Output:** an exposure dashboard + scenario console per client, with point forecasts as a humble secondary feature carrying honest uncertainty bands.

## Initial client targets

Two specific clients drive the first import/export vertical:

1. **Paris auto importer** — primary exposures: EUR/USD, ro-ro ocean freight, inland diesel, EU/US duties.
2. **US textile importer** — primary exposures: CNY/USD (or relevant Asia source FX), container ocean freight, US inland trucking, duties.

Cost-structure entries for both live in `docs/cost_structures.md`.

## Layout

```
truflation-operate/
├── README.md                   # this file
├── ingest/
│   ├── eia_diesel.py           # retail + wholesale diesel direct from EIA-via-FRED
│   ├── fred_freight.py         # PPI trucking + freight-related FRED series
│   └── fred_fx.py              # major FX rates (EUR, CNY, MXN, CAD, INR vs USD)
├── verticals/
│   └── (import_export_*.py — to come)
├── scenarios/
│   └── (exposure_quantify.py — wires shock_scenario / conditional_forecast)
├── results/
│   └── (per-vertical CSVs and scenario outputs)
└── docs/
    ├── cost_structures.md      # client cost-share weights with sources
    └── findings_*.md           # honest per-vertical evaluation
```

## Reuses

- `src/thales/models/archetypes/bvar_minnesota.py` — the BVAR engine (642 lines, includes IRF, FEVD, `conditional_forecast`, `shock_scenario`).
- `src/thales/cost_structures.py` — base registry, extended here for import/export.
- `src/thales/vintage/` — point-in-time DuckDB store; all new ingests write under fresh source tags.
- `src/thales/ingest/{fred,eia,truf_network}.py` — fetch primitives reused.

## Honest scope

This is **not** a Truflation Operate product launch. It's the model and scenario-engine layer that a product would consume. UI surfaces (`truflation.com/operate/restaurants` etc. as described in doc 08) are out of scope here.
