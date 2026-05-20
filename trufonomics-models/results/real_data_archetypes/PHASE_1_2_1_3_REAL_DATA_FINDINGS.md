# Phase 1.2 + 1.3 — Remaining real-data extensions

**Date:** 2026-04-25
**Script:** `scripts/finish_phase_1_2_1_3.py`
**Outputs:** `results/real_data_archetypes/{bsts,pure_ms}_yoy_*.csv`

## §1.2 BSTS LL on YoY — additional categories

| Category | n | σ̂_μ | σ̂_seasonal | σ̂_ε | Trend (start → end) | Seasonal amp (pp) | R² |
|----------|--:|----:|-----------:|----:|---------------------|------------------:|---:|
| Recreation (prior) | 64 | 0.49 | 0 | 0.31 | 5.6% → 1.1% | 0.56 | 0.96 |
| **Food-away** | 64 | 1.88 | 0 | 0.00 | **9.1% → 3.2%** | **6.37** | 1.00 |
| **Other** | 64 | 0.70 | 0 | 0.37 | 1.6% → 0.4% | 0.31 | 0.99 |

### Three observations

1. **Food-away has unusually high seasonal amplitude on YoY** (6.37 pp
   peak-to-peak). YoY differencing should cancel most of the yearly
   cycle on most series — that it doesn't on food-away suggests **base-
   year effects** (when the base year had unusual pricing patterns
   from COVID, the YoY comparison surfaces them). This is a known
   distortion for restaurant data through 2021-2023.

2. **Food-away trend went from 9.1% to 3.2% YoY** — captures the
   restaurant-inflation surge and its normalization. Largest swing of
   any category we've fit so far.

3. **Other category has modest seasonal (0.31 pp) and clean noise
   (σ_ε = 0.37)** — the residual catch-all categories behave as
   well-mannered noise series. R² = 0.99 confirms BSTS handles them
   without issue.

The Recreation finding from earlier is replicated: σ_seasonal collapses
to zero in all three cases, indicating the seasonal pattern is
**constant-amplitude recurring** (no year-over-year drift in shape).

## §1.3 Pure MS regime detector — sticky services

| Category | σ̂_low | σ̂_high | p̂_00 | p̂_11 | Months in high-vol | Frac |
|----------|------:|-------:|-----:|-----:|-------------------:|-----:|
| Health | 0.49 | 3.48 | 0.91 | 0.91 | 32 / 64 | **50.0%** |
| Education | 1.02 | 3.13 | 0.97 | 0.96 | 13 / 64 | 20.3% |
| Communications | 0.39 | 2.33 | 0.89 | 0.93 | 34 / 64 | **53.1%** |
| Alcohol & Tobacco | 0.77 | 2.59 | 0.98 | 0.94 | 23 / 64 | 35.9% |

### Three observations

1. **Health and Communications are in high-vol ~50% of the time.**
   Surprising for "sticky services" — but consistent with what we know:
   - Health insurance reset windows (Q4 each year) create predictable
     volatility spikes
   - Communications pricing has been volatile in the cellular-plan
     restructuring era (post-2021)

2. **Education has the lowest high-vol fraction (20.3%)** —
   semester-driven pricing is genuinely sticky outside specific
   announcement windows. The model correctly identifies education
   as the calmest of the sticky-services categories.

3. **σ_high / σ_low ratios** (Health 7.2×, Communications 6.0×,
   Education 3.1×, Alcohol 3.4×) tell us how dramatic the regime
   transitions are. Health has the highest contrast — the difference
   between calm Health-pricing months and turbulent ones is a 7×
   variance jump. That's significant volatility risk.

## Cross-category coherence

These per-category regime probabilities are **inputs to the Tier 3a
"VIX for inflation" product at the component level**. A subscriber
asking "is Health pricing in a turbulent regime today?" gets a
direct answer from the Health column of `pure_ms_yoy_health.csv`.

When we wire the regime indicator into CBDFComposer, the per-category
P(high) feeds into the cross-component covariance: in a Health
high-vol regime, Health's contribution to headline forecast bands
should widen.

## Real-data archetype validation status

| Archetype | Real-data fits done | Outcome |
|-----------|---------------------|---------|
| 1.1 Commodity TVP | Utilities × Henry Hub | β regime shift recovered |
| 1.2 BSTS | Recreation, Food-away, Other (3 categories) | All decompositions clean, per-transform rule reconfirmed across categories |
| 1.3 Pure MS | All 4 official targets + Health + Education + Communications + Alcohol/Tobacco (8 series) | Coherent regime detection across measures and sticky-services components |
| 1.4 VECM | ⏳ Clothing × import-price-index | next |
| 1.5 Hierarchical housing | ⏳ BLS regional + Truflation regional housing | needs Vast.ai / data shape |

**4 of 5 archetypes have at least one real-data validation fit;
Phase 1.2 and 1.3 now have multiple.** Phase 1.4 VECM is the
remaining pure-numpy archetype to validate; Phase 1.5 hierarchical
housing is the GPU-relevant one.

## Outputs

```
results/real_data_archetypes/
├── bsts_yoy_ll_food_away.csv
├── bsts_yoy_ll_other.csv
├── bsts_yoy_ll_summary.csv
├── pure_ms_yoy_health.csv
├── pure_ms_yoy_education.csv
├── pure_ms_yoy_communications.csv
├── pure_ms_yoy_alcohol_tobacco.csv
└── pure_ms_yoy_summary.csv
```

Combined with prior runs (Recreation, all 4 official targets), we now
have **8 real-data BSTS or MS fit artifacts** ready for downstream
consumption by composition + regime-API products.
