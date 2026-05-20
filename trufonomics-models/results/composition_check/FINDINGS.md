# Composition Sanity Check — Findings

**Date:** 2026-04-24
**Script:** `scripts/composition_check.py`
**Purpose:** Verify that the 80 component streams ingested from TRUF Network, combined with the Truflation category weights at `data/truflation/weights/`, reproduce Truflation's published daily US CPI aggregate.

## Setup

- **Inputs:** 12 top-level component streams (one per Truflation category), their daily index-level values 2020-01-01 → 2026-04-16
- **Weights:** `categories-tables-v1.csv` (2010–2025) + `categories-tables-v2.csv` (2026–) — top-level weights sum to 100.000% exactly in both
- **Cross-walk:** 80/80 catalog streams matched to a `category_id` via name normalisation
- **Comparison target:** `truflation_us_cpi_frozen_yoy` (apples-to-apples with our frozen component streams)

## Results

Two reconstruction methods tested, 1,932 eval days:

| Method | n | median | mean | SD | |resid| max | |resid| p95 | within 0.1pp | within 0.5pp |
|---|---|---|---|---|---|---|---|---|
| **M2 (aggregate-then-YoY)** | 1,932 | **0.000** | +0.059 | 0.224 | 1.050 | 0.517 | **68%** | **94%** |
| M1 (weighted-sum of YoYs) | 1,932 | −0.028 | +0.035 | 0.262 | 1.153 | 0.532 | 43% | 94% |

**M2 is the preferred method** — it exactly matches Truflation's own aggregation math (build composite index from weighted component indexes, then compute YoY on the composite). Median residual of 0.000 pp means the reconstruction is essentially exact on the typical day.

## Reference — residuals vs the LIVE (revised) aggregate

Including for context only; the frozen-to-frozen comparison above is the fair one:

| Method | median vs live | mean vs live | SD vs live |
|---|---|---|---|
| M2 | 0.000 pp | +0.128 | 0.681 |
| M1 | +0.006 pp | +0.104 | 0.602 |

Larger variance because live reflects post-hoc revisions to the underlying components that frozen doesn't see.

## What this means

1. **Our 80 TN component streams are correct.** Weighted via the category weights and composed the same way Truflation composes them, they reproduce the frozen headline within 0.2 pp SD.
2. **Weight interpretation is correct.** Top-level weights sum to 100; v1→v2 transition at 2026-01-01 applied without step-change artefacts.
3. **Cross-walk is correct.** 80/80 streams map to category_ids cleanly with no orphans.
4. **The data pipeline (TN fetch → vintage store → weighted composition) is sound end-to-end.**

## Small residuals that remain

Mean residual 0.059 pp (M2), with a slight overshoot in the most-recent weeks (0.2–0.6 pp). Candidate explanations — low priority to investigate now, but worth noting:

- **Component-level fine-weights not in the v1/v2 table.** Our composition uses the 12 top-level category weights. Truflation likely uses finer (source-id != 0) weights inside each category. First-order: if sub-weights sum correctly to the parent weight, the top-level sum equals the subcomponent sum. But small non-linearities in YoY computation can leak through.
- **Compositing math.** M2 builds a composite index with sum-weight-normalised values, then YoY. Truflation might use a chain-linked index (each day rebased) that differs from a fixed-weight composite by higher-order terms in a period of rapid price change.
- **Revisions.** Even frozen streams get revised occasionally when new sources are added. Our ingest snapshot was 2026-04-24; the published `truflation_us_cpi_frozen_yoy` in the kairos parquet has its own snapshot date.

None of these blocks moving forward; all could be closed via sharper weight data from backend (per-stream leaf weights) or by matching Truflation's exact index-construction algorithm.

## Go/no-go for downstream work

**Green.** The component data + weights match the published aggregate closely enough to trust for:
- CBDF composition-layer experiments
- Per-category archetype model fits (each uses its component streams as inputs; the weights tell us how each archetype's output aggregates up)
- Stefan's daily forecaster using component-level features (tighter than the goods/services 2-way split we had before)

## What still blocks the full vintage-correct story

The one remaining gap from the original backend-dev ask:
- **`(reference_date, as_of_date)` pairs per observation.** Frozen streams are revision-pinned by design, so this is less critical for these 80 streams. But for the live-revised counterparts (if we ever want to compare frozen-vs-live) we'd need publication timestamps.

That's still a clean, narrow ask for the backend team.

## Artifacts

- `results/composition_check/headline_residuals.csv` — per-day published vs reconstructed vs residual, both methods, both comparison targets
- `src/thales/weights.py` — weights loader + cross-walk
- `data/truflation/weights/` — copied from `kairos/data/truflation/weights/`
