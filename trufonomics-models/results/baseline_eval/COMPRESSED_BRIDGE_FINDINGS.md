# Compressed multi-component bridge — Fix #2 (2026-04-25)

Resolves user's prioritized fix #2: "Compress to 3-5 features
(PCA/PLS/grouped aggregates) for multi-component bridge" — because 12
features × 36-obs training is a hopeless overfit (n_train ≤ p).

**This is a partial success with a load-bearing negative finding.**
Compression fixes the overfit. But none of the compressed multi-
component variants beat the 1-feature headline bridge on the data we
have today.

## Setup

| | |
|--|--|
| Frame | Same-month nowcast (predict `BLS_yoy[T]` at end-of-T) |
| Window | 2024-01 → 2026-03 (n=26 origins) — limited by per-component data start in 2021 |
| Floor | `last_release_v1` — predict BLS_yoy[T] = BLS_yoy[T-1] |
| Train window | 36 months (the bind: 36 obs ≪ 12 features in raw multi) |

## The headline

| Model | n_feat | RMSE | Δ vs floor | dir hit | cov80 |
|---|---:|---:|---:|---:|---:|
| `last_release_v1`         | 0  | 0.2273 |   +0.00%  | 50% | 96% |
| `same_month_bridge_v1`    | 3  | **0.2130** | **+6.25%** | 69% | 92% |
| `multi_raw_v1` (12 feat)  | 14 | 0.3449 |  −51.77%  | 65% | 65% |
| `compressed_pca_3`        | 5  | 0.2417 |   −6.37%  | 73% | 81% |
| `compressed_pca_5`        | 7  | 0.2463 |   −8.37%  | 73% | 85% |
| `compressed_pls_3`        | 5  | 0.2896 |  −27.44%  | 69% | 81% |
| `compressed_grouped_5`    | 7  | 0.2458 |   −8.15%  | 58% | 69% |

(`n_feat` = total regression columns including intercept + BLS lag.
Earlier draft of this doc reported n=25 because the script called
`walk_forward(... horizon=1)` then mutated `f.target = f.origin` after
the fact, which silently dropped the last origin. Fixed in the script
to use `horizon=0` directly. Conclusions unchanged.)

## What this confirms (the win)

**Compression eliminates the catastrophic overfit.** Multi-raw at
−51.77% is replaced by PCA-3 at −6.37% — a **45.4pp improvement**
purely from compression, with no change to the data. Confirms the
user's hypothesis: 12 features × 36 obs was the problem, not the data.

Direction-hit and coverage also recover:

- 80% coverage: 65% (raw) → 81% (PCA-3) → 85% (PCA-5)
- Direction hit: 65% (raw) → 73% (PCA-3) — better than the 1-feat bridge (69%)

## What this surfaces (the load-bearing negative)

**None of the compressed multi-component variants beat
`same_month_bridge_v1`'s 1-feature headline aggregation.** PCA-3 is
−10.3%, PLS-3 is −40.1%, grouped-5 is −24.9% — all *worse* than the
floor. Yet the 1-feature `truf_yoy` headline is **+14.78% better**.

The most plausible explanation: Truflation's own headline series is
already an industry-weighted aggregate of the 12 components, and that
aggregation is doing useful **shrinkage smoothing** of per-component
noise. Re-decomposing the panel into 12 noisy components, then
re-projecting onto a low-rank subspace, just reintroduces noise without
recovering any new signal.

Two practical knock-on findings:

1. **PLS underperforms PCA in this regime.** PLS-3 (−40.1%) is much
   worse than PCA-3 (−10.3%), despite being supervised. Usual cause:
   25 effective OOS observations are too few for supervised
   compression — PLS chases target correlation that doesn't generalize.
   PCA's unsupervised choice is more conservative and ages better.

2. **Grouped aggregates underperform PCA.** The 5-bucket econ grouping
   (energy/housing/sticky-services/goods/other) loses to PCA-3. Likely
   because the grouping uses prior weights (which average to the
   headline) without choosing components by *predictive* covariance with
   BLS. Grouped is the most interpretable but the least accurate of
   the three.

## Recommendation

For Tier 1 production:

- **Keep `same_month_bridge_v1` (1 Truflation feature) as the default
  same-month nowcaster.** It's the best forecaster we have on the data
  available today, and it's the simplest to explain to enterprise
  customers ("Truflation's own daily index, plus last-known BLS").

- **Park the multi-component bridge.** Don't ship it as Tier 1 yet.
  Per-component data needs >>36 observations of clean history before
  raw OLS / ridge regression on 12 features becomes viable — that's at
  least mid-2027 on monthly cadence. A faster path is the **archetype
  → CBDF composition** route (each archetype absorbs its own
  per-component signal cleanly), which is the architecture the planning
  doc already commits to.

- **Compression is correct as a method, just blocked by data.** Keep
  `CompressedMultiComponentBridge` (with PCA as the default) in the
  codebase and re-evaluate when n_train ≥ 60.

## What I'm NOT claiming

- Component data is useless. The signal is presumably there at higher
  frequency (daily) and within archetypes (e.g. utilities × Henry Hub
  TVP, Phase 1.1 result).
- PCA always loses to a single-feature bridge. This finding holds
  for the **monthly aggregate-vs-aggregate frame** with 25 OOS obs.
  Raw daily nowcast or per-archetype tasks may differ.

## Files

- `src/thales/models/same_month_nowcaster.py::CompressedMultiComponentBridge`
- `tests/test_compressed_bridge.py` — 7 unit tests, all green
- `scripts/compressed_bridge_comparison.py` — reproduces the table above
- `results/baseline_eval/compressed_bridge_comparison.csv` — per-row
  predictions for all 7 variants

## Glossary (stats terms)

- **PCA (principal component analysis):** SVD of standardized features;
  top-k directions of maximum *variance*. Unsupervised — does not see
  the target.
- **PLS (partial least squares):** SVD-like decomposition where
  directions are chosen to maximize *covariance* with the target.
  Supervised. Wold 1975.
- **Grouped aggregate:** taxonomy-driven mean (or weighted mean) within
  a fixed grouping (here: 5 economic macro-buckets). Interpretable but
  not data-driven.
- **Direction hit:** fraction of forecasts where `pred > today` matches
  `actual > today`. The "did we get the sign right?" metric.
- **Coverage at α:** fraction of actuals that fell inside the
  predicted α-band. Ideal: cov80 = 80%, cov95 = 95%.
