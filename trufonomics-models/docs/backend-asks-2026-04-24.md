# Backend Asks — 2026-04-24

Draft to send Jarryd when he's next available. Consolidated from the Thales-v2 Phase 0 state.

---

Hey Jarryd —

Phase 0 Thales is in a good place (80 component streams, full history 2020-2026, weights loader working, composition reconstructs the published frozen headline within 0.000 pp median residual across 1,932 days). A few items I need from the backend side to close remaining gaps. Priority order:

## 1. Per-observation `(reference_date, as_of_date)` pairs — the biggest remaining data asset

Right now all 80 TRUF Network component streams we pull are tagged with `as_of_date = ingest_date`. Works fine for forward use but doesn't support vintage-correct backtesting against historical origins. What I'd like:

- For each of the 80 streams, the publication-date timestamp per observation (the moment Truflation first made that (reference_date, value) public). Even just the **first publication date** per observation — no revision history needed if the streams are genuinely frozen.
- If live (non-frozen) counterparts of the same 80 streams exist, same ask: per-observation `as_of_date` when each value was first made public and when it was revised (if applicable).

This is the one unblocker for the pseudo-real-time evaluation story in `docs/pre-registration/001-initial-nowcast-methodology.md`. Without it, any pre-2026 backtest we claim is weaker than it could be.

## 2. Leaf-level (source_id != 0) weights — if exposed anywhere

The weight CSVs at `kairos/data/truflation/weights/categories-tables-v{1,2}.csv` have source-level rows (e.g., `source_id=496`, `table=com_numbeo_us_cereals`, `relative_importance=0.135`) in addition to the aggregate (source_id=0) rows. Right now our composition uses the top-level 12 category weights and gets 0.000 pp median residual vs published frozen headline, with 0.2-0.3 pp overshoot in recent weeks. The small residual is likely from Truflation aggregating at a finer level (source-feed-by-source-feed rather than category-by-category).

If the leaf-level weights are exposed somewhere accessible (MariaDB, API, or just a larger CSV), I'd like the full per-source weights. With them we should be able to close the residual to ~0 across all dates, not just median.

## 3. One missing subcategory

Our 80-row stream catalog covers 80 of 81 taxonomy nodes in `categories-metadata.csv`. The one missing node:

- `category_id=158`, name `"Miscellaneous products and services"`, parent `"Other"` (category_id=89)

Could you send the `tn_stream_id` for that one? Low weight (fraction of the 3.36% "Other" top-level), but it'd close the coverage check.

## 4. PCE-specific component catalog

We have the 80 CPI-family components. Does Truflation expose a parallel catalog of PCE-family components? Planning doc calls for BEA PCE nowcasts as a secondary target alongside BLS CPI. If there's a separate `pce_streams_catalog.csv` (or similar) with tn_stream_ids for PCE components, I'd like that.

## 5. Live-pull access timing

The TN Network SDK path works via subprocess on macOS (we use a Linux-wheel-compatible Python to avoid the pure-arm64 segfault); Linux deployment should be clean. When Thales moves to Vast.ai for production, we'll hit the TN gateway from Linux and that pattern simplifies. No action needed from you unless you hit an auth / rate-limit wall we should know about.

---

## What we have that's working today (for context, not asks)

- 80 component streams ingested (2020-2026, ~176k rows)
- 12 category + 36 subcategory + 32 component taxonomy nodes covered
- Weights loader reading v1 (2010-2025) + v2 (2026-) correctly
- Composition math validated: frozen aggregate reconstructed within 0.000 pp median residual
- Full Cleveland Fed historical nowcast archive (2013-2026) for benchmarking
- FRED ALFRED vintages for 38/47 covariate series
- 42 tests passing in CI

None of those are blocked on you. The five items above are.
