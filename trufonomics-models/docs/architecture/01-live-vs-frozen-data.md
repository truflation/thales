# Live vs frozen data — backend architecture for Tier-3 transmission products

**Status:** open questions captured 2026-04-26. Decisions pending sign-off
before Phase 3 productionization.

This doc captures three architectural axes that came up while building
Phase 3.1 (logistics transmission VAR), plus the recommended defaults.
All three are independent — they can be decided separately and
re-litigated as the product matures.

## Context

A shipping company customer hits the Thales API:
> "Given +20% diesel over 12 months, what's my fuel/labor/maintenance/
> freight P&L impact?"

The backend must (a) load model coefficients, (b) load current state of
endogenous variables, (c) project forward via `shock_scenario()`, (d)
return a customer-facing report. **What "current state" and "model"
mean** is the architectural question.

## Axis 1 — Training data (when we fit the BVAR)

| Pattern | Pros | Cons | Cost |
|---|---|---|---|
| **Refit on every API request** | Always uses freshest data | Same customer hitting at 9:00 vs 9:01 may get different numbers because of intra-hour data revisions | High — wastes compute at scale |
| **Refit nightly** ⭐ | Reproducible during a business day; auditable model version | One-day data lag in coefficients (negligible — Minnesota prior keeps coefficients near random walk) | Low — once per night |
| **Refit weekly** | Even more stable; cheaper | Misses recent regime changes | Lowest |
| **Refit quarterly** | Maximally auditable | Stale during fast-moving periods | Lowest |

**Recommendation: nightly retrain.** BVAR coefficients drift ~0.1% day-
over-day under the Minnesota prior — the structural transmission
ratios (e.g. 8% diesel→freight pass-through) don't move. Every
customer report stamps `model_version: v2026-04-26-bvar-logistics-v1`.

**Open question:** does every model bump require a customer-facing
changelog? Yes for material structural changes (re-specified vector,
new variables); no for the daily refresh.

## Axis 2 — Inference state (the "now" we project from)

| Pattern | Pros | Cons |
|---|---|---|
| **Live at request time** | Captures intra-day moves | Two customers asking the same question at different times of day get different answers — confusing UX |
| **End-of-day snapshot** ⭐ | Reproducible; "today's view" is a single canonical state; vintage-disciplined audit trail | Misses intra-day surprises by up to 24h |
| **Hourly snapshot** | Compromise | Adds infra complexity for marginal value |

**Recommendation: end-of-day snapshot at NYMEX close (16:00 ET).**
Customer-facing report stamps `data_as_of: 2026-04-26 16:00 ET`. If
they re-run the scenario tomorrow with the same inputs, the diff is
attributable: "your fuel cost forecast moved +1.2% because diesel
rose 3¢ overnight."

**Open question:** what about during high-volatility events (OPEC
announcement at 11:00 AM)? Two options:
1. Trigger an emergency snapshot when oil moves > 3% intraday
2. Show a "stale-state warning" if state is > X hours old AND
   underlying spot prices have moved > Y% since snapshot

## Axis 3 — Vintage discipline (historical revisions)

This is the most important axis — gets it wrong = silently optimistic
backtests.

**Empirical evidence from this project:**
- Truflation API value for 2026-04-24 was **1.7597%** on Friday morning
- Same date now reports **1.8246%** — a **6.5 bp upward revision** in
  2 days. The series is *revisable*.
- FRED BLS series get even bigger revisions — annual benchmark
  revisions can shift quarterly numbers by ±0.5pp.

| Use case | Vintage discipline |
|---|---|
| **Training the model** | Use point-in-time vintages (ALFRED for FRED, frozen Truflation series). Fitting on revised values gives the model "future information" it couldn't have known at training time → backtest looks better than reality. |
| **Live inference** | Use the latest vintage. Customer wants "best available estimate now." |
| **Backtesting strategies** | Use first-observed vintages everywhere. Anything else is cheating. |

**The vintage store is the right substrate** — `as_of_date` per row
gives us reproducibility for free. The discipline is in *consistently
using* it.

**Open question:** for FRED series not in ALFRED (e.g. TRUCKD11,
PCU48414841 — see `BVAR_LOGISTICS_5VAR_FINDINGS.md`), we have only
the latest vintage. Three options:
1. Use latest with explicit caveat "look-ahead bias on slow-revising
   series"
2. Snapshot daily and accumulate our own vintage history (slow build,
   takes years to accumulate)
3. Skip those series in backtests; only model live-going-forward

The pragmatic answer is (1) for slow-revising series + (2) for fast-
moving ones (truck rates revise more than wages).

## Production data flow (recommended)

```
┌─────────────────────────────────────────────────────────────────┐
│  Cron: ingest daily snapshots                                   │
│  ├─ FRED panel (logistics + macro)                              │
│  ├─ Truflation feed (live + frozen series)                      │
│  ├─ FMP commodity prices (HOUSD, CLUSD, NGUSD, etc.)            │
│  └─ Each row tagged as_of_date=today                            │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│  Vintage store (DuckDB) — append-only, multi-vintage            │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ↓ (nightly job)
┌─────────────────────────────────────────────────────────────────┐
│  Model registry                                                 │
│  ├─ BVAR coefficients fit on as_of_date≤today                   │
│  ├─ Versioned: v2026-04-26-bvar-logistics-v1                    │
│  └─ Includes Σ, IRF tables pre-computed                         │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ↓ (API request)
┌─────────────────────────────────────────────────────────────────┐
│  Inference path                                                 │
│  ├─ Pull latest model version                                   │
│  ├─ Pull state vector at end-of-yesterday close                 │
│  ├─ Run shock_scenario() with customer's scenario inputs        │
│  └─ Return result with full lineage                             │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│  Customer report                                                │
│  ├─ point estimates + bands                                     │
│  ├─ model_version: ...                                          │
│  ├─ data_as_of: ...                                             │
│  ├─ scenario_inputs: ...                                        │
│  └─ "what changed since last query" diff (if applicable)        │
└─────────────────────────────────────────────────────────────────┘
```

## Customer-facing implications

The shipping company sees:
- "As of close yesterday, your +20% diesel scenario projects +$5.3M
  / 12 months for your operating profile."
- "Tomorrow: +$5.4M (+0.1M vs yesterday) — attributable to diesel
  rising 2.3% overnight."

This is what enterprise procurement asks for: **defensible numbers
with audit trails**, not real-time WebSocket streams. Nobody hedges
against an intra-second oil tick — they hedge against the close-of-
day state of the world.

## Open questions for sign-off

1. **Snapshot frequency**: end-of-day vs hourly vs intra-day-trigger?
2. **Model retrain cadence**: nightly vs weekly vs quarterly?
3. **Model versioning**: how do we communicate model bumps to
   customers? Email? Changelog? In-product banner?
4. **Vintage gap series**: do we accept look-ahead bias on slow-
   revising series (TRUCKD11 etc.) or wait until we've accumulated
   our own vintage history (12+ months of daily snapshots)?
5. **Backfill discipline**: when we add a new variable to the panel,
   do we backfill its vintage history (impossible) or only count it
   from the date we started snapshotting (limits backtest length)?
6. **State at the customer level**: do we store per-customer state
   (so customer-A's "as-of" can differ from customer-B's), or is
   "today's snapshot" the same for everyone?

## Decision log

(To be filled in as decisions are made.)

| Date | Decision | Rationale | Decider |
|---|---|---|---|
| (pending) | Nightly model refit | Coefficients are slow-moving | — |
| (pending) | EoD snapshot at NYMEX close | Standard market-data convention | — |
| (pending) | Vintage discipline: ALFRED where available, latest-only with caveat elsewhere | Pragmatic given FRED's coverage gaps | — |
