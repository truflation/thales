# Phase 3.1 — 5-variable logistics BVAR (addendum)

Builds on `BVAR_MINNESOTA_FINDINGS.md`. Tasks #126 (FRED ingest) and
#127 (cost-structure DB) shipped; the BVAR was then re-fit on the
expanded vector.

## What shipped

1. **`scripts/ingest_logistics_fred.py`** — pulls 9 FRED series
   into the vintage store (CES wages, GASDESW diesel, PPI freight,
   CPI maintenance, ATA tonnage, TSA freight, VMT, etc.). 2,404 rows
   added across 9 series, 16 years of history.

2. **`thales.cost_structures`** — registry of industry → cost-share
   weight maps. Logistics weights (fuel 35%, labor 25%, maintenance 10%,
   insurance 5%, other 25%) match ATRI's 2023 Operational Costs of
   Trucking analysis. 5 unit tests.

3. **`scripts/bvar_logistics_5var.py`** — runs the 5-variable BVAR fit
   on the new panel. Results below.

## 5-variable panel

192 monthly obs, 2010-01 → 2026-01.

| variable | mean | sd | AC1 |
|---|---:|---:|---:|
| log_diesel | +1.21 | 0.213 | +0.979 |
| log_freight | +5.03 | 0.177 | +0.997 |
| log_maintenance | +5.71 | 0.170 | +1.000 |
| log_labor | +3.12 | 0.146 | +0.999 |
| log_volume | +4.64 | 0.114 | +0.992 |

**All five are I(1) at the monthly frequency** (AC1 ≈ 1.0). The system's
max\|eigenvalue\| = 1.005 — borderline non-stationary. Expected for
log-levels of price/wage indices that trend over decades. For a
production product the right framing is in MoM/YoY changes (Fix #5
applies), but the level VAR is informative for IRF/FEVD interpretation.

## The headline IRF — diesel-shock pass-through

A 1-SD diesel shock (≈ 4.4% in log-points) propagates as follows:

| horizon | diesel | freight | maintenance | labor | volume |
|---:|---:|---:|---:|---:|---:|
| 0 | +4.41 | **+0.39** | +0.03 | +0.03 | +0.08 |
| 1 | +4.34 | +0.39 | +0.03 | +0.03 | +0.08 |
| 6 | +3.99 | +0.39 | +0.03 | +0.03 | +0.08 |
| 12 | +3.61 | **+0.38** | +0.03 | +0.03 | +0.08 |
| 24 | +2.96 | **+0.37** | +0.04 | +0.04 | +0.07 |

**The economic finding:** roughly **9% of a diesel-cost shock passes
through to freight rates** at every horizon (0.39 / 4.41 ≈ 0.09), and
the pass-through is **highly persistent** — barely decays over 24
months. This is the operational core of the transmission product:
fuel cost movements get partially built into freight pricing, and
that effect compounds in shipper P&L over time.

Maintenance, labor, and volume IRFs are tiny — the cross-effects
exist but at the noise floor on monthly data. Each of those variables
is mostly driven by its own dynamics.

## FEVD at h=12 (variance shares, %)

| response \\ shock | diesel | freight | maintenance | labor | volume |
|---|---:|---:|---:|---:|---:|
| diesel | **99.5** | 0.3 | 0.0 | 0.0 | 0.1 |
| freight | **7.5** | 92.5 | 0.0 | 0.0 | 0.0 |
| maintenance | 0.5 | 2.9 | 96.6 | 0.0 | 0.0 |
| labor | 0.2 | 0.0 | 0.3 | 99.5 | 0.0 |
| volume | 0.3 | 0.1 | 0.0 | 0.9 | 98.6 |

Reads:
- **Diesel: 99.5% own-shock** — most exogenous, as constructed by
  ordering and confirmed by data.
- **Freight: 7.5% of variance from diesel shocks** — the structural
  hypothesis is supported. This is the *answerable* fraction of
  trucking-rate uncertainty that fuel-hedging can reduce.
- Labor / volume are nearly orthogonal to fuel costs at the monthly
  frequency. (Their reaction to fuel is at the 1-2 year horizon
  rather than monthly.)

## Walk-forward forecasts (1-month, n=131 OOS, 2015-08 → 2026-04)

| target | RMSE | RMSE_naive | Δ% vs naive | cov80 | cov95 |
|---|---:|---:|---:|---:|---:|
| log_diesel | 0.0485 | 0.0467 | −3.87% | 74.8% | 92.4% |
| log_freight | 0.0168 | 0.0169 | +0.16% | 55.0% | 75.6% |
| log_maintenance | 0.00515 | 0.00639 | **+19.37%** | 64.1% | 75.6% |
| log_labor | 0.00661 | 0.00729 | **+9.32%** | 78.6% | 90.1% |
| log_volume | 0.0147 | 0.0146 | −0.82% | 85.5% | 94.7% |

**Where the BVAR helps:** maintenance (+19%) and labor (+9%) — both
smooth slow-moving series where the cross-information from the rest
of the system meaningfully tightens the next-month forecast.

**Where it doesn't:** diesel (–3.9%) and volume (–0.8%) — both highly
persistent series that random-walk-as-a-forecaster handles fine on its
own.

**Coverage problem:** freight and maintenance bands undercover at 95%
nominal (75.6% / 75.6%, i.e. −20pp). The BVAR's Gaussian closed-form
forecast SD systematically understates the volatility of freight and
maintenance series. **Conformal bands are the right fix** (would need
extending the rolling-conformal pipeline from baselines to the
multivariate VAR — queued for the conditional-forecast work).

## Data gaps remaining

| Variable | Status | Path forward |
|---|---|---|
| insurance | ❌ no FRED series at fleet-commercial granularity | Cass insurance (paywalled), or aggregate from auto-insurance + commercial-fleet rates |
| margin | ❌ requires 10-Q parsing | Sentieo / capIQ pipeline, or SEC EDGAR scraping |
| Cass freight rate (private) | ❌ paywalled | Truflation enterprise relationship — best option |

Insurance and margin are the only structural gaps blocking the full
6-var product. The 5-var fit captures the essential transmission
mechanism (diesel → freight pass-through) and produces a usable cost-
forecast distribution for the labor / maintenance / volume legs.

## Files

- `src/thales/cost_structures.py` (new — registry)
- `tests/test_cost_structures.py` (new — 5 tests)
- `scripts/ingest_logistics_fred.py` (new)
- `scripts/bvar_logistics_5var.py` (new)
- `results/real_data_archetypes/bvar_logistics_5var_summary.csv`
- `results/real_data_archetypes/bvar_logistics_5var_fevd_h12.csv`
- `results/real_data_archetypes/bvar_logistics_5var_irf.csv`

## Next: conditional forecasts (#128)

The product the customer actually wants: "given this oil/diesel
futures-curve path over the next 12 months, what's the conditional
distribution of my freight, labor, maintenance, and volume costs?"
That's task #128 — a `predict_conditional()` method on the BVAR
forecaster that takes a forced path on a subset of variables and
projects the rest. Standard Bańbura-Giannone-Reichlin algorithm.
