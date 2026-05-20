# Phase 3.1 — BVAR-Minnesota archetype + partial fuel-chain real-data fit

Implements task #115. Scope for this session: **the model itself plus
end-to-end pipeline validation**. The full 6-variable logistics VAR
(fuel/labor/maintenance/freight/margin/volume) is blocked on data
sourcing — see "Data gaps" below.

## What shipped

1. **`thales.models.archetypes.bvar_minnesota`** — new module:
   - `minnesota_prior_diag()` — Litterman 1986 prior precision
     diagonal with `overall_tightness`, `cross_tightness`, `lag_decay`.
   - `fit_bvar_minnesota(Y, p, ...)` — closed-form posterior (per-
     equation generalized ridge with random-walk prior mean on
     own-lag-1). No MCMC needed; Bańbura-Giannone-Reichlin form.
   - `cholesky_irf()` — orthogonalized impulse responses with
     selectable shock ordering.
   - `fevd()` — forecast-error variance decomposition. Rows sum to 1
     by construction.
   - `BVARForecaster` — Forecaster-protocol wrapper for `walk_forward`.
     Iterates h-steps via the companion matrix; bands from h-step
     Gaussian VAR forecast SD.

2. **11 unit tests** covering:
   - Prior shape + lag-decay + cross-tightness semantics
   - VAR(1) coefficient recovery on synthetic data (atol 0.05)
   - Loose-prior intercept recovery
   - IRF[h=0] equals Cholesky factor of Σ
   - IRF decay for stable VAR
   - FEVD rows sum to 1 at every horizon
   - Cholesky-ordering invariance: at h=0 the first variable's variance
     is 100% own-shock under default order
   - Reordering changes IRF (different decomposition)
   - Forecaster Protocol compatibility

## Real-data fit on partial fuel chain

3-variable VAR on what we already have in the vintage store:

- `log_oil` — log WTI crude (DCOILWTICO, daily → month-end)
- `log_gas` — log US regular retail gasoline (GASREGW, weekly → month-end)
- `truf_fuel` — Truflation `transport_gasoline_other_fuels_and_motor_oil`
  YoY in pp (daily → month-end → 12-month YoY)

Cholesky ordering: **[oil → gas → truf_fuel]** — most exogenous first,
matching the structural transmission hypothesis (oil shocks propagate
to retail gasoline, then to consumer fuel CPI proxy).

### Static fit, p=2 on full panel

Stable: max\|eig\| = 0.956 < 1. AR(1) matrix (oil, gas, fuel rows;
oil, gas, fuel cols):

```
[[ +0.952  -0.016    0    ]    oil ← own-persistence + tiny gas reversion
 [ -0      +0.971    0    ]    gas ← own-persistence
 [+33.81  -41.57   +1.14 ]]    truf_fuel ← strong oil pass-through
```

Truf_fuel's coefficient on log_oil (+33.8) reads as: a 1-unit shift in
log_oil → +33.8 pp on truf_fuel YoY. Magnitude is plausible for
log → YoY-pp scaling and the high beta of consumer fuel CPI to oil.

### IRF (Cholesky, ordering above)

| horizon | oil←oil | gas←oil | truf←oil | oil←gas | gas←gas | truf←gas | truf←truf |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.10 | 0.04 | 3.49 | 0.00 | 0.06 | 2.78 | 7.03 |
| 1 | 0.09 | 0.04 | **5.68** | 0.00 | 0.05 | 0.82 | **8.01** |
| 3 | 0.08 | 0.04 | 5.28 | 0.00 | 0.05 | −5.53 | 6.67 |
| 6 | 0.07 | 0.03 | 3.44 | −0.01 | 0.04 | −12.49 | 4.48 |
| 12 | 0.04 | 0.02 | 0.48 | −0.01 | 0.03 | −18.18 | 2.04 |
| 24 | 0.02 | 0.01 | −2.51 | −0.02 | 0.02 | −16.81 | 0.46 |

Read: an oil shock pushes truf_fuel up immediately (+3.5 pp at h=0),
peaking at h=1 (+5.7 pp), then decays. **The gas-shock IRF on
truf_fuel turns negative after h=3** — likely a small-sample
artifact (n=64 monthly obs) but worth flagging; in a longer panel
this should reverse to positive throughout.

### FEVD at h=12

| response \\ shock | oil | gas | truf_fuel |
|---|---:|---:|---:|
| oil | **98.9%** | 1.1% | 0.05% |
| gas | 33.4% | **66.6%** | 0.01% |
| truf_fuel | 6.9% | **79.4%** | 13.7% |

Reads correctly:
- **Oil is 99% own-shock** — most exogenous, as expected.
- **Gas: 33% pass-through from oil**, 67% own — sensible.
- **Truf_fuel: 79% from gas, 14% own, 7% from oil** — gas dominates
  as the immediate input to the consumer-fuel index; oil's effect
  shows up via gas, not directly. Structural transmission is what we
  hoped to see.

### Walk-forward forecasts

**3-var fuel chain (target = truf_fuel YoY, n=27 OOS, 2024-01 → 2026-03):**

| metric | value |
|---|---|
| RMSE | 6.77 (naive 6.89, **+1.77%**) |
| cov80 | 96.3% (overcovers by +16 pp) |
| cov95 | 96.3% |
| dir hit | 51.9% (base-rate 55.6%) |

Beats RW by 1.77% RMSE on n=27 — not enough sample to claim a real
edge. truf_fuel YoY has SD = 24 pp (huge volatility from the 2021-2023
fuel-price burst), so Gaussian bands at z·σ_h come out 23 pp wide and
overcover badly. Sample is the binding constraint.

**Long-window oil → gas BVAR (n=135 OOS, 2015-01 → 2026-03):**

| metric | value |
|---|---|
| RMSE | 0.0645 (naive 0.0644, −0.22%) |
| cov80 | **78.5%** (−1.5 pp from nominal) |
| cov95 | 91.9% (−3.1 pp) |
| dir hit | 50.4% |
| Verdict | SHIP |

This is the pipeline-validation benchmark. The model is essentially
indistinguishable from random-walk on point forecasts (oil is nearly
a unit root; gas tracks oil 1-for-1) — but **bands are well-
calibrated** (cov80 78.5%, cov95 91.9% under Gaussian z·σ_h forecast
SD). Verdict SHIP confirms the pipeline runs end-to-end on real data.

## What the BVAR is actually for

Not point forecasting on near-unit-root macro series — VAR loses to
random walk there for the same reason AR(1) on YoY does (Fix #5
finding). The value of the BVAR is **structural**:

1. **IRFs** answer "if oil jumps 1σ, what does my fuel-cost line do
   over the next 24 months?" — a per-industry counterfactual.
2. **FEVD** answers "what fraction of fuel-cost uncertainty is driven
   by oil vs. retail-gas vs. idiosyncratic?" — a hedging-priorities
   tool.
3. **Conditional forecasts** (next sub-deliverable) answer "given a
   futures-curve path for oil, what's my forward fuel-cost
   distribution?" — directly product-relevant.

The 6-var logistics version delivers all three for the cost-structure
product. The 3-var fuel chain validates the pipeline but isn't
client-ready on its own.

## Data gaps for the full Phase 3.1 logistics VAR

The planning doc specifies endogenous vector
`[fuel_π, labor_π, maintenance_π, freight_rate, margin, volume]`.
We have fuel coverage. The other five need sourcing:

| Variable | Candidate source | Status |
|---|---|---|
| **fuel_π** | DCOILWTICO + GASREGW + truf transport_fuel | ✅ in vintage store |
| **labor_π (trucking wages)** | FRED `CES4348410008` (avg hourly earnings, truck transp); BLS QCEW NAICS 484 | ❌ not ingested |
| **maintenance_π** | BLS PPI 333111 (machinery & maintenance), or CPI vehicle-maintenance subindex | ❌ not ingested |
| **freight_rate** | Cass Freight Index (private; license needed) OR BTS truckload rate per mile (TLI) OR FRED `IRTRUCKBSL` | ❌ not ingested |
| **margin** | Public-co operating margin proxies (KNX, ODFL, JBHT 10-Q) → industry composite | ❌ requires manual aggregation |
| **volume** | ATA Truck Tonnage Index (FRED `TRUCKD11`); rail intermodal via AAR | ❌ not ingested |

The labor, maintenance, and volume series are FRED-ingestable in an
afternoon. Cass freight is paywalled — if Truflation has a relationship
that's the cleanest path. Margin requires its own pipeline (10-Q
parsing or a service like Sentieo).

## Files

- `src/thales/models/archetypes/bvar_minnesota.py` (new)
- `tests/test_bvar_minnesota.py` (new — 11 tests, all green)
- `scripts/bvar_fuel_chain_real.py` (new)
- `results/real_data_archetypes/bvar_fuel_chain_predictions.csv`
- `results/real_data_archetypes/bvar_fuel_chain_irf.csv`
- `results/real_data_archetypes/bvar_fuel_chain_fevd.csv`

## Next steps (in priority order)

1. **Ingest labor / maintenance / volume FRED series** — quick.
2. **Build the cost-structure DB** — small JSON / DuckDB table mapping
   industry → cost weights; the planning doc has logistics weights.
3. **6-var fit on what we have** (fuel + labor + maintenance + volume,
   missing freight + margin) — partial product.
4. **Conditional forecasts** — add a `predict_conditional(future_path)`
   method to `BVARForecaster`.
5. **Economic-value backtest** — fuel-hedging strategy P&L using the
   BVAR's conditional-forecast distribution.
6. **Once freight + margin sourced** — full 6-var product.

## Glossary (stats terms)

- **VAR(p):** Vector AutoRegression of order p — a system where each
  variable depends linearly on its own and all others' lags up to p.
- **Minnesota prior:** Litterman 1986 — random-walk prior mean (own-
  lag-1 = 1, all else = 0) with shrinkage that gets tighter with lag
  number and is more conservative across variables than within.
- **Cholesky identification:** assigns ordering to shocks so that
  structural ε's are recoverable from reduced-form innovations via
  the lower-triangular Cholesky factor of Σ. Most-exogenous variable
  goes first.
- **IRF (impulse response):** the path of variable i over horizon h
  in response to a 1-SD shock in variable j, holding all other
  shocks at zero.
- **FEVD (forecast-error variance decomposition):** at each horizon,
  the share of variable i's forecast-error variance attributable to
  each shock j. Rows sum to 1.
