# Next-release print forecasters — CPI + PCE

**Date:** 2026-05-15
**Coverage:** Monthly BLS CPI YoY + BEA PCE YoY official print forecasts

This doc tracks the four production forecasters for the next BLS CPI
and BEA PCE prints, plus the Cleveland Fed nowcast comparator, with
their architectures, validated track records, and standing
predictions.

## The four forecasters

| Target | Forecaster | Architecture | Density bands? |
|---|---|---|:---:|
| BLS CPI | Thales standalone | Headline-only MoM-AR(1) on CPIAUCSL log-MoM | ✓ |
| BLS CPI | **BLS-native CBDF** | 11 BLS subindexes + BLS weights + per-component MoM-AR(1) + M2 composition | ✗ (TODO) |
| BEA PCE | PCE standalone | Headline-only MoM-AR(1) on PCEPI log-MoM | ✓ |
| BEA PCE | **PCE-native CBDF** | 3 BEA components (Durable, Nondurable, Services) + OLS weights + per-component MoM-AR(1) + linear composition | ✗ (TODO) |

Plus: **Cleveland Fed nowcast** as external comparator for both CPI
and PCE. Cleveland publishes point-only nowcasts, daily-updating, on
both targets.

## Standing predictions (as of 2026-05-15)

### April 2026 CPI (released 2026-05-12, actual = **3.7792%**)

| Forecaster | Prediction | Error vs 3.7792% |
|---|---:|---:|
| **BLS-native CBDF** | **3.7359%** | **−4.33 bp** ← closest |
| Oliver (Head of Data) | 3.70% | −7.92 bp |
| Thales standalone | 3.8559% (± 0.16 pp 80%) | +7.67 bp |
| Cleveland Fed | 3.56% | −21.92 bp |

**BLS-native CBDF cut Thales standalone's error roughly in half** —
first head-to-head win for the components-based approach.

### May 2026 CPI (next release ~June 11)

Forward-looking from BLS-native CBDF at origin 2026-04-30: **3.9891%**.
Thales standalone re-run pending after April BLS data ingest (and
fixing the `shift(12)` data-quality bug; see below).

### April 2026 PCE (next release ~May 30)

| Forecaster | Prediction | 80% band |
|---|---:|---|
| PCE-native CBDF | **3.7820%** | n/a |
| PCE standalone | 3.7643% | [3.6198, 3.8556] (± 0.12 pp) |
| Cleveland Fed PCE nowcast | 3.7300% | n/a |

Tight 5 bp agreement across all three. Implied house view:
**~3.76% ± 0.04 pp** (range across methods). All three forecast a
~28 bp jump from March's 3.50% YoY, mirroring CPI's March→April +49 bp
move.

## Validated track records

### CPI side

| Forecaster | Reference | Track record |
|---|---|---|
| Thales standalone (MoM-AR(1)) | vs Stock-Watson DFM, n=25 | +37.6% RMSE reduction, p=0.0003 |
| Cleveland + Thales blend | vs Cleveland alone, n=36 | +67.8% RMSE reduction, p=0.04 |
| **BLS-native CBDF** | Single print (April 2026) | error 4.33 bp vs standalone 7.67 bp |

The Bridged-CBDF earlier (Truflation components → bridged to BLS)
scored +25.6 – 30.6% RMSE vs Stock-Watson DFM, worse than standalone's
+37.6%. Removing the Truflation cross-source noise by using BLS
components directly closed that gap on the April print. Multi-month
walk-forward evaluation pending.

### PCE side

- PCE standalone and PCE-native CBDF: no walk-forward track record yet
  — both built 2026-05-15.
- Composition validation for PCE-native CBDF: residual SD = 0.093 pp,
  median bias +0.011 pp, 84% within 0.1 pp on 183 historical months.
  Cleaner than BLS-native (median +0.078 pp, SD 0.107 pp).

## Architecture differences in one paragraph each

**Thales standalone (CPI) / PCE standalone:** Sees only the headline
series itself. Fits a single AR(1) on monthly log-percentage-changes,
forecasts one month ahead, composes back to YoY via the identity
`YoY[T+1] = YoY[T] + MoM[T+1] − MoM[T+1−12]`. Bands via bootstrap of
AR(1) calibration residuals (24-month window).

**BLS-native / PCE-native CBDF:** Sees official sub-components
separately and forecasts each one's MoM-AR(1) independently. Composes
forecasts via the official (BLS) or empirically-calibrated (PCE)
weights. For PCE the weights are OLS-fitted (Durable 7.8%, Nondurable
23.2%, Services 66.9%, Σ = 0.98) since BEA's chain-type Fisher
aggregation isn't exactly linear; the residual after fitting is
median +0.011 pp / SD 0.093 pp over 183 months — clean enough.
Anchor-offset to actual headline YoY at origin so the forecast lines
up with the latest known print.

**Cleveland Fed:** Proprietary structural model with real-time
high-frequency inputs (daily oil, weekly retail, BLS subindex
publications as they release). Daily-updating point forecast. No bands
exposed. Point-only.

## Why the three differ on the same target

| What it sees | Standalone | Native CBDF | Cleveland Fed |
|---|:---:|:---:|:---:|
| Only the headline series | ✓ | | |
| Sub-component price levels | | ✓ | ✓ |
| Sub-component official weights | | ✓ | ✓ |
| Daily / weekly intra-month data | | | ✓ |
| Density (bands) | ✓ | ✗ (TODO) | ✗ |
| Closed-source? | open | open | closed |

April CPI illustrated this:
- Standalone extrapolated last month's momentum, slightly over-shot (+7.67 bp)
- BLS-native CBDF caught that shelter and food were softening, came closest (−4.33 bp)
- Cleveland under-weighted the late-March WTI gasoline move, missed low (−21.92 bp)

## What's done, what's pending

### Shipped
- BLS standalone + density bands (`forecast_next_bls_cpi.py`)
- BLS-native CBDF, point only (`forecast_next_bls_cpi_blsnative.py`)
- PCE standalone + density bands (`forecast_next_bea_pce.py`)
- PCE-native CBDF, point only (`forecast_next_bea_pce_native.py`)
- Cleveland Fed scraper auto-refresh (CPI + PCE both)
- BLS panel auto-refresh (27 series including 11 components used by CBDF)
- BEA PCE component ingest (3 chain-type price indexes from FRED ALFRED)

### Pending
- **Density bands on CBDF models** (both CPI and PCE). Implementation:
  bootstrap per-component AR(1) residuals → sample paths → compose
  through M2 → quantile bands. ~30 min each.
- **Walk-forward eval for both CBDF models.** Need to score against
  historical prints (n ≥ 24) to get DM-significant track records.
- **`shift(12)` data-quality bug** in legacy scripts (Thales
  standalone). October 2025 BLS data is missing from the vintage
  store; positional `shift(12)` lands on March 2025 instead of April
  2025 for April-2026 YoY, giving wrong values. New BLS-native CBDF
  uses date-based YoY lookup which is robust. Legacy script needs the
  same fix.
- **BLS-native CBDF on PCE** — currently the PCE-native CBDF uses
  3 BEA components. Adding 6-9 more granular sub-components (e.g.,
  Health Care services, Food services, Housing & Utilities) would
  match the 11-component BLS-native richness. Would require ingesting
  more FRED series.
- **Blend** (Cleveland + standalone + CBDF) for both CPI and PCE.

## Reproduce (one-liner)

```bash
cd /Users/kluless/kairos/trufonomics-models && \
  uv run python -m thales.ingest.bls && \
  uv run python -m thales.ingest.fred_alfred --series PCEPI --target && \
  uv run python -m thales.ingest.cleveland_fed && \
  uv run python scripts/ingest_pce_components.py && \
  uv run python scripts/forecast_next_bls_cpi.py && \
  uv run python scripts/forecast_next_bls_cpi_blsnative.py && \
  uv run python scripts/forecast_next_bea_pce.py && \
  uv run python scripts/forecast_next_bea_pce_native.py
```

All four forecasts refresh in ~30 seconds.

## Files

- `scripts/forecast_next_bls_cpi.py` — Thales standalone (CPI)
- `scripts/forecast_next_bls_cpi_blsnative.py` — BLS-native CBDF
- `scripts/forecast_next_bea_pce.py` — PCE standalone
- `scripts/forecast_next_bea_pce_native.py` — PCE-native CBDF
- `scripts/ingest_pce_components.py` — BEA component ingest (one-off)
- `results/next_release_forecast/*.json` — per-day per-target outputs

## Glossary

- **MoM-AR(1) on log-MoM.** AR(1) regression on the monthly
  log-percentage-change of the level. The simplest non-trivial model
  for monthly inflation that captures momentum without overfitting.
- **M2 composition.** Composite_level = Σ_c w_c × level_c (then YoY on
  the composite). The "weighted-sum-of-rebased-levels" approach
  validated in `composition_check.py` at 0.000 pp median residual on
  Truflation.
- **Anchor offset.** Constant additive correction so the forecast at
  origin = actual headline YoY at origin. Removes any composition
  residual from the forecasted YoY without distorting the dynamics.
- **CBDF.** Component-Based Dynamic Factor — composes per-component
  forecasts into a headline forecast via the accounting identity (or
  approximation thereof for PCE Fisher chain-type indexes). After
  O'Keeffe & Petrova 2025 (NY Fed SR 1152).
- **Standalone.** Forecaster that sees only the aggregate target —
  no components, no external data. The minimal baseline.
