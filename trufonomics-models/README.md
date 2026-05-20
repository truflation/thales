# Thales

Truflation's foundational econometric model stack for inflation nowcasting and forecasting. Composes official BLS CPI and BEA PCE releases from component-level data, attaches calibrated uncertainty bands, and combines model outputs into committee point forecasts.

## What ships in production

- **Headline CPI committee** — three forecasters of the next BLS All-Items CPI release (`scripts/cpi_committee.py`).
- **Headline PCE committee** — three forecasters of the next BEA PCEPI release (`scripts/pce_committee.py`).
- **Core CPI forecasters** — standalone, BLS-native CBDF v2, and Truflation-weighted variants (`scripts/forecast_next_bls_core_cpi*.py`).
- **97-origin walk-forward backtest** of the CPI committee against persistence (`scripts/backtest_cpi_committee.py`): +34.1% RMSE reduction vs persistence, DM p = 0.0095, 80% band empirical coverage 84.5%.
- **Daily Truflation YoY committee** for the daily CPI and PCE indexes (`scripts/daily_committee.py`).

Methodology is documented in `docs/products/` and in the Obsidian vault under `15 - trufonomics/thales/` (Thales — CPI Forecasting Methodology, Thales — PCE Forecasting Methodology).

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/truflation/thales.git
cd thales/trufonomics-models
uv sync                                 # install deps from pyproject.toml
cp .env.example .env                    # fill in API keys
```

`.env` keys required for the production forecasters:

| Variable                | Source                                            | Used by                          |
|-------------------------|---------------------------------------------------|----------------------------------|
| `FRED_API_KEY`          | https://fred.stlouisfed.org/docs/api/api_key.html | FRED + ALFRED vintage ingest     |
| `BLS_API_KEY`           | https://data.bls.gov/registrationEngine/          | BLS subindex panel               |
| `EIA_API_KEY`           | https://www.eia.gov/opendata/                     | Energy covariates                |
| `TRUFLATION_API_KEY`    | Truflation enterprise                             | Daily Truflation stream ingest   |

## Reproducing the production forecasts

The vintage store lives at `data/vintage_store/thales.duckdb` and is the single source of truth for every forecaster. It is **gitignored** because it's a binary DuckDB file. The steps below populate it from scratch — total runtime is ~5 minutes for the data pulls.

### 1. Populate the vintage store

Run these from the `trufonomics-models/` directory. Each is idempotent — re-running just appends new vintage rows.

```bash
# 1a. Official target series — CPIAUCSL, CPILFESL, PCEPI, PCEPILFE (ALFRED vintage)
uv run python -m thales.ingest.fred_alfred --targets

# 1b. BLS subindex panel — 35 series including headline + 11 components + core decomposition
uv run python -m thales.ingest.bls

# 1c. Three BEA chain-type PCE components — durables, non-durables, services
uv run python scripts/ingest_pce_components.py

# 1d. Cleveland Fed nowcast (CPI + PCE) — used as benchmark + blend partner
uv run python -m thales.ingest.cleveland_fed
```

The Truflation category weights (`data/truflation/weights/categories-tables-v2.csv`) and stream catalogue (`data/truflation/streams_catalog.csv`) are committed to the repo, so no separate download is needed for them.

### 2. Run the CPI forecasters

```bash
# Three CPI forecasters at the next BLS release
uv run python scripts/forecast_next_bls_cpi.py              # Thales standalone (MoM-AR(1) on CPIAUCSL)
uv run python scripts/forecast_next_bls_cpi_blsnative.py    # BLS-native CBDF (11 subindexes + BLS weights)
uv run python scripts/forecast_next_bls_cpi_trufweights.py  # Truflation-weighted CBDF (11 subindexes + Truflation weights)

# Combine into canonical committee — writes results/next_release_forecast/committee_cpi_*.json
uv run python scripts/cpi_committee.py
```

Output: a JSON with each individual forecaster's point + 80%/95% bands, the committee average, and a stamped `as_of_date`.

### 3. Run the PCE forecasters

```bash
uv run python scripts/forecast_next_bea_pce.py              # PCE standalone (MoM-AR(1) on PCEPI)
uv run python scripts/forecast_next_bea_pce_native.py       # PCE-native CBDF (3 BEA chain indexes + OLS weights)
uv run python scripts/forecast_next_bea_pce_trufweights.py  # Truflation-weighted CBDF (11 BLS subindexes + Truflation PCE weights, auto-switches to nowcast mode when April BLS data is available)

# Combine — writes results/next_release_forecast/committee_pce_*.json
uv run python scripts/pce_committee.py
```

### 4. Run the Core CPI forecasters

```bash
uv run python scripts/forecast_next_bls_core_cpi.py             # standalone MoM-AR(1) on CPILFESL
uv run python scripts/forecast_next_bls_core_cpi_blsnative.py   # BLS-native CBDF v2 (SACL1E + SASLE, OLS-fit)
uv run python scripts/forecast_next_bls_core_cpi_trufweights.py # Truflation-weighted variant
```

(No committee script for Core CPI yet — the three are produced and reported individually.)

### 5. Run the backtest

```bash
uv run python scripts/backtest_cpi_committee.py                  # 2018-01 → latest, 97 origins
# Optional narrower window:
uv run python scripts/backtest_cpi_committee.py --start 2020-01-31
```

Output: `results/next_release_forecast/backtest_cpi_committee.csv` with per-origin errors and aggregate stats printed to stdout (RMSE, MAE, DM test, 80% band coverage).

## Repo layout

```
trufonomics-models/
├── src/thales/
│   ├── ingest/                     # FRED, ALFRED, BLS, Cleveland Fed, EIA, Truflation ingest modules
│   ├── vintage/                    # Point-in-time DuckDB vintage store
│   ├── models/                     # Forecaster classes (baselines, DFM, MoM-composed, LSTM, …)
│   ├── synthetic/                  # Synthetic DGPs for archetype recovery tests
│   ├── evaluation/                 # Metrics, harness, scoring DB, density evaluation
│   ├── targets.py                  # Date-aware YoY lookup helpers
│   └── weights.py                  # Truflation weights loader + BLS cross-walk
│
├── scripts/                        # Production forecast + backtest entry points
│   ├── forecast_next_bls_cpi*.py   # CPI forecasters (§2)
│   ├── forecast_next_bea_pce*.py   # PCE forecasters (§3)
│   ├── forecast_next_bls_core_cpi*.py  # Core CPI forecasters (§4)
│   ├── cpi_committee.py            # Canonical CPI committee
│   ├── pce_committee.py            # Canonical PCE committee
│   ├── backtest_cpi_committee.py   # Walk-forward CPI backtest
│   ├── regime_transition_probe.py  # Composite transition score
│   ├── daily_committee.py          # Daily Truflation YoY committee
│   └── ingest_pce_components.py    # One-off BEA PCE chain indexes ingest
│
├── data/
│   ├── vintage_store/              # DuckDB store (gitignored — populated by ingest)
│   ├── truflation/weights/         # Category weight CSVs (tracked)
│   └── truflation/streams_catalog.csv
│
├── results/
│   ├── next_release_forecast/      # Production forecasts + backtest outputs
│   ├── baseline_eval/              # Earlier comparator runs
│   ├── daily_forecast_live/        # Daily live forecast track record
│   └── …
│
├── docs/
│   ├── products/                   # Per-tier product specs
│   ├── planning/                   # Architecture, evaluation framework, checklist
│   ├── pre-registration/           # Dated commit-before-eval docs
│   └── architecture/
│
└── tests/                          # Unit + statistical regression tests
```

## Architecture (one-screen summary)

**Five archetype models** under the hood — one per category-level generating process:

1. **Commodity pass-through** — TVP-VECM with stochastic volatility (Utilities, fuel, food-at-home)
2. **Rate-sensitive durables** — hierarchical DFM-SSM (Housing: owned + rented split)
3. **Sticky administered services** — UC + SV + Markov-switching (Health, Education, Communications, Alcohol/Tobacco)
4. **Import-exposed tradables** — VECM with tariff regime dummy (Clothing, durables)
5. **Discretionary demand-cycle** — BSTS with dual seasonality (Recreation, food-away, Other)

**Composition layer** — Component-Based Dynamic Factor (CBDF; O'Keeffe & Petrova 2025, NY Fed SR 1152) respects the accounting identity and composes archetype outputs into headline nowcasts with density.

**Regime layer** — UC-SV-MS on headline + sticky/flexible decomposition (Bils-Klenow methodology).

**Transmission layer** — Bayesian VARs with Minnesota priors, one per industry vertical.

Details: `docs/planning/01-architecture.md`.

## Evaluation philosophy

- **Vintage discipline** — every ingest is point-in-time, every backtest walks forward. The vintage store is Tier 0.
- **Three evaluation tiers** — synthetic recovery → historical pseudo-real-time walk-forward → live production monitoring.
- **Regime-stratified always** — full-sample averages hide everything. Report by regime (stable, COVID, surge, disinflation, tariff, shutdowns).
- **Density over point** — CRPS, PIT, calibration are primary. Institutional buyers benchmark on density.
- **Pre-registration** — comparators, windows, metrics, significance thresholds committed *before* running evaluations. See `docs/pre-registration/`.
- **DM and GW tests mandatory** — claimed improvements without significance tests are noise.

## Benchmarks tracked

- **Primary:** Cleveland Fed Inflation Nowcasting Model (https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting)
- **Secondary:** Survey of Professional Forecasters (Philadelphia Fed)
- **Tertiary:** Bloomberg / Blue Chip consensus, Dallas Fed trimmed mean PCE, Atlanta Fed sticky-price CPI
- **Internal baseline:** Path A nowcast from the earlier kairos work (+42% MSE vs persistence on aggregate same-month task)

## Relationship to kairos

This subproject sits inside the `kairos/` workspace as a deliberately-isolated stack. Kairos hosts the older Heimdall / TSFM track and is referenced only as baseline numbers to beat. No cross-imports between the two; tech stacks are incompatible by design.
