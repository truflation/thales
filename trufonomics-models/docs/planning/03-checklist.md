# 03 — Build Checklist

Phased, actionable todo list. Reality-synced 2026-04-25 against what's actually in the repo. Phases 0, 1.1-1.5, and 2.1 have working synthetic-recovery / structural deliverables. Phase 2.2, 3 and per-archetype real-data fits remain.

---

## Phase 0 — Foundation (weeks 1–4, zero component data needed)

Everything in Phase 0 can be done with only the aggregate Truflation data you already have plus public covariates.

### 0.1 Repo scaffolding

- [x] Create repo `trufonomics-models/` with directory structure (src/ layout under `src/thales/`, adapted from the plan's flat layout)
- [x] `uv init`, install core deps: `duckdb`, `fredapi`, `polars`, `pandas`, `pyarrow`, `numpy`, `scipy`, `scikit-learn`, `properscoring`, `requests`, `beautifulsoup4`, `lxml`, `trufnetwork-sdk-py` (via remote wheel). **Deferred:** `jax`, `dynamax`, `numpyro`, `pymc`, `statsmodels`, `cmdstanpy`, `streamlit` — install when the Phase 1 models actually need them
- [x] `CLAUDE.md` + `docs/planning/` committed
- [x] `.env.example` with `FRED_API_KEY`, `BLS_API_KEY`, `EIA_API_KEY`, `TRUF_PRIVATE_KEY`, `TRUFLATION_API_KEY` slots
- [ ] Basic CI (GitHub Actions) — **not set up; 42 tests pass locally in ~4:42**

### 0.2 Vintage data store (Tier 0, critical)

- [x] DuckDB schema `(series_id, reference_date, as_of_date, value, source, source_hash, ingestion_ts)` — primary key `(series_id, reference_date, as_of_date, source)`
- [x] Python API: `VintageStore.get_vintage(series_id, as_of_date) -> pd.Series`
- [x] Python API: `VintageStore.snapshot(as_of=date) -> dict[series_id, Series]`
- [x] Unit tests: point-in-time correctness (`as_of_date ≤ X`)
- [x] Unit tests: revision handling (both vintages retrievable, `revisions()` returns full history)
- [x] Append-only enforcement (INSERT-only; conflicts on same PK with different value raise `ValueError`)
- [x] Source hash logging per ingestion (sha256[:16] of the raw payload)
- [x] Integration live: 333k+ rows, 7 sources, `data/vintage_store/thales.duckdb`

### 0.3 Public data ingest

- [x] FRED ingest — **47 series**, tagged `as_of_date=today` (ALFRED migration is a TODO; see task #72). Covers rates, yield curve, credit spreads, inflation expectations, commodities, FX, labor, housing, GDP, money, credit
- [x] BLS direct ingest — **23 CUSR subindex series** via BLS v2 API (headline, core, sticky, food, energy, shelter, transport, medical, apparel, recreation, edu/comm)
- [x] EIA ingest — **6 series** (WTI, Brent, Henry Hub daily + US + PADD-1 + PADD-5 retail gasoline weekly)
- [ ] NOAA climate (HDD/CDD by region) — **not built**
- [ ] Baltic Dry / SCFI / Drewry shipping costs — **not built**
- [ ] ALFRED-backed vintage handling per-series (ingest currently tags all history with today's as_of) — **TODO**, task #72

### 0.4 Comparator data

- [x] **Cleveland Fed Nowcast** — both the forward HTML scraper AND the full historical JSON backfill. 54k historical rows 2013-08-31 → present + daily forward archive going forward. Email to `public.information@clev.frb.org` not needed — the JSON endpoints `.../webcharts/inflationnowcasting/nowcast_{month,year,quarter}.json` expose the full vintage archive publicly
- [ ] SPF (Philadelphia Fed) — **not ingested**
- [ ] Atlanta Fed GDPNow archive — **not ingested**
- [ ] NY Fed Staff Nowcast archive — **not ingested**
- [ ] Blue Chip — **skipped (paid access)**

### 0.5 Truflation ingest

- [x] API credentials setup — TRUF_PRIVATE_KEY in .env, TN SDK working via subprocess pattern on macOS (augustus venv-bridge Python as worker)
- [x] **80 component streams** ingested (full Truflation CPI taxonomy: 12 categories + 36/37 subcategories + 32/32 components; 1 subcategory missing, id 158 Misc products & services under Other). Daily index-level, 2020-01-01 → 2026-04-16, ~2,300 obs each, 176k rows total
- [ ] Daily *component* ingest cron at 01:00 UTC — **not wired**; currently one-shot ingest. Last refresh 2026-04-16. Forward-fill in the live forecaster covers the gap on the headline number but the attribution table flatlines until refresh. Task: schedule the cron (lightweight once the subprocess pattern is stable on Linux). Separate from the *forecast* cron in §0.13 below
- [x] Vintage metadata stored: `source='truf_network'`, `as_of_date` per pull
- [x] **Weights loader + cross-walk**: `src/thales/weights.py` reads `categories-tables-v1.csv` (2010-2025) and `categories-tables-v2.csv` (2026-) plus `categories-metadata.csv`. 80/80 catalog streams cross-walked to category_ids by name. Top-level 12 weights sum to 100.000%
- [x] **Composition sanity check**: reconstructed frozen headline from 80 components × 12 top-level weights matches published `truflation_us_cpi_frozen_yoy` with median residual 0.000 pp, 94% of days within 0.5 pp (Method 2 aggregate-then-YoY). See `results/composition_check/FINDINGS.md`

### 0.6 Synthetic DGPs (critical before real models)

- [x] `src/thales/synthetic/commodity_passthrough.py` — GBM commodity, drifting β_t, stochastic volatility. Template established
- [ ] `synthetic/rate_sensitive_hierarchical.py` — **not built**
- [ ] `synthetic/uc_sv_ms.py` — **not built**
- [x] `synthetic/vecm_tariff.py` — bivariate cointegrated tradables with tariff regime dummy. Reproducibility + spread-shift sanity checks passing
- [x] `synthetic/bsts_discretionary.py` — local linear trend + dummy seasonal + noise. Centered-seasonal sanity check passing
- [ ] HRNN synthetic DGP — **new, per methodology review 2026-04-24**
- [ ] Calibration to empirical moments — partial; commodity DGP has sensible defaults but not explicitly moment-matched to Truflation gasoline
- [ ] Realistic defects (missing obs, revisions, outliers) — not added

### 0.7 Evaluation harness

- [x] `src/thales/evaluation/metrics.py` — RMSE, MAE, MASE, directional accuracy, CRPS (samples + Gaussian), log_score_gaussian, quantile_loss, PIT, PIT KS p-value, interval coverage, sharpness, Brier, log_loss, roc_auc, bootstrap_ci. 14 tests
- [x] `src/thales/evaluation/tests.py` — Diebold-Mariano (two-sided, one-sided, squared/absolute loss), Clark-West (nested), Giacomini-White (unconditional + conditional with test instruments), KS uniform, Newey-West HAC lag-3. 9 tests
- [x] `src/thales/evaluation/harness.py` — walk-forward simulator. `Forecast` dataclass + `Forecaster` Protocol + `walk_forward` (no-peek slicing) + `attach_actuals` + `score` → `ScoreBlock` (RMSE / MAE / RMSE-vs-naive / 80%/95% coverage / mean width / directional accuracy / CRPS / SHIP gate). 14 tests passing
- [x] `src/thales/evaluation/scoring_db.py` — DuckDB-backed cross-model scoreboard. Schema: `forecasts(model_id, target_series, origin_date, target_date, point, lo80/hi80/lo95/hi95, metadata_json)` + `scoring(... actual, today_baseline, error, hit_80/95, direction_hit)`. Read-side returns frame schema-identical to `harness.attach_actuals` so any code that scores one works on both. 9 tests passing
- [ ] Self-test harness on synthetic DGP — next; commodity DGP recovery via the new harness (closes the integration loop)

### 0.8 Dashboard

- [ ] Streamlit skeleton — **not built**; all results so far are CSV + markdown findings notes
- [ ] Per-model metric tables, fan charts, PIT, reliability, regime-stratified, horizon-stratified, DM matrix — not built

### 0.9 Parameter recovery tests (Tier 1 synthetic)

- [x] `tests/test_commodity_recovery.py` — 8 tests passing (reproducibility, drift bounds, static-OLS bias under TVP, etc.)
- [x] **`tests/test_commodity_archetype.py`** — 9 tests passing on the actual TVP estimator (mean recovery, path Pearson > 0.7, MAE < 0.07, TVP beats OLS, σ_ε / σ_β recovery, determinism). The model recovery gate-1 test.
- [x] **`tests/test_bsts_archetype.py`** — 10 tests passing on the BSTS estimator (trend Pearson > 0.95, seasonal Pearson > 0.7, decomposition R² > 0.90, σ_ε within factor of 2, determinism). Phase 1.2 gate-1 test.
- [x] **`tests/test_vecm_archetype.py`** — 14 tests passing on the VECM estimator (α_1, α_2 within 0.02, θ within 1.5, μ within 1.5, σ_i within 15%, ρ within 0.10, signs correct, determinism). Phase 1.4 gate-1 test.
- [ ] `tests/test_housing_hierarchical_recovery.py` — blocked on DGP
- [ ] `tests/test_ucsv_ms_recovery.py` — blocked on DGP
- [ ] `tests/test_vecm_tariff_recovery.py` — blocked on DGP
- [ ] `tests/test_bsts_discretionary_recovery.py` — blocked on DGP
- [ ] `tests/test_regime_model_recovery.py` — blocked on model
- [ ] `tests/test_transmission_var_recovery.py` — blocked on model
- [ ] `tests/test_cbdf_composition_recovery.py` — blocked on composition layer
- [ ] PPC tests for each Bayesian model — blocked on models

### 0.10 Three baseline nowcasts (end-to-end test of pipeline)

- [x] **Official targets ingested** via ALFRED — `CPIAUCSL`, `CPILFESL`, `PCEPI`, `PCEPILFE` under `source='fred_alfred_target'` (5,494 vintage rows, 193-206 as_of dates per series; revisions properly tracked). Loader `src/thales/targets.py` provides `load_target_yoy`, `load_nowcast_comparator`, `load_panel`
- [x] **Baseline 1 (persistence)** — `PersistenceBaseline` in `src/thales/models/baselines.py`. RMSE on +1m YoY: **0.2424 (CPI), 0.2679 (PCE), 0.1840 (Core PCE)** over 122 origins 2015-12 → 2026-02. The floor every Thales archetype must beat. See `results/baseline_eval/FINDINGS.md`
- [x] **Baseline 2 (AR(1))** — `AR1Baseline` in `src/thales/models/baselines.py`. Loses to persistence by 3-5% RMSE on every target (consistent with Stock-Watson 2007: monthly YoY is near-unit-root, the AR(1) coefficient hurts more than it helps)
- [x] **Baseline 3 (Path A retargeted)** — `PathAForecaster` in `src/thales/models/baselines.py`. 2-feature OLS (persistence + Truflation YoY) → BLS CPI / BEA PCE YoY[T+1]. Beats persistence on **every** target: +9.50% on CPI, +8.66% on Core CPI, +5.18% on PCE, +5.87% on Core PCE. Directional lift even more dramatic on core measures (+20pp on Core CPI, +17pp on Core PCE). Same-month +42% claim doesn't transfer fully to +1m frame, but 5-9% RMSE reduction at +1m is real signal — the 2-feature architecture pays its keep at a frame where most papers report zero skill
- [x] **Cleveland Fed comparator — native h=0 eval** — `scripts/eval_clevfed_native.py` + `clevfed_native_FINDINGS.md`. Cleveland Fed in its native same-month frame: **+54.80% RMSE reduction vs last-release on Headline CPI** (88.8% direction acc), +24% on Core CPI / PCE, but **−24% on Core PCE** (series too smooth for any model to beat persistence). This is the institutional comparator bar. Forward-month scrape (the +1m comparator) deferred to a follow-up
- [x] **Walk-forward harness end-to-end** — runs `persistence_v1`, `ar1_v1`, `clevfed_v1` through `walk_forward` → `attach_actuals` → `score`, persists to DuckDB `results/baseline_eval/scoring.duckdb`. Validates the harness design.
- [x] **CRPS + PIT density scoring plumbed across the model zoo** — `src/thales/evaluation/density.py` (samples helpers + `DensityBlock`); every Forecaster (Persistence, AR(1), PathA, MoM-composed, Stock-Watson DFM, SameMonthBridge, BridgedCBDF) emits `Forecast.samples`; `score()` auto-computes CRPS / PIT KS p-value / density coverage / sharpness from a samples column on the prediction frame. `scripts/density_eval.py` runs the full zoo. Headline: MoM-composed AR(1) wins CRPS (0.152 vs DFM 0.263, +42 %); Bridged-CBDF posts the lowest CRPS overall (0.145) on n=13. See `results/baseline_eval/DENSITY_EVAL_FINDINGS.md`.
- [x] **Split-conformal bands** — `AR1Baseline` and `PathAForecaster` now support `calib_months` parameter. New model_ids `ar1_conformal_v1` / `patha_conformal_v1` use 24-month holdout calibration. Coverage on Core series improves toward nominal (Core CPI 64.2 → 78.8%; Core PCE 62.4 → 70.6%). Point RMSE degrades because shorter training misses the 2022-2024 surge — follow-up is rolling-conformal (sliding calibration window) or shorter fixed window
- [ ] Evaluation dashboard — blocked on 0.8

**Prior Thales v1 (kairos) work** validated the "beat naive" gate separately: Path A nowcast +42% MSE reduction vs persistence on aggregate same-month BLS; Gated v2 +59%. Treat as reference baselines for this repo's work but not a substitute for the formal §0.10 evaluation in the Thales-v2 pipeline.

### 0.11 Pre-registration doc

- [x] **Draft skeleton committed** at `docs/pre-registration/001-initial-nowcast-methodology.md` (2026-04-24) — comparators, windows, metrics, thresholds, ablations, HRNN/TSFM decisions from methodology review
- [ ] Lock date pending `evaluation/harness.py` build — cannot run pre-registered evaluation until the spine exists

### 0.12 Infrastructure — Vast.ai

- [ ] Template image with CUDA 12.x + jax[cuda] + dynamax + NumPyro — **not built**; not needed yet since no MCMC / GPU work has started
- [ ] Deploy script — not built
- [ ] Expected setup when the first archetype SSM fit is ready (Phase 1.1)

### 0.13 Live day-ahead pilot (Stefan-facing track record)

Operational track-record infrastructure for the day-ahead LIVE Truflation YoY forecaster. Internal pilot for ≥7 days before any external publication.

- [x] `scripts/forecast_live_tomorrow.py` — Ridge + split-conformal bands + median bias correction on LIVE Truflation YoY (Feed API target, 12 components from vintage store fwd-filled). Saves JSON to `results/daily_forecast_live/forecast_live_<origin>.json`, logs point to vintage store under `thales_daily_forecast_live`
- [x] `scripts/score_yesterday.py` — pulls Feed API, scores yesterday's prediction against today's published value (signed error, 80% / 95% hit, direction hit), writes idempotent row to `results/daily_forecast_live/scoring.csv`
- [x] `scripts/weekly_rollup.py` — n predictions, MAE, RMSE vs naive, 80%/95% coverage, direction vs base-rate, SHIP/HOLD verdict using same gates as the historical backtest (cov80 ±7pp, cov95 ±4pp)
- [x] `scripts/run_daily.sh` + `scripts/com.thales.dailyforecast.plist` — launchd agent runs scoring then forecasting daily at 09:00 local (`RunAtLoad=true` to catch up on missed wake). Logs to `~/Library/Logs/thales/forecast.{out,err}.log`
- [x] **Day 0 logged: origin=2026-04-24 → target=2026-04-25 forecast 1.7507% ↓** (today=1.7597%, 80% band [1.7215, 1.7786]). Backtest at origin: 80% cov 76.7%, 95% cov 92.2%, MAE 0.0411 pp, direction 57.8% vs base-rate-up 56.7% — SHIP gate passed
- [ ] First scored row (target=2026-04-25) — **pending** Feed API publishing Apr 25 actual; tomorrow's 09:00 cron writes it
- [ ] First weekly rollup verdict — **pending** ~2026-05-01 (7 scored rows)
- [ ] Promote scoring CSV → DuckDB (`results/daily_forecast_live/scoring.duckdb`) when a second model variant lands and we want one queryable scoreboard
- [ ] Migrate cron from laptop launchd to a $5/mo VPS or Mac mini once the pilot extends past 30 days or starts missing wake-cycles

---

## Phase 1 — Archetypes (weeks 5–16, requires component data)

Sequence chosen to climb difficulty gradually and unlock highest-impact models fastest.

### 1.1 Commodity pass-through (easiest)
- [x] **TVP estimation core (Kalman + RTS, no SV yet)** in pure numpy — `src/thales/models/archetypes/commodity.py`. Recovers true β path on synthetic DGP with Pearson **0.999**, MAE **0.041**, vs static OLS MAE 0.80. **+94.9% improvement.** 9/9 recovery tests passing. See `results/archetype_recovery/commodity_recovery_FINDINGS.md`. **This is gate-1 evidence for the architecture.**
- [ ] Layer SV onto ε_t (NumPyro / MCMC) — Phase 1.2 work
- [ ] Add VECM cointegration layer — Phase 1.2 work
- [x] **Real-data fit on Truflation Utilities × Henry Hub natural gas** — `scripts/tvp_utilities_henryhub.py`. β evolves coherently from −0.26 (COVID decoupling, 2020) → +0.10 (normal regime, 2024-26). Static OLS gives β=0.043 (time-average, hides the sign flip). TVP recovers the dynamics cleanly. Largest 60-day β shift in Sep 2020 = post-COVID re-coupling. **First real-data archetype validation passes.** See `results/real_data_archetypes/FINDINGS.md`
- [ ] Extend to fuel portion of Transportation
- [ ] Extend to food-at-home portion of Food
- [ ] Full evaluation vs AR(1), ARMA-GARCH, naive persistence
- [ ] Ablations: constant β vs TVP β, SV vs homoskedastic
- [ ] Ship to dashboard

### 1.2 BSTS discretionary
- [x] **BSTS estimation core** in pure numpy (multivariate Kalman + RTS, K=13 state for monthly seasonal). Local linear trend + dummy seasonal + irregular. **Trend Pearson 0.9999, seasonal Pearson 0.9965, decomposition R² 0.9997** on synthetic. 14/14 recovery tests passing. `src/thales/models/archetypes/bsts.py`. See `results/archetype_recovery/bsts_recovery_FINDINGS.md`. **Identifiability flag documented + local-level variant added** + **empirical resolution via real-CPI experiment**: LLT clearly preferred for level series (Δ log-lik +44 to +101), LL clearly preferred for YoY series (Δ log-lik mostly < 1, AIC favors LL). Per-transform production rule established. Both variants validated.
- [x] **Fit on Recreation & culture (real data)** — `scripts/bsts_recreation_culture.py`. Both LLT-on-level (R² 1.0000) and LL-on-YoY (R² 0.9625) variants applied. Per-transform rule confirmed empirically: level seasonal amplitude 3× larger than YoY. σ_seasonal collapsed to zero in both — captured constant-amplitude yearly cycle (correct for Recreation). Trend says Recreation YoY ran 5.6% in 2021 → 1.1% by 2026. See `results/real_data_archetypes/bsts_recreation_FINDINGS.md`
- [x] **Extend to Food-away (real data)** — trend captured surge from 9.1% → 3.2% YoY. Unusually high seasonal amplitude (6.37 pp) attributed to base-year effects from COVID-era restaurant pricing. R² = 1.000.
- [x] **Extend to Other category (real data)** — modest disinflation 1.6% → 0.4% YoY, clean residual noise σ_ε = 0.37. R² = 0.99.
- [x] **Tier-1 first cut — same-month nowcast frame eval** — `scripts/gate2_same_month_nowcast.py`. **Thales bridge (α + β·BLS_lag1 + γ·truf_yoy) achieves +32.56% RMSE reduction vs last-release on BLS Headline CPI YoY** at h=0 frame, n=115 origins (2016-08 → 2026-03). Direction 72.2% vs 54.8% base-rate (+17pp lift). Cleveland Fed leads by ~22pp on RMSE reduction. **First gate-2 institutional-grade result.** See `results/baseline_eval/GATE_2_SAME_MONTH_FINDINGS.md`
- [ ] Ship to dashboard

### 1.3 UC-SV-MS sticky services
- [x] **Markov-switching variance core (MS layer)** in pure numpy — Hamilton 1989 forward filter + Kim 1994 backward smoother. Recovers σ_low (+2.4%), σ_high (-6%), p_00 (within 0.04pp), p_11 (within 2.5pp). Smoothed regime classification 94.8% accuracy vs 78.6% base rate (+16pp lift). 13/13 tests passing. `src/thales/models/archetypes/regime_switching.py`. See `results/archetype_recovery/regime_switching_phase_1_3_FINDINGS.md`
- [x] **UC + MS layer** — continuous trend state + Kim 1994 collapsing trick (4 → 2 branches per step). Multi-start MLE (5-7 restarts) to escape local optima. σ_η within factor 2, σ_low/σ_high within 30%, level Pearson > 0.85, regime classification > 80%. 11/11 tests passing.
- [x] **SV layer** — stochastic volatility via NumPyro NUTS (Kim-Shephard 1998 spec, non-centered parameterization). Recovers μ_h, φ, σ_h within tolerance, 0 divergences, h-path Pearson 0.78, 90% band coverage 94.4%. 10/10 tests passing (marked slow). `src/thales/models/archetypes/sv.py`. NumPyro + JAX 0.4 + jaxlib pinned in `pyproject.toml`
- [x] **Full UC + MS + SV composed model** — single NumPyro model with all three latent processes coexisting. Discrete regime marginalized via Hamilton forward in log-space (NUTS samples only continuous). Smoothed regime probabilities reconstructed via Kim smoother on posterior-mean params. **11/11 recovery tests passing** in 5:53 (CPU): σ_low/σ_high recovered within factor 2, level Pearson > 0.6, h-path Pearson > 0.3, regime classification ≥ base rate, < 10% NUTS divergences. `src/thales/models/archetypes/uc_sv_ms.py`
- [x] **Pure MS on Health (real data)** — σ_low=0.49, σ_high=3.48 (7.2× contrast), high-vol 50% of months. Largest variance contrast among sticky services
- [x] **Pure MS on Education (real data)** — σ_low=1.02, σ_high=3.13 (3.1× contrast), high-vol only 20.3% of months — calmest sticky-services category
- [x] **Pure MS on Communications (real data)** — σ_low=0.39, σ_high=2.33 (6.0× contrast), high-vol 53% of months — turbulence consistent with cellular-plan restructuring era
- [x] **Pure MS on Alcohol & Tobacco (real data)** — σ_low=0.77, σ_high=2.59 (3.4× contrast), high-vol 35.9% of months
- [ ] Validate regime detection on known repricing windows (Q4 health insurance, fall education, tobacco tax changes) — has data; needs window-by-window narrative analysis
- [ ] Ablation: with vs without regime-switching
- [ ] Ship

### 1.4 VECM tradables
- [x] **VECM core** in pure numpy (per-equation OLS with known β=(1,-1) and tariff dummy) — `src/thales/models/archetypes/vecm.py`. Recovers α_1, α_2, μ, θ, σ_1, σ_2, ρ within 10% on synthetic. 14/14 tests passing. See `results/archetype_recovery/vecm_recovery_FINDINGS.md`. **Tariff regime dummy validated** as a separable parameter
- [x] Johansen's procedure for unknown β (statsmodels) — `JohansenGatedVECM` shipped (Fix #4): per-origin Johansen trace test gates VECM vs three fallback methods (ARDL/bridge/AR1). Real-data finding: gate fires VECM 100% on theory-cointegrated pairs (Truflation Clothing × BLS Apparel CPI); only ~30% on borderline pairs. Bridge fallback is dangerously wrong on spurious cointegration; AR(1) is the safest. See `results/real_data_archetypes/JOHANSEN_GATED_VECM_FINDINGS.md`
- [ ] Bayesian VECM with Minnesota priors in NumPyro — Phase 1.4+ work
- [x] **Fit on Clothing & footwear (real data)** — `scripts/vecm_clothing_real.py`. Truflation Clothing × BLS Apparel CPI, n=74 monthly, 2020-01 → 2026-03. Estimator converges cleanly, produces interpretable coefficients (residual correlation ρ=0.50, sensible). **Honest finding**: assumed β=(1,-1) cointegrating vector may not exactly hold for this pair — both α_1 and α_2 negative (non-standard error-correction sign pattern). Johansen's procedure for β estimation flagged as Phase 1.4 follow-up. Tariff dummy captures small +0.02 spread shift with correct sign post-April-2025. See `results/real_data_archetypes/vecm_clothing_FINDINGS.md`
- [ ] Extend to Housing durables, imported Food (after Johansen's β estimation lands)
- [x] **Tariff regime dummy validated against known April 2025 shifts (real data)** — small but visible θ ≈ +0.02 with correct sign; magnitude limited by 11-month post-tariff window
- [ ] Ship

### 1.5 Hierarchical housing SSM (flagship, hardest)
- [ ] Implement in dynamax with JAX
- [ ] National affordability factor + regional idiosyncratic + rate environment
- [ ] Owned block (mortgage-rate-driven) + rented block (Zillow/Trulia)
- [ ] Mariano-Murasawa for mixed frequencies
- [ ] Two-step estimation (DGR 2011) for initial values, then EM
- [ ] Validate against CoreLogic and Case-Shiller backcast
- [ ] GPU required — A100 on Vast.ai for final fits
- [ ] Ship — this is the model you show investors

---

## Phase 2 — Composition and regime (weeks 17–22)

### 2.1 CBDF composition
- [x] **2.1a — Weighted composition core** with accounting-identity-respecting weighted sum + Monte Carlo bands. `WeightedComposer` in `src/thales/models/composition/weighted.py`. Component attribution + per-component contributions in metadata. 12/12 tests passing including a real-Truflation-weights smoke test
- [x] **2.1b — CBDF cross-component correlation** via fitted multivariate-Gaussian residual covariance. `CBDFComposer` in `src/thales/models/composition/cbdf.py`. Captures O'Keeffe-Petrova-style cross-component dependence (fuel shock hits utilities + food + transport together). Shrinkage to diagonal for short panels (n < 2× n_comp) + PSD safeguard. Falls back to independent draws if covariance not fit. 9/9 tests passing including positive/negative correlation band-width effects
- [x] **End-to-end demo on real Truflation data** — `scripts/demo_phase_2_1_end_to_end.py`. 12 top-level component series → CBDFComposer with real 2026 v2 weights → composed headline forecast. Composition residual median -0.014 pp vs direct headline persistence (within composition-check FINDINGS tolerance). Validates the full Phase 2.1 pipeline end-to-end on real data
- [ ] Joint estimation via EM (optional, start with sequential) — Phase 2.1c, after first real-data composition
- [ ] Full headline BLS CPI and BEA PCE nowcast with density — needs real-data per-component archetype fits + composition wired through harness
- [ ] Multi-horizon: -15 to +12
- [ ] Full evaluation vs Cleveland Fed, SPF, Bloomberg
- [x] Ablation: CBDF vs standard DFM (the O'Keeffe-Petrova improvement claim) — `scripts/okeefe_headtohead.py` + `results/baseline_eval/OKEEFE_HEADTOHEAD_FINDINGS.md`. Headline result: **MoM-composed AR(1) beats Stock-Watson DFM by +37.6% RMSE on BLS Headline CPI YoY (n=25 OOS, DM p=0.0003)**. Operational claim: **Cleveland Fed + Thales beats Cleveland Fed alone by +67.8% RMSE (n=36, DM p=0.04)**. Direct CBDF on inflation does NOT beat DFM (−74% to −80%) because Truflation 12-component weighted sum ≠ BLS headline (structural ~50bp gap). **Bridged CBDF** (CBDF nowcast → rolling-OLS BLS bridge) **beats DFM by +25.6% / +30.6%, p<0.0001 on n=11**. MoM-composed direct-target still wins on the overlap (DM p=0.0002), but Bridged-CBDF is competitive and may dominate at longer horizons / scenario forecasts.
- [x] **Bridged-CBDF wired as a Forecaster class** — `src/thales/models/composition/bridged_cbdf.py` (rolling-OLS bridge with rolling-conformal residuals + Gaussian fallback, prediction-for-next-period contract on `inner_pred_col`). 6 unit tests including coefficient recovery on a synthetic bridge DGP. Closes the architecture-doc queued item. The class is generic — bridges any Truflation-scale signal to BLS, not just CBDF.
- [ ] Regime-stratified DM tests
- [ ] Ship — flagship public product

### 2.2 Regime model
- [x] **First production application — UC+SV+MS on real BLS Headline CPI YoY** — `scripts/regime_on_headline_cpi.py`. **Honest finding documented**: the UC layer absorbs all variance, regime mechanism stays dormant. Synthetic recovery passed for matching DGP; real-data application reveals architecture mismatch — YoY is already differenced, no genuine level walk for UC to identify. Three latents compete for one observed series; MCMC picks the level-dominant mode. See `results/regime/FINDINGS.md`. **Lesson**: every archetype must be re-validated on real target before claiming production-ready
- [x] **Phase 2.2b — MS+SV combined model built and tested.** `src/thales/models/archetypes/ms_sv.py`. 11/11 synthetic recovery tests passing (slow). **Real-data finding**: even MS+SV is over-parameterized for monthly CPI YoY — SV layer absorbs variance instead of UC, P(high) max 0.10 on 2022 surge. See `results/regime/PHASE_2_2_RESOLUTION.md`
- [x] **Phase 2.2 resolved — Pure MS (Hamilton 2-state) is the correctly-specified regime model for monthly YoY**. P(high) = 1.00 across 2021-04 → 2023-05 (entire post-COVID surge). 2014-12 → 2015-12 also correctly flagged (oil collapse era). Production rule documented: pure MS for short mean-reverting targets, UC+MS for trending levels, SV+MS for long volatile series. `fit_hamilton_2state` ships as the regime detector
- [x] **Pure MS applied to all 4 official targets** (CPI, Core CPI, PCE, Core PCE YoY). Cross-target coherence excellent: all flag post-COVID surge (P=0.81-1.00); only Headline measures flag the oil collapse (correct — Core excludes energy); **Core CPI and Core PCE are STILL in high-vol regime today (60 months and counting)** even though Headline returned to low-vol — genuine economic finding about persistent core inflation volatility. See `results/regime/PURE_MS_ALL_TARGETS_FINDINGS.md`
- [x] **Phase 2.2c — UC+SV+MS on CPI level** (CPIAUCSL index) — `scripts/uc_sv_ms_on_cpi_level.py`. **UC layer now works correctly**: level Pearson 1.000 with raw, σ_eta=0.42/month meaningful. σ_low=0.043, σ_high=0.180 (4.2× contrast). MS becomes selective — flags only Nov-Dec 2021 (sharpest level acceleration period). **Production rule confirmed**: UC+SV+MS for trending level series; pure MS for already-differenced YoY
- [x] **Price-change frequency classification (Bils-Klenow methodology)** — `scripts/sticky_flex_regime.py`. Canonical approximation: sticky ≈ Core CPI YoY (CPILFESL); flex ≈ Headline − Core (food + energy residual). Sticky σ_high/σ_low = 9.3× contrast; flex σ_high/σ_low = 3.9× contrast.
- [x] **Sticky trend and flex trend latent states** — smoothed YoY + P(high) regime probabilities for both buckets emitted as monthly time series
- [x] **Regime probability feed** — daily-updatable feed of (P_sticky_high, P_flex_high, sticky-flex gap). At 2026-03-31: sticky=0.997 ON, flex=0.059 OFF — **services-driven regime** (matches Fed's "core stickiness" framing)
- [x] **Event study validation against known regime shifts** — Sticky: 2011-Q1 + 2021-03 → 2026-03 continuous (60mo). Flex: 2011-01 → 2012-02 (energy reflation), 2014-12 → 2016-08 (oil collapse), 2021-03 → 2023-08 (post-COVID). Cross-bucket: 18.7% both high, 16.5% services-only, 17.0% energy-only, 47.8% calm
- [ ] Economic value backtest (TIPS breakeven strategy) — Phase 3 / requires TIPS price data
- [ ] Ship as separate API product — "VIX for inflation" — model layer ready, API/billing infra is separate work

---

## Phase 3 — Transmission and multi-country (weeks 23+)

### 3.1 First transmission VAR — logistics
- [x] Industry cost structure database — `src/thales/cost_structures.py` registry, ATRI-aligned weights for logistics + restaurants + 4 Phase 3.3 verticals
- [x] Bayesian VAR with Minnesota prior — `src/thales/models/archetypes/bvar_minnesota.py` (closed-form BGR posterior, no MCMC). 19 unit tests including synthetic recovery, Cholesky IRF / FEVD invariants, conditional forecast + shock scenario
- [x] Endogenous: 5-var (fuel + labor + maintenance + freight + volume) — insurance/margin paywalled, deferred to Phase 3.1.x
- [x] Cholesky identification, category shocks first — diesel-shock IRF: 8% pass-through to freight rates at every horizon, persistent
- [x] IRF and FEVD computation — `cholesky_irf()`, `fevd()` shipped
- [x] Conditional forecasts — `conditional_forecast()` (DLS-style) + `shock_scenario()` (IRF-driven, the right tool for hedging-decision support since it engages contemporaneous Σ-correlation channel)
- [x] Economic value backtest (fuel-hedging strategy) — internal validation only. v2 with β-optimal hedge ratio: static DBO −15.5% σ-reduction (within 3pp of theoretical R² ceiling 18.5%). BVAR-modulated dynamic strategies don't beat static at monthly cadence. **Conclusion: monthly vol signal not sharp enough to ship as a feature; not customer-facing.** See `results/real_data_archetypes/FUEL_HEDGE_BACKTEST_FINDINGS.md`
- [x] Client-facing API contract: input cost structure → get forward P&L path — exposure-analytics output (cost-line $ + IRF + FEVD), framed per `docs/architecture/02-product-boundary-no-advice.md` (no advice / no recommendations). For $100M shipper: +20% diesel = +$5.3M total cost-line exposure (6.27% of opex)

### 3.2 Restaurants transmission VAR
- [x] Cost structure: food COGS 30%, labor 30%, rent 8%, utilities 4%, other 28% — registered in `cost_structures.py`
- [x] Same structure as logistics, different endogenous vector — 6-var BVAR(1) on **MoM log-differences** (not log-levels — Σ-correlation amplification was inflating IRFs in the level frame; same lesson as Fix #5)
- [x] Customer scenario tool — `scripts/bvar_restaurants_6var.py` produces dollar P&L exposure for food-shock scenarios, with explicit "co-movement not causation" framing for output-side variables
- [ ] Menu-pricing timing backtest — deferred (similar caveat as 3.1e: monthly cadence too noisy for actionable claims)

### 3.3 Additional verticals (prioritized)
- [x] Mid-market retail — 5-var BVAR; max\|eig\|=0.516 STABLE; mom_sales +46.57% RMSE reduction; +20pp wholesale shock = +$976k cost exposure ($10M shipper)
- [x] Healthcare operators — 4-var BVAR; max\|eig\|=0.538 STABLE; mom_pharma +26.88% RMSE reduction; +20pp pharma shock = +$262k
- [x] Real estate operators — 5-var BVAR; max\|eig\|=0.844 STABLE; mom_labor +26.62% RMSE reduction; +20pp construction-materials shock = +$963k
- [x] Manufacturing verticals — 5-var BVAR; max\|eig\|=0.546 STABLE; mom_logistics +27.48% RMSE reduction; +20pp raw-materials shock = +$1.72M
- [x] **Cross-vertical pattern (6 verticals total)**: BVAR reliably beats RW on cost-side variables (+15-30%), weak/harmful on demand/output-side (manufacturing IP −19%, restaurant traffic −24%). Clean product story: "we forecast YOUR cost lines, you forecast YOUR demand."
- See `results/real_data_archetypes/BVAR_PHASE33_FINDINGS.md`

### 3.4 Multi-country
- [ ] UK model replication (category weights, data sources localized)
- [ ] Additional countries as Truflation expands coverage
- [ ] Global regime synthesis

### 3.5 Research output
- [ ] Fed-grade working paper on CBDF applied to daily component data
- [ ] Submission to journals (IJF, JBES)
- [ ] Conference presentations (NBER, SED)

---

## Week-by-week quick view (first 8 weeks)

| Week | Focus | Deliverable |
| --- | --- | --- |
| 1 | Repo scaffold, vintage store, FRED + BLS ingest | Working `get_vintage()` API with tests |
| 2 | Cleveland Fed scrape, comparator ingest, Truflation API | Cleveland Fed history archived, daily scrape running |
| 3 | Synthetic DGPs + parameter recovery tests | All 8 recovery tests passing in CI |
| 4 | Evaluation harness + metrics + dashboard v0 | Baseline nowcasts scored vs Cleveland Fed |
| 5 | Pre-registration doc + Vast.ai setup | Committed pre-reg, GPU template |
| 6–7 | Commodity pass-through archetype | Utilities category nowcast live |
| 8 | BSTS discretionary archetype | Recreation nowcast live, two archetypes shipped |

Stop and course-correct if:
- Week 2 you can't get Cleveland Fed data. Pivot to daily scrape forward-only.
- Week 4 baselines don't beat naive. Pipeline is broken — debug before continuing.
- Week 6 vintage discipline feels burdensome. Lock it in — the alternative is a year of fake backtests.
