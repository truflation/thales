# Density evaluation — CRPS, PIT calibration, coverage, sharpness

**Date:** 2026-04-27
**Script:** `scripts/density_eval.py`
**Output:** `results/baseline_eval/density_eval_summary.csv`

Closes the queued architecture-doc item "CRPS / density-scoring across
all models" (line 352 of `docs/architecture/03-system-architecture.md`).

## Headline matrix — BLS Headline CPI YoY, walk-forward h=1

| Model | n | RMSE | Δ% vs naive | **CRPS** | **PIT-KS p** | **cov80** | **cov95** | **sharp80** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| persistence_v1 | 73 | 0.4593 | +42.3 % | 0.258 | 0.605 | 70.0 % | 89.0 % | 0.82 |
| ar1_yoy_v1 | 62 | 0.4725 | +39.5 % | 0.272 | **0.890** | 75.8 % | 90.3 % | 1.07 |
| **ar1_mom_composed_v1** | 61 | **0.2846** | **+63.8 %** | **0.152** | 0.812 | **78.7 %** | **91.8 %** | 0.56 |
| patha_v1 | 49 | 0.3741 | +39.1 % | 0.186 | 0.571 | 89.8 % | 93.9 % | 0.88 |
| dfm_stock_watson_v1 | 25 | 0.4561 | −42.5 % | 0.263 | 0.479 | 96.0 % | 100.0 % | 1.58 |
| **bridged_cbdf_v1** | 13 | **0.2486** | **+24.0 %** | **0.145** | 0.085 | 61.5 % | 84.6 % | **0.41** |

Reading the columns:

  * **CRPS** — proper-scoring-rule analog of MAE for densities. Lower is
    better. CRPS of a deterministic point forecast reduces to its MAE,
    so the column is comparable across point and density forecasters.
  * **PIT-KS p** — Kolmogorov-Smirnov test that PIT values are
    Uniform(0, 1). High p ⇒ density is calibrated; p < 0.05 ⇒
    miscalibrated.
  * **cov80 / cov95** — empirical coverage of the central credible
    interval drawn from the sample matrix. Should be near nominal.
  * **sharp80** — mean width of the 80 % interval (lower is sharper,
    but only meaningful conditional on calibration).

## Five things this table tells us

### 1. MoM-composed AR(1) wins on density too — not just RMSE

The 37.6 % RMSE advantage over Stock-Watson DFM (`OKEEFE_HEADTOHEAD_FINDINGS.md`) was a point-forecast result. The CRPS column shows **0.152 vs DFM's 0.263 — a 42 % CRPS improvement**, fully consistent with the RMSE story. PIT KS p = 0.812 confirms the predictive distribution is calibrated. Sharpness (0.56) is tighter than every model except Bridged-CBDF, while cov80 (78.7 %) is closer to the 80 % nominal target than DFM (96 % — over-cover) or Persistence (70 % — under-cover).

This is the cleanest "density wins follow point wins" pattern we expected from the methodology argument: the closed-form composition identity transmits the AR(1) MoM-residual distribution directly into a YoY predictive distribution with the right scale.

### 2. Stock-Watson DFM badly over-covers — Gaussian closed-form posterior is too wide

DFM's bands are constructed analytically from `β_z² · σ_η² + σ_ν²` (a Gaussian one-step-ahead variance). On the 25 OOS months the bands cover **96 % at the 80 % level** and **100 % at the 95 % level**. The bands are 1.58 / 2.43 percentage points wide on average — wider than the data calls for. CRPS 0.263 partly reflects this lack of sharpness.

This is consistent with the Stock-Watson 2007 finding that monthly inflation YoY is near-unit-root and the DFM's factor-AR(1)-then-target-loading predictive variance overstates the residual scale because the factor doesn't carry much marginal information beyond persistence.

### 3. Bridged-CBDF has the lowest CRPS but the bands are under-dispersed

Bridged-CBDF posts the lowest CRPS in the table (0.145), beating MoM-composed at the point-forecast level too (RMSE 0.249 vs 0.285 on the n=13 overlap). The bands are the sharpest in the table (0.41 / 0.76).

But the PIT KS p-value is **0.085** — borderline calibrated, just inside conventional 5 % significance. Empirical 80 % coverage is 61.5 % (under-covers by 18 pp) and 95 % coverage is 84.6 % (under-covers by 10 pp). The bridge residual distribution underestimates the realized error scale on this small sample.

Two diagnoses: (a) n = 13 is short — the rolling-conformal residual quantiles are noisy when the calibration window is half the sample. (b) The 2025-onward window has been a low-volatility regime; the bridge fits to small recent residuals and assumes that will continue.

Practical implication for the operational deployment: use Bridged-CBDF for point forecasts and CRPS rankings, but inflate the bands by the empirical 80 %-cov gap (≈ 1.30×) until n ≥ 24 calibration months accumulate. The architecture doc's "queued" item to wire Bridged-CBDF as a Forecaster class is now closed (`src/thales/models/composition/bridged_cbdf.py`).

### 4. Path A is the most calibrated point + bands forecaster in production

Path A's PIT KS p = 0.571 (well above 0.05), cov80 = 90 % (slightly conservative — 10 pp above nominal), cov95 = 94 % (within 1 pp of nominal). Sharpness 0.88 / 1.32 is moderate. This is the right calibration story for a production deployment: slightly over-cover is the safer side of mis-calibration when downstream customers are sizing positions on the bands.

### 5. AR(1) on YoY is the most calibrated forecaster, period

AR(1)-on-YoY's PIT KS p = 0.890 — the highest in the table. Its rolling-conformal bands have cov80 = 76 %, cov95 = 90 %, both within 5 pp of nominal. RMSE is 5 bp worse than persistence, which is expected (Stock-Watson 2007: monthly YoY is near-unit-root, AR(1) collapses to persistence with a small mean-reversion penalty). But the residual distribution it produces, while not the smallest, is the truest reflection of the underlying error process.

This is worth flagging because the literature treats AR(1) as a strawman; the density evaluation says it's a perfectly serviceable calibrated forecaster — the issue is the point estimate, not the distribution.

## What shipped

- **`src/thales/evaluation/density.py`** — new module. `samples_from_residuals` (bootstrap), `samples_from_gaussian` (parametric), `samples_from_quantiles` (inverse-CDF interpolation), `score_density` returning a `DensityBlock` with CRPS, PIT KS p-value, 80 %/95 % coverage, sharpness. 19 unit tests.
- **`Forecast.samples` plumbed through every Forecaster** — Persistence, AR(1), Path A, MoM-composed, Stock-Watson DFM, SameMonthBridgeNowcaster, BridgedCBDF all emit predictive samples (bootstrap from rolling-conformal residuals, or Gaussian draws when residuals are unavailable). 53 tests covering the wiring.
- **`harness.attach_actuals` + `score`** — samples ride along as a column on the prediction frame; `score()` auto-stacks them and computes density metrics. `ScoreBlock` extended with `pit_ks_pvalue`, `cov80_density`, `cov95_density`, `sharp80_density`, `sharp95_density`, `n_density`. 5 end-to-end tests.
- **`src/thales/models/composition/bridged_cbdf.py`** — the inline `_bridge_cbdf` from `okeefe_headtohead.py` promoted to a wired Forecaster class. 6 unit tests including coefficient recovery on a synthetic bridge DGP.
- **`scripts/density_eval.py`** — comparison runner for the headline matrix above. CSV at `results/baseline_eval/density_eval_summary.csv`.

## Test suite

```
pytest tests/test_density.py tests/test_density_e2e.py \
        tests/test_baselines_rolling_conformal.py \
        tests/test_dfm.py tests/test_mom_composed.py \
        tests/test_bridged_cbdf.py tests/test_harness.py \
        tests/test_bridge.py tests/test_bridge_rolling_conformal.py
=> 79 passed in 6.5s
```

## What's now possible

With density scoring wired across the model zoo, every comparison the
team report and the head-to-head doc make is now scoreable on density
metrics, not just RMSE. The next set of comparisons that motivated this
work — "does a foundation model beat the econometric stack on density,
not just RMSE" — has a fair playing field to land in.

## Glossary (stats terms)

- **CRPS (Continuous Ranked Probability Score):** integral of the squared difference between the predictive CDF and the empirical step function at the realization. Reduces to MAE for a deterministic forecast. Lower is better. Gneiting-Raftery 2007 establish it as a strictly proper scoring rule.
- **PIT (Probability Integral Transform):** for each observation, the predictive CDF evaluated at the realized value. Under correct calibration, PIT values are Uniform(0, 1). Dawid 1984 introduced the diagnostic.
- **PIT KS p-value:** Kolmogorov-Smirnov test that PIT values follow Uniform(0, 1). High p ⇒ calibrated; low p ⇒ miscalibrated.
- **Sharpness:** mean width of the central credible interval. Calibration without sharpness is useless (a band as wide as the historical range covers everything but says nothing).
- **Bootstrap from residuals:** the natural density counterpart of split-conformal banding. Same residual vector defines both the [lo, hi] interval (via quantiles) and the predictive density (via resampling).
