# Stefan Day-Ahead Forecaster — Historical Backtest

**Date:** 2026-04-24
**Script:** `scripts/stefan_historical_backtest.py`
**Window:** 2026-01-22 → 2026-04-14 (83 origins)

## Method recap

12 top-level Truflation CPI category components → per-category walk-forward OLS (persistence + 1–2 exogenous daily covariates) → weighted composition via 2026 v2 category weights → bootstrap residual bands.

## Headline metrics

| Metric | Model | Naive `y[T+1]=y[T]` |
|---|---|---|
| RMSE | 0.4021 pp | 0.3988 pp |
| MAE | 0.3484 pp | 0.3450 pp |
| RMSE reduction vs naive | **-0.83%** | — |
| Directional accuracy | **42.2%** | — (base rate up: 3.6%) |
| 80% band coverage | **2.4%** (nominal 80%) | — |
| 95% band coverage | **8.4%** (nominal 95%) | — |
| Mean 80% band width | 0.0868 pp | — |
| Mean 95% band width | 0.2014 pp | — |

## Last 30 origins

- RMSE: 0.3483 pp
- 80% coverage: 3.3%
- Directional accuracy: 3.3%

## Honest interpretation

- Model ties naive on RMSE (-0.8%). Day-ahead is autocorrelation-dominated; expected.
- 80% band coverage = 2.4% is under nominal. Bands under-state uncertainty.
- Directional accuracy 42.2% near coin-flip (base rate 3.6%). Avoid claiming direction in the post.

## Caveats

- **Vintage approximation.** TN component streams tagged with `as_of=ingest_date` (not true first-publication date). For daily-frequency frozen streams the difference is ≤1 day (Truflation's 24h QC delay). Small leak; documented in pre-reg §2.5.
- **Target = Truflation's own frozen YoY**, not BLS CPI. Predicting a different number than the institutional nowcast product.
- **n=83 is a small sample** for claims about calibration. 200+ days would tighten the coverage estimate.
- **Bootstrap bands assume per-component residual independence.** Components co-move (gas and utilities both ride nat gas). True multivariate residual distribution would give slightly different (likely wider) bands.

## Ship / no-ship verdict

**Hold.** Bands are off nominal by more than ±7pp on 80% or ±4pp on 95%. Before Stefan posts: investigate why bands are off. Likely candidates — residual correlation across components, regime mismatch, or need for ALFRED vintages to prevent subtle leak. Re-run with wider/narrower bands or more data.

## Artifacts

- `results/daily_forecast/historical_backtest.csv` — per-origin predictions + bands + realized values