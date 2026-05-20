# 02 — Evaluation

Evaluation is the moat. Frontier econometric methods are public; the shop that deploys them with rigorous evaluation wins institutional trust.

## Contents

- [Philosophy — three tiers](#philosophy--three-tiers)
- [Metrics](#metrics)
- [Regime-stratified evaluation](#regime-stratified-evaluation)
- [Horizon-stratified evaluation](#horizon-stratified-evaluation)
- [Tier 1 — Synthetic recovery](#tier-1--synthetic-recovery)
- [Tier 2 — Historical pseudo-real-time](#tier-2--historical-pseudo-real-time)
- [Tier 3 — Live production monitoring](#tier-3--live-production-monitoring)
- [Comparator set](#comparator-set)
- [Stress tests](#stress-tests)
- [Economic value backtests](#economic-value-backtests)
- [Reporting](#reporting)
- [Governance and pre-registration](#governance-and-pre-registration)

---

## Philosophy — three tiers

A single "out-of-sample RMSE" number is marketing, not evaluation. Real evaluation answers three questions, each with its own tier:

**Tier 1 — Synthetic (does the math work?).** Generate data from a known DGP, fit the model, check if it recovers the truth. Catches bugs, specification errors, non-identification. Fast to run. Every archetype and every latent state gets a recovery test.

**Tier 2 — Historical pseudo-real-time (does it work on the past?).** Walk-forward through vintages, produce forecasts, score. This is what we publish. Slow — budget GPU time.

**Tier 3 — Live production (does it still work?).** Every forecast logged, scored when truth arrives. Triggers retraining and retirement.

All three always. No skipping.

---

## Metrics

Implement all of these in one `evaluation/metrics.py` module with a consistent interface. Every model pushes through the same pipeline.

### Point accuracy

| Metric | Lower/higher | Notes |
| --- | --- | --- |
| RMSE | lower better | Standard, easy to communicate |
| MAE | lower better | Robust to outliers, often more informative than RMSE for inflation |
| MASE (mean absolute scaled error) | lower better | Scaled against naive baseline, comparable across series |
| Directional accuracy | higher better | Did you call the sign? Matters for tradeable products |

### Density accuracy

| Metric | Lower/higher | Notes |
| --- | --- | --- |
| CRPS (Continuous Ranked Probability Score) | lower better | **Primary density metric.** Generalizes MAE to distributions |
| Log predictive score | higher better | What O'Keeffe-Petrova report. Sensitive to tail calibration, unstable if model assigns near-zero probability to realized value |
| Quantile loss (pinball) at {5, 10, 25, 50, 75, 90, 95} | lower better | For risk management clients (VaR-style use) |

### Calibration

- **PIT histogram** with KS test of uniformity. U-shaped = overconfident. Hump = underconfident.
- **Reliability diagram** — bin forecast probabilities, compare to realized frequencies
- **Interval coverage** — does the 80% band contain truth 80% of the time empirically?
- **Sharpness** — average width of 50/80/95 intervals, meaningful only conditional on calibration

### For regime/classification models

| Metric | Lower/higher | Notes |
| --- | --- | --- |
| Brier score | lower better | Primary classification metric |
| Log loss | lower better | Penalizes confident wrong calls harder |
| ROC AUC | higher better | For binary regime calls |
| Calibration plot | visual | Bin probabilities, check realized frequencies |
| Regime timing lead (days) | higher better | Days before realized regime change probability crosses threshold. This is the "alpha" metric |

### Statistical significance

- **Diebold-Mariano test** — p-value on "is my forecast loss different from the comparator's?"
- **Giacomini-White test** — modern version, handles nested models and parameter estimation uncertainty properly

Always report DM and GW alongside raw improvements. Without them, a claimed "beat" is often noise.

Libraries: [`properscoring`](https://github.com/properscoring/properscoring) for CRPS and log score. Roll your own PIT, reliability, DM, GW — 20 lines each.

---

## Regime-stratified evaluation

**The thing most shops miss.** Average metrics over regime-heterogeneous samples are misleading by construction.

Pre-defined regime windows (committed to pre-registration doc):

| Regime | Window | Notes |
| --- | --- | --- |
| Stable low inflation | 2015–2019 | Pre-pandemic baseline |
| COVID shock | Mar 2020 – Dec 2020 | Initial disruption |
| Inflation surge | Jan 2021 – Jul 2023 | Stimulus, supply chain, rapid acceleration |
| Disinflation | Aug 2023 – Dec 2024 | Unwinding of surge |
| Stable low (return) | 2024–present baseline | |
| Tariff regime | Apr 2025 – present | Active tariff policy |
| Government shutdown | Oct 2025 and Jan 2026 | BLS release disruption — special flag |

Report full metric table for each subwindow, not just full sample. Institutional buyers stratify whether we do or not. Better to front-run them.

---

## Horizon-stratified evaluation

Every model evaluated at multiple horizons:

| Horizon | Meaning | Buyer |
| --- | --- | --- |
| h = −1 | Backcast (post-release) | Validation, internal |
| h = 0 | Nowcast at release | Macro funds, CPI traders |
| h = −15 days | Nowcast 2 weeks before release | Macro funds |
| h = +1 month | Short forecast | Treasurers, hedgers |
| h = +3 months | Planning forecast | CFOs |
| h = +6 months | Annual planning | CFOs, procurement |
| h = +12 months | Strategic | Real estate, lenders |

Report RMSE-by-horizon curves vs random walk. Model should beat random walk at every horizon. If only at h = 0, your "forecasting ability" is really a real-time index advantage — be honest about it.

---

## Tier 1 — Synthetic recovery

For every archetype, write a DGP and a recovery test. Template:

```python
def test_commodity_passthrough_recovers_beta():
    # Generate: beta_t as bounded random walk, sigma_t as SV
    true_beta, true_sigma, data = simulate_commodity_passthrough(
        T=2000, beta_0=0.35, beta_drift=0.01, sv_persistence=0.98
    )
    model = CommodityPassthrough()
    posterior = model.fit(data)
    beta_hat = posterior["beta_t"].mean(axis=0)
    assert correlation(beta_hat, true_beta) > 0.9
    assert coverage(posterior["beta_t"], true_beta, 0.9) > 0.85
    assert coverage(posterior["beta_t"], true_beta, 0.5) > 0.45
```

Specific recovery tests by archetype:

| Archetype | Recovery targets |
| --- | --- |
| Commodity pass-through | β_t, σ_t, ECM adjustment speed, IRF match |
| Rate-sensitive housing | Affordability factor, regional loadings, tenure-share weights; separates national from regional correctly |
| UC-SV-MS sticky services | Regime sequence (Viterbi), regime-conditional persistence, trend innovation variance; no false-triggers on noise |
| VECM tradables | Cointegrating vector, speed of adjustment, tariff dummies; Johansen trace stat correct on synthetic cointegrated data |
| BSTS discretionary | Trend, dual seasonals (daily + annual separated), cycle frequency and damping |
| CBDF composition | Aggregate recovered with correct density (joint calibration, not just marginal) |
| Regime model | Classification accuracy, timing lead, false-transition rate, regime inertia |
| Transmission VAR | IRF recovery with known shock structure |

### Posterior predictive checks

For Bayesian models, simulate from posterior and compare to actual data on moments *not* explicitly fit: higher-order autocorrelation, kurtosis, cross-series correlation, extreme event frequency. Misspecification shows here even when point estimates fit fine.

---

## Tier 2 — Historical pseudo-real-time

The walk-forward simulator:

```python
for forecast_date in eval_dates:
    info_set = vintage_store.snapshot(as_of=forecast_date)
    model = ModelClass()
    model.fit(info_set)
    for horizon in [-15, -7, 0, +1, +3, +6, +12]:
        target_date = forecast_date + horizon_offset(horizon)
        forecast = model.forecast(target_date)
        scoring_db.insert(
            model_version, forecast_date, target_date, horizon,
            forecast.mean, forecast.quantiles, forecast.samples
        )
```

### Two configurations

1. **Expanding window** — all data up to `forecast_date`, refit from scratch each time. Most honest, most expensive.
2. **Rolling window** — fixed-length window (e.g., 10 years). Tests stability, addresses concept drift.

Run both. Compare.

### Refit cadence

Nightly refits = slow but honest. Weekly refits with daily Kalman filtering = standard production pattern. Test both.

### Ablation tests

Every component that costs complexity gets an ablation. If removing it doesn't meaningfully lose accuracy/calibration, simplify.

Ablations to run:
- CBDF vs standard DFM (the O'Keeffe-Petrova comparison, on our data)
- Time-varying pass-through β_t vs constant β
- Regime-switching UC-SV vs plain UC-SV
- Hierarchical housing SSM vs flat panel SSM
- Full mixed-frequency Mariano-Murasawa vs quarterly aggregation
- Bayesian VECM with Minnesota priors vs frequentist VECM

### Revision stability

Log every forecast revision. Compute:
- **Revision variance** — how much does h = +6 drift as data accumulates?
- **Revision efficiency** — news-driven or noise-driven?
- **Final-revision convergence** — monotonic to realization or oscillating?

Present as "forecast evolution" chart: show how June 2026 CPI forecast evolved January through June. Rare in published work.

---

## Tier 3 — Live production monitoring

Every forecast automatically scored. Alerts on:

| Trigger | Condition | Action |
| --- | --- | --- |
| Calibration drift | Rolling 20-forecast PIT KS p-value < 0.05 | Investigate |
| CRPS regime break | Rolling 20-forecast CRPS exceeds 95th historical percentile | Alert |
| Comparator crossover | Stop beating Cleveland Fed on rolling window | Investigate |
| Input distribution drift | Covariate distribution shifts outside training support | Alert |
| Reproducibility break | Weekly random forecast re-run fails bit-identity check | Critical alert — data corruption |

Simple thresholds, interpretable alerts. Not ML-based drift detection.

---

## Comparator set

Comparators determine what we can claim. Set explicitly in pre-registration.

| Model class | Primary comparator | Secondary | Sanity check |
| --- | --- | --- | --- |
| Archetype nowcasts (category) | AR(1), ARMA-GARCH benchmark | Static version of same model (ablation) | Naive persistence |
| CBDF headline nowcast | [Cleveland Fed Inflation Nowcast](https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting) | [SPF](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters), Bloomberg consensus | Random walk |
| Regime model | St. Louis Fed recession probability, Atlanta Fed sticky-price CPI, Dallas Fed trimmed mean PCE | None robust | None |
| Transmission VAR | S&P Global / ISM industry PPI forecasts | Cost-structure-weighted sector PPI | Industry aggregate |

**Important caveat on Cleveland Fed:** they publish point nowcasts only, not densities. Their density construction (historical error rolling window, described in [WP 2406](https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/publications/working-papers/2024/wp2406.pdf)) isn't on the website. For density comparison: reconstruct their density from their point + rolling window of their errors, or compare to SPF densities where available.

---

## Stress tests

Before going live, subject every model to synthetic shocks:

- 10σ outlier injection — does model blow up or widen posterior appropriately?
- Sudden level shift — does regime model detect within expected lead?
- Data provider outage (5 days missing) — does Kalman filter handle cleanly?
- Historical revision — does vintage store isolate from prior forecasts?
- Deliberately misspecified priors — does posterior update enough to overcome?

Put in CI. Produces "known failure modes" document shipped with each model.

---

## Economic value backtests

For regime model and transmission VARs, statistical metrics are necessary but insufficient.

**Regime model:** Simple trading rule on TIPS breakevens based on regime probability. Go long breakevens when P(persistent) > 70%, short when < 30%. Report Sharpe, max drawdown, hit rate 2018–2026. If Sharpe > 0.8 out-of-sample, it's a real product.

**Logistics transmission VAR:** Simulated hedging strategy for a trucking firm using diesel futures, timed by IRF signals. P&L vs unhedged and vs fixed-ratio. Value saved per unit volume.

**Restaurants transmission VAR:** Menu-price-timing rule based on forward food/wage inflation. Margin preservation vs 6-month-lagged naive repricing.

These close enterprise sales.

---

## Reporting

Three audiences, three formats:

### Internal dashboard
Streamlit app on scoring database. Every model, every horizon, every regime. Filterable, drillable. Nightly updates. Operations cockpit.

### Quarterly evaluation report
Structured PDF-able doc. Executive summary → methodology (pointer to pre-reg) → metric tables (point, density, regime-stratified, horizon-stratified) → DM test matrices → economic value backtests → known failure modes → roadmap. Dated, versioned, archived.

### Public live track record
URL with live nowcast vs realization history, updated after every major release. Cleveland Fed does this. We do too. Over 18 months this compounds into unfakeable credibility.

---

## Governance and pre-registration

**Pre-registration doc:** committed to repo *before* results known. Specifies comparators, metrics, windows, significance thresholds, ablation set. Amendable with timestamped diffs, never silently edited.

**Model versioning:** every model has a semantic version. Evaluation results keyed to version. New version → published comparison report vs previous on identical windows. No silent upgrades.

**Data versioning:** vintage store is append-only. Every ingestion logged with source hash. Any historical forecast reproducible bit-exactly.

Pre-registration template → `docs/pre-registration/` (create when first eval is ready).

---

## The inversion

Build the evaluation harness, metrics, synthetic DGPs, scoring DB, and comparator pipeline **before** the real models. Iteration velocity goes up 5–10x. Ship-broken probability drops near zero.

This is why Phase 0 looks heavy on infrastructure. It's the moat.
