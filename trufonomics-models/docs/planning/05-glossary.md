# 05 — Glossary

Alphabetized reference for terminology used across the stack.

---

**Ablation test.** Removing a model component and re-evaluating. If performance doesn't degrade meaningfully, the component is not justified.

**AR(1).** Autoregressive process of order 1 — each observation depends on the previous one plus noise. Simplest persistence model, common baseline.

**ARMA-GARCH.** Autoregressive Moving Average with Generalized Autoregressive Conditional Heteroskedasticity. Time series with mean dynamics (ARMA) and volatility dynamics (GARCH). Standard econometric benchmark.

**Backcast.** Forecast of a past period for which official data exists, typically for model validation. Horizon h = −1 or less.

**Bayesian VAR.** Vector autoregression estimated via Bayesian methods, typically with Minnesota priors that shrink coefficients toward parsimonious structure. Standard in macro forecasting.

**BSTS.** Bayesian Structural Time Series. A class of state-space model decomposing a series into trend + seasonal + cycle + covariate effects. Google's `CausalImpact` uses this internally.

**Brier score.** Mean squared error for probabilistic classifications. Primary metric for regime models. Lower is better.

**Calibration.** Whether forecast probabilities match realized frequencies. A 90% interval should contain the realization 90% of the time across many forecasts. Distinct from sharpness.

**CBDF.** Component-Based Dynamic Factor model. NY Fed 2025 paper (O'Keeffe & Petrova) combining bottom-up component modeling with dynamic factor structure that respects GDP accounting identity.

**Cointegration.** Long-run statistical relationship between non-stationary series. Their linear combination is stationary even though each is not. Foundation for VECM models.

**Credible interval.** Bayesian analogue of confidence interval. The 80% credible interval has 80% posterior probability of containing the true value.

**CRPS (Continuous Ranked Probability Score).** Integral of (forecast CDF − realized indicator)² over the real line. Generalizes MAE to full distributions. Primary density metric. Lower is better.

**Density forecast.** The full probability distribution over possible outcomes, not just a point. Required for risk management and all institutional buyers.

**DFM.** Dynamic Factor Model. Latent factors drive many observed series. Widely used at central banks for nowcasting.

**Diebold-Mariano test (DM test).** Hypothesis test for whether two forecast error sequences have the same expected loss. Produces p-value on "is model A actually better than model B?"

**DGP.** Data-generating process. The true (often unknown) mechanism producing observed data. Synthetic DGPs are DGPs we define and simulate from to test models.

**Doz-Giannone-Reichlin (DGR) two-step.** Classic two-step estimation of large dynamic factor models: PCA for initial factor estimates, then EM or likelihood-based refinement.

**ECM / VECM.** Error Correction Model / Vector ECM. Models how deviations from a long-run equilibrium drive short-run dynamics. Standard tool for cointegrated series.

**EM algorithm.** Expectation-Maximization. Iterative estimation for models with latent variables. Standard for many state-space models.

**Fan chart.** Visualization of forecast central estimate with expanding credible intervals over horizon. Bank of England convention.

**FEVD (Forecast Error Variance Decomposition).** Share of forecast uncertainty attributable to each structural shock. Tells a client "40% of margin risk is fuel, 30% wages, etc."

**Giacomini-White test (GW test).** Modern version of DM test, handles nested models and parameter estimation uncertainty properly. Use alongside DM.

**Horizon.** Forecast distance from the date the forecast is made. h = 0 is a nowcast of the current period, h = +6 is a six-period-ahead forecast.

**Impulse response function (IRF).** Forecast path of a variable in response to a unit shock in another variable. Core output of VARs, fundamental to transmission VAR product.

**Information set.** Everything known on a given date, respecting vintages and release lags. Denoted 𝓘_t. The pseudo-real-time harness reconstructs 𝓘_t for every historical forecast date.

**Jagged edge (ragged edge).** The end of a real-time dataset where different series have different last-observed dates because of different release lags. The reason mixed-frequency Kalman filtering exists.

**Kalman filter.** Optimal estimator for linear-Gaussian state-space models. Recursively updates state estimates as new observations arrive.

**Kim-Nelson filter.** Hamilton-style regime-switching embedded in Kalman filter. Used for Markov-switching state-space models.

**Log predictive score.** Log of predictive density evaluated at the realized value. Higher is better. Can be unstable in tails — a model assigning near-zero probability to a realized value gets crushed.

**MAE.** Mean Absolute Error. Robust to outliers.

**MASE.** Mean Absolute Scaled Error. Scaled against a naive baseline so it's comparable across series.

**Mariano-Murasawa cumulator.** Standard trick for handling mixed-frequency data in state-space models. Creates cumulator states that aggregate high-frequency observations to match low-frequency releases.

**Minnesota prior.** Bayesian VAR prior that shrinks coefficients toward a "random walk with drift" baseline. Tight prior on own lags, looser on cross-variable lags.

**Mixed-frequency model.** Handles data at different observation frequencies (daily Truflation, weekly rates, monthly BLS) without forcing everything to lowest frequency.

**Nowcast.** Model-based prediction of a current or very recent period for which the official number has not been released. Horizon ≈ 0. Distinct from forecast (future period) and measurement (settled current value).

**OER (Owners' Equivalent Rent).** BLS method for imputing housing costs for homeowners — asking renters what they would pay, then applying to owned units. Truflation uses actual mortgage-based costs instead, creating a persistent methodological wedge vs BLS.

**Particle filter.** Monte Carlo method for nonlinear/non-Gaussian state-space models. Use when Kalman filter assumptions break.

**PIT (Probability Integral Transform).** Value of forecast CDF at realization. Under correct calibration, PIT values are Uniform(0,1). Primary calibration diagnostic.

**Point forecast.** Single-number prediction, no uncertainty. Cleveland Fed publishes point nowcasts; we publish density.

**Point-in-time / vintage.** The exact version of a dataset as it existed on a specific past date, before subsequent revisions. Essential for honest backtests.

**Posterior predictive check (PPC).** Simulating from the posterior and comparing simulated data to actual on moments not explicitly fit. Primary misspecification diagnostic for Bayesian models.

**PPI.** Producer Price Index. Industry-level prices. Used for transmission VAR training data.

**Pre-registration.** Committing to evaluation methodology (comparators, metrics, windows) *before* running the evaluation. Credibility weapon against accusations of p-hacking.

**Pseudo real-time evaluation.** Walk-forward through history using only vintages available at each historical date. The honest backtest.

**Quantile loss (pinball loss).** Asymmetric loss function for a specific quantile forecast. Aggregated across quantiles gives a full density score.

**Random walk forecast.** Today's value projected forward as the forecast for all future horizons. Universal sanity-check baseline — any model that doesn't beat random walk is not forecasting.

**Regime-switching.** Discrete regime variable whose current state determines model parameters. Handles structural breaks and regime-dependent dynamics (stable vs shock, transitory vs persistent).

**Reliability diagram.** Binned plot of forecast probabilities vs realized frequencies. Calibration diagnostic for probabilistic classifiers.

**RMSE.** Root Mean Squared Error. Standard point forecast metric. Sensitive to outliers.

**ROC AUC.** Area Under Receiver Operating Characteristic curve. Classification metric, threshold-independent.

**Sharpness.** How tight the forecast distribution is. Meaningful only conditional on being calibrated. A wide-uniform distribution is calibrated but useless.

**SPF.** Survey of Professional Forecasters, Philadelphia Fed. Quarterly survey of economists' forecasts. Benchmark comparator.

**SSM (State Space Model).** Class of models with observed measurements and unobserved states evolving over time. Encompasses Kalman filter, BSTS, dynamic factor models, regime switching.

**Stochastic volatility (SV).** Volatility itself follows a time-series process, not constant. Essential for honest forecast bands during shock regimes.

**Structural break.** Discrete change in data-generating process parameters. Regime models detect these.

**TVP.** Time-varying parameter. Coefficients drift over time as latent states rather than being constant.

**Trend-cycle decomposition.** Splitting observed series into slow-moving trend + faster cycle + noise. UC-SV does this.

**UC-SV-MS.** Unobserved Components with Stochastic Volatility and Markov Switching. Inflation's workhorse model at central banks — trend, cycle, regime-conditional persistence, time-varying volatility.

**Vintage store.** Database of data keyed by reference date × as-of date. The foundation of honest backtesting.

**Walk-forward simulation.** Evaluation technique: at each historical date, reconstruct information set, refit model, generate forecasts, move forward. The honest backtest methodology.
