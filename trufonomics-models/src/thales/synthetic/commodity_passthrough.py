"""Synthetic DGP for the Commodity pass-through archetype.

Used to validate estimation code — generate data from a known DGP, fit a
model, verify recovery of the true parameters. This is the template for
every archetype's recovery test per docs/planning/02-evaluation.md §Tier 1.

The DGP matches the structure specified in §Archetype 1 of
docs/planning/01-architecture.md:

    log(p_t) = α + β_t · log(c_t) + ε_t
    ε_t  ~ N(0, exp(h_t))
    β_t  = β_{t-1} + η^β_t,   η^β_t ~ N(0, σ_β²),  β_t ∈ [0, 1]
    h_t  = ρ · h_{t-1} + η^h_t, η^h_t ~ N(0, σ_h²)          (stoch. vol.)
    log(c_t) = log(c_{t-1}) + μ_c + η^c_t,  η^c_t ~ N(0, σ_c²)  (GBM commodity)

Default parameters roughly match empirical gasoline pass-through levels
(mean β ≈ 0.35, slow drift, moderate stochastic volatility).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PassthroughDGP:
    """Ground-truth parameters and realised state paths for one simulation."""
    commodity: np.ndarray        # log-level (T,)
    retail: np.ndarray            # log-level (T,)
    true_beta: np.ndarray         # (T,) time-varying pass-through
    true_log_sigma: np.ndarray   # (T,) log stochastic volatility
    alpha: float                  # intercept
    commodity_drift: float        # μ_c
    beta_drift_sd: float          # σ_β
    sv_persistence: float         # ρ
    sv_shock_sd: float            # σ_h


def simulate_commodity_passthrough(
    T: int = 2000,
    beta_0: float = 0.35,
    beta_drift_sd: float = 0.005,
    alpha: float = 0.0,
    log_commodity_0: float = 4.0,        # ≈ log(55), reasonable for oil
    commodity_drift: float = 0.0001,
    commodity_shock_sd: float = 0.015,
    sv_persistence: float = 0.98,
    sv_shock_sd: float = 0.15,
    sv_log_mean: float = -5.0,            # ≈ log(0.007²), small noise floor
    seed: int = 0,
) -> PassthroughDGP:
    """Generate a synthetic commodity–retail pass-through path with SV.

    Returns a ``PassthroughDGP`` with both observables and the latent truths.
    Default parameters are calibrated so that retail residual SD is roughly
    consistent with empirical daily retail-price innovation sizes.
    """
    rng = np.random.default_rng(seed)

    # Commodity path as GBM
    c_shocks = rng.normal(0.0, commodity_shock_sd, size=T)
    log_c = np.empty(T)
    log_c[0] = log_commodity_0
    for t in range(1, T):
        log_c[t] = log_c[t - 1] + commodity_drift + c_shocks[t]

    # Time-varying β as bounded random walk
    beta_shocks = rng.normal(0.0, beta_drift_sd, size=T)
    beta = np.empty(T)
    beta[0] = beta_0
    for t in range(1, T):
        beta[t] = np.clip(beta[t - 1] + beta_shocks[t], 0.0, 1.0)

    # Stochastic volatility: log σ_t²
    sv_shocks = rng.normal(0.0, sv_shock_sd, size=T)
    log_sigma = np.empty(T)
    log_sigma[0] = sv_log_mean
    for t in range(1, T):
        log_sigma[t] = (sv_log_mean
                         + sv_persistence * (log_sigma[t - 1] - sv_log_mean)
                         + sv_shocks[t])

    # Noise with SV
    noise = rng.normal(0.0, 1.0, size=T) * np.exp(log_sigma / 2.0)
    log_retail = alpha + beta * log_c + noise

    return PassthroughDGP(
        commodity=log_c,
        retail=log_retail,
        true_beta=beta,
        true_log_sigma=log_sigma,
        alpha=alpha,
        commodity_drift=commodity_drift,
        beta_drift_sd=beta_drift_sd,
        sv_persistence=sv_persistence,
        sv_shock_sd=sv_shock_sd,
    )


def static_ols_recovery(dgp: PassthroughDGP) -> tuple[float, float, float]:
    """Fit static OLS ``log_retail = a + b · log_commodity + ε`` and return
    (intercept, slope, residual_sd).

    For a TVP-VECM DGP with small β drift this should recover the *mean* β.
    Phase 0 sanity check; the full TVP-SV recovery test comes when the real
    state-space model class lands.
    """
    X = np.column_stack([np.ones_like(dgp.commodity), dgp.commodity])
    coefs, *_ = np.linalg.lstsq(X, dgp.retail, rcond=None)
    pred = X @ coefs
    resid = dgp.retail - pred
    return float(coefs[0]), float(coefs[1]), float(resid.std(ddof=1))
