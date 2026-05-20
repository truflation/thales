"""Synthetic DGP for the BSTS-discretionary archetype (Phase 1.2).

Bayesian Structural Time Series with three latent components:

  Observation:  y_t  =  μ_t  +  s_t  +  ε_t,                  ε_t ~ N(0, σ_ε²)
  Trend:        μ_t  =  μ_{t-1}  +  δ_{t-1}  +  η^μ_t,        η^μ_t ~ N(0, σ_μ²)
  Slope:        δ_t  =  δ_{t-1}  +  η^δ_t,                     η^δ_t ~ N(0, σ_δ²)
  Seasonal:     s_t  =  −Σ_{k=1..S-1} s_{t-k}  +  η^s_t,       η^s_t ~ N(0, σ_s²)

Captures the pattern of CPI categories like Recreation, Food-away, and
"All Other" — moderate trend with strong yearly seasonality and noise.

Default parameters use S=12 (monthly seasonal), modest σ_μ / σ_δ (slow
trend), seasonal amplitude on the same order of the noise. Caller
supplies the seasonal amplitude as a tuple of S values to make recovery
visually tractable.

Companion to ``commodity_passthrough.py`` — same Tier-1 evaluation
discipline (synthetic ground truth → fit → recover).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BSTSDGP:
    """Ground-truth components and observable for one simulation."""
    y: np.ndarray              # (T,) observable
    trend: np.ndarray          # (T,) μ_t  — local-linear-trend level
    slope: np.ndarray          # (T,) δ_t  — slope of the trend
    seasonal: np.ndarray       # (T,) s_t  — current-period seasonal
    noise: np.ndarray          # (T,) ε_t
    sigma_mu: float
    sigma_delta: float
    sigma_seasonal: float
    sigma_eps: float
    period: int                # S


def simulate_bsts_discretionary(
    T: int = 600,
    period: int = 12,
    initial_level: float = 100.0,
    initial_slope: float = 0.05,
    seasonal_pattern: np.ndarray | None = None,
    sigma_mu: float = 0.05,
    sigma_delta: float = 0.005,
    sigma_seasonal: float = 0.1,
    sigma_eps: float = 0.5,
    seed: int = 0,
) -> BSTSDGP:
    """Generate a synthetic BSTS path. Returns ``BSTSDGP`` with all latents.

    ``seasonal_pattern`` is a length-``period`` array of seasonal effects
    that the seasonal walk drifts away from gradually. If None, a
    sinusoidal pattern of amplitude 2 is used (small relative to a level
    around 100).

    Defaults give a path that's qualitatively close to a discretionary
    CPI category index: slow trend, modest annual swing, modest noise.
    Caller can intensify any component by raising the corresponding σ.
    """
    rng = np.random.default_rng(seed)

    if seasonal_pattern is None:
        seasonal_pattern = 2.0 * np.sin(2.0 * np.pi * np.arange(period) / period)
    seasonal_pattern = np.asarray(seasonal_pattern, dtype=float)
    if len(seasonal_pattern) != period:
        raise ValueError(f"seasonal_pattern length {len(seasonal_pattern)} != "
                          f"period {period}")
    # Center on zero so it doesn't shift the level
    seasonal_pattern = seasonal_pattern - seasonal_pattern.mean()

    trend = np.empty(T)
    slope = np.empty(T)
    seasonal = np.empty(T)
    noise = rng.normal(0.0, sigma_eps, size=T)

    trend[0] = initial_level
    slope[0] = initial_slope

    # Seed seasonal with the static pattern for the first cycle.
    for t in range(min(period - 1, T)):
        seasonal[t] = seasonal_pattern[t]

    for t in range(1, T):
        slope[t] = slope[t - 1] + rng.normal(0.0, sigma_delta)
        trend[t] = trend[t - 1] + slope[t - 1] + rng.normal(0.0, sigma_mu)
        if t >= period - 1:
            # Dummy seasonal: s_t = -Σ_{k=1..S-1} s_{t-k} + η_s
            past = seasonal[t - period + 1: t]
            seasonal[t] = -past.sum() + rng.normal(0.0, sigma_seasonal)

    y = trend + seasonal + noise

    return BSTSDGP(
        y=y, trend=trend, slope=slope, seasonal=seasonal, noise=noise,
        sigma_mu=sigma_mu, sigma_delta=sigma_delta,
        sigma_seasonal=sigma_seasonal, sigma_eps=sigma_eps,
        period=period,
    )
