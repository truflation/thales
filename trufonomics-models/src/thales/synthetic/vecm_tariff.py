"""Synthetic DGP for the VECM-tradables archetype (Phase 1.4).

Bivariate cointegrated system with a tariff regime dummy:

  Δy_1t = α_1 (z_{t-1} − μ_t) + ε_{1t}
  Δy_2t = α_2 (z_{t-1} − μ_t) + ε_{2t}

  z_t   = y_{1t} − y_{2t}                       (cointegrating relation)
  μ_t   = μ_0 + θ · D_{t-1}                     (tariff-regime shifted equilibrium)

with ``D_t`` a 0/1 tariff-regime dummy (e.g. April 2025 onward = 1).
``α_1`` is typically negative and ``α_2`` typically positive — clothing
falls when above its long-run relation to the import index, imports rise.

Both y series are individually I(1) (random-walk-like); their spread
``z_t`` is stationary around ``μ_t`` and reverts at speed determined by
``α_1 − α_2``. The tariff regime shifts the equilibrium spread by ``θ``.

Calibrated (very loosely) on a clothing-vs-import-index pair: clothing
falls slowly toward equilibrium (~1% per month adjustment), imports
rise quickly (~5% per month). Mean spread ≈ 0; tariff shock shifts it
positively.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VECMDGP:
    """Ground-truth parameters and realised paths for one VECM simulation."""
    y1: np.ndarray              # (T,) tradable index 1 (e.g. clothing)
    y2: np.ndarray              # (T,) tradable index 2 (e.g. imports)
    z: np.ndarray               # (T,) cointegrating residual y1 - y2
    regime: np.ndarray          # (T,) 0/1 tariff regime dummy
    alpha_1: float              # speed of adjustment for y1
    alpha_2: float              # speed of adjustment for y2
    mu_0: float                 # baseline long-run equilibrium spread
    theta: float                # tariff-regime shift in equilibrium
    sigma_1: float
    sigma_2: float
    rho: float                  # correlation of ε_1, ε_2


def simulate_vecm_tariff(
    T: int = 600,
    initial_y1: float = 100.0,
    initial_y2: float = 100.0,
    alpha_1: float = -0.05,
    alpha_2: float = +0.10,
    mu_0: float = 0.0,
    theta: float = 5.0,
    regime_start: int | None = None,
    sigma_1: float = 0.4,
    sigma_2: float = 0.6,
    rho: float = 0.0,
    seed: int = 0,
) -> VECMDGP:
    """Generate a bivariate cointegrated path with tariff regime shift.

    ``regime_start`` is the index from which D_t flips to 1. Default
    `None` puts it 60% through the series.
    """
    rng = np.random.default_rng(seed)

    if regime_start is None:
        regime_start = int(T * 0.6)
    regime = np.zeros(T, dtype=int)
    regime[regime_start:] = 1

    y1 = np.empty(T)
    y2 = np.empty(T)
    z = np.empty(T)
    y1[0] = initial_y1
    y2[0] = initial_y2
    z[0] = y1[0] - y2[0]

    # Bivariate normal innovations with correlation rho
    cov = np.array([[sigma_1 ** 2, rho * sigma_1 * sigma_2],
                       [rho * sigma_1 * sigma_2, sigma_2 ** 2]])
    L = np.linalg.cholesky(cov)
    eta = (L @ rng.normal(size=(2, T)))

    for t in range(1, T):
        mu_t = mu_0 + theta * regime[t - 1]
        ec = z[t - 1] - mu_t   # error-correction term
        d_y1 = alpha_1 * ec + eta[0, t]
        d_y2 = alpha_2 * ec + eta[1, t]
        y1[t] = y1[t - 1] + d_y1
        y2[t] = y2[t - 1] + d_y2
        z[t] = y1[t] - y2[t]

    return VECMDGP(
        y1=y1, y2=y2, z=z, regime=regime,
        alpha_1=alpha_1, alpha_2=alpha_2,
        mu_0=mu_0, theta=theta,
        sigma_1=sigma_1, sigma_2=sigma_2, rho=rho,
    )
