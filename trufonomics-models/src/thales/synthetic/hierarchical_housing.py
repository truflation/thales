"""Synthetic DGP for the hierarchical-housing flagship archetype (Phase 1.5).

Hierarchical dynamic factor model:

    F_t       =  F_{t-1}  +  η^F_t,             η^F_t ~ N(0, σ_F²)        (national)
    λ_{r,t}   =  ρ_r · λ_{r,t-1}  +  ν_{r,t},   ν_{r,t} ~ N(0, σ_{λ,r}²)  (regional AR(1))
    y_{r,t}   =  β_r · F_t  +  λ_{r,t}  +  ε_{r,t},  ε_{r,t} ~ N(0, σ_{ε,r}²)

with ``r ∈ {1, …, R}`` regions. Default R=4 matches the U.S. Census
divisions (Northeast / Midwest / South / West).

The structure captures:
  * **National housing factor** F_t — a common slow-moving shock (e.g.
    mortgage rate environment, national demand) drives all regions.
  * **Regional idiosyncratic** λ_{r,t} — AR(1) deviations specific to a
    region (e.g. local population shocks, state-level policy).
  * **Region-specific loadings** β_r — how strongly each region tracks
    the national factor (West typically higher, Midwest lower).

Used to validate the JAX-based hierarchical estimator (`fit_hierarchical_housing`).
Phase 1.5+ extensions:
  * Owned vs rented split (extra 2nd-tier hierarchy)
  * Mixed frequency (Mariano-Murasawa: Case-Shiller monthly but lagged)
  * Mortgage-rate environment as exogenous regressor on F_t
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HierarchicalHousingDGP:
    """Realised paths + ground-truth params for one hierarchical-housing simulation."""
    y: np.ndarray                  # (T, R) regional observables
    national_factor: np.ndarray    # (T,) F_t
    regional_idio: np.ndarray      # (T, R) λ_{r,t}
    sigma_F: float                 # national factor innovation SD
    rhos: np.ndarray               # (R,) regional AR(1) persistences
    sigma_lambdas: np.ndarray      # (R,) regional idiosyncratic SD
    betas: np.ndarray              # (R,) regional loadings on F
    sigma_eps: np.ndarray          # (R,) observation noise SD per region
    region_names: list[str]


def simulate_hierarchical_housing(
    T: int = 360,                          # 30 yrs of monthly data
    region_names: tuple[str, ...] = ("NE", "MW", "S", "W"),
    sigma_F: float = 0.15,
    rhos: tuple[float, ...] = (0.85, 0.80, 0.90, 0.88),
    sigma_lambdas: tuple[float, ...] = (0.20, 0.15, 0.18, 0.22),
    betas: tuple[float, ...] = (1.0, 0.7, 1.1, 1.3),
    sigma_eps: tuple[float, ...] = (0.10, 0.10, 0.10, 0.10),
    initial_F: float = 0.0,
    seed: int = 0,
) -> HierarchicalHousingDGP:
    """Generate a hierarchical-housing path with R regions and known latents.

    Defaults loosely calibrated to U.S. monthly housing CPI YoY-growth
    dynamics: national factor walks slowly, regional AR(1) are persistent,
    West (β=1.3) tracks national more aggressively than Midwest (β=0.7).
    """
    rng = np.random.default_rng(seed)
    R = len(region_names)
    if not (len(rhos) == len(sigma_lambdas) == len(betas) == len(sigma_eps) == R):
        raise ValueError("rhos / sigma_lambdas / betas / sigma_eps must all "
                          "have length R")

    # National factor as random walk
    F = np.empty(T)
    F[0] = initial_F
    for t in range(1, T):
        F[t] = F[t - 1] + rng.normal(0.0, sigma_F)

    # Regional idiosyncratic AR(1)
    lam = np.empty((T, R))
    for r in range(R):
        lam[0, r] = rng.normal(0.0, sigma_lambdas[r] /
                                  np.sqrt(max(1 - rhos[r] ** 2, 1e-3)))
        for t in range(1, T):
            lam[t, r] = (rhos[r] * lam[t - 1, r]
                          + rng.normal(0.0, sigma_lambdas[r]))

    # Observable
    y = np.empty((T, R))
    for r in range(R):
        eps_r = rng.normal(0.0, sigma_eps[r], size=T)
        y[:, r] = betas[r] * F + lam[:, r] + eps_r

    return HierarchicalHousingDGP(
        y=y,
        national_factor=F,
        regional_idio=lam,
        sigma_F=sigma_F,
        rhos=np.asarray(rhos, dtype=float),
        sigma_lambdas=np.asarray(sigma_lambdas, dtype=float),
        betas=np.asarray(betas, dtype=float),
        sigma_eps=np.asarray(sigma_eps, dtype=float),
        region_names=list(region_names),
    )
