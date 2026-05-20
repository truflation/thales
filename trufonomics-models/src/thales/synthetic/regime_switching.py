"""Synthetic DGP for the Markov-switching variance archetype (Phase 1.3).

The simplest Hamilton 1989 model:

  y_t  =  μ  +  ε_t,         ε_t ~ N(0, σ²_{S_t})
  S_t  ∈  {0, 1}             (low-vol / high-vol regime)
  P(S_t = j | S_{t-1} = i)  =  P_{ij}    (Markov transition matrix)

Captures the "calm vs turbulent" regime pattern of inflation series:
long stretches of low-volatility steady-state interrupted by high-vol
shocks (commodity spikes, COVID, tariff repricing).

The full UC-SV-MS (Phase 1.3 production) layers an unobserved-
components level on top and adds stochastic volatility within each
regime. That requires MCMC. This synthetic-recovery DGP is the
**Hamilton-only** core, which is the necessary first step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MarkovSwitchingDGP:
    """Realised regime path + observable for one Markov-switching simulation."""
    y: np.ndarray
    regime: np.ndarray              # int regime label (0 or 1)
    mu: float
    sigma_low: float
    sigma_high: float
    p_stay_low: float               # P(S_t=0 | S_{t-1}=0)
    p_stay_high: float              # P(S_t=1 | S_{t-1}=1)


def simulate_markov_switching(
    T: int = 600,
    mu: float = 0.0,
    sigma_low: float = 0.5,
    sigma_high: float = 2.0,
    p_stay_low: float = 0.95,
    p_stay_high: float = 0.85,
    initial_regime: int = 0,
    seed: int = 0,
) -> MarkovSwitchingDGP:
    """Generate a 2-state Markov-switching variance path.

    ``p_stay_low`` and ``p_stay_high`` are the diagonal entries of the
    transition matrix — typical inflation regimes spend long stretches
    in the low-vol state and shorter stretches in high-vol.
    """
    rng = np.random.default_rng(seed)
    regime = np.empty(T, dtype=int)
    regime[0] = int(initial_regime)
    for t in range(1, T):
        if regime[t - 1] == 0:
            regime[t] = 0 if rng.random() < p_stay_low else 1
        else:
            regime[t] = 1 if rng.random() < p_stay_high else 0

    sigmas = np.where(regime == 0, sigma_low, sigma_high)
    eps = rng.normal(0.0, sigmas)
    y = mu + eps

    return MarkovSwitchingDGP(
        y=y, regime=regime, mu=mu,
        sigma_low=sigma_low, sigma_high=sigma_high,
        p_stay_low=p_stay_low, p_stay_high=p_stay_high,
    )


# ─── UC + MS (Phase 1.3 expanded — adds continuous level state) ───────────


@dataclass
class UCMSDGP:
    """Unobserved-Components + Markov-Switching DGP — Phase 1.3 core."""
    y: np.ndarray                  # observable
    mu_path: np.ndarray            # latent level state μ_t
    regime: np.ndarray             # 0/1 regime path
    sigma_eta: float               # state-evolution SD (level walk)
    sigma_low: float               # observation SD in regime 0
    sigma_high: float              # observation SD in regime 1
    p_stay_low: float
    p_stay_high: float


@dataclass
class SVDGP:
    """Stochastic-volatility (no regime, no level walk) DGP.

    Canonical Kim-Shephard 1998 specification:
        y_t      = exp(h_t / 2) · ε_t,        ε_t ~ N(0, 1)
        h_t      = μ_h + φ (h_{t-1} - μ_h) + ν_t,   ν_t ~ N(0, σ_h²)

    h_t is the latent log-volatility; φ controls persistence (typically
    > 0.9 for daily financial data, > 0.7 for monthly).
    """
    y: np.ndarray            # observable
    h_path: np.ndarray       # latent log-volatility path
    mu_h: float              # long-run mean of h
    phi: float               # AR(1) persistence
    sigma_h: float           # innovation SD of h


def simulate_sv(
    T: int = 1000,
    mu_h: float = -2.0,        # exp(-2/2) ≈ 0.37 — moderate baseline vol
    phi: float = 0.95,
    sigma_h: float = 0.3,
    seed: int = 0,
) -> SVDGP:
    """Generate a stochastic-volatility path with known latents."""
    rng = np.random.default_rng(seed)
    h = np.empty(T)
    h[0] = mu_h
    for t in range(1, T):
        h[t] = mu_h + phi * (h[t - 1] - mu_h) + rng.normal(0, sigma_h)
    eps = rng.normal(0, 1, size=T)
    y = np.exp(h / 2) * eps
    return SVDGP(y=y, h_path=h, mu_h=mu_h, phi=phi, sigma_h=sigma_h)


@dataclass
class UCSVMSDGP:
    """Full UC + SV + MS DGP — Phase 1.3 complete.

    All three latent processes:
      μ_t  (continuous level, random walk)
      h_t  (continuous log-vol, AR(1) zero-mean)
      S_t  (discrete regime, Markov)

    Observation:
        y_t = μ_t + ε_t,  ε_t ~ N(0, σ²_{S_t} · exp(h_t))

    Total observation variance is σ²_{S_t} multiplied by exp(h_t).
    σ_{S_t} is the regime-level baseline; exp(h_t) is the within-regime
    log-vol modulation. h_t is centered at 0 by AR(1) zero-mean
    parameterization, so σ_{S_t} interpretation is clean.
    """
    y: np.ndarray
    mu_path: np.ndarray
    h_path: np.ndarray
    regime: np.ndarray
    sigma_eta: float
    sigma_low: float
    sigma_high: float
    p_stay_low: float
    p_stay_high: float
    phi: float
    sigma_h: float


@dataclass
class MSSVDGP:
    """MS + SV combined DGP — no level-walk state.

        y_t  =  μ  +  ε_t,        ε_t ~ N(0, σ²_{S_t} · exp(h_t))
        S_t  Markov on {0, 1}
        h_t  =  φ h_{t-1} + ν_t,   ν_t ~ N(0, σ_h²)   zero-mean AR(1)

    For already-differenced (mean-reverting) YoY targets where adding a
    UC level walk over-fits and absorbs regime variance.
    """
    y: np.ndarray
    h_path: np.ndarray
    regime: np.ndarray
    mu: float
    sigma_low: float
    sigma_high: float
    p_stay_low: float
    p_stay_high: float
    phi: float
    sigma_h: float


def simulate_ms_sv(
    T: int = 400,
    mu: float = 2.5,
    sigma_low: float = 0.3,
    sigma_high: float = 1.2,
    p_stay_low: float = 0.95,
    p_stay_high: float = 0.85,
    phi: float = 0.95,
    sigma_h: float = 0.3,
    initial_regime: int = 0,
    seed: int = 0,
) -> MSSVDGP:
    """Generate a MS + SV path (no level walk)."""
    rng = np.random.default_rng(seed)

    regime = np.empty(T, dtype=int)
    regime[0] = int(initial_regime)
    for t in range(1, T):
        if regime[t - 1] == 0:
            regime[t] = 0 if rng.random() < p_stay_low else 1
        else:
            regime[t] = 1 if rng.random() < p_stay_high else 0

    h_path = np.empty(T)
    h_path[0] = rng.normal(0.0, sigma_h / np.sqrt(max(1 - phi**2, 1e-3)))
    for t in range(1, T):
        h_path[t] = phi * h_path[t - 1] + rng.normal(0.0, sigma_h)

    sigma_regime = np.where(regime == 0, sigma_low, sigma_high)
    eps = rng.normal(0.0, 1.0, size=T) * sigma_regime * np.exp(h_path / 2)
    y = mu + eps

    return MSSVDGP(
        y=y, h_path=h_path, regime=regime,
        mu=mu, sigma_low=sigma_low, sigma_high=sigma_high,
        p_stay_low=p_stay_low, p_stay_high=p_stay_high,
        phi=phi, sigma_h=sigma_h,
    )


def simulate_uc_sv_ms(
    T: int = 400,
    initial_level: float = 0.0,
    sigma_eta: float = 0.05,
    sigma_low: float = 0.4,
    sigma_high: float = 1.5,
    p_stay_low: float = 0.95,
    p_stay_high: float = 0.85,
    phi: float = 0.95,
    sigma_h: float = 0.3,
    initial_regime: int = 0,
    seed: int = 0,
) -> UCSVMSDGP:
    """Generate a full UC + SV + MS path with all latents tracked.

    h_t is generated as a zero-mean AR(1) (stationary mean = 0 → exp(h)
    has multiplicative effect of 1 on average). σ_{low/high} provides the
    regime-level baseline.
    """
    rng = np.random.default_rng(seed)

    # Regime path
    regime = np.empty(T, dtype=int)
    regime[0] = int(initial_regime)
    for t in range(1, T):
        if regime[t - 1] == 0:
            regime[t] = 0 if rng.random() < p_stay_low else 1
        else:
            regime[t] = 1 if rng.random() < p_stay_high else 0

    # Level path
    eta = rng.normal(0.0, sigma_eta, size=T)
    mu_path = np.empty(T)
    mu_path[0] = initial_level
    for t in range(1, T):
        mu_path[t] = mu_path[t - 1] + eta[t]

    # Log-vol path: zero-mean AR(1)
    h_path = np.empty(T)
    h_path[0] = rng.normal(0.0, sigma_h / np.sqrt(max(1 - phi**2, 1e-3)))
    for t in range(1, T):
        h_path[t] = phi * h_path[t - 1] + rng.normal(0.0, sigma_h)

    sigma_regime = np.where(regime == 0, sigma_low, sigma_high)
    eps = rng.normal(0.0, 1.0, size=T) * sigma_regime * np.exp(h_path / 2)
    y = mu_path + eps

    return UCSVMSDGP(
        y=y, mu_path=mu_path, h_path=h_path, regime=regime,
        sigma_eta=sigma_eta,
        sigma_low=sigma_low, sigma_high=sigma_high,
        p_stay_low=p_stay_low, p_stay_high=p_stay_high,
        phi=phi, sigma_h=sigma_h,
    )


def simulate_uc_ms(
    T: int = 600,
    initial_level: float = 0.0,
    sigma_eta: float = 0.05,
    sigma_low: float = 0.4,
    sigma_high: float = 1.5,
    p_stay_low: float = 0.95,
    p_stay_high: float = 0.85,
    initial_regime: int = 0,
    seed: int = 0,
) -> UCMSDGP:
    """Generate a UC+MS path: latent level random walk + regime-switched
    observation variance.

    State equations:
        μ_t = μ_{t-1} + η_t,    η_t ~ N(0, σ_η²)
        S_t Markov on {0, 1} with transition matrix P
    Observation:
        y_t = μ_t + ε_t,        ε_t ~ N(0, σ²_{S_t})

    The regime affects ONLY the observation noise — the level walks at
    a constant rate σ_η regardless of regime. (A natural extension is to
    also switch σ_η per regime; not modeled here.)
    """
    rng = np.random.default_rng(seed)
    regime = np.empty(T, dtype=int)
    regime[0] = int(initial_regime)
    for t in range(1, T):
        if regime[t - 1] == 0:
            regime[t] = 0 if rng.random() < p_stay_low else 1
        else:
            regime[t] = 1 if rng.random() < p_stay_high else 0

    eta = rng.normal(0.0, sigma_eta, size=T)
    mu_path = np.empty(T)
    mu_path[0] = initial_level
    for t in range(1, T):
        mu_path[t] = mu_path[t - 1] + eta[t]

    sigmas = np.where(regime == 0, sigma_low, sigma_high)
    eps = rng.normal(0.0, sigmas)
    y = mu_path + eps

    return UCMSDGP(
        y=y, mu_path=mu_path, regime=regime,
        sigma_eta=sigma_eta,
        sigma_low=sigma_low, sigma_high=sigma_high,
        p_stay_low=p_stay_low, p_stay_high=p_stay_high,
    )
