"""Phase 1.4 archetype model — VECM tradables with tariff dummy.

Bivariate Vector Error Correction Model with a known cointegrating
vector β = (1, −1) (so the cointegrating relation is the *spread*
``z_t = y_{1t} − y_{2t}``) and a tariff regime dummy:

    Δy_t  =  α (z_{t-1} − μ − θ D_{t-1}) + ε_t,    ε_t ~ N(0, Σ)

per equation (i = 1, 2):

    Δy_{it}  =  α_i z_{t-1}  +  c_i  +  γ_i D_{t-1}  +  ε_{it}

with ``c_i = −α_i μ`` and ``γ_i = −α_i θ``. Linear in the parameters,
estimated by per-equation OLS — exact MLE under joint normality of the
residuals (the SUR efficiency gain is zero when both equations have the
same right-hand side).

After fitting, recover the structural parameters by:

    μ_i = −c_i / α_i,   θ_i = −γ_i / α_i

These should agree across the two equations under the null that the
model is correctly specified. The cross-equation residual covariance is
the empirical covariance of the OLS residuals.

Deliberately scoped:

  * **β = (1, −1) is assumed known.** Johansen's procedure for unknown β
    is the textbook approach (and what statsmodels.VECM does), but for
    the clothing-vs-imports use case the equilibrium spread is theory-
    motivated, not data-discovered. Phase 1.5+ work can swap in Johansen
    if needed.
  * **No SV** — homoskedastic Σ. SV layer arrives with Phase 1.2 SV work.
  * **Single regime dummy.** Multiple structural breaks would need
    multiple dummies; trivial extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VECMFit:
    """OLS fit of the bivariate VECM equations."""
    alpha_1: float           # speed of adjustment for y_1
    alpha_2: float           # speed of adjustment for y_2
    mu_1: float              # equilibrium spread implied by equation 1
    mu_2: float              # equilibrium spread implied by equation 2
    theta_1: float           # tariff shift implied by equation 1
    theta_2: float           # tariff shift implied by equation 2
    c_1: float               # intercept of equation 1 in OLS form
    c_2: float
    gamma_1: float           # tariff dummy coef of equation 1 in OLS form
    gamma_2: float
    sigma_1: float           # residual SD eq 1
    sigma_2: float           # residual SD eq 2
    rho: float               # residual correlation
    n_train: int


def fit_vecm(y1: np.ndarray, y2: np.ndarray,
               regime: np.ndarray) -> VECMFit:
    """Fit the bivariate VECM with known β=(1,-1) and tariff dummy.

    ``y1``, ``y2``, ``regime`` are 1D arrays of equal length T.
    Differences are taken so the regression has T-1 rows.
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    regime = np.asarray(regime, dtype=float)
    T = len(y1)
    if not (len(y2) == T == len(regime)):
        raise ValueError("y1, y2, regime must have the same length")
    if T < 50:
        raise ValueError(f"need ≥50 observations, got {T}")

    z = y1 - y2
    dy1 = np.diff(y1)
    dy2 = np.diff(y2)
    z_lag = z[:-1]
    d_lag = regime[:-1]

    # Per-equation OLS: Δy_i = α_i z_{t-1} + c_i + γ_i D_{t-1} + ε_i
    X = np.column_stack([z_lag, np.ones_like(z_lag), d_lag])

    coefs_1, *_ = np.linalg.lstsq(X, dy1, rcond=None)
    coefs_2, *_ = np.linalg.lstsq(X, dy2, rcond=None)

    alpha_1, c_1, gamma_1 = (float(coefs_1[0]), float(coefs_1[1]),
                                  float(coefs_1[2]))
    alpha_2, c_2, gamma_2 = (float(coefs_2[0]), float(coefs_2[1]),
                                  float(coefs_2[2]))

    # Structural parameters: μ_i = -c_i/α_i, θ_i = -γ_i/α_i
    mu_1 = -c_1 / alpha_1 if alpha_1 != 0 else float("nan")
    mu_2 = -c_2 / alpha_2 if alpha_2 != 0 else float("nan")
    theta_1 = -gamma_1 / alpha_1 if alpha_1 != 0 else float("nan")
    theta_2 = -gamma_2 / alpha_2 if alpha_2 != 0 else float("nan")

    # Residual covariance
    res_1 = dy1 - X @ coefs_1
    res_2 = dy2 - X @ coefs_2
    sigma_1 = float(np.std(res_1, ddof=3))
    sigma_2 = float(np.std(res_2, ddof=3))
    if sigma_1 > 0 and sigma_2 > 0:
        rho = float(np.corrcoef(res_1, res_2)[0, 1])
    else:
        rho = 0.0

    return VECMFit(
        alpha_1=alpha_1, alpha_2=alpha_2,
        mu_1=mu_1, mu_2=mu_2,
        theta_1=theta_1, theta_2=theta_2,
        c_1=c_1, c_2=c_2,
        gamma_1=gamma_1, gamma_2=gamma_2,
        sigma_1=sigma_1, sigma_2=sigma_2, rho=rho,
        n_train=len(z_lag),
    )
