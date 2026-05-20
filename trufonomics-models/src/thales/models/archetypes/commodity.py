"""Phase 1.1 archetype model — Commodity pass-through (TVP linear regression).

Pure-numpy Kalman filter + RTS smoother for a 1D state-space model:

    Observation:  y_t = α + β_t · x_t + ε_t,   ε_t ~ N(0, σ²)
    State:        β_t = β_{t-1} + η^β_t,        η^β_t ~ N(0, σ_β²)

with ``y = log(retail_price)`` and ``x = log(commodity_price)``. Captures
the time-varying pass-through coefficient between commodity and retail.

This is the **filtered/smoothed** version — full TVP-VECM-SV with
stochastic volatility lives at Phase 1.2 and requires MCMC (PyMC /
NumPyro). The current model:

  * Recovers ``β_t`` smoothed estimates within ~0.05 of the true latent
    path on the synthetic DGP (Pearson > 0.85, MAE < 0.05).
  * Fits hyperparameters ``α, σ², σ_β²`` by ML via scipy.optimize.minimize.
  * Returns the full posterior path of β plus α and the noise variances.

Deliberately does NOT model:

  * Stochastic volatility on ε_t (Phase 1.2)
  * Cointegration / VECM error-correction layer (Phase 1.2)
  * Multiple commodities or hierarchical regional pass-through (Phase 1.5)

This module's role is to **validate the TVP estimation core** in
isolation, independent of the SV and VECM extensions. Once recovery is
proven here, the SV and VECM extensions are additive surgery on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class TVPFit:
    """Posterior summary from a TVP-Commodity model fit."""
    alpha: float
    sigma_eps: float        # observation noise SD
    sigma_beta: float       # state innovation SD
    beta_filtered: np.ndarray  # E[β_t | y_1..t]   shape (T,)
    beta_smoothed: np.ndarray  # E[β_t | y_1..T]   shape (T,)
    log_likelihood: float
    n_iter: int


def _kalman_filter(y: np.ndarray, x: np.ndarray,
                     alpha: float, sigma_eps: float, sigma_beta: float,
                     beta_0: float = 0.5,
                     P_0: float = 1.0,
                     ) -> tuple[np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, float]:
    """Forward pass. Returns (β_filt, P_filt, β_pred, P_pred, log_lik).

    All shape (T,). β_pred[t] is the one-step-ahead prediction of β_t
    given observations 1..t-1; β_filt[t] is the update after seeing y_t.
    """
    T = len(y)
    sigma2_eps = sigma_eps ** 2
    sigma2_beta = sigma_beta ** 2

    beta_pred = np.empty(T)
    P_pred = np.empty(T)
    beta_filt = np.empty(T)
    P_filt = np.empty(T)
    log_lik = 0.0

    beta_pred[0] = beta_0
    P_pred[0] = P_0
    for t in range(T):
        # Predict (already done for t=0)
        if t > 0:
            beta_pred[t] = beta_filt[t - 1]
            P_pred[t] = P_filt[t - 1] + sigma2_beta
        # Innovation
        v = y[t] - alpha - x[t] * beta_pred[t]
        S = x[t] ** 2 * P_pred[t] + sigma2_eps
        K = P_pred[t] * x[t] / S
        beta_filt[t] = beta_pred[t] + K * v
        P_filt[t] = (1.0 - K * x[t]) * P_pred[t]
        # Log-likelihood contribution
        log_lik += -0.5 * (np.log(2.0 * np.pi * S) + v * v / S)

    return beta_filt, P_filt, beta_pred, P_pred, float(log_lik)


def _rts_smoother(beta_filt: np.ndarray, P_filt: np.ndarray,
                    beta_pred: np.ndarray, P_pred: np.ndarray,
                    sigma_beta: float) -> np.ndarray:
    """RTS smoother. Backward pass to compute E[β_t | y_1..T]."""
    T = len(beta_filt)
    sigma2_beta = sigma_beta ** 2
    beta_smoothed = np.empty(T)
    P_smoothed = np.empty(T)
    beta_smoothed[-1] = beta_filt[-1]
    P_smoothed[-1] = P_filt[-1]
    for t in range(T - 2, -1, -1):
        # Smoother gain
        P_pred_next = P_filt[t] + sigma2_beta
        if P_pred_next > 0:
            J = P_filt[t] / P_pred_next
        else:
            J = 0.0
        beta_smoothed[t] = beta_filt[t] + J * (beta_smoothed[t + 1]
                                                  - beta_filt[t])
        P_smoothed[t] = P_filt[t] + J ** 2 * (P_smoothed[t + 1]
                                                 - P_pred_next)
    return beta_smoothed


def _neg_log_lik(theta: np.ndarray, y: np.ndarray,
                   x: np.ndarray, beta_0: float, P_0: float) -> float:
    """For scipy.optimize.minimize. theta = [alpha, log_sigma_eps, log_sigma_beta].

    Log-parameterized to keep variances positive without bounds.
    """
    alpha = theta[0]
    sigma_eps = np.exp(theta[1])
    sigma_beta = np.exp(theta[2])
    _, _, _, _, ll = _kalman_filter(y, x, alpha, sigma_eps, sigma_beta,
                                       beta_0=beta_0, P_0=P_0)
    return -ll


def fit_tvp_commodity(commodity: np.ndarray, retail: np.ndarray,
                        beta_0: float = 0.5,
                        P_0: float = 1.0,
                        max_iter: int = 200,
                        seed_alpha: float | None = None,
                        ) -> TVPFit:
    """Fit α, σ_ε, σ_β by maximum likelihood; return smoothed β path.

    ``commodity``, ``retail`` are 1D log-level arrays of equal length.
    ``beta_0`` is the prior mean for β_0 (init the filter); ``P_0`` is its
    prior variance. Defaults are deliberately diffuse (P_0 = 1.0) so
    initial uncertainty doesn't dominate.

    Likelihood is concave-ish but not globally so — use a sensible
    starting point near OLS for α and modest noise SDs.
    """
    y = np.asarray(retail, dtype=float)
    x = np.asarray(commodity, dtype=float)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("commodity and retail must be 1D arrays of same length")
    if len(y) < 50:
        raise ValueError(f"need ≥50 observations, got {len(y)}")

    # OLS-based initial guess
    X = np.column_stack([np.ones_like(x), x])
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_init = float(coefs[0]) if seed_alpha is None else float(seed_alpha)
    resid = y - X @ coefs
    sigma_eps_init = float(np.std(resid, ddof=1))
    sigma_beta_init = max(sigma_eps_init / 10.0, 1e-3)

    theta0 = np.array([alpha_init, np.log(sigma_eps_init),
                          np.log(sigma_beta_init)])

    res = minimize(
        _neg_log_lik, theta0,
        args=(y, x, beta_0, P_0),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-5, "fatol": 1e-6},
    )

    alpha_hat = float(res.x[0])
    sigma_eps_hat = float(np.exp(res.x[1]))
    sigma_beta_hat = float(np.exp(res.x[2]))

    beta_filt, P_filt, beta_pred, P_pred, log_lik = _kalman_filter(
        y, x, alpha_hat, sigma_eps_hat, sigma_beta_hat,
        beta_0=beta_0, P_0=P_0)
    beta_smoothed = _rts_smoother(beta_filt, P_filt, beta_pred, P_pred,
                                      sigma_beta_hat)

    return TVPFit(
        alpha=alpha_hat,
        sigma_eps=sigma_eps_hat,
        sigma_beta=sigma_beta_hat,
        beta_filtered=beta_filt,
        beta_smoothed=beta_smoothed,
        log_likelihood=log_lik,
        n_iter=int(res.nit),
    )
