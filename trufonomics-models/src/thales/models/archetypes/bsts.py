"""Phase 1.2 archetype model — BSTS discretionary.

Bayesian Structural Time Series with three components:

  * **Local linear trend** — `μ_t = μ_{t-1} + δ_{t-1} + η^μ_t`
  * **Drifting slope**     — `δ_t = δ_{t-1} + η^δ_t`
  * **Dummy seasonal**     — `s_t = -Σ_{k=1..S-1} s_{t-k} + η^s_t`

Observation: `y_t = μ_t + s_t + ε_t`.

State vector (dimension `K = 2 + (period − 1)`):

    α_t = [μ_t, δ_t, s_t, s_{t-1}, s_{t-2}, …, s_{t-(period-2)}]

Transition matrix `T_mat` (K×K):

    | 1  1  0  0 …  0 |          μ ← μ + δ + ηᵘ
    | 0  1  0  0 …  0 |          δ ← δ + ηᵟ
    | 0  0 −1 −1 … −1 |          s_t ← -Σ s_{t-1..t-(S-1)}
    | 0  0  1  0 …  0 |          s_{t-1} ← s_t (shift register)
    | 0  0  0  1 …  0 |
    |       ⋮          |
    | 0  0  0  0 …  0 |

Observation matrix Z = [1, 0, 1, 0, 0, …, 0] (picks μ + current s).

State noise Q is diagonal: σ_μ² on (0,0), σ_δ² on (1,1), σ_s² on (2,2),
zeros elsewhere (the shift-register rows are deterministic).

Hyperparameters fit by ML via scipy.optimize.minimize on log-σ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class BSTSFit:
    """Posterior summary from a BSTS fit."""
    sigma_mu: float
    sigma_delta: float
    sigma_seasonal: float
    sigma_eps: float

    trend_smoothed: np.ndarray      # (T,)
    slope_smoothed: np.ndarray      # (T,)
    seasonal_smoothed: np.ndarray   # (T,)

    log_likelihood: float
    n_iter: int
    period: int


def _build_state_space(period: int,
                         sigma_mu: float, sigma_delta: float,
                         sigma_seasonal: float, sigma_eps: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Construct (T_mat, Z, Q, R) for a BSTS with given hyperparameters.

    K = 2 + (period - 1) is the state dimension.
    """
    K = 2 + (period - 1)
    T_mat = np.zeros((K, K))
    # Local linear trend
    T_mat[0, 0] = 1.0
    T_mat[0, 1] = 1.0
    T_mat[1, 1] = 1.0
    # Seasonal: s_t = -Σ s_{t-1..t-(S-1)}
    T_mat[2, 2:K] = -1.0
    # Shift register: s_{t-k} ← s_{t-k+1}
    for k in range(period - 2):
        T_mat[3 + k, 2 + k] = 1.0

    Z = np.zeros(K)
    Z[0] = 1.0   # μ
    Z[2] = 1.0   # current s

    Q = np.zeros((K, K))
    Q[0, 0] = sigma_mu ** 2
    Q[1, 1] = sigma_delta ** 2
    Q[2, 2] = sigma_seasonal ** 2

    R = sigma_eps ** 2
    return T_mat, Z, Q, R


def _kalman_filter(y: np.ndarray, T_mat: np.ndarray, Z: np.ndarray,
                     Q: np.ndarray, R: float,
                     a0: np.ndarray, P0: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, float]:
    """Multivariate Kalman filter forward pass.

    Returns (a_filt, P_filt, a_pred, P_pred, log_lik) all aligned at length T.
    a_pred[t] is the prediction for time-t state given y_1..y_{t-1};
    a_filt[t] is the update after seeing y_t.
    """
    T = len(y)
    K = T_mat.shape[0]
    a_pred = np.empty((T, K))
    P_pred = np.empty((T, K, K))
    a_filt = np.empty((T, K))
    P_filt = np.empty((T, K, K))
    log_lik = 0.0

    a_pred[0] = a0
    P_pred[0] = P0
    for t in range(T):
        if t > 0:
            a_pred[t] = T_mat @ a_filt[t - 1]
            P_pred[t] = T_mat @ P_filt[t - 1] @ T_mat.T + Q
        # Innovation
        v = y[t] - Z @ a_pred[t]
        S = Z @ P_pred[t] @ Z + R
        K_gain = P_pred[t] @ Z / S    # shape (K,)
        a_filt[t] = a_pred[t] + K_gain * v
        # P_filt = (I - K Z) P_pred
        IKZ = np.eye(K) - np.outer(K_gain, Z)
        P_filt[t] = IKZ @ P_pred[t]
        # Log-likelihood
        log_lik += -0.5 * (np.log(2.0 * np.pi * S) + v * v / S)

    return a_filt, P_filt, a_pred, P_pred, float(log_lik)


def _rts_smoother(a_filt: np.ndarray, P_filt: np.ndarray,
                    a_pred: np.ndarray, P_pred: np.ndarray,
                    T_mat: np.ndarray) -> np.ndarray:
    """Backward RTS smoother. Returns (T, K) smoothed state means."""
    T = len(a_filt)
    a_sm = np.empty_like(a_filt)
    a_sm[-1] = a_filt[-1]
    P_sm_curr = P_filt[-1].copy()
    for t in range(T - 2, -1, -1):
        # Smoother gain
        try:
            P_pred_next_inv = np.linalg.inv(P_pred[t + 1])
        except np.linalg.LinAlgError:
            P_pred_next_inv = np.linalg.pinv(P_pred[t + 1])
        J = P_filt[t] @ T_mat.T @ P_pred_next_inv
        a_sm[t] = a_filt[t] + J @ (a_sm[t + 1] - a_pred[t + 1])
        P_sm_curr = P_filt[t] + J @ (P_sm_curr - P_pred[t + 1]) @ J.T
    return a_sm


def _neg_log_lik(theta: np.ndarray, y: np.ndarray, period: int,
                   a0: np.ndarray, P0: np.ndarray) -> float:
    """For scipy.optimize.minimize. theta = log(σ_μ, σ_δ, σ_s, σ_ε)."""
    sigma_mu, sigma_delta, sigma_seasonal, sigma_eps = np.exp(theta)
    T_mat, Z, Q, R = _build_state_space(period, sigma_mu, sigma_delta,
                                              sigma_seasonal, sigma_eps)
    _, _, _, _, ll = _kalman_filter(y, T_mat, Z, Q, R, a0, P0)
    return -ll


def fit_bsts(y: np.ndarray, period: int = 12,
              max_iter: int = 400,
              ) -> BSTSFit:
    """Fit a BSTS model to a 1D series. Hyperparameters by ML.

    ``period`` is the seasonal period (12 for monthly, 7 for weekly,
    365 for daily-with-yearly).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if len(y) < 2 * period + 10:
        raise ValueError(f"need ≥{2 * period + 10} obs for period={period}, "
                          f"got {len(y)}")

    # Initial state mean: detrended level near initial y, slope ~0,
    # seasonal centered on first cycle.
    K = 2 + (period - 1)
    a0 = np.zeros(K)
    a0[0] = float(y[0])
    a0[1] = float((y[period] - y[0]) / period) if len(y) > period else 0.0
    # Seasonal slot: empirical first-cycle deviation from a rough trend
    if len(y) >= period:
        rough_trend = np.linspace(y[0], y[period - 1], period)
        a0[2: 2 + (period - 1)] = (y[: period - 1] - rough_trend[: period - 1])
    P0 = np.eye(K) * 1.0  # diffuse prior

    # Sensible scale init: σ_ε ~ residual SD from rough detrend
    rough_resid = y - np.linspace(y[0], y[-1], len(y))
    sigma_eps_init = float(np.std(rough_resid))
    sigma_mu_init = sigma_eps_init / 5.0
    sigma_delta_init = sigma_eps_init / 50.0
    sigma_seasonal_init = sigma_eps_init / 5.0

    theta0 = np.log([sigma_mu_init, sigma_delta_init,
                       sigma_seasonal_init, sigma_eps_init])

    res = minimize(
        _neg_log_lik, theta0,
        args=(y, period, a0, P0),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-5},
    )
    sigma_mu, sigma_delta, sigma_seasonal, sigma_eps = np.exp(res.x)

    T_mat, Z, Q, R = _build_state_space(period, sigma_mu, sigma_delta,
                                              sigma_seasonal, sigma_eps)
    a_filt, P_filt, a_pred, P_pred, log_lik = _kalman_filter(
        y, T_mat, Z, Q, R, a0, P0)
    a_sm = _rts_smoother(a_filt, P_filt, a_pred, P_pred, T_mat)

    return BSTSFit(
        sigma_mu=float(sigma_mu),
        sigma_delta=float(sigma_delta),
        sigma_seasonal=float(sigma_seasonal),
        sigma_eps=float(sigma_eps),
        trend_smoothed=a_sm[:, 0],
        slope_smoothed=a_sm[:, 1],
        seasonal_smoothed=a_sm[:, 2],
        log_likelihood=log_lik,
        n_iter=int(res.nit),
        period=period,
    )


# ─── Local-level variant (no slope component) ────────────────────────────


def _build_state_space_ll(period: int,
                              sigma_mu: float, sigma_seasonal: float,
                              sigma_eps: float
                              ) -> tuple[np.ndarray, np.ndarray,
                                            np.ndarray, float]:
    """Local-level (no slope) state-space construction.

    State α_t = [μ_t, s_t, s_{t-1}, …, s_{t-(period-2)}], dim K = 1 + (period-1).

    Compared to the LLT version, this drops both the δ state and the
    σ_δ hyperparameter — eliminating the σ_μ vs σ_δ identifiability
    issue documented in Phase 1.2 FINDINGS.
    """
    K = 1 + (period - 1)
    T_mat = np.zeros((K, K))
    T_mat[0, 0] = 1.0                   # μ random walk
    T_mat[1, 1:K] = -1.0                # seasonal sum-to-zero constraint
    for k in range(period - 2):
        T_mat[2 + k, 1 + k] = 1.0       # shift register

    Z = np.zeros(K)
    Z[0] = 1.0
    Z[1] = 1.0

    Q = np.zeros((K, K))
    Q[0, 0] = sigma_mu ** 2
    Q[1, 1] = sigma_seasonal ** 2

    R = sigma_eps ** 2
    return T_mat, Z, Q, R


def _neg_log_lik_ll(theta: np.ndarray, y: np.ndarray, period: int,
                       a0: np.ndarray, P0: np.ndarray) -> float:
    sigma_mu, sigma_seasonal, sigma_eps = np.exp(theta)
    T_mat, Z, Q, R = _build_state_space_ll(period, sigma_mu,
                                                  sigma_seasonal, sigma_eps)
    _, _, _, _, ll = _kalman_filter(y, T_mat, Z, Q, R, a0, P0)
    return -ll


@dataclass
class BSTSLocalLevelFit:
    """Posterior summary from a local-level BSTS fit (no slope state)."""
    sigma_mu: float
    sigma_seasonal: float
    sigma_eps: float
    trend_smoothed: np.ndarray
    seasonal_smoothed: np.ndarray
    log_likelihood: float
    n_iter: int
    period: int


def fit_bsts_local_level(y: np.ndarray, period: int = 12,
                           max_iter: int = 400,
                           ) -> BSTSLocalLevelFit:
    """Fit a local-level BSTS (μ random walk + seasonal + irregular).

    Drops the slope state δ vs ``fit_bsts``. Three hyperparameters:
    σ_μ, σ_s, σ_ε. No σ_μ / σ_δ identifiability issue. Recommended when
    the underlying trend doesn't have a strong drift component
    (true for most CPI YoY series).

    For data with a clear deterministic trend, prefer the LLT version
    (``fit_bsts``) and accept the parameter-decomposition non-uniqueness
    — the combined trend recovers regardless.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if len(y) < 2 * period + 10:
        raise ValueError(f"need ≥{2 * period + 10} obs for period={period}, "
                          f"got {len(y)}")

    K = 1 + (period - 1)
    a0 = np.zeros(K)
    a0[0] = float(y[0])
    if len(y) >= period:
        rough_trend = np.linspace(y[0], y[period - 1], period)
        a0[1: 1 + (period - 1)] = (y[: period - 1]
                                       - rough_trend[: period - 1])
    P0 = np.eye(K) * 1.0

    rough_resid = y - np.linspace(y[0], y[-1], len(y))
    sigma_eps_init = float(np.std(rough_resid))
    sigma_mu_init = sigma_eps_init / 5.0
    sigma_seasonal_init = sigma_eps_init / 5.0
    theta0 = np.log([sigma_mu_init, sigma_seasonal_init, sigma_eps_init])

    res = minimize(
        _neg_log_lik_ll, theta0,
        args=(y, period, a0, P0),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-5},
    )
    sigma_mu, sigma_seasonal, sigma_eps = np.exp(res.x)

    T_mat, Z, Q, R = _build_state_space_ll(period, sigma_mu,
                                                  sigma_seasonal, sigma_eps)
    a_filt, P_filt, a_pred, P_pred, log_lik = _kalman_filter(
        y, T_mat, Z, Q, R, a0, P0)
    a_sm = _rts_smoother(a_filt, P_filt, a_pred, P_pred, T_mat)

    return BSTSLocalLevelFit(
        sigma_mu=float(sigma_mu),
        sigma_seasonal=float(sigma_seasonal),
        sigma_eps=float(sigma_eps),
        trend_smoothed=a_sm[:, 0],
        seasonal_smoothed=a_sm[:, 1],
        log_likelihood=log_lik,
        n_iter=int(res.nit),
        period=period,
    )
