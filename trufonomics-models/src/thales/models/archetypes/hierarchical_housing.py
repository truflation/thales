"""Phase 1.5 archetype — Hierarchical Housing Dynamic Factor Model.

JAX-based multivariate Kalman filter + RTS smoother for a hierarchical
dynamic factor model:

    State α_t  =  [ F_t,  λ_{1,t},  λ_{2,t},  …,  λ_{R,t} ]   dim K = 1 + R

    Transition T (K × K):
        F_t      =  F_{t-1}                    (random walk)
        λ_{r,t}  =  ρ_r · λ_{r,t-1}             (AR(1) per region)

    Observation Z (R × K):
        y_{r,t}  =  β_r · F_t  +  λ_{r,t}  +  ε_{r,t}

    Q (K × K)  =  diag(σ_F², σ_{λ,1}², …, σ_{λ,R}²)
    R_obs (R × R) =  diag(σ_{ε,1}², …, σ_{ε,R}²)

Hyperparameter set: σ_F, ρ_r, σ_{λ,r}, β_r, σ_{ε,r} → 1 + 4·R parameters
for R regions. Fit by ML via JAX-LBFGS; runs on CPU (slow) or GPU (fast).

Built JAX-native end-to-end so the same code runs on the Vast.ai A100
without modification — set ``JAX_PLATFORMS=cuda`` and the filter
JIT-compiles to GPU kernels.

Phase 1.5+ extensions (not implemented here):
  * Owned vs rented sub-block — duplicate state structure with
    cross-block correlations
  * Mariano-Murasawa mixed-frequency observation — Case-Shiller is
    monthly with 2-month lag
  * Mortgage-rate exogenous regressor on F_t — adds one slope param
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.optimize import minimize as jax_minimize


@dataclass
class HierarchicalHousingFit:
    """Posterior summary from a hierarchical-housing fit."""
    sigma_F: float
    rhos: np.ndarray            # (R,)
    sigma_lambdas: np.ndarray   # (R,)
    betas: np.ndarray           # (R,)
    sigma_eps: np.ndarray       # (R,)

    F_smoothed: np.ndarray      # (T,) — smoothed national factor
    lambda_smoothed: np.ndarray # (T, R) — smoothed regional idiosyncratic

    log_likelihood: float
    n_iter: int
    region_names: list[str]


# ─── State-space construction ─────────────────────────────────────────────


def _pack_params(sigma_F: float, rhos: jnp.ndarray,
                   sigma_lambdas: jnp.ndarray, betas: jnp.ndarray,
                   sigma_eps: jnp.ndarray) -> jnp.ndarray:
    """Flatten params for optimizer. Log/logit-transform for unconstrained
    optimization."""
    R = len(rhos)
    return jnp.concatenate([
        jnp.atleast_1d(jnp.log(sigma_F)),
        # logit ρ ∈ (0, 1) — assume positive-AR housing
        jnp.log(rhos / (1.0 - rhos + 1e-9)),
        jnp.log(sigma_lambdas),
        betas,                                      # unconstrained
        jnp.log(sigma_eps),
    ])


def _unpack_params(theta: jnp.ndarray, R: int) -> tuple:
    """Inverse of _pack_params."""
    idx = 0
    sigma_F = jnp.exp(theta[idx]); idx += 1
    logit_rho = theta[idx: idx + R]; idx += R
    rhos = 1.0 / (1.0 + jnp.exp(-logit_rho))
    sigma_lambdas = jnp.exp(theta[idx: idx + R]); idx += R
    betas = theta[idx: idx + R]; idx += R
    sigma_eps = jnp.exp(theta[idx: idx + R])
    return sigma_F, rhos, sigma_lambdas, betas, sigma_eps


def _build_matrices(sigma_F: jnp.ndarray, rhos: jnp.ndarray,
                      sigma_lambdas: jnp.ndarray, betas: jnp.ndarray,
                      sigma_eps: jnp.ndarray) -> tuple:
    """Construct (T_mat, Z, Q, R_obs) for a given parameter vector."""
    R = len(rhos)
    K = 1 + R
    # Transition matrix
    T_mat = jnp.zeros((K, K))
    T_mat = T_mat.at[0, 0].set(1.0)                        # F random walk
    T_mat = T_mat.at[1:, 1:].set(jnp.diag(rhos))           # regional AR(1)

    # Observation matrix Z (R × K)
    Z = jnp.zeros((R, K))
    Z = Z.at[:, 0].set(betas)                              # β loading on F
    Z = Z.at[jnp.arange(R), 1 + jnp.arange(R)].set(1.0)    # 1 on own λ_r

    # State noise Q
    Q = jnp.zeros((K, K))
    Q = Q.at[0, 0].set(sigma_F ** 2)
    Q = Q.at[jnp.arange(1, K), jnp.arange(1, K)].set(sigma_lambdas ** 2)

    # Observation noise R
    R_obs = jnp.diag(sigma_eps ** 2)

    return T_mat, Z, Q, R_obs


# ─── Kalman filter / smoother in JAX ─────────────────────────────────────


def _kalman_log_lik(y: jnp.ndarray, T_mat: jnp.ndarray, Z: jnp.ndarray,
                      Q: jnp.ndarray, R_obs: jnp.ndarray,
                      a0: jnp.ndarray, P0: jnp.ndarray) -> jnp.ndarray:
    """Forward Kalman filter, returns scalar log-likelihood.

    JAX-native, scan-based, JIT-compilable. y has shape (T, R) (already
    stacked by row).
    """
    T, R = y.shape
    K = T_mat.shape[0]

    def step(carry, y_t):
        a_filt_prev, P_filt_prev, log_lik_acc = carry
        # Predict
        a_pred = T_mat @ a_filt_prev
        P_pred = T_mat @ P_filt_prev @ T_mat.T + Q
        # Innovation
        v = y_t - Z @ a_pred                                # (R,)
        S = Z @ P_pred @ Z.T + R_obs                         # (R, R)
        # Kalman gain
        S_inv = jnp.linalg.inv(S)
        K_gain = P_pred @ Z.T @ S_inv                        # (K, R)
        a_filt = a_pred + K_gain @ v
        P_filt = P_pred - K_gain @ Z @ P_pred
        # Log-density of innovation
        sign, logdet = jnp.linalg.slogdet(S)
        log_lik_t = -0.5 * (R * jnp.log(2.0 * jnp.pi) + logdet
                              + v @ S_inv @ v)
        return (a_filt, P_filt, log_lik_acc + log_lik_t), None

    init = (a0, P0, jnp.array(0.0))
    (_, _, log_lik), _ = lax.scan(step, init, y)
    return log_lik


def _kalman_filter_full(y: jnp.ndarray, T_mat: jnp.ndarray, Z: jnp.ndarray,
                          Q: jnp.ndarray, R_obs: jnp.ndarray,
                          a0: jnp.ndarray, P0: jnp.ndarray
                          ) -> tuple[jnp.ndarray, jnp.ndarray,
                                       jnp.ndarray, jnp.ndarray]:
    """Forward filter returning full state path (for smoothing)."""
    T, R = y.shape

    def step(carry, y_t):
        a_filt_prev, P_filt_prev = carry
        a_pred = T_mat @ a_filt_prev
        P_pred = T_mat @ P_filt_prev @ T_mat.T + Q
        v = y_t - Z @ a_pred
        S = Z @ P_pred @ Z.T + R_obs
        S_inv = jnp.linalg.inv(S)
        K_gain = P_pred @ Z.T @ S_inv
        a_filt = a_pred + K_gain @ v
        P_filt = P_pred - K_gain @ Z @ P_pred
        return (a_filt, P_filt), (a_filt, P_filt, a_pred, P_pred)

    init = (a0, P0)
    _, (a_filt, P_filt, a_pred, P_pred) = lax.scan(step, init, y)
    return a_filt, P_filt, a_pred, P_pred


def _rts_smoother(a_filt: jnp.ndarray, P_filt: jnp.ndarray,
                    a_pred: jnp.ndarray, P_pred: jnp.ndarray,
                    T_mat: jnp.ndarray) -> jnp.ndarray:
    """Backward RTS smoother. Returns smoothed means."""
    T = a_filt.shape[0]
    a_sm_last = a_filt[-1]

    def step(a_sm_next, idx):
        t = idx
        try_inv = jnp.linalg.inv(P_pred[t + 1])
        J = P_filt[t] @ T_mat.T @ try_inv
        a_sm = a_filt[t] + J @ (a_sm_next - a_pred[t + 1])
        return a_sm, a_sm

    indices = jnp.arange(T - 2, -1, -1)
    _, a_sm_rev = lax.scan(step, a_sm_last, indices)
    a_sm_full = jnp.concatenate([a_sm_rev[::-1], a_sm_last[None, :]])
    return a_sm_full


# ─── Fitting ──────────────────────────────────────────────────────────────


def _neg_log_lik_jax(theta: jnp.ndarray, y: jnp.ndarray,
                       a0: jnp.ndarray, P0: jnp.ndarray, R: int) -> jnp.ndarray:
    sigma_F, rhos, sigma_lambdas, betas, sigma_eps = _unpack_params(theta, R)
    T_mat, Z, Q, R_obs = _build_matrices(sigma_F, rhos, sigma_lambdas,
                                                betas, sigma_eps)
    return -_kalman_log_lik(y, T_mat, Z, Q, R_obs, a0, P0)


def fit_hierarchical_housing(y: np.ndarray,
                                  region_names: list[str] | None = None,
                                  max_iter: int = 200,
                                  ) -> HierarchicalHousingFit:
    """Fit the hierarchical housing DFM by ML using JAX-LBFGS.

    ``y`` is a (T, R) array of regional series. Returns the fitted
    parameters plus smoothed F_t and λ_{r,t} paths.
    """
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim != 2:
        raise ValueError(f"y must be (T, R); got shape {y_arr.shape}")
    T, R = y_arr.shape
    if T < 50:
        raise ValueError(f"need ≥50 obs, got T={T}")
    if region_names is None:
        region_names = [f"r{r}" for r in range(R)]
    if len(region_names) != R:
        raise ValueError(f"region_names has length {len(region_names)} ≠ R={R}")

    # Initial guesses
    sigma_F_init = 0.10
    rhos_init = jnp.full((R,), 0.80)
    sigma_lambdas_init = jnp.full((R,), 0.15)
    betas_init = jnp.ones((R,))
    sigma_eps_init = jnp.full((R,), 0.10)

    theta0 = _pack_params(sigma_F_init, rhos_init, sigma_lambdas_init,
                              betas_init, sigma_eps_init)

    # Diffuse initial state
    K = 1 + R
    a0 = jnp.zeros(K)
    P0 = jnp.eye(K) * 10.0

    y_jax = jnp.asarray(y_arr)

    def loss(theta):
        return _neg_log_lik_jax(theta, y_jax, a0, P0, R)

    res = jax_minimize(loss, theta0, method="BFGS",
                          options={"maxiter": max_iter})
    theta_hat = res.x

    sigma_F, rhos, sigma_lambdas, betas, sigma_eps = _unpack_params(
        theta_hat, R)
    T_mat, Z, Q, R_obs = _build_matrices(sigma_F, rhos, sigma_lambdas,
                                                betas, sigma_eps)
    a_filt, P_filt, a_pred, P_pred = _kalman_filter_full(
        y_jax, T_mat, Z, Q, R_obs, a0, P0)
    a_sm = _rts_smoother(a_filt, P_filt, a_pred, P_pred, T_mat)

    log_lik = float(-loss(theta_hat))

    return HierarchicalHousingFit(
        sigma_F=float(sigma_F),
        rhos=np.asarray(rhos),
        sigma_lambdas=np.asarray(sigma_lambdas),
        betas=np.asarray(betas),
        sigma_eps=np.asarray(sigma_eps),
        F_smoothed=np.asarray(a_sm[:, 0]),
        lambda_smoothed=np.asarray(a_sm[:, 1:]),
        log_likelihood=log_lik,
        n_iter=int(res.nit),
        region_names=region_names,
    )
