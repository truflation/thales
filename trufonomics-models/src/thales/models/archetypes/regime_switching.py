"""Phase 1.3 archetype model — Markov-switching variance (Hamilton 1989).

Two-state Markov-switching model for the observation variance:

  y_t  =  μ  +  ε_t,        ε_t ~ N(0, σ²_{S_t})
  S_t  ∈  {0, 1}            with transition matrix P

Hamilton's 1989 filter computes:

  ξ_{t|t}    =  P(S_t = · | y_{1:t})    — filtered regime probabilities
  ξ_{t|T}    =  P(S_t = · | y_{1:T})    — smoothed (Kim 1994)

Hyperparameters fit by ML over (μ, σ_0, σ_1, p_00, p_11) using
scipy.optimize, log-parameterized on σ and logit-parameterized on p
to keep them in valid ranges.

Pure numpy. Two-state restriction keeps the filter recursion
tractable; extending to N regimes is a trivial enlargement of the
filter loop.

What this is NOT:

  * **Not UC-SV-MS** — there's no unobserved-components level state, no
    stochastic volatility within each regime. Those layers add a
    continuous Kalman state (UC) and break linear-Gaussianity (SV);
    both are doable but require MCMC for the full posterior.
  * **Not multi-state** — restricted to 2 regimes for clarity. 3+
    regimes is a trivial loop generalization but rarely empirically
    distinguishable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class HamiltonFit:
    """Hamilton-filter posterior summary."""
    mu: float
    sigma_low: float
    sigma_high: float
    p_stay_low: float           # P(S_t=0 | S_{t-1}=0)
    p_stay_high: float          # P(S_t=1 | S_{t-1}=1)
    smoothed_prob_high: np.ndarray   # (T,) P(S_t = 1 | y_{1:T})
    filtered_prob_high: np.ndarray   # (T,) P(S_t = 1 | y_{1:t})
    log_likelihood: float
    n_iter: int


def _hamilton_filter(y: np.ndarray, mu: float,
                       sigma_low: float, sigma_high: float,
                       p00: float, p11: float
                       ) -> tuple[np.ndarray, np.ndarray, float]:
    """Hamilton 1989 forward filter for 2-state Markov-switching variance.

    Returns ``(xi_filt, xi_pred, log_lik)`` each of shape ``(T, 2)`` (or
    a scalar for log_lik). ``xi_filt[t, j] = P(S_t=j | y_{1:t})`` and
    ``xi_pred[t, j] = P(S_t=j | y_{1:t-1})``.
    """
    T = len(y)
    P = np.array([[p00, 1.0 - p00],
                    [1.0 - p11, p11]])
    sigmas = np.array([sigma_low, sigma_high])
    xi_filt = np.empty((T, 2))
    xi_pred = np.empty((T, 2))

    # Stationary initial distribution from P
    if 0.0 < p00 < 1.0 and 0.0 < p11 < 1.0:
        pi_low = (1.0 - p11) / (2.0 - p00 - p11)
    else:
        pi_low = 0.5
    xi_pred[0] = np.array([pi_low, 1.0 - pi_low])

    log_lik = 0.0
    for t in range(T):
        # Likelihood at each regime
        diff = y[t] - mu
        log_dens = -0.5 * (np.log(2 * np.pi) + 2 * np.log(sigmas)
                              + (diff / sigmas) ** 2)
        # Stable normalization via max-subtraction
        m = log_dens.max()
        f = np.exp(log_dens - m)
        joint = xi_pred[t] * f
        z = joint.sum()
        if z <= 0.0 or not np.isfinite(z):
            return xi_filt, xi_pred, -1e10
        xi_filt[t] = joint / z
        log_lik += np.log(z) + m

        if t < T - 1:
            xi_pred[t + 1] = P.T @ xi_filt[t]

    return xi_filt, xi_pred, float(log_lik)


def _kim_smoother(xi_filt: np.ndarray, xi_pred: np.ndarray,
                   p00: float, p11: float) -> np.ndarray:
    """Kim 1994 backward smoother. Returns ``xi_smooth`` shape (T, 2)."""
    T = xi_filt.shape[0]
    P = np.array([[p00, 1.0 - p00],
                    [1.0 - p11, p11]])
    xi_smooth = np.empty_like(xi_filt)
    xi_smooth[-1] = xi_filt[-1]
    for t in range(T - 2, -1, -1):
        ratio = np.where(xi_pred[t + 1] > 0,
                          xi_smooth[t + 1] / xi_pred[t + 1], 0.0)
        xi_smooth[t] = xi_filt[t] * (P @ ratio)
        z = xi_smooth[t].sum()
        if z > 0:
            xi_smooth[t] /= z
    return xi_smooth


def _neg_log_lik(theta: np.ndarray, y: np.ndarray) -> float:
    """theta = (μ, log σ_low, log σ_high, logit p00, logit p11)."""
    mu = theta[0]
    sigma_low = float(np.exp(theta[1]))
    sigma_high = float(np.exp(theta[2]))
    p00 = 1.0 / (1.0 + np.exp(-theta[3]))
    p11 = 1.0 / (1.0 + np.exp(-theta[4]))
    # Order constraint: σ_low ≤ σ_high. Penalize if violated.
    if sigma_low > sigma_high:
        return 1e10
    _, _, ll = _hamilton_filter(y, mu, sigma_low, sigma_high, p00, p11)
    return -ll


def fit_hamilton_2state(y: np.ndarray,
                          max_iter: int = 500,
                          ) -> HamiltonFit:
    """Fit a 2-state Markov-switching variance model by ML.

    Uses Nelder-Mead on (μ, log σ_low, log σ_high, logit p00, logit p11).
    The constraint σ_low ≤ σ_high is enforced via a penalty (label-
    switching prevention).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if len(y) < 50:
        raise ValueError(f"need ≥50 obs, got {len(y)}")

    # Initial guesses
    mu_init = float(np.median(y))
    sigma_init = float(np.std(y - mu_init, ddof=1))
    sigma_low_init = sigma_init * 0.7
    sigma_high_init = sigma_init * 1.5
    p00_init = 0.95
    p11_init = 0.85

    def _logit(p):
        return float(np.log(p / (1.0 - p)))

    theta0 = np.array([
        mu_init,
        np.log(sigma_low_init),
        np.log(sigma_high_init),
        _logit(p00_init),
        _logit(p11_init),
    ])

    res = minimize(_neg_log_lik, theta0, args=(y,),
                      method="Nelder-Mead",
                      options={"maxiter": max_iter,
                               "xatol": 1e-4, "fatol": 1e-4})
    mu = float(res.x[0])
    sigma_low = float(np.exp(res.x[1]))
    sigma_high = float(np.exp(res.x[2]))
    p00 = 1.0 / (1.0 + np.exp(-res.x[3]))
    p11 = 1.0 / (1.0 + np.exp(-res.x[4]))

    xi_filt, xi_pred, log_lik = _hamilton_filter(
        y, mu, sigma_low, sigma_high, p00, p11)
    xi_smooth = _kim_smoother(xi_filt, xi_pred, p00, p11)

    return HamiltonFit(
        mu=mu,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        p_stay_low=float(p00),
        p_stay_high=float(p11),
        smoothed_prob_high=xi_smooth[:, 1],
        filtered_prob_high=xi_filt[:, 1],
        log_likelihood=float(log_lik),
        n_iter=int(res.nit),
    )


# ─── UC + MS via Kim 1994 collapsing ─────────────────────────────────────


@dataclass
class UCMSFit:
    """Posterior summary from a UC+MS fit (latent level + regime-switched obs noise)."""
    sigma_eta: float
    sigma_low: float
    sigma_high: float
    p_stay_low: float
    p_stay_high: float
    mu_smoothed: np.ndarray         # (T,) E[μ_t | y_{1:T}, all regimes]
    smoothed_prob_high: np.ndarray  # (T,) P(S_t = 1 | y_{1:T})
    filtered_prob_high: np.ndarray  # (T,) P(S_t = 1 | y_{1:t})
    log_likelihood: float
    n_iter: int


def _kim_filter_ucms(y: np.ndarray,
                       sigma_eta: float, sigma_low: float, sigma_high: float,
                       p00: float, p11: float,
                       mu0: float, P0: float
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray, np.ndarray, float]:
    """Kim 1994 forward filter for UC + 2-state MS.

    Returns:
        mu_filt:  (T, 2)  E[μ_t | y_{1:t}, S_t = j], collapsed per regime
        P_filt:   (T, 2)  Var[μ_t | y_{1:t}, S_t = j], collapsed
        mu_pred:  (T, 2)  E[μ_t | y_{1:t-1}, S_t = j], pre-update predicted means
        P_pred:   (T, 2)  Var[...] pre-update
        xi_filt:  (T, 2)  P(S_t = j | y_{1:t})
        xi_pred:  (T, 2)  P(S_t = j | y_{1:t-1})
        log_lik:  scalar
    """
    T = len(y)
    se2 = sigma_eta ** 2
    sigmas2 = np.array([sigma_low ** 2, sigma_high ** 2])
    P_trans = np.array([[p00, 1.0 - p00],
                          [1.0 - p11, p11]])

    mu_filt = np.empty((T, 2))
    P_filt = np.empty((T, 2))
    mu_pred = np.empty((T, 2))
    P_pred = np.empty((T, 2))
    xi_filt = np.empty((T, 2))
    xi_pred = np.empty((T, 2))

    # Initial regime distribution (stationary)
    if 0.0 < p00 < 1.0 and 0.0 < p11 < 1.0:
        pi_low = (1.0 - p11) / (2.0 - p00 - p11)
    else:
        pi_low = 0.5
    xi_pred[0] = np.array([pi_low, 1.0 - pi_low])
    # Initial level under both regimes (same prior on μ)
    mu_pred[0] = np.array([mu0, mu0])
    P_pred[0] = np.array([P0, P0])

    log_lik = 0.0
    for t in range(T):
        # For each (i = previous regime, j = current regime), the predicted
        # state is just the previous regime's filtered state propagated:
        # μ_t|t-1[i, j] = μ_filt[t-1, i],  P_t|t-1[i, j] = P_filt[t-1, i] + σ_η²
        if t == 0:
            mu_pred_pair = np.tile(mu_pred[0][:, None], (1, 2))   # (i, j)
            P_pred_pair = np.tile(P_pred[0][:, None], (1, 2))
        else:
            mu_pred_pair = np.tile(mu_filt[t - 1][:, None], (1, 2))
            P_pred_pair = np.tile(P_filt[t - 1][:, None] + se2, (1, 2))

        # Innovation under each (i, j)
        v = y[t] - mu_pred_pair                       # (2, 2)
        S = P_pred_pair + sigmas2[None, :]            # (2, 2) — S_y depends on j
        K = P_pred_pair / S                            # Kalman gain
        mu_pair = mu_pred_pair + K * v                # (2, 2) updated means
        P_pair = (1.0 - K) * P_pred_pair              # (2, 2) updated variances

        # Conditional log-likelihood under each (i, j)
        log_dens = -0.5 * (np.log(2 * np.pi * S) + v ** 2 / S)
        m = log_dens.max()
        f = np.exp(log_dens - m)                      # (2, 2)

        # Joint posterior: P(S_{t-1}=i, S_t=j | y_{1:t}) ∝ ξ_{t-1|t-1}[i] * P_{ij} * f[i,j]
        if t == 0:
            prev_xi = xi_pred[0]                       # (2,)
        else:
            prev_xi = xi_filt[t - 1]
        joint = prev_xi[:, None] * P_trans * f         # (2, 2)
        z = joint.sum()
        if z <= 0.0 or not np.isfinite(z):
            return (mu_filt, P_filt, mu_pred, P_pred,
                    xi_filt, xi_pred, -1e10)
        joint /= z
        log_lik += np.log(z) + m                       # contribution to log-lik

        # Marginal: P(S_t = j | y_{1:t}) = sum_i joint[i, j]
        xi_filt[t] = joint.sum(axis=0)                  # (2,)
        # Save prediction for output
        # xi_pred[t+1] computed at next step

        # Collapse: for each j, weighted average over i
        for j in range(2):
            wj = joint[:, j]                           # P(S_{t-1}=i, S_t=j | y_{1:t})
            sw = wj.sum()
            if sw > 0:
                w_norm = wj / sw                       # weights for collapsing
                mu_filt[t, j] = (w_norm * mu_pair[:, j]).sum()
                # Collapse variance: weighted P + cross terms
                d = mu_pair[:, j] - mu_filt[t, j]
                P_filt[t, j] = (w_norm * (P_pair[:, j] + d ** 2)).sum()
            else:
                mu_filt[t, j] = mu_pred_pair[0, j]
                P_filt[t, j] = P_pred_pair[0, j]

        # Save predicted mu/P for next step's bookkeeping
        mu_pred[t] = mu_pred_pair[0]   # arbitrary — these are i-independent
        P_pred[t] = P_pred_pair[0]
        if t < T - 1:
            xi_pred[t + 1] = P_trans.T @ xi_filt[t]

    return mu_filt, P_filt, mu_pred, P_pred, xi_filt, xi_pred, float(log_lik)


def _kim_smoother_ucms(mu_filt: np.ndarray, P_filt: np.ndarray,
                          xi_filt: np.ndarray, xi_pred: np.ndarray,
                          sigma_eta: float, p00: float, p11: float,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Approximate Kim smoother for UC + MS.

    Two-step approach:
      1. Smooth the regime probabilities via the standard Kim 1994 backward
         recursion (same as in Hamilton-only).
      2. Compute smoothed level: marginal over smoothed regime probs of the
         per-regime filtered means. (Strict optimal Kim smoothing also
         smooths the level conditional on each regime path; the marginal
         approximation is standard practice — see Kim & Nelson 1999 §5.4.)
    """
    T = mu_filt.shape[0]
    P_trans = np.array([[p00, 1.0 - p00],
                          [1.0 - p11, p11]])
    xi_smooth = np.empty_like(xi_filt)
    xi_smooth[-1] = xi_filt[-1]
    for t in range(T - 2, -1, -1):
        ratio = np.where(xi_pred[t + 1] > 0,
                          xi_smooth[t + 1] / xi_pred[t + 1], 0.0)
        xi_smooth[t] = xi_filt[t] * (P_trans @ ratio)
        s = xi_smooth[t].sum()
        if s > 0:
            xi_smooth[t] /= s

    # Marginal smoothed level: weighted average over regimes
    mu_smoothed = (xi_smooth * mu_filt).sum(axis=1)
    return mu_smoothed, xi_smooth


def _neg_log_lik_ucms(theta: np.ndarray, y: np.ndarray,
                         mu0: float, P0: float) -> float:
    """theta = (log σ_η, log σ_low, log σ_high, logit p_00, logit p_11)."""
    sigma_eta = float(np.exp(theta[0]))
    sigma_low = float(np.exp(theta[1]))
    sigma_high = float(np.exp(theta[2]))
    p00 = 1.0 / (1.0 + np.exp(-theta[3]))
    p11 = 1.0 / (1.0 + np.exp(-theta[4]))
    if sigma_low > sigma_high:
        return 1e10
    _, _, _, _, _, _, ll = _kim_filter_ucms(
        y, sigma_eta, sigma_low, sigma_high, p00, p11, mu0, P0)
    return -ll


def fit_uc_ms(y: np.ndarray,
                max_iter: int = 600,
                n_restarts: int = 5,
                ) -> UCMSFit:
    """Fit UC + 2-state MS by ML using Kim 1994 collapsing filter.

    Estimates (σ_η, σ_low, σ_high, p_00, p_11). The level constant is
    absorbed into the level state itself (which starts diffuse). The
    σ_low ≤ σ_high constraint is enforced via penalty.

    Multi-start optimization: the likelihood is notoriously multi-modal
    in UC+MS (Kim & Nelson 1999 §5.3). Trying ``n_restarts`` starting
    points covering different (σ_low, σ_high) contrasts and keeping the
    best log-lik fit is the standard recommendation.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if len(y) < 50:
        raise ValueError(f"need ≥50 obs, got {len(y)}")

    mu0 = float(np.median(y))           # diffuse-ish prior on μ_0
    P0 = max(np.var(y), 1.0)            # high prior variance

    sigma_init = float(np.std(y - mu0, ddof=1))

    def _logit(p):
        return float(np.log(p / (1.0 - p)))

    # Multi-start grid: vary σ_low / σ_high contrast and σ_η scale
    # Keep first start same as old single-start init for backwards comp
    starts = [
        # (σ_eta_factor, σ_low_factor, σ_high_factor, p00, p11)
        (0.10,  0.70,  1.30,  0.95, 0.85),
        (0.05,  0.40,  1.50,  0.97, 0.85),
        (0.20,  0.50,  1.50,  0.95, 0.80),
        (0.05,  0.30,  2.00,  0.98, 0.80),
        (0.30,  0.60,  1.20,  0.90, 0.85),
        (0.08,  0.45,  1.80,  0.97, 0.90),
        (0.15,  0.35,  2.50,  0.98, 0.85),
    ][: max(n_restarts, 1)]

    best_res = None
    best_ll = -np.inf
    for s_eta, s_lo, s_hi, p00, p11 in starts:
        theta0 = np.array([
            np.log(sigma_init * s_eta),
            np.log(sigma_init * s_lo),
            np.log(sigma_init * s_hi),
            _logit(p00),
            _logit(p11),
        ])
        try:
            res = minimize(_neg_log_lik_ucms, theta0, args=(y, mu0, P0),
                              method="Nelder-Mead",
                              options={"maxiter": max_iter,
                                       "xatol": 1e-4, "fatol": 1e-4})
        except Exception:   # noqa: BLE001
            continue
        ll = -res.fun
        if np.isfinite(ll) and ll > best_ll:
            best_res = res
            best_ll = ll

    if best_res is None:
        raise RuntimeError("UC+MS optimization failed from all starting points")
    res = best_res

    sigma_eta = float(np.exp(res.x[0]))
    sigma_low = float(np.exp(res.x[1]))
    sigma_high = float(np.exp(res.x[2]))
    p00 = float(1.0 / (1.0 + np.exp(-res.x[3])))
    p11 = float(1.0 / (1.0 + np.exp(-res.x[4])))

    mu_filt, P_filt, mu_pred, P_pred, xi_filt, xi_pred, log_lik = (
        _kim_filter_ucms(y, sigma_eta, sigma_low, sigma_high,
                            p00, p11, mu0, P0))
    mu_smoothed, xi_smooth = _kim_smoother_ucms(
        mu_filt, P_filt, xi_filt, xi_pred, sigma_eta, p00, p11)

    return UCMSFit(
        sigma_eta=sigma_eta,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        p_stay_low=p00,
        p_stay_high=p11,
        mu_smoothed=mu_smoothed,
        smoothed_prob_high=xi_smooth[:, 1],
        filtered_prob_high=xi_filt[:, 1],
        log_likelihood=float(log_lik),
        n_iter=int(res.nit),
    )
