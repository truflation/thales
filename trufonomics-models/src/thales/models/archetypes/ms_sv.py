"""Phase 2.2b — MS + SV combined model (no UC layer).

For already-differenced YoY targets. The full UC + SV + MS spec
over-parameterizes monthly CPI YoY: the level walk absorbs regime
variance and the MS mechanism stays dormant (see
``results/regime/FINDINGS.md``). This module ships the same MS + SV
architecture but with a CONSTANT μ instead of a random-walking μ_t,
which is the right structural assumption for series that have already
been differenced.

    Observation:  y_t  =  μ  +  ε_t,
                  ε_t  ~  N(0, σ²_{S_t} · exp(h_t))
    Log-vol:      h_t  =  φ h_{t-1} + ν_t,   ν_t ~ N(0, σ_h²)
    Regime:       S_t  ∈ {0, 1}              Markov on transition matrix P

Inference is the same as in ``uc_sv_ms.py``: discrete S_t marginalized
in the likelihood via Hamilton 1989 forward algorithm in log-space;
NUTS samples continuous parameters + h_t path. After fitting, smoothed
regime probabilities are reconstructed via Kim 1994 backward smoother.

Use this for:
  * Monthly CPI YoY (BLS Headline / Core / PCE / Core PCE)
  * Any YoY or differenced inflation series
  * Stationary financial returns

Use the full ``uc_sv_ms.py`` for:
  * CPI level series (where the level walk is genuine secular drift)
  * Synthetic data generated with a level walk
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS


@dataclass
class MSSVFit:
    """Posterior summary for the MS + SV model."""
    mu: float
    sigma_low: float
    sigma_high: float
    p_stay_low: float
    p_stay_high: float
    phi: float
    sigma_h: float

    h_smoothed: np.ndarray              # (T,) posterior mean of h_t
    smoothed_prob_high: np.ndarray      # (T,) reconstructed via Kim smoother

    n_samples: int
    n_warmup: int
    diverging: int


def _hmm_forward_loglik(y: jnp.ndarray, mu: jnp.ndarray, h: jnp.ndarray,
                          sigma_low: jnp.ndarray, sigma_high: jnp.ndarray,
                          p00: jnp.ndarray, p11: jnp.ndarray) -> jnp.ndarray:
    """Hamilton forward algorithm in log-space for MS + SV (no UC).

    Returns log P(y_{1:T} | continuous params) with regime path
    integrated out. ``mu`` is a SCALAR (constant level), not a path.
    """
    T = y.shape[0]
    sigmas = jnp.stack([sigma_low, sigma_high])

    pi_low = (1.0 - p11) / (2.0 - p00 - p11 + 1e-9)
    log_xi_init = jnp.log(jnp.stack([pi_low, 1.0 - pi_low]) + 1e-12)

    P = jnp.array([[p00, 1.0 - p00],
                     [1.0 - p11, p11]])
    log_P = jnp.log(P + 1e-12)

    sigma_t = sigmas[None, :] * jnp.exp(h[:, None] / 2.0)   # (T, 2)
    # Constant μ — broadcasts across time
    log_lik = (-0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma_t)
                - 0.5 * ((y[:, None] - mu) / sigma_t) ** 2)   # (T, 2)

    def step(log_xi_prev, t):
        log_xi_pred = logsumexp(log_xi_prev[:, None] + log_P, axis=0)
        log_joint = log_xi_pred + log_lik[t]
        log_z = logsumexp(log_joint)
        log_xi_new = log_joint - log_z
        return log_xi_new, log_z

    _, log_zs = jax.lax.scan(step,
                                jnp.where(jnp.isfinite(log_xi_init),
                                            log_xi_init,
                                            jnp.log(jnp.array([0.5, 0.5]))),
                                jnp.arange(T))
    return log_zs.sum()


def _model(y: jnp.ndarray) -> None:
    """NumPyro model for MS + SV (no UC).

    Constant μ (one scalar) replaces the random-walking μ_t state from
    the full UC+SV+MS. This is the structural fix for already-
    differenced targets.
    """
    T = y.shape[0]

    # Constant level
    mu = numpyro.sample("mu", dist.Normal(jnp.median(y), 5.0))

    # Regime variances
    sigma_low = numpyro.sample("sigma_low", dist.HalfNormal(2.0))
    sigma_diff = numpyro.sample("sigma_diff", dist.HalfNormal(3.0))
    sigma_high = numpyro.deterministic("sigma_high", sigma_low + sigma_diff)

    # Transition probabilities
    p00 = numpyro.sample("p_stay_low", dist.Beta(20.0, 1.5))
    p11 = numpyro.sample("p_stay_high", dist.Beta(10.0, 2.0))

    # SV process
    phi = numpyro.sample("phi", dist.Beta(20.0, 1.5))
    sigma_h = numpyro.sample("sigma_h", dist.HalfNormal(0.5))

    # Log-vol path via non-centered AR(1)
    h_init_scale = sigma_h / jnp.sqrt(1.0 - phi ** 2 + 1e-6)
    h_0 = numpyro.sample("h_0", dist.Normal(0.0, h_init_scale))
    eta_unit = numpyro.sample(
        "h_innov_unit",
        dist.Normal(0.0, 1.0).expand([T - 1]).to_event(1))

    def h_step(carry, eta_t):
        h_prev = carry
        h_t = phi * h_prev + sigma_h * eta_t
        return h_t, h_t

    _, h_rest = jax.lax.scan(h_step, h_0, eta_unit)
    h = jnp.concatenate([jnp.atleast_1d(h_0), h_rest])
    numpyro.deterministic("h", h)

    # Marginalized HMM likelihood
    log_lik = _hmm_forward_loglik(y, mu, h, sigma_low, sigma_high, p00, p11)
    numpyro.factor("y_likelihood", log_lik)


def _kim_smoother_regimes(y: np.ndarray, mu: float, h: np.ndarray,
                            sigma_low: float, sigma_high: float,
                            p00: float, p11: float) -> np.ndarray:
    """Reconstruct smoothed P(S_t = 1 | y_{1:T}) from posterior-mean params."""
    T = len(y)
    P = np.array([[p00, 1.0 - p00], [1.0 - p11, p11]])
    sigmas = np.array([sigma_low, sigma_high])

    pi_low = (1.0 - p11) / (2.0 - p00 - p11 + 1e-9)
    xi_filt = np.empty((T, 2))
    xi_pred = np.empty((T, 2))
    xi_pred[0] = np.array([pi_low, 1.0 - pi_low])

    for t in range(T):
        sigma_t = sigmas * np.exp(h[t] / 2)
        log_lik = (-0.5 * np.log(2 * np.pi) - np.log(sigma_t)
                    - 0.5 * ((y[t] - mu) / sigma_t) ** 2)
        m = log_lik.max()
        f = np.exp(log_lik - m)
        joint = xi_pred[t] * f
        z = joint.sum()
        if z > 0:
            xi_filt[t] = joint / z
        else:
            xi_filt[t] = np.array([0.5, 0.5])
        if t < T - 1:
            xi_pred[t + 1] = P.T @ xi_filt[t]

    xi_smooth = np.empty_like(xi_filt)
    xi_smooth[-1] = xi_filt[-1]
    for t in range(T - 2, -1, -1):
        ratio = np.where(xi_pred[t + 1] > 0,
                          xi_smooth[t + 1] / xi_pred[t + 1], 0.0)
        xi_smooth[t] = xi_filt[t] * (P @ ratio)
        s = xi_smooth[t].sum()
        if s > 0:
            xi_smooth[t] /= s

    return xi_smooth[:, 1]


def fit_ms_sv(y: np.ndarray,
                num_warmup: int = 500,
                num_samples: int = 500,
                num_chains: int = 1,
                seed: int = 0,
                progress_bar: bool = False,
                ) -> MSSVFit:
    """Fit MS + SV (no UC) via NumPyro NUTS.

    Marginalizes the discrete regime path inside the likelihood.
    Smoothed regime probabilities reconstructed post-hoc via Kim
    smoother on posterior-mean params/h-path.
    """
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim != 1:
        raise ValueError("y must be 1D")
    if len(y_arr) < 50:
        raise ValueError(f"need ≥50 obs, got {len(y_arr)}")

    rng_key = jax.random.PRNGKey(seed)
    kernel = NUTS(_model, target_accept_prob=0.95)
    mcmc = MCMC(kernel,
                  num_warmup=num_warmup,
                  num_samples=num_samples,
                  num_chains=num_chains,
                  progress_bar=progress_bar)
    mcmc.run(rng_key, y=jnp.asarray(y_arr))
    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()

    mu = float(np.mean(samples["mu"]))
    sigma_low = float(np.mean(samples["sigma_low"]))
    sigma_high = float(np.mean(samples["sigma_high"]))
    p00 = float(np.mean(samples["p_stay_low"]))
    p11 = float(np.mean(samples["p_stay_high"]))
    phi = float(np.mean(samples["phi"]))
    sigma_h = float(np.mean(samples["sigma_h"]))

    h_smoothed = np.asarray(samples["h"]).mean(axis=0)
    smoothed_prob_high = _kim_smoother_regimes(
        y_arr, mu, h_smoothed, sigma_low, sigma_high, p00, p11)

    diverging = 0
    if "diverging" in extra:
        diverging = int(np.asarray(extra["diverging"]).sum())

    return MSSVFit(
        mu=mu,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        p_stay_low=p00,
        p_stay_high=p11,
        phi=phi,
        sigma_h=sigma_h,
        h_smoothed=h_smoothed,
        smoothed_prob_high=smoothed_prob_high,
        n_samples=num_samples,
        n_warmup=num_warmup,
        diverging=diverging,
    )
