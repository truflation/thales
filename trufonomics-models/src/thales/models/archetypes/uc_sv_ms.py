"""Full UC + MS + SV composed model — Phase 1.3 complete.

Layers:

  Observation:  y_t  =  μ_t  +  ε_t,
                ε_t  ~  N(0, σ²_{S_t} · exp(h_t))
  Level (UC):   μ_t  =  μ_{t-1}  +  η_t,    η_t ~ N(0, σ_η²)
  Log-vol (SV): h_t  =  φ h_{t-1} + ν_t,     ν_t ~ N(0, σ_h²)   zero-mean AR(1)
  Regime (MS):  S_t  ∈ {0, 1}              Markov on transition matrix P

Inference strategy:

  * **Continuous parameters** (σ_η, σ_low, σ_high, φ, σ_h, p_00, p_11)
    plus the **continuous latent paths** (μ_t, h_t) are sampled by NUTS.
  * The **discrete regime path** S_t is **marginalized inside the
    likelihood** via the Hamilton 1989 forward algorithm. This avoids
    the discrete-state sampling problem (NUTS doesn't handle discrete
    latents) and produces an exact marginal likelihood as a function of
    the continuous params.
  * After fitting, **smoothed regime probabilities** can be reconstructed
    via the Kim 1994 smoother on the posterior-mean parameters (or on
    each posterior sample for full posterior of regime path — slow but
    proper).

This is the "no shortcuts" full Phase 1.3 archetype. All three latent
processes coexist; identification depends on each latent being driven
by structurally different evolution (level walks slowly, log-vol AR(1)
modulates noise, regime jumps the variance floor).

Performance: with T=300 and 500/500 warmup/samples, expect ~2-5 minutes
on CPU. Scales linearly with T. GPU/Vast.ai gives ~5-10× speedup.
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
class UCSVMSFit:
    """Posterior summary for the full UC + SV + MS model."""
    sigma_eta: float
    sigma_low: float
    sigma_high: float
    p_stay_low: float
    p_stay_high: float
    phi: float
    sigma_h: float

    mu_smoothed: np.ndarray              # (T,) posterior mean of μ_t
    h_smoothed: np.ndarray               # (T,) posterior mean of h_t
    smoothed_prob_high: np.ndarray       # (T,) reconstructed via Kim smoother

    n_samples: int
    n_warmup: int
    diverging: int

    # Optional posterior samples (only populated when fit_uc_sv_ms is called
    # with return_samples=True). Each is shape (n_samples, ...).
    posterior_samples: dict | None = None


def _hmm_forward_loglik(y: jnp.ndarray, mu: jnp.ndarray, h: jnp.ndarray,
                          sigma_low: jnp.ndarray, sigma_high: jnp.ndarray,
                          p00: jnp.ndarray, p11: jnp.ndarray) -> jnp.ndarray:
    """Hamilton forward algorithm in log-space.

    Returns the marginal log-likelihood log P(y_{1:T} | continuous params)
    with the regime path S_t integrated out.
    """
    T = y.shape[0]
    sigmas = jnp.stack([sigma_low, sigma_high])    # (2,)

    # Stationary initial regime distribution
    pi_low = (1.0 - p11) / (2.0 - p00 - p11 + 1e-9)
    log_xi_init = jnp.log(jnp.stack([pi_low, 1.0 - pi_low]) + 1e-12)

    # log P_{ij}: 2x2 transition matrix in log-space
    P = jnp.array([[p00, 1.0 - p00],
                     [1.0 - p11, p11]])
    log_P = jnp.log(P + 1e-12)

    sigma_t = sigmas[None, :] * jnp.exp(h[:, None] / 2.0)   # (T, 2)
    # Observation log-density per regime
    log_lik = (-0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma_t)
                - 0.5 * ((y[:, None] - mu[:, None]) / sigma_t) ** 2)  # (T, 2)

    def step(log_xi_prev, t):
        # Predict: log P(S_t=j | y_{1:t-1}) = logsumexp_i (log P(S_{t-1}=i | y_{1:t-1}) + log P_{ij})
        log_xi_pred = logsumexp(log_xi_prev[:, None] + log_P, axis=0)
        # Update with likelihood
        log_joint = log_xi_pred + log_lik[t]
        log_z = logsumexp(log_joint)
        log_xi_new = log_joint - log_z
        return log_xi_new, log_z

    # Run scan
    _, log_zs = jax.lax.scan(step,
                                jnp.where(jnp.isfinite(log_xi_init),
                                            log_xi_init,
                                            jnp.log(jnp.array([0.5, 0.5]))),
                                jnp.arange(T))
    return log_zs.sum()


def _model(y: jnp.ndarray, sigma_eta_prior_scale: float = 0.5) -> None:
    """NumPyro model for full UC + SV + MS.

    Discrete regime marginalized in likelihood; continuous latents and
    parameters sampled via NUTS.

    ``sigma_eta_prior_scale`` controls how flexible the level walk can
    be. With a broad prior (default 0.5), the level can absorb regime
    jumps and the MS mechanism stays dormant; tighten to (~0.05) for
    monthly CPI YoY where regimes should drive most of the inter-period
    variance.
    """
    T = y.shape[0]

    # Hyperparameter priors
    sigma_eta = numpyro.sample(
        "sigma_eta", dist.HalfNormal(sigma_eta_prior_scale))
    sigma_low = numpyro.sample("sigma_low", dist.HalfNormal(2.0))
    sigma_diff = numpyro.sample("sigma_diff", dist.HalfNormal(3.0))
    sigma_high = numpyro.deterministic("sigma_high", sigma_low + sigma_diff)
    p00 = numpyro.sample("p_stay_low", dist.Beta(20.0, 1.5))
    p11 = numpyro.sample("p_stay_high", dist.Beta(10.0, 2.0))
    phi = numpyro.sample("phi", dist.Beta(20.0, 1.5))
    sigma_h = numpyro.sample("sigma_h", dist.HalfNormal(0.5))

    # Level path via non-centered random walk
    mu_innov_unit = numpyro.sample("mu_innov_unit",
                                          dist.Normal(0.0, 1.0).expand([T]).to_event(1))
    mu_innov = mu_innov_unit * sigma_eta
    mu_innov = mu_innov.at[0].set(0.0)  # first innov sets initial level mean = 0
    mu0 = numpyro.sample("mu0", dist.Normal(jnp.median(y), 5.0))
    mu = mu0 + jnp.cumsum(mu_innov)
    numpyro.deterministic("mu", mu)

    # Log-vol path via non-centered AR(1)
    h_init_scale = sigma_h / jnp.sqrt(1.0 - phi ** 2 + 1e-6)
    h_0 = numpyro.sample("h_0", dist.Normal(0.0, h_init_scale))
    eta_unit = numpyro.sample("h_innov_unit",
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


def _kim_smoother_regimes(y: np.ndarray, mu: np.ndarray, h: np.ndarray,
                            sigma_low: float, sigma_high: float,
                            p00: float, p11: float) -> np.ndarray:
    """Reconstruct smoothed P(S_t=1 | y_{1:T}) given posterior-mean
    parameter and latent-path estimates. Hamilton forward → Kim backward."""
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
                    - 0.5 * ((y[t] - mu[t]) / sigma_t) ** 2)
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

    # Backward Kim smoother
    xi_smooth = np.empty_like(xi_filt)
    xi_smooth[-1] = xi_filt[-1]
    for t in range(T - 2, -1, -1):
        ratio = np.where(xi_pred[t + 1] > 0,
                          xi_smooth[t + 1] / xi_pred[t + 1], 0.0)
        xi_smooth[t] = xi_filt[t] * (P @ ratio)
        s = xi_smooth[t].sum()
        if s > 0:
            xi_smooth[t] /= s

    return xi_smooth[:, 1]   # P(S_t = 1 | y_{1:T})


def fit_uc_sv_ms(y: np.ndarray,
                     num_warmup: int = 500,
                     num_samples: int = 500,
                     num_chains: int = 1,
                     seed: int = 0,
                     progress_bar: bool = False,
                     sigma_eta_prior_scale: float = 0.5,
                     return_samples: bool = False,
                     ) -> UCSVMSFit:
    """Fit the full UC + SV + MS composed model via NumPyro NUTS.

    Marginalizes the discrete regime path inside the likelihood
    (Hamilton forward), so NUTS samples only continuous params + the
    μ_t and h_t paths. Smoothed regime probabilities are reconstructed
    post-hoc via Kim smoother on posterior-mean params/paths.

    ``sigma_eta_prior_scale``: HalfNormal scale on σ_η. Default 0.5 is
    appropriate for synthetic-recovery testing where the level genuinely
    walks. For real monthly CPI YoY, tighten to ~0.05 — otherwise the
    level absorbs regime variance and the MS layer is dormant.
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
    mcmc.run(rng_key, y=jnp.asarray(y_arr),
                sigma_eta_prior_scale=sigma_eta_prior_scale)
    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()

    sigma_eta = float(np.mean(samples["sigma_eta"]))
    sigma_low = float(np.mean(samples["sigma_low"]))
    sigma_high = float(np.mean(samples["sigma_high"]))
    p00 = float(np.mean(samples["p_stay_low"]))
    p11 = float(np.mean(samples["p_stay_high"]))
    phi = float(np.mean(samples["phi"]))
    sigma_h = float(np.mean(samples["sigma_h"]))

    mu_smoothed = np.asarray(samples["mu"]).mean(axis=0)
    h_smoothed = np.asarray(samples["h"]).mean(axis=0)

    # Reconstruct smoothed regime probabilities at posterior-mean params
    smoothed_prob_high = _kim_smoother_regimes(
        y_arr, mu_smoothed, h_smoothed,
        sigma_low, sigma_high, p00, p11)

    diverging = 0
    if "diverging" in extra:
        diverging = int(np.asarray(extra["diverging"]).sum())

    posterior = None
    if return_samples:
        # Keep only the params + final-timestep latents needed for forecasting.
        # Full mu/h paths can be huge; we only need the last value per draw.
        posterior = {
            "sigma_eta": np.asarray(samples["sigma_eta"]),
            "sigma_low": np.asarray(samples["sigma_low"]),
            "sigma_high": np.asarray(samples["sigma_high"]),
            "p_stay_low": np.asarray(samples["p_stay_low"]),
            "p_stay_high": np.asarray(samples["p_stay_high"]),
            "phi": np.asarray(samples["phi"]),
            "sigma_h": np.asarray(samples["sigma_h"]),
            "mu_T": np.asarray(samples["mu"])[:, -1],
            "h_T": np.asarray(samples["h"])[:, -1],
        }

    return UCSVMSFit(
        sigma_eta=sigma_eta,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        p_stay_low=p00,
        p_stay_high=p11,
        phi=phi,
        sigma_h=sigma_h,
        mu_smoothed=mu_smoothed,
        h_smoothed=h_smoothed,
        smoothed_prob_high=smoothed_prob_high,
        n_samples=num_samples,
        n_warmup=num_warmup,
        diverging=diverging,
        posterior_samples=posterior,
    )


def forecast_uc_sv_ms(fit: UCSVMSFit,
                          horizons: list[int],
                          n_paths: int = 500,
                          seed: int = 0,
                          ) -> dict[int, np.ndarray]:
    """Monte-Carlo forecast paths from a fitted UC+SV+MS model.

    For each horizon h (in periods of the original y), draws ``n_paths``
    sample values of y_{T+h} by:
      1. Sampling a posterior draw of (params, μ_T, h_T).
      2. Sampling the regime path forward via the Markov chain, starting
         from the posterior smoothed prob at T.
      3. Walking μ forward as a random walk with σ_η innovations.
      4. Walking h forward as a zero-mean AR(1).
      5. Drawing y_{T+h} = μ_{T+h} + ε with ε ~ N(0, σ_{S_{T+h}}^2 · exp(h_{T+h})).

    If posterior samples weren't kept, falls back to plug-in (point
    estimates of params + smoothed final state).

    Returns {h: np.ndarray of shape (n_paths,)}.
    """
    rng = np.random.default_rng(seed)
    H = max(horizons)

    if fit.posterior_samples is not None:
        post = fit.posterior_samples
        n_post = len(post["mu_T"])
        # Sample (with replacement) one draw per path
        idx = rng.integers(0, n_post, n_paths)
        sigma_eta = post["sigma_eta"][idx]
        sigma_low = post["sigma_low"][idx]
        sigma_high = post["sigma_high"][idx]
        p00 = post["p_stay_low"][idx]
        p11 = post["p_stay_high"][idx]
        phi = post["phi"][idx]
        sigma_h = post["sigma_h"][idx]
        mu_T = post["mu_T"][idx]
        h_T = post["h_T"][idx]
    else:
        # Plug-in: tile point estimates + smoothed final state
        sigma_eta = np.full(n_paths, fit.sigma_eta)
        sigma_low = np.full(n_paths, fit.sigma_low)
        sigma_high = np.full(n_paths, fit.sigma_high)
        p00 = np.full(n_paths, fit.p_stay_low)
        p11 = np.full(n_paths, fit.p_stay_high)
        phi = np.full(n_paths, fit.phi)
        sigma_h = np.full(n_paths, fit.sigma_h)
        mu_T = np.full(n_paths, fit.mu_smoothed[-1])
        h_T = np.full(n_paths, fit.h_smoothed[-1])

    # Initial regime: sample S_T ~ Bernoulli(prob_high_smoothed[-1])
    p_high_T = float(fit.smoothed_prob_high[-1])
    S = (rng.random(n_paths) < p_high_T).astype(int)

    # Forward simulate to max horizon
    mu_path = mu_T.copy()
    h_path = h_T.copy()
    snapshots: dict[int, np.ndarray] = {}

    for step in range(1, H + 1):
        # μ random walk
        mu_path = mu_path + rng.normal(0, sigma_eta)
        # h zero-mean AR(1)
        h_path = phi * h_path + rng.normal(0, sigma_h)
        # Regime transition: P(stay | low) = p00, P(stay | high) = p11
        u = rng.random(n_paths)
        stay = np.where(S == 0, u < p00, u < p11)
        S = np.where(stay, S, 1 - S)
        # Observation
        sigma = np.where(S == 0, sigma_low, sigma_high) * np.exp(h_path / 2.0)
        y_step = mu_path + rng.normal(0, sigma)
        if step in horizons:
            snapshots[step] = y_step.copy()

    return snapshots
