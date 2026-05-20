"""Phase 1.3 SV layer — Stochastic Volatility via NumPyro/MCMC.

Canonical Kim-Shephard 1998 stochastic-volatility model:

    y_t  =  exp(h_t / 2) · ε_t,            ε_t ~ N(0, 1)
    h_t  =  μ_h + φ (h_{t-1} − μ_h) + ν_t,  ν_t ~ N(0, σ_h²)

Inference via NumPyro NUTS sampler. The latent log-volatility path
``h_t`` is sampled as part of the posterior — non-centered
parameterization to avoid the standard funnel pathology (Betancourt
2017).

This module is the SV building block. Combining SV with UC (level
walk) and MS (regime switching) yields the full Phase 1.3 archetype;
the layers compose because each is a separate latent process.

Returns posterior-mean point estimates plus the full posterior of
``h_t``. CRPS and band scoring should consume the full posterior, not
just the mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


@dataclass
class SVFit:
    """Posterior summary from an SV NumPyro fit."""
    mu_h: float                   # posterior mean
    phi: float                    # posterior mean
    sigma_h: float                # posterior mean
    h_smoothed: np.ndarray        # (T,) posterior mean of h_t
    h_q05: np.ndarray             # (T,) 5% quantile of h_t
    h_q95: np.ndarray             # (T,) 95% quantile of h_t
    n_samples: int
    n_chains: int
    n_warmup: int
    diverging: int                # NUTS divergences — should be ≈ 0


def _sv_model(y: np.ndarray) -> None:
    """NumPyro model for canonical SV.

    Non-centered parameterization on h: sample standard-normal innovations
    and reconstruct h via affine transform. Avoids Neal's funnel.
    """
    T = len(y)

    mu_h = numpyro.sample("mu_h", dist.Normal(0.0, 5.0))
    phi = numpyro.sample("phi", dist.Beta(20.0, 1.5))   # prior near 0.93
    sigma_h = numpyro.sample("sigma_h", dist.HalfNormal(1.0))

    # Stationary initial condition for h_0
    h_init_scale = sigma_h / jnp.sqrt(1.0 - phi ** 2 + 1e-6)
    h_0 = numpyro.sample("h_0", dist.Normal(mu_h, h_init_scale))

    # Non-centered innovations
    eta = numpyro.sample("eta",
                            dist.Normal(0.0, 1.0).expand([T - 1]).to_event(1))

    def step(carry, eta_t):
        h_prev = carry
        h_t = mu_h + phi * (h_prev - mu_h) + sigma_h * eta_t
        return h_t, h_t

    _, h_rest = jax.lax.scan(step, h_0, eta)
    h = jnp.concatenate([jnp.atleast_1d(h_0), h_rest])
    numpyro.deterministic("h", h)

    # Likelihood
    sigma_t = jnp.exp(h / 2.0)
    numpyro.sample("y", dist.Normal(0.0, sigma_t), obs=y)


def fit_sv(y: np.ndarray,
              num_warmup: int = 500,
              num_samples: int = 1000,
              num_chains: int = 1,
              seed: int = 0,
              progress_bar: bool = False,
              ) -> SVFit:
    """Fit the canonical SV model on ``y`` via NUTS.

    ``num_warmup`` and ``num_samples`` are per-chain. ``num_chains`` > 1
    parallelizes over CPU/GPU devices via JAX.
    """
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim != 1:
        raise ValueError("y must be 1D")
    if len(y_arr) < 50:
        raise ValueError(f"need ≥50 obs, got {len(y_arr)}")

    rng_key = jax.random.PRNGKey(seed)
    kernel = NUTS(_sv_model, target_accept_prob=0.95)
    mcmc = MCMC(kernel,
                  num_warmup=num_warmup,
                  num_samples=num_samples,
                  num_chains=num_chains,
                  progress_bar=progress_bar)
    mcmc.run(rng_key, y=jnp.asarray(y_arr))
    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()

    h_samples = np.asarray(samples["h"])      # (S, T)
    h_smoothed = h_samples.mean(axis=0)
    h_q05 = np.quantile(h_samples, 0.05, axis=0)
    h_q95 = np.quantile(h_samples, 0.95, axis=0)

    diverging = 0
    if "diverging" in extra:
        diverging = int(np.asarray(extra["diverging"]).sum())

    return SVFit(
        mu_h=float(np.mean(samples["mu_h"])),
        phi=float(np.mean(samples["phi"])),
        sigma_h=float(np.mean(samples["sigma_h"])),
        h_smoothed=h_smoothed,
        h_q05=h_q05,
        h_q95=h_q95,
        n_samples=num_samples,
        n_chains=num_chains,
        n_warmup=num_warmup,
        diverging=diverging,
    )
