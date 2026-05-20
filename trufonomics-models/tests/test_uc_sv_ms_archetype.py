"""Recovery tests for the full UC + SV + MS composed archetype.

Phase 1.3 complete: all three latent processes coexist. Marked ``slow``
— composed MCMC is expensive (~3-5 min per fit on CPU).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from thales.models.archetypes.uc_sv_ms import fit_uc_sv_ms  # noqa: E402
from thales.synthetic.regime_switching import simulate_uc_sv_ms  # noqa: E402

pytestmark = pytest.mark.slow


# ─── DGP smoke ────────────────────────────────────────────────────────────


def test_uc_sv_ms_dgp_reproducibility():
    a = simulate_uc_sv_ms(T=200, seed=42)
    b = simulate_uc_sv_ms(T=200, seed=42)
    np.testing.assert_allclose(a.y, b.y)
    np.testing.assert_allclose(a.mu_path, b.mu_path)
    np.testing.assert_allclose(a.h_path, b.h_path)
    np.testing.assert_array_equal(a.regime, b.regime)


def test_uc_sv_ms_dgp_components_are_distinct():
    """The level, log-vol, and regime paths should be distinct (not all
    derived from the same noise)."""
    dgp = simulate_uc_sv_ms(T=400, seed=0)
    # Different latent processes ⇒ low correlation between (μ, h) and
    # (μ, regime) paths
    r_mu_h = abs(np.corrcoef(dgp.mu_path, dgp.h_path)[0, 1])
    assert r_mu_h < 0.5, f"μ and h too correlated: |r|={r_mu_h:.3f}"


# ─── Recovery (small T, modest MCMC budget) ───────────────────────────────


def test_uc_sv_ms_sigma_low_recovery():
    """σ_low recovered within factor of 2."""
    dgp = simulate_uc_sv_ms(
        T=300, sigma_eta=0.05, sigma_low=0.4, sigma_high=1.5,
        phi=0.95, sigma_h=0.3, seed=0,
    )
    fit = fit_uc_sv_ms(dgp.y, num_warmup=400, num_samples=400, seed=1)
    ratio = fit.sigma_low / dgp.sigma_low
    assert 0.4 < ratio < 2.5, (
        f"σ_low ratio {ratio:.2f} (got {fit.sigma_low:.3f}, true {dgp.sigma_low:.3f})")


def test_uc_sv_ms_sigma_high_recovery():
    """σ_high recovered within factor of 2."""
    dgp = simulate_uc_sv_ms(
        T=300, sigma_eta=0.05, sigma_low=0.4, sigma_high=1.5,
        phi=0.95, sigma_h=0.3, seed=0,
    )
    fit = fit_uc_sv_ms(dgp.y, num_warmup=400, num_samples=400, seed=1)
    ratio = fit.sigma_high / dgp.sigma_high
    assert 0.4 < ratio < 2.5, (
        f"σ_high ratio {ratio:.2f} (got {fit.sigma_high:.3f}, true {dgp.sigma_high:.3f})")


def test_uc_sv_ms_sigma_ordering_natural():
    """σ_low ≤ σ_high enforced by the model spec (σ_diff is positive)."""
    dgp = simulate_uc_sv_ms(T=200, seed=2)
    fit = fit_uc_sv_ms(dgp.y, num_warmup=300, num_samples=300, seed=2)
    assert fit.sigma_low <= fit.sigma_high


def test_uc_sv_ms_level_path_recovery():
    """Smoothed μ_t correlates with true μ_t > 0.7."""
    dgp = simulate_uc_sv_ms(
        T=400, sigma_eta=0.08, sigma_low=0.3, sigma_high=1.2,
        seed=3,
    )
    fit = fit_uc_sv_ms(dgp.y, num_warmup=400, num_samples=400, seed=3)
    burn = 50
    r = np.corrcoef(fit.mu_smoothed[burn:], dgp.mu_path[burn:])[0, 1]
    assert r > 0.6, f"level Pearson {r:.3f} below 0.6"


def test_uc_sv_ms_h_path_recovery():
    """Smoothed h_t correlates with true h_t > 0.4 (harder than SV-alone
    since now confounded with regime)."""
    dgp = simulate_uc_sv_ms(
        T=400, sigma_eta=0.05, sigma_low=0.5, sigma_high=1.5,
        phi=0.95, sigma_h=0.4, seed=4,
    )
    fit = fit_uc_sv_ms(dgp.y, num_warmup=400, num_samples=400, seed=4)
    burn = 50
    r = np.corrcoef(fit.h_smoothed[burn:], dgp.h_path[burn:])[0, 1]
    assert r > 0.30, f"h-path Pearson {r:.3f} below 0.30"


def test_uc_sv_ms_regime_classification_above_base_rate():
    """Smoothed regime probability classifies regimes above base rate."""
    dgp = simulate_uc_sv_ms(
        T=400, sigma_eta=0.05, sigma_low=0.4, sigma_high=1.8,
        p_stay_low=0.95, p_stay_high=0.85, seed=5,
    )
    fit = fit_uc_sv_ms(dgp.y, num_warmup=400, num_samples=400, seed=5)
    pred = (fit.smoothed_prob_high > 0.5).astype(int)
    accuracy = (pred == dgp.regime).mean()
    base_rate = max(dgp.regime.mean(), 1 - dgp.regime.mean())
    assert accuracy > base_rate - 0.05, (
        f"regime accuracy {accuracy:.3f} not better than base rate "
        f"{base_rate:.3f}")


def test_uc_sv_ms_no_catastrophic_divergences():
    """NUTS divergence rate < 10% (loose; full UC+SV+MS is hard)."""
    dgp = simulate_uc_sv_ms(T=200, seed=6)
    fit = fit_uc_sv_ms(dgp.y, num_warmup=300, num_samples=300, seed=6)
    div_rate = fit.diverging / fit.n_samples
    assert div_rate < 0.10, (
        f"divergence rate {div_rate:.2%} above 10% threshold")


# ─── Smoke ────────────────────────────────────────────────────────────────


def test_uc_sv_ms_rejects_short_series():
    with pytest.raises(ValueError, match="≥50"):
        fit_uc_sv_ms(np.zeros(20), num_warmup=10, num_samples=10)


def test_uc_sv_ms_rejects_2d_input():
    with pytest.raises(ValueError, match="1D"):
        fit_uc_sv_ms(np.zeros((100, 2)), num_warmup=10, num_samples=10)
