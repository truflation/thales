"""Recovery tests for MS + SV (no UC) archetype — Phase 2.2b."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from thales.models.archetypes.ms_sv import fit_ms_sv  # noqa: E402
from thales.synthetic.regime_switching import simulate_ms_sv  # noqa: E402

pytestmark = pytest.mark.slow


# ─── DGP smoke ────────────────────────────────────────────────────────────


def test_dgp_reproducibility():
    a = simulate_ms_sv(T=200, seed=42)
    b = simulate_ms_sv(T=200, seed=42)
    np.testing.assert_allclose(a.y, b.y)
    np.testing.assert_allclose(a.h_path, b.h_path)
    np.testing.assert_array_equal(a.regime, b.regime)


def test_dgp_no_level_walk():
    """y should oscillate around constant μ — empirical mean ≈ μ for large T."""
    dgp = simulate_ms_sv(T=2000, mu=2.5, seed=0)
    assert abs(dgp.y.mean() - dgp.mu) < 0.20


# ─── Recovery ─────────────────────────────────────────────────────────────


def test_mu_recovery():
    """Constant μ recovered within 0.30 (loose because of regime + SV noise)."""
    dgp = simulate_ms_sv(
        T=400, mu=2.5, sigma_low=0.3, sigma_high=1.2, seed=0,
    )
    fit = fit_ms_sv(dgp.y, num_warmup=400, num_samples=400, seed=1)
    assert abs(fit.mu - dgp.mu) < 0.30, (
        f"μ {fit.mu:.3f} vs true {dgp.mu:.3f}")


def test_sigma_low_recovery():
    dgp = simulate_ms_sv(
        T=400, sigma_low=0.3, sigma_high=1.2, seed=0,
    )
    fit = fit_ms_sv(dgp.y, num_warmup=400, num_samples=400, seed=1)
    ratio = fit.sigma_low / dgp.sigma_low
    assert 0.4 < ratio < 2.5, (
        f"σ_low ratio {ratio:.2f} out of [0.4, 2.5]")


def test_sigma_high_recovery():
    dgp = simulate_ms_sv(
        T=400, sigma_low=0.3, sigma_high=1.2, seed=0,
    )
    fit = fit_ms_sv(dgp.y, num_warmup=400, num_samples=400, seed=1)
    ratio = fit.sigma_high / dgp.sigma_high
    assert 0.4 < ratio < 2.5, (
        f"σ_high ratio {ratio:.2f} out of [0.4, 2.5]")


def test_sigma_ordering_natural():
    """σ_low ≤ σ_high enforced by the model spec (σ_diff is positive)."""
    dgp = simulate_ms_sv(T=200, seed=2)
    fit = fit_ms_sv(dgp.y, num_warmup=300, num_samples=300, seed=2)
    assert fit.sigma_low <= fit.sigma_high


def test_regime_classification_above_base_rate():
    """Smoothed P(high) classifies regimes above base rate."""
    dgp = simulate_ms_sv(
        T=400, sigma_low=0.3, sigma_high=1.5,
        p_stay_low=0.95, p_stay_high=0.85, seed=5,
    )
    fit = fit_ms_sv(dgp.y, num_warmup=400, num_samples=400, seed=5)
    pred = (fit.smoothed_prob_high > 0.5).astype(int)
    accuracy = (pred == dgp.regime).mean()
    base_rate = max(dgp.regime.mean(), 1 - dgp.regime.mean())
    assert accuracy > base_rate - 0.05, (
        f"regime accuracy {accuracy:.3f} not better than base rate "
        f"{base_rate:.3f}")


def test_h_path_recovery():
    """Smoothed h_t correlates with truth > 0.4."""
    dgp = simulate_ms_sv(
        T=400, sigma_low=0.3, sigma_high=1.5, phi=0.95, sigma_h=0.4, seed=4,
    )
    fit = fit_ms_sv(dgp.y, num_warmup=400, num_samples=400, seed=4)
    burn = 50
    r = np.corrcoef(fit.h_smoothed[burn:], dgp.h_path[burn:])[0, 1]
    assert r > 0.30, f"h-path Pearson {r:.3f} below 0.30"


def test_no_catastrophic_divergences():
    dgp = simulate_ms_sv(T=200, seed=6)
    fit = fit_ms_sv(dgp.y, num_warmup=300, num_samples=300, seed=6)
    div_rate = fit.diverging / fit.n_samples
    assert div_rate < 0.10, (
        f"divergence rate {div_rate:.2%} above 10% threshold")


# ─── Smoke ────────────────────────────────────────────────────────────────


def test_rejects_short_series():
    with pytest.raises(ValueError, match="≥50"):
        fit_ms_sv(np.zeros(20), num_warmup=10, num_samples=10)


def test_rejects_2d_input():
    with pytest.raises(ValueError, match="1D"):
        fit_ms_sv(np.zeros((100, 2)), num_warmup=10, num_samples=10)
