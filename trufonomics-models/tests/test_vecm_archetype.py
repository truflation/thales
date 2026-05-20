"""Recovery tests for the VECM-tradables archetype model.

Phase 1.4 gate-1 evidence. The OLS estimator should recover the true
α_1, α_2, μ_0, θ from synthetic data. Standard tolerances per
state-of-the-art VECM literature (Lütkepohl 2005 §6).
"""

from __future__ import annotations

import numpy as np
import pytest

from thales.models.archetypes.vecm import fit_vecm
from thales.synthetic.vecm_tariff import simulate_vecm_tariff


# ─── Adjustment-speed recovery ────────────────────────────────────────────


def test_alpha_1_recovery():
    dgp = simulate_vecm_tariff(
        T=600, alpha_1=-0.05, alpha_2=+0.10,
        mu_0=0.0, theta=5.0, sigma_1=0.4, sigma_2=0.6,
        seed=0,
    )
    fit = fit_vecm(dgp.y1, dgp.y2, dgp.regime)
    assert abs(fit.alpha_1 - dgp.alpha_1) < 0.02, (
        f"α_1 estimate {fit.alpha_1:.4f} vs true {dgp.alpha_1:.4f}")


def test_alpha_2_recovery():
    dgp = simulate_vecm_tariff(
        T=600, alpha_1=-0.05, alpha_2=+0.10,
        mu_0=0.0, theta=5.0, sigma_1=0.4, sigma_2=0.6,
        seed=0,
    )
    fit = fit_vecm(dgp.y1, dgp.y2, dgp.regime)
    assert abs(fit.alpha_2 - dgp.alpha_2) < 0.02, (
        f"α_2 estimate {fit.alpha_2:.4f} vs true {dgp.alpha_2:.4f}")


def test_alpha_signs_correct():
    """α_1 should be < 0 and α_2 > 0 (clothing falls when above
    equilibrium; imports rise)."""
    dgp = simulate_vecm_tariff(T=400, seed=1)
    fit = fit_vecm(dgp.y1, dgp.y2, dgp.regime)
    assert fit.alpha_1 < 0, f"α_1 should be negative, got {fit.alpha_1:.4f}"
    assert fit.alpha_2 > 0, f"α_2 should be positive, got {fit.alpha_2:.4f}"


# ─── Tariff-shift recovery ────────────────────────────────────────────────


def test_theta_recovery_eq1():
    dgp = simulate_vecm_tariff(
        T=600, alpha_1=-0.05, alpha_2=+0.10, mu_0=0.0, theta=5.0,
        sigma_1=0.4, sigma_2=0.6, seed=2,
    )
    fit = fit_vecm(dgp.y1, dgp.y2, dgp.regime)
    # Tighter tolerance because θ is well-identified once you have
    # enough post-regime observations
    assert abs(fit.theta_1 - dgp.theta) < 1.5, (
        f"θ_1 estimate {fit.theta_1:.3f} vs true {dgp.theta:.3f}")


def test_theta_recovery_eq2():
    dgp = simulate_vecm_tariff(
        T=600, alpha_1=-0.05, alpha_2=+0.10, mu_0=0.0, theta=5.0,
        sigma_1=0.4, sigma_2=0.6, seed=2,
    )
    fit = fit_vecm(dgp.y1, dgp.y2, dgp.regime)
    assert abs(fit.theta_2 - dgp.theta) < 1.5, (
        f"θ_2 estimate {fit.theta_2:.3f} vs true {dgp.theta:.3f}")


def test_theta_estimates_agree_across_equations():
    """The two equations imply the same θ — they should agree (within
    sampling noise) on synthetic data.

    Tolerance 2.0: each θ_i = -γ_i/α_i is a ratio of two noisy estimates,
    so SD compounds. With σ_1=0.4, α_1=-0.05, T_post≈240, SD(θ_1)≈0.5;
    similarly SD(θ_2)≈0.4. SD of the difference ≈ 0.65, so 2σ ≈ 1.3.
    Set threshold at 2.0 to keep false-failure rate near 5%.
    """
    dgp = simulate_vecm_tariff(T=600, theta=5.0, seed=3)
    fit = fit_vecm(dgp.y1, dgp.y2, dgp.regime)
    assert abs(fit.theta_1 - fit.theta_2) < 2.0, (
        f"θ estimates disagree: eq1={fit.theta_1:.3f}, eq2={fit.theta_2:.3f}")


# ─── Equilibrium-spread recovery ──────────────────────────────────────────


def test_mu_recovery():
    """When μ_0 is non-zero, both equations should recover it."""
    dgp = simulate_vecm_tariff(
        T=800, alpha_1=-0.06, alpha_2=+0.10,
        mu_0=3.0, theta=2.0,    # both non-zero
        sigma_1=0.4, sigma_2=0.5, seed=4,
    )
    fit = fit_vecm(dgp.y1, dgp.y2, dgp.regime)
    # Loose tolerance — μ is the harder of the structural parameters to
    # identify because it depends on c_i/α_i, both of which are noisy
    assert abs(fit.mu_1 - dgp.mu_0) < 1.5, (
        f"μ_1 estimate {fit.mu_1:.3f} vs true {dgp.mu_0:.3f}")
    assert abs(fit.mu_2 - dgp.mu_0) < 1.5, (
        f"μ_2 estimate {fit.mu_2:.3f} vs true {dgp.mu_0:.3f}")


# ─── Residual covariance ──────────────────────────────────────────────────


def test_sigma_recovery():
    dgp = simulate_vecm_tariff(
        T=800, sigma_1=0.4, sigma_2=0.6, rho=0.0, seed=5)
    fit = fit_vecm(dgp.y1, dgp.y2, dgp.regime)
    assert abs(fit.sigma_1 - dgp.sigma_1) / dgp.sigma_1 < 0.15
    assert abs(fit.sigma_2 - dgp.sigma_2) / dgp.sigma_2 < 0.15


def test_rho_recovery_when_correlated():
    dgp = simulate_vecm_tariff(
        T=1000, sigma_1=0.4, sigma_2=0.6, rho=0.5, seed=6)
    fit = fit_vecm(dgp.y1, dgp.y2, dgp.regime)
    assert abs(fit.rho - dgp.rho) < 0.10, (
        f"ρ estimate {fit.rho:.3f} vs true {dgp.rho:.3f}")


# ─── Structural break / tariff identification ────────────────────────────


def test_pre_post_regime_spread_shifts():
    """Empirical sanity: the spread y1-y2 should shift visibly after the
    tariff regime starts."""
    dgp = simulate_vecm_tariff(
        T=800, mu_0=0.0, theta=8.0,    # large shift
        regime_start=400, seed=7,
    )
    pre = dgp.z[200:400].mean()  # post burn-in, pre regime
    post = dgp.z[600:].mean()    # post regime
    assert post - pre > 4.0, (
        f"spread shift pre={pre:.3f} post={post:.3f} too small for θ=8.0")


# ─── Smoke / determinism ─────────────────────────────────────────────────


def test_fit_is_deterministic():
    dgp = simulate_vecm_tariff(T=300, seed=99)
    fit_a = fit_vecm(dgp.y1, dgp.y2, dgp.regime)
    fit_b = fit_vecm(dgp.y1, dgp.y2, dgp.regime)
    assert fit_a.alpha_1 == pytest.approx(fit_b.alpha_1)
    assert fit_a.theta_1 == pytest.approx(fit_b.theta_1)


def test_rejects_short_series():
    short = np.zeros(20)
    with pytest.raises(ValueError, match="≥50"):
        fit_vecm(short, short, short)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        fit_vecm(np.zeros(100), np.zeros(50), np.zeros(100))


# ─── DGP smoke check ─────────────────────────────────────────────────────


def test_dgp_reproducibility():
    a = simulate_vecm_tariff(T=200, seed=42)
    b = simulate_vecm_tariff(T=200, seed=42)
    np.testing.assert_allclose(a.y1, b.y1)
    np.testing.assert_allclose(a.y2, b.y2)
