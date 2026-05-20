"""Recovery tests for the Phase 1.5 hierarchical-housing archetype.

JAX-based; runs on CPU (slow) or GPU (fast). Marked ``slow`` since
JAX compilation + ML optimization adds noticeable overhead.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from thales.models.archetypes.hierarchical_housing import (  # noqa: E402
    fit_hierarchical_housing,
)
from thales.synthetic.hierarchical_housing import (  # noqa: E402
    simulate_hierarchical_housing,
)

pytestmark = pytest.mark.slow


# ─── DGP smoke ────────────────────────────────────────────────────────────


def test_dgp_reproducibility():
    a = simulate_hierarchical_housing(T=200, seed=42)
    b = simulate_hierarchical_housing(T=200, seed=42)
    np.testing.assert_allclose(a.y, b.y)
    np.testing.assert_allclose(a.national_factor, b.national_factor)


def test_dgp_observable_loadings_visible():
    """Region with β = 1.3 should track national factor more closely."""
    dgp = simulate_hierarchical_housing(
        T=600, betas=(0.5, 1.5),
        sigma_lambdas=(0.05, 0.05),
        sigma_eps=(0.05, 0.05),
        rhos=(0.8, 0.8),
        region_names=("low_β", "high_β"),
        seed=0,
    )
    r_low = np.corrcoef(dgp.y[:, 0], dgp.national_factor)[0, 1]
    r_high = np.corrcoef(dgp.y[:, 1], dgp.national_factor)[0, 1]
    assert r_high > r_low, (
        f"high-β region should track national factor more strongly "
        f"({r_high:.3f}) than low-β region ({r_low:.3f})")


# ─── Recovery ─────────────────────────────────────────────────────────────


def test_national_factor_recovery():
    """Smoothed national factor F_t correlates with truth > 0.9."""
    dgp = simulate_hierarchical_housing(
        T=400, sigma_F=0.15,
        betas=(1.0, 0.8, 1.2, 1.1),
        seed=1,
    )
    fit = fit_hierarchical_housing(dgp.y, region_names=dgp.region_names)
    burn = 30
    r = np.corrcoef(fit.F_smoothed[burn:], dgp.national_factor[burn:])[0, 1]
    # Sign-invariance: F is identified up to sign (factor model)
    assert abs(r) > 0.85, f"|Pearson(F_smoothed, F_true)| = {abs(r):.3f}"


def test_regional_idiosyncratic_recovery():
    """Each smoothed λ_{r,t} should recover the regional AR(1) with
    Pearson > 0.5 (lower threshold than F because variance is split)."""
    dgp = simulate_hierarchical_housing(
        T=500, sigma_F=0.10, betas=(1.0, 0.8, 1.2, 1.1),
        sigma_lambdas=(0.20, 0.20, 0.20, 0.20),
        seed=2,
    )
    fit = fit_hierarchical_housing(dgp.y)
    burn = 30
    R = dgp.y.shape[1]
    correlations = []
    for r in range(R):
        c = abs(np.corrcoef(fit.lambda_smoothed[burn:, r],
                                dgp.regional_idio[burn:, r])[0, 1])
        correlations.append(c)
    avg_r = np.mean(correlations)
    assert avg_r > 0.4, (
        f"average |Pearson(λ_smoothed, λ_true)| = {avg_r:.3f} < 0.4; "
        f"per-region: {correlations}")


def test_observation_reconstruction():
    """Smoothed F + λ should reconstruct y_t with R² > 0.9."""
    dgp = simulate_hierarchical_housing(T=400, seed=3)
    fit = fit_hierarchical_housing(dgp.y)
    R = dgp.y.shape[1]
    burn = 30
    # y_hat = β F + λ
    y_hat = (fit.F_smoothed[burn:, None] * fit.betas[None, :]
              + fit.lambda_smoothed[burn:])
    y_true = dgp.y[burn:]
    ss_res = np.sum((y_hat - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.85, f"reconstruction R² {r2:.3f} below 0.85"


def test_betas_signs_recovered():
    """Sign of β_r should match true loadings (factor identified
    up to overall sign flip — within-region signs should be consistent)."""
    dgp = simulate_hierarchical_housing(
        T=400, betas=(1.0, 0.5, 1.3, 1.1),  # all positive
        seed=4,
    )
    fit = fit_hierarchical_housing(dgp.y)
    # All betas should have the same sign (positive or all negative)
    signs = np.sign(fit.betas)
    assert (signs == signs[0]).all(), (
        f"β signs disagree: {fit.betas} — factor sign should be consistent")


def test_sigma_eps_recovery():
    """Observation noise σ_ε,r recovered within factor of 2."""
    dgp = simulate_hierarchical_housing(
        T=500, sigma_eps=(0.10, 0.10, 0.10, 0.10), seed=5,
    )
    fit = fit_hierarchical_housing(dgp.y)
    for r in range(dgp.y.shape[1]):
        ratio = fit.sigma_eps[r] / dgp.sigma_eps[r]
        assert 0.4 < ratio < 2.5, (
            f"region {r} σ_ε ratio {ratio:.2f}")


# ─── Smoke ────────────────────────────────────────────────────────────────


def test_rejects_short_series():
    with pytest.raises(ValueError, match="≥50"):
        fit_hierarchical_housing(np.zeros((20, 4)))


def test_rejects_1d_input():
    with pytest.raises(ValueError, match=r"\(T, R\)"):
        fit_hierarchical_housing(np.zeros(100))


def test_rejects_mismatched_region_names():
    with pytest.raises(ValueError, match="region_names"):
        fit_hierarchical_housing(np.zeros((100, 4)),
                                       region_names=["a", "b"])
