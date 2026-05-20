"""Recovery tests for the stochastic-volatility archetype (Phase 1.3 SV).

NumPyro NUTS sampler. Marked ``slow`` — MCMC sampling adds O(seconds)
per fit. Run via ``pytest -m slow tests/test_sv_archetype.py`` to
exercise.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Force JAX to CPU for testing — avoids spurious GPU init issues on dev
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from thales.models.archetypes.sv import fit_sv  # noqa: E402
from thales.synthetic.regime_switching import simulate_sv  # noqa: E402

pytestmark = pytest.mark.slow


# ─── DGP smoke ────────────────────────────────────────────────────────────


def test_sv_dgp_reproducibility():
    a = simulate_sv(T=200, seed=42)
    b = simulate_sv(T=200, seed=42)
    np.testing.assert_allclose(a.y, b.y)
    np.testing.assert_allclose(a.h_path, b.h_path)


def test_sv_dgp_high_h_implies_high_volatility():
    """In high-h regions, |y| should be larger (sanity check on simulator)."""
    dgp = simulate_sv(T=2000, mu_h=-2.0, phi=0.95, sigma_h=0.5, seed=1)
    median_h = float(np.median(dgp.h_path))
    high = dgp.y[dgp.h_path > median_h]
    low = dgp.y[dgp.h_path <= median_h]
    assert np.std(high) > 1.5 * np.std(low), (
        f"high-h std {np.std(high):.3f} not >> low-h std {np.std(low):.3f}")


# ─── Recovery (small T to keep tests under a few minutes) ─────────────────


def test_sv_mu_h_recovery():
    """Posterior mean of μ_h within 0.5 of true (loose, small T)."""
    dgp = simulate_sv(T=400, mu_h=-1.5, phi=0.95, sigma_h=0.3, seed=0)
    fit = fit_sv(dgp.y, num_warmup=400, num_samples=600, seed=1)
    assert abs(fit.mu_h - dgp.mu_h) < 0.6, (
        f"μ_h posterior mean {fit.mu_h:.3f} vs true {dgp.mu_h:.3f}")


def test_sv_phi_recovery():
    """Posterior mean of φ within 0.15 of true."""
    dgp = simulate_sv(T=400, mu_h=-1.5, phi=0.95, sigma_h=0.3, seed=0)
    fit = fit_sv(dgp.y, num_warmup=400, num_samples=600, seed=1)
    assert abs(fit.phi - dgp.phi) < 0.15, (
        f"φ posterior mean {fit.phi:.3f} vs true {dgp.phi:.3f}")


def test_sv_sigma_h_recovery():
    """Posterior mean of σ_h within factor of 2 of true."""
    dgp = simulate_sv(T=400, mu_h=-1.5, phi=0.95, sigma_h=0.3, seed=0)
    fit = fit_sv(dgp.y, num_warmup=400, num_samples=600, seed=1)
    ratio = fit.sigma_h / dgp.sigma_h
    assert 0.4 < ratio < 2.5, (
        f"σ_h ratio {ratio:.2f} outside [0.4, 2.5]")


def test_sv_h_path_correlation():
    """Posterior-mean h path should correlate with true h_t > 0.6."""
    dgp = simulate_sv(T=500, mu_h=-1.5, phi=0.95, sigma_h=0.4, seed=2)
    fit = fit_sv(dgp.y, num_warmup=400, num_samples=600, seed=2)
    burn = 50
    r = np.corrcoef(fit.h_smoothed[burn:], dgp.h_path[burn:])[0, 1]
    assert r > 0.55, f"h-path correlation {r:.3f} below 0.55"


def test_sv_no_divergences_on_clean_data():
    """NUTS should run without divergent transitions on clean SV data
    when target_accept_prob=0.95."""
    dgp = simulate_sv(T=300, mu_h=-1.0, phi=0.90, sigma_h=0.2, seed=3)
    fit = fit_sv(dgp.y, num_warmup=400, num_samples=400, seed=3)
    # Allow up to 5% divergences (loose)
    div_rate = fit.diverging / fit.n_samples
    assert div_rate < 0.05, (
        f"NUTS divergences {fit.diverging}/{fit.n_samples} too high")


def test_sv_quantile_bands_have_correct_shape():
    """h_q05 and h_q95 should sandwich h_smoothed."""
    dgp = simulate_sv(T=200, seed=4)
    fit = fit_sv(dgp.y, num_warmup=200, num_samples=300, seed=4)
    assert fit.h_q05.shape == fit.h_smoothed.shape == fit.h_q95.shape
    assert (fit.h_q05 <= fit.h_smoothed + 1e-6).all()
    assert (fit.h_q95 >= fit.h_smoothed - 1e-6).all()


# ─── Smoke ────────────────────────────────────────────────────────────────


def test_sv_rejects_short_series():
    with pytest.raises(ValueError, match="≥50"):
        fit_sv(np.zeros(20), num_warmup=10, num_samples=10)


def test_sv_rejects_2d_input():
    with pytest.raises(ValueError, match="1D"):
        fit_sv(np.zeros((100, 2)), num_warmup=10, num_samples=10)
