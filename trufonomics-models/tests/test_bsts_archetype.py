"""Recovery tests for the BSTS-discretionary archetype model.

Phase 1.2 gate-1 evidence. Three-component recovery: trend, seasonal,
noise. Tighter than 1.1 because the state space is 13D not 1D — much
more room for component-mixing identifiability problems.

Tests:

  1. **Trend recovery** — Pearson(smoothed trend, true trend) > 0.95.
     Trend is the easiest component since it's the largest in absolute
     terms; this is a sanity gate.
  2. **Seasonal recovery** — Pearson(smoothed seasonal, true seasonal)
     > 0.7 over the second-and-later cycles. Seasonal is harder because
     it's small relative to trend; threshold is honest about that.
  3. **Decomposition completeness** — fitted (trend + seasonal) explains
     > 90% of variance of y minus noise.
  4. **σ_ε recovery** — within factor of 2 of true noise SD.
  5. **Determinism + smoke checks**.
"""

from __future__ import annotations

import numpy as np
import pytest

from thales.models.archetypes.bsts import fit_bsts, fit_bsts_local_level
from thales.synthetic.bsts_discretionary import simulate_bsts_discretionary


# ─── Trend recovery ──────────────────────────────────────────────────────


def test_trend_recovery_correlation():
    dgp = simulate_bsts_discretionary(
        T=400, period=12, initial_level=100.0, initial_slope=0.05,
        sigma_mu=0.05, sigma_delta=0.005,
        sigma_seasonal=0.05, sigma_eps=0.5, seed=0,
    )
    fit = fit_bsts(dgp.y, period=12)
    burn = 24  # 2 cycles of burn-in
    r = np.corrcoef(fit.trend_smoothed[burn:], dgp.trend[burn:])[0, 1]
    assert r > 0.95, f"trend correlation {r:.3f} below 0.95"


def test_trend_recovery_mae():
    dgp = simulate_bsts_discretionary(
        T=400, period=12, sigma_eps=0.5, seed=1)
    fit = fit_bsts(dgp.y, period=12)
    burn = 24
    mae = np.mean(np.abs(fit.trend_smoothed[burn:] - dgp.trend[burn:]))
    # Trend is around 100; 2.5% MAE on level is acceptable
    assert mae < 2.5, f"trend MAE {mae:.3f} above 2.5"


# ─── Seasonal recovery ───────────────────────────────────────────────────


def test_seasonal_recovery_correlation():
    """Strong seasonal pattern should be recovered to Pearson > 0.7."""
    # Strong seasonal so it's identifiable from noise
    pattern = 5.0 * np.sin(2 * np.pi * np.arange(12) / 12)
    dgp = simulate_bsts_discretionary(
        T=600, period=12, seasonal_pattern=pattern,
        sigma_seasonal=0.1, sigma_eps=0.5, seed=2,
    )
    fit = fit_bsts(dgp.y, period=12)
    burn = 24
    r = np.corrcoef(fit.seasonal_smoothed[burn:],
                       dgp.seasonal[burn:])[0, 1]
    assert r > 0.7, f"seasonal correlation {r:.3f} below 0.7"


# ─── Decomposition completeness ──────────────────────────────────────────


def test_decomposition_explains_signal():
    """Smoothed (trend + seasonal) should explain > 90% of var(y - noise)."""
    pattern = 3.0 * np.sin(2 * np.pi * np.arange(12) / 12)
    dgp = simulate_bsts_discretionary(
        T=400, period=12, seasonal_pattern=pattern,
        sigma_eps=0.5, seed=3,
    )
    fit = fit_bsts(dgp.y, period=12)
    burn = 24
    fitted_signal = fit.trend_smoothed[burn:] + fit.seasonal_smoothed[burn:]
    true_signal = dgp.trend[burn:] + dgp.seasonal[burn:]
    # R² of fitted vs true signal
    ss_res = np.sum((fitted_signal - true_signal) ** 2)
    ss_tot = np.sum((true_signal - true_signal.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.90, f"decomposition R² {r2:.3f} below 0.90"


# ─── Hyperparameter recovery ─────────────────────────────────────────────


def test_sigma_eps_recovery_within_factor_2():
    dgp = simulate_bsts_discretionary(
        T=400, period=12, sigma_eps=0.5, seed=4)
    fit = fit_bsts(dgp.y, period=12)
    ratio = fit.sigma_eps / dgp.sigma_eps
    assert 0.5 < ratio < 2.0, (
        f"σ_ε estimate {fit.sigma_eps:.4f} vs true {dgp.sigma_eps:.4f} "
        f"(ratio {ratio:.2f})")


# ─── Determinism + smoke checks ──────────────────────────────────────────


def test_fit_is_deterministic_for_same_data():
    dgp = simulate_bsts_discretionary(T=200, period=12, seed=99)
    fit_a = fit_bsts(dgp.y, period=12)
    fit_b = fit_bsts(dgp.y, period=12)
    np.testing.assert_allclose(fit_a.trend_smoothed, fit_b.trend_smoothed,
                                  rtol=1e-6)


def test_rejects_short_series():
    short = np.zeros(20)
    with pytest.raises(ValueError, match="≥"):
        fit_bsts(short, period=12)


def test_rejects_2d_input():
    bad = np.zeros((100, 2))
    with pytest.raises(ValueError, match="1D"):
        fit_bsts(bad, period=12)


# ─── DGP smoke checks ────────────────────────────────────────────────────


def test_dgp_seasonal_is_centered():
    dgp = simulate_bsts_discretionary(T=600, period=12, seed=5)
    # After enough observations the empirical seasonal should average ~0
    burn = 24
    assert abs(dgp.seasonal[burn:].mean()) < 0.5, (
        f"seasonal mean {dgp.seasonal[burn:].mean():.3f} "
        f"should be near zero")


def test_dgp_reproducibility():
    a = simulate_bsts_discretionary(T=200, seed=42)
    b = simulate_bsts_discretionary(T=200, seed=42)
    np.testing.assert_allclose(a.y, b.y)
    np.testing.assert_allclose(a.trend, b.trend)


# ─── Local-level variant tests ────────────────────────────────────────────


def test_local_level_trend_recovery():
    """Local-level BSTS (no slope state) recovers trend Pearson > 0.95."""
    dgp = simulate_bsts_discretionary(
        T=400, period=12, sigma_mu=0.05, sigma_delta=0.005,
        sigma_seasonal=0.05, sigma_eps=0.5, seed=0,
    )
    fit = fit_bsts_local_level(dgp.y, period=12)
    burn = 24
    r = np.corrcoef(fit.trend_smoothed[burn:], dgp.trend[burn:])[0, 1]
    assert r > 0.95, f"local-level trend correlation {r:.3f} below 0.95"


def test_local_level_sigma_eps_recovery_within_factor_2():
    """Local-level should recover σ_ε at least as well as LLT."""
    dgp = simulate_bsts_discretionary(
        T=400, period=12, sigma_eps=0.5, seed=4)
    fit = fit_bsts_local_level(dgp.y, period=12)
    ratio = fit.sigma_eps / dgp.sigma_eps
    assert 0.5 < ratio < 2.0, (
        f"σ_ε estimate {fit.sigma_eps:.4f} vs true {dgp.sigma_eps:.4f}")


def test_local_level_decomposition_explains_signal():
    """Local-level decomposition R² > 0.90."""
    pattern = 3.0 * np.sin(2 * np.pi * np.arange(12) / 12)
    dgp = simulate_bsts_discretionary(
        T=400, period=12, seasonal_pattern=pattern,
        sigma_eps=0.5, seed=3,
    )
    fit = fit_bsts_local_level(dgp.y, period=12)
    burn = 24
    fitted = fit.trend_smoothed[burn:] + fit.seasonal_smoothed[burn:]
    truth = dgp.trend[burn:] + dgp.seasonal[burn:]
    ss_res = np.sum((fitted - truth) ** 2)
    ss_tot = np.sum((truth - truth.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.90, f"local-level decomposition R² {r2:.3f} below 0.90"


def test_local_level_has_no_sigma_delta_param():
    """Smoke check: BSTSLocalLevelFit has no slope-related fields."""
    dgp = simulate_bsts_discretionary(T=200, seed=0)
    fit = fit_bsts_local_level(dgp.y, period=12)
    assert not hasattr(fit, "sigma_delta")
    assert not hasattr(fit, "slope_smoothed")
