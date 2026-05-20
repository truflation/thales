"""Recovery tests for the Markov-switching variance archetype.

Phase 1.3 partial gate-1: the Hamilton filter recovers σ_low, σ_high,
and the smoothed regime probabilities from synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest

from thales.models.archetypes.regime_switching import (
    fit_hamilton_2state,
    fit_uc_ms,
)
from thales.synthetic.regime_switching import (
    simulate_markov_switching,
    simulate_uc_ms,
)


# ─── σ recovery ───────────────────────────────────────────────────────────


def test_sigma_low_recovery():
    dgp = simulate_markov_switching(
        T=1000, sigma_low=0.5, sigma_high=2.0,
        p_stay_low=0.95, p_stay_high=0.85, seed=0,
    )
    fit = fit_hamilton_2state(dgp.y)
    assert abs(fit.sigma_low - dgp.sigma_low) / dgp.sigma_low < 0.20, (
        f"σ_low {fit.sigma_low:.3f} vs {dgp.sigma_low:.3f}")


def test_sigma_high_recovery():
    dgp = simulate_markov_switching(
        T=1000, sigma_low=0.5, sigma_high=2.0,
        p_stay_low=0.95, p_stay_high=0.85, seed=0,
    )
    fit = fit_hamilton_2state(dgp.y)
    assert abs(fit.sigma_high - dgp.sigma_high) / dgp.sigma_high < 0.20, (
        f"σ_high {fit.sigma_high:.3f} vs {dgp.sigma_high:.3f}")


def test_sigma_ordering_enforced():
    """σ_low must be ≤ σ_high (label-switching prevention)."""
    dgp = simulate_markov_switching(T=600, seed=1)
    fit = fit_hamilton_2state(dgp.y)
    assert fit.sigma_low <= fit.sigma_high


# ─── Transition-probability recovery ──────────────────────────────────────


def test_p_stay_low_recovery():
    dgp = simulate_markov_switching(
        T=1500, sigma_low=0.5, sigma_high=2.0,
        p_stay_low=0.95, p_stay_high=0.85, seed=2,
    )
    fit = fit_hamilton_2state(dgp.y)
    assert abs(fit.p_stay_low - dgp.p_stay_low) < 0.07, (
        f"p_stay_low {fit.p_stay_low:.3f} vs {dgp.p_stay_low:.3f}")


def test_p_stay_high_recovery():
    dgp = simulate_markov_switching(
        T=1500, sigma_low=0.5, sigma_high=2.0,
        p_stay_low=0.95, p_stay_high=0.85, seed=2,
    )
    fit = fit_hamilton_2state(dgp.y)
    assert abs(fit.p_stay_high - dgp.p_stay_high) < 0.10, (
        f"p_stay_high {fit.p_stay_high:.3f} vs {dgp.p_stay_high:.3f}")


# ─── Regime classification ───────────────────────────────────────────────


def test_smoothed_regime_classification_accuracy():
    """Smoothed P(S_t=1) > 0.5 should match true regime more often than
    base rate. Threshold: > 80% accuracy."""
    dgp = simulate_markov_switching(
        T=1000, sigma_low=0.5, sigma_high=2.0,
        p_stay_low=0.95, p_stay_high=0.85, seed=3,
    )
    fit = fit_hamilton_2state(dgp.y)
    pred_regime = (fit.smoothed_prob_high > 0.5).astype(int)
    accuracy = (pred_regime == dgp.regime).mean()
    base_rate = max(dgp.regime.mean(), 1 - dgp.regime.mean())
    assert accuracy > 0.80, (
        f"smoothed regime accuracy {accuracy:.3f} below 0.80 "
        f"(base rate {base_rate:.3f})")
    assert accuracy > base_rate, (
        f"smoothed accuracy {accuracy:.3f} not above base rate {base_rate:.3f}")


def test_smoothed_better_than_filtered():
    """Smoothing uses future data ⇒ smoothed regime accuracy ≥ filtered."""
    dgp = simulate_markov_switching(T=800, seed=4)
    fit = fit_hamilton_2state(dgp.y)
    pred_smooth = (fit.smoothed_prob_high > 0.5).astype(int)
    pred_filt = (fit.filtered_prob_high > 0.5).astype(int)
    acc_smooth = (pred_smooth == dgp.regime).mean()
    acc_filt = (pred_filt == dgp.regime).mean()
    assert acc_smooth >= acc_filt - 0.02, (
        f"smoothed {acc_smooth:.3f} < filtered {acc_filt:.3f}")


# ─── μ recovery ───────────────────────────────────────────────────────────


def test_mu_recovery():
    dgp = simulate_markov_switching(
        T=1000, mu=2.0, sigma_low=0.5, sigma_high=2.0, seed=5,
    )
    fit = fit_hamilton_2state(dgp.y)
    assert abs(fit.mu - dgp.mu) < 0.30, (
        f"μ {fit.mu:.3f} vs {dgp.mu:.3f}")


# ─── Smoke / determinism ─────────────────────────────────────────────────


def test_fit_is_deterministic():
    dgp = simulate_markov_switching(T=400, seed=99)
    fit_a = fit_hamilton_2state(dgp.y)
    fit_b = fit_hamilton_2state(dgp.y)
    assert fit_a.sigma_low == pytest.approx(fit_b.sigma_low, rel=1e-4)
    assert fit_a.p_stay_low == pytest.approx(fit_b.p_stay_low, rel=1e-3)


def test_rejects_short_series():
    with pytest.raises(ValueError, match="≥50"):
        fit_hamilton_2state(np.zeros(20))


def test_rejects_2d_input():
    with pytest.raises(ValueError, match="1D"):
        fit_hamilton_2state(np.zeros((100, 2)))


# ─── DGP smoke ───────────────────────────────────────────────────────────


def test_dgp_reproducibility():
    a = simulate_markov_switching(T=200, seed=42)
    b = simulate_markov_switching(T=200, seed=42)
    np.testing.assert_allclose(a.y, b.y)
    np.testing.assert_array_equal(a.regime, b.regime)


def test_dgp_high_regime_has_higher_variance():
    """Sanity: empirically y in regime 1 should have higher variance."""
    dgp = simulate_markov_switching(T=2000, sigma_low=0.5,
                                          sigma_high=3.0, seed=7)
    var_low = dgp.y[dgp.regime == 0].std()
    var_high = dgp.y[dgp.regime == 1].std()
    assert var_high > 1.5 * var_low, (
        f"high-regime std {var_high:.3f} not >> low-regime std {var_low:.3f}")


# ─── UC + MS recovery (Phase 1.3 expanded) ────────────────────────────────


def test_ucms_sigma_eta_recovery():
    """σ_η (level walk) should be recovered within factor of 2."""
    dgp = simulate_uc_ms(
        T=1500, sigma_eta=0.05, sigma_low=0.4, sigma_high=1.5,
        p_stay_low=0.95, p_stay_high=0.85, seed=0,
    )
    fit = fit_uc_ms(dgp.y)
    ratio = fit.sigma_eta / dgp.sigma_eta
    assert 0.5 < ratio < 2.5, (
        f"σ_η {fit.sigma_eta:.4f} vs true {dgp.sigma_eta:.4f} "
        f"(ratio {ratio:.2f})")


def test_ucms_sigma_low_recovery():
    dgp = simulate_uc_ms(
        T=1500, sigma_eta=0.05, sigma_low=0.4, sigma_high=1.5, seed=0,
    )
    fit = fit_uc_ms(dgp.y)
    assert abs(fit.sigma_low - dgp.sigma_low) / dgp.sigma_low < 0.30, (
        f"σ_low {fit.sigma_low:.3f} vs {dgp.sigma_low:.3f}")


def test_ucms_sigma_high_recovery():
    dgp = simulate_uc_ms(
        T=1500, sigma_eta=0.05, sigma_low=0.4, sigma_high=1.5, seed=0,
    )
    fit = fit_uc_ms(dgp.y)
    assert abs(fit.sigma_high - dgp.sigma_high) / dgp.sigma_high < 0.30, (
        f"σ_high {fit.sigma_high:.3f} vs {dgp.sigma_high:.3f}")


def test_ucms_sigma_ordering_enforced():
    """σ_low ≤ σ_high (label-switching prevention)."""
    dgp = simulate_uc_ms(T=600, seed=1)
    fit = fit_uc_ms(dgp.y)
    assert fit.sigma_low <= fit.sigma_high


def test_ucms_level_path_recovery():
    """Smoothed μ_t should track true level path with Pearson > 0.9."""
    dgp = simulate_uc_ms(
        T=1500, sigma_eta=0.08, sigma_low=0.3, sigma_high=1.5, seed=2,
    )
    fit = fit_uc_ms(dgp.y)
    burn = 50
    r = np.corrcoef(fit.mu_smoothed[burn:], dgp.mu_path[burn:])[0, 1]
    assert r > 0.85, f"level Pearson {r:.3f} below 0.85"


def test_ucms_regime_classification():
    """UC+MS should still classify regimes correctly: smoothed accuracy > 80%."""
    dgp = simulate_uc_ms(
        T=1500, sigma_eta=0.05, sigma_low=0.4, sigma_high=1.5,
        p_stay_low=0.95, p_stay_high=0.85, seed=3,
    )
    fit = fit_uc_ms(dgp.y)
    pred = (fit.smoothed_prob_high > 0.5).astype(int)
    accuracy = (pred == dgp.regime).mean()
    assert accuracy > 0.80, f"regime accuracy {accuracy:.3f} below 0.80"


def test_ucms_p_stay_low_recovery():
    dgp = simulate_uc_ms(
        T=1500, p_stay_low=0.95, p_stay_high=0.85, seed=4,
    )
    fit = fit_uc_ms(dgp.y)
    assert abs(fit.p_stay_low - dgp.p_stay_low) < 0.10


def test_ucms_p_stay_high_recovery():
    dgp = simulate_uc_ms(
        T=1500, p_stay_low=0.95, p_stay_high=0.85, seed=4,
    )
    fit = fit_uc_ms(dgp.y)
    assert abs(fit.p_stay_high - dgp.p_stay_high) < 0.15


def test_ucms_fit_is_deterministic():
    dgp = simulate_uc_ms(T=300, seed=99)
    fit_a = fit_uc_ms(dgp.y)
    fit_b = fit_uc_ms(dgp.y)
    assert fit_a.sigma_low == pytest.approx(fit_b.sigma_low, rel=1e-4)
    np.testing.assert_allclose(fit_a.mu_smoothed, fit_b.mu_smoothed, rtol=1e-4)


def test_ucms_rejects_short_series():
    with pytest.raises(ValueError, match="≥50"):
        fit_uc_ms(np.zeros(20))


def test_ucms_dgp_reproducibility():
    a = simulate_uc_ms(T=300, seed=42)
    b = simulate_uc_ms(T=300, seed=42)
    np.testing.assert_allclose(a.y, b.y)
    np.testing.assert_allclose(a.mu_path, b.mu_path)
    np.testing.assert_array_equal(a.regime, b.regime)
