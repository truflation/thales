"""Recovery tests for the commodity pass-through synthetic DGP.

The pattern these tests establish for every archetype:

  1. Reproducibility — same seed produces identical output.
  2. Distributional sanity — drawn paths match the stated generating process
     in mean and variance.
  3. Baseline recovery — a simple OLS recovers the *time-average* pass-through
     when β drifts slowly. (Full TVP-SV recovery lands with the real model
     class, per the Phase 1 checklist.)
"""

from __future__ import annotations

import numpy as np

from thales.synthetic.commodity_passthrough import (
    PassthroughDGP,
    simulate_commodity_passthrough,
    static_ols_recovery,
)


def test_simulation_is_reproducible() -> None:
    a = simulate_commodity_passthrough(T=500, seed=42)
    b = simulate_commodity_passthrough(T=500, seed=42)
    assert np.allclose(a.retail, b.retail)
    assert np.allclose(a.true_beta, b.true_beta)


def test_different_seeds_produce_different_paths() -> None:
    a = simulate_commodity_passthrough(T=500, seed=1)
    b = simulate_commodity_passthrough(T=500, seed=2)
    assert not np.allclose(a.retail, b.retail)


def test_beta_stays_bounded() -> None:
    # With small drift SD, β should stay in [0, 1]
    dgp = simulate_commodity_passthrough(T=3000, beta_0=0.35,
                                            beta_drift_sd=0.005, seed=0)
    assert dgp.true_beta.min() >= 0.0
    assert dgp.true_beta.max() <= 1.0


def test_beta_drift_magnitude_matches_spec() -> None:
    dgp = simulate_commodity_passthrough(T=5000, beta_0=0.35,
                                            beta_drift_sd=0.005, seed=0)
    diffs = np.diff(dgp.true_beta)
    # Empirical SD of β innovations should be ≈ σ_β (0.005)
    # Note: clipping to [0, 1] slightly shrinks SD when path wanders to edges
    assert 0.003 <= diffs.std(ddof=1) <= 0.007


def test_commodity_path_is_gbm() -> None:
    """Log-commodity diffs should have mean = drift and SD = shock_sd."""
    dgp = simulate_commodity_passthrough(
        T=5000, commodity_drift=0.0005, commodity_shock_sd=0.02, seed=0)
    log_diffs = np.diff(dgp.commodity)
    assert abs(log_diffs.mean() - 0.0005) < 0.001
    assert abs(log_diffs.std(ddof=1) - 0.02) < 0.002


def test_static_ols_is_in_plausible_range_but_biased() -> None:
    """Static OLS on a TVP-β DGP where log-commodity follows a random walk
    is *biased* — there's an omitted-variable effect between the drifting β
    and the unit-root X. The test documents that the estimate is in a
    plausible [0.1, 0.9] range but does NOT assert tight recovery of the
    time-mean β. Tight recovery is the job of the TVP-VECM-SV model in
    Phase 1; this static fit is the "simplest useful baseline" that the
    real model must eventually beat.
    """
    dgp = simulate_commodity_passthrough(
        T=5000, beta_0=0.35, beta_drift_sd=0.004, seed=0)
    _, beta_hat, _ = static_ols_recovery(dgp)
    # Loose sanity: estimate is on the same unit scale, not blown up
    assert 0.1 <= beta_hat <= 0.9, f"OLS β={beta_hat:.3f} out of plausible range"
    # And positive (retail does follow commodity directionally)
    assert beta_hat > 0.0


def test_static_ols_estimate_stable_with_small_drift() -> None:
    """When β barely drifts, static OLS should be much closer to mean β.

    This confirms the bias in the previous test comes from TVP specifically,
    not from some bug in the fit code.
    """
    dgp = simulate_commodity_passthrough(
        T=5000, beta_0=0.35, beta_drift_sd=0.00005,   # nearly fixed
        seed=0)
    mean_true_beta = float(dgp.true_beta.mean())
    _, beta_hat, _ = static_ols_recovery(dgp)
    assert abs(beta_hat - mean_true_beta) < 0.02, (
        f"OLS β={beta_hat:.4f}, mean-true β={mean_true_beta:.4f}")


def test_residual_sd_tracks_stochastic_volatility() -> None:
    """Residual SD from OLS should be on the same order as mean exp(log_sigma/2)."""
    dgp = simulate_commodity_passthrough(T=3000, seed=0)
    _, _, resid_sd = static_ols_recovery(dgp)
    true_avg_sigma = float(np.exp(dgp.true_log_sigma / 2).mean())
    # OLS residual SD includes the TVP-β drift, so it will be ≥ true σ mean.
    # Check same order of magnitude (within factor of 5).
    assert 0.2 * true_avg_sigma <= resid_sd <= 5.0 * true_avg_sigma


def test_dataclass_exposes_true_latents() -> None:
    dgp = simulate_commodity_passthrough(T=200, seed=0)
    assert isinstance(dgp, PassthroughDGP)
    assert dgp.commodity.shape == (200,)
    assert dgp.true_beta.shape == (200,)
    assert dgp.true_log_sigma.shape == (200,)
