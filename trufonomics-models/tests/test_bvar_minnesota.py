"""Synthetic recovery + IRF/FEVD tests for Phase 3.1 BVAR-Minnesota."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thales.evaluation.harness import attach_actuals, score, walk_forward
from thales.models.archetypes.bvar_minnesota import (
    BVARForecaster,
    cholesky_irf,
    conditional_forecast,
    fevd,
    fit_bvar_minnesota,
    minnesota_prior_diag,
    shock_scenario,
)


# ── Synthetic DGP ────────────────────────────────────────────────────────


def _simulate_var1(A: np.ndarray, sigma: np.ndarray, T: int = 500,
                      seed: int = 0) -> np.ndarray:
    """Simulate Y_t = A Y_{t-1} + ε_t, ε_t ~ N(0, Σ). Returns (T, k)."""
    rng = np.random.default_rng(seed)
    k = A.shape[0]
    L = np.linalg.cholesky(sigma)
    Y = np.zeros((T, k))
    Y[0] = rng.normal(0, 1, k)
    for t in range(1, T):
        eps = L @ rng.normal(0, 1, k)
        Y[t] = A @ Y[t - 1] + eps
    return Y


# ── Prior helper ─────────────────────────────────────────────────────────


def test_minnesota_prior_shape_and_layout():
    """Prior diagonal should be length k·(k·p + 1) — equation-stacked."""
    k, p = 3, 2
    sigma_diag = np.array([1.0, 2.0, 0.5])
    sd = minnesota_prior_diag(k, p, sigma_diag)
    assert sd.shape == (k * (k * p + 1),)
    # Intercepts (positions 0, k*p+1, 2*(k*p+1)) should be huge (loose)
    sd_eq = sd.reshape(k, k * p + 1)
    for i in range(k):
        assert sd_eq[i, 0] >= 100.0


def test_minnesota_prior_lag_decay_shrinks_higher_lags():
    """Higher lags should have tighter (smaller) prior SD when lag_decay>0."""
    k, p = 2, 3
    sd = minnesota_prior_diag(k, p, np.ones(k), lag_decay=1.0)
    sd_eq = sd.reshape(k, k * p + 1)
    # For equation 0, own-lag-1 vs own-lag-3 SDs are at positions 1 and 1+2*k
    own_l1 = sd_eq[0, 1]
    own_l3 = sd_eq[0, 1 + 2 * k]
    assert own_l3 < own_l1


def test_cross_tightness_shrinks_off_diagonal():
    sd_strict = minnesota_prior_diag(2, 1, np.ones(2), cross_tightness=0.1)
    sd_loose = minnesota_prior_diag(2, 1, np.ones(2), cross_tightness=1.0)
    sd_strict_eq = sd_strict.reshape(2, 3)
    sd_loose_eq = sd_loose.reshape(2, 3)
    # Cross coefficient for equation 0 is at position 2 (lag-1 of var-1)
    assert sd_strict_eq[0, 2] < sd_loose_eq[0, 2]


# ── Recovery on synthetic VAR(1) ─────────────────────────────────────────


def test_recover_var1_coefficients_well():
    """Generous overall_tightness + long sample → posterior mean should
    be close to the true A matrix."""
    A = np.array([
        [0.7, 0.2],
        [0.1, 0.6],
    ])
    sigma = np.array([[0.5, 0.1], [0.1, 0.3]])
    Y = _simulate_var1(A, sigma, T=2000, seed=42)
    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=2.0,
                                  cross_tightness=1.0, lag_decay=0.0)
    A_hat = fit.coefs[:, 1:3]    # drop intercept
    np.testing.assert_allclose(A_hat, A, atol=0.05)


def test_intercept_loose_prior_doesnt_shrink_it_hard():
    """Intercept prior is set wide (1e3) so a non-zero true mean is recovered."""
    A = np.array([[0.5, 0.0], [0.0, 0.5]])
    sigma = np.eye(2)
    Y = _simulate_var1(A, sigma, T=1000, seed=1)
    Y = Y + np.array([3.0, -2.0])    # add constant offset
    fit = fit_bvar_minnesota(Y, p=1)
    intercept = fit.coefs[:, 0]
    expected_intercept = np.array([3.0, -2.0]) * (1 - 0.5)    # (I - A)·μ
    np.testing.assert_allclose(intercept, expected_intercept, atol=0.15)


# ── IRF + FEVD ───────────────────────────────────────────────────────────


def test_irf_zero_lag_equals_cholesky():
    """At horizon 0, irf[0, :, :] must equal the Cholesky factor of Σ."""
    A = np.array([[0.5, 0.0], [0.0, 0.5]])
    sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    Y = _simulate_var1(A, sigma, T=1000, seed=2)
    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=2.0)
    irf = cholesky_irf(fit, h=10)
    L = np.linalg.cholesky(fit.sigma)
    np.testing.assert_allclose(irf[0], L, atol=1e-9)


def test_irf_decays_for_stable_var():
    """A stable VAR (eigenvalues < 1) should have IRF decaying to 0."""
    A = np.array([[0.7, 0.1], [0.05, 0.6]])
    sigma = np.eye(2)
    Y = _simulate_var1(A, sigma, T=1000, seed=3)
    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=2.0)
    irf = cholesky_irf(fit, h=40)
    # Magnitudes at h=40 should be much smaller than h=0
    norms = np.linalg.norm(irf.reshape(irf.shape[0], -1), axis=1)
    assert norms[-1] < 0.05 * norms[0]


def test_fevd_rows_sum_to_one():
    """Variance shares must sum to 1 along the shock axis at every h."""
    A = np.array([[0.7, 0.1], [0.05, 0.6]])
    sigma = np.eye(2)
    Y = _simulate_var1(A, sigma, T=500, seed=4)
    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=2.0)
    f = fevd(fit, h=20)
    # f has shape (h+1, k, k); sum across last axis (shocks) should = 1
    sums = f[1:].sum(axis=-1)    # skip h=0 where ratio is undefined when total=0
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)


def test_fevd_cholesky_first_variable_pure_own_shock_at_h0():
    """Under default Cholesky ordering, the first variable's contemporaneous
    forecast variance (h=0) is exactly 100% own-shock — by construction
    of the lower-triangular factor (B[0, 1:] = 0)."""
    A = np.array([[0.6, 0.05], [0.05, 0.6]])
    sigma = np.eye(2)
    Y = _simulate_var1(A, sigma, T=500, seed=5)
    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=2.0)
    f = fevd(fit, h=20)
    assert f[0, 0, 0] == pytest.approx(1.0, abs=1e-9)
    assert f[0, 0, 1] == pytest.approx(0.0, abs=1e-9)
    # By h=20 the lag-transmission has acted, so cross-share should
    # be non-trivial (>0).
    assert f[20, 0, 1] > 0.0


def test_cholesky_order_changes_identification():
    """Reversing the Cholesky order should change the IRF (different
    ordering ⇒ different decomposition of Σ)."""
    A = np.array([[0.5, 0.0], [0.0, 0.5]])
    sigma = np.array([[1.0, 0.6], [0.6, 1.0]])
    Y = _simulate_var1(A, sigma, T=500, seed=6)
    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=2.0)
    irf_default = cholesky_irf(fit, h=5)
    irf_reversed = cholesky_irf(fit, h=5, cholesky_order=[1, 0])
    # Should differ at h=0 (the Cholesky factor changes)
    assert not np.allclose(irf_default[0], irf_reversed[0])


# ── Forecaster Protocol ──────────────────────────────────────────────────


# ── Conditional forecasts ────────────────────────────────────────────────


def test_conditional_forecast_forced_path_exact():
    """The forced variable's posterior mean over draws must equal the
    forced path exactly — by construction of the override."""
    A = np.array([[0.7, 0.1], [0.1, 0.6]])
    sigma = np.eye(2) * 0.5
    Y = _simulate_var1(A, sigma, T=400, seed=10)
    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=2.0)
    h = 6
    forced = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    out = conditional_forecast(
        fit, history=Y, forced_paths={0: forced}, h=h,
        n_samples=400, seed=11)
    np.testing.assert_allclose(out["mean"][:, 0], forced, atol=1e-9)
    # Quantiles for the forced variable must also equal the forced path
    np.testing.assert_allclose(out["q05"][:, 0], forced, atol=1e-9)
    np.testing.assert_allclose(out["q95"][:, 0], forced, atol=1e-9)


def test_conditional_forecast_free_variable_responds_to_forced():
    """The free variable's mean trajectory under a forced rising path on
    var-0 should differ from its mean under a flat path — the
    conditioning is informative."""
    A = np.array([[0.5, 0.0], [0.5, 0.5]])    # var-1 strongly responds to var-0
    sigma = np.eye(2) * 0.1
    Y = _simulate_var1(A, sigma, T=400, seed=12)
    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=2.0)
    h = 8
    rising = np.linspace(1.0, 3.0, h)
    flat = np.full(h, 1.0)
    out_rising = conditional_forecast(
        fit, history=Y, forced_paths={0: rising}, h=h,
        n_samples=400, seed=13)
    out_flat = conditional_forecast(
        fit, history=Y, forced_paths={0: flat}, h=h,
        n_samples=400, seed=13)
    # Under rising var-0, var-1's path should end up higher than under flat
    assert out_rising["mean"][-1, 1] > out_flat["mean"][-1, 1]


def test_conditional_forecast_free_variable_uncertainty_widens_with_horizon():
    """Free-variable bands should widen with horizon (forecast variance
    accumulates over h steps)."""
    A = np.array([[0.7, 0.0], [0.1, 0.6]])
    sigma = np.eye(2) * 0.3
    Y = _simulate_var1(A, sigma, T=400, seed=14)
    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=2.0)
    h = 12
    forced = np.zeros(h)
    out = conditional_forecast(
        fit, history=Y, forced_paths={0: forced}, h=h,
        n_samples=600, seed=15)
    # Width of var-1's 90% band should increase with h
    width_h1 = out["q95"][0, 1] - out["q05"][0, 1]
    width_h12 = out["q95"][-1, 1] - out["q05"][-1, 1]
    assert width_h12 > width_h1


def test_conditional_forecast_validates_horizon_mismatch():
    A = np.eye(2) * 0.5
    sigma = np.eye(2)
    Y = _simulate_var1(A, sigma, T=200, seed=16)
    fit = fit_bvar_minnesota(Y, p=1)
    with pytest.raises(ValueError, match="length h"):
        conditional_forecast(fit, history=Y,
                                  forced_paths={0: np.zeros(5)}, h=10,
                                  n_samples=10)


def test_conditional_forecast_validates_var_index():
    A = np.eye(2) * 0.5
    sigma = np.eye(2)
    Y = _simulate_var1(A, sigma, T=200, seed=17)
    fit = fit_bvar_minnesota(Y, p=1)
    with pytest.raises(ValueError, match="out of range"):
        conditional_forecast(fit, history=Y,
                                  forced_paths={5: np.zeros(3)}, h=3,
                                  n_samples=10)


# ── Shock scenarios ──────────────────────────────────────────────────────


def test_shock_scenario_recovers_requested_shock_size_at_h0():
    """If we ask for a +0.20 shock to var-0, the h=0 response of var-0
    should be exactly +0.20 (by construction of the scaling)."""
    A = np.array([[0.6, 0.0], [0.3, 0.5]])
    sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    Y = _simulate_var1(A, sigma, T=400, seed=20)
    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=2.0)
    traj = shock_scenario(fit, baseline=Y[-1], shock_var_idx=0,
                                 shock_size=0.20, h=10)
    assert traj[0, 0] == pytest.approx(0.20, abs=1e-9)


def test_shock_scenario_propagates_to_other_variables():
    """A diesel-style shock should produce non-trivial response on the
    correlated partner variable via Σ — that's the contemporaneous
    transmission `conditional_forecast` misses."""
    A = np.array([[0.5, 0.0], [0.0, 0.5]])      # zero AR cross-effects
    sigma = np.array([[1.0, 0.7], [0.7, 1.0]])  # strong contemporaneous correlation
    Y = _simulate_var1(A, sigma, T=400, seed=21)
    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=2.0)
    traj = shock_scenario(fit, baseline=Y[-1], shock_var_idx=0,
                                 shock_size=1.0, h=5)
    # var-1 at h=0 should respond — and decisively non-zero
    assert abs(traj[0, 1]) > 0.4    # ≈ correlation under unit shock


def test_shock_scenario_decays_for_stable_var():
    A = np.array([[0.7, 0.1], [0.1, 0.6]])
    sigma = np.eye(2)
    Y = _simulate_var1(A, sigma, T=400, seed=22)
    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=2.0)
    traj = shock_scenario(fit, baseline=Y[-1], shock_var_idx=0,
                                 shock_size=1.0, h=40)
    assert abs(traj[40, 0]) < 0.05 * abs(traj[0, 0])


def test_bvar_forecaster_rolling_conformal_band_on_synthetic():
    """The rolling-conformal band path should produce a non-trivial
    band different from Gaussian, with the same point forecast."""
    A = np.array([[0.7, 0.1], [0.05, 0.6]])
    sigma = np.eye(2)
    Y = _simulate_var1(A, sigma, T=200, seed=99)
    idx = pd.date_range("2018-01-31", periods=200, freq="ME")
    panel = pd.DataFrame(Y, columns=["v1", "v2"], index=idx)
    fc_g = BVARForecaster(var_cols=["v1", "v2"], target_col="v1",
                                horizon=1, p=1, train_min=60,
                                band_method="gaussian")
    fc_c = BVARForecaster(var_cols=["v1", "v2"], target_col="v1",
                                horizon=1, p=1, train_min=60,
                                calib_months=24,
                                band_method="rolling_conformal")
    f_g = fc_g.fit_predict(panel, panel.index[150], panel.index[151])
    f_c = fc_c.fit_predict(panel, panel.index[150], panel.index[151])
    # Same point forecast (band method should not affect it).
    assert f_g.point == pytest.approx(f_c.point, rel=1e-12)
    # Band metadata reflects the chosen method.
    assert f_g.metadata["band_source"] == "gaussian"
    assert f_c.metadata["band_source"] == "rolling_conformal"
    assert f_c.metadata["n_calib"] >= 9


def test_bvar_forecaster_walk_forward_compat():
    A = np.array([[0.7, 0.1], [0.05, 0.6]])
    sigma = np.eye(2)
    Y = _simulate_var1(A, sigma, T=300, seed=7)
    idx = pd.date_range("2020-01-31", periods=300, freq="ME")
    panel = pd.DataFrame(Y, columns=["v1", "v2"], index=idx)
    fc = BVARForecaster(var_cols=["v1", "v2"], target_col="v1",
                              horizon=1, p=1, train_min=60)
    origins = panel.index[100:120]
    forecasts = walk_forward(fc, panel, "v1", origins, horizon=1)
    assert len(forecasts) > 0
    df = attach_actuals(forecasts, panel["v1"])
    block = score(df)
    assert np.isfinite(block.rmse)
    # On a stable VAR, the BVAR should be at least competitive with naive
    assert block.rmse_naive is not None
    assert block.rmse < block.rmse_naive * 1.5    # not catastrophically worse
