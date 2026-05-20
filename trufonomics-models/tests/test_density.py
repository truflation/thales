"""Tests for thales.evaluation.density — sample emission + density block.

Covers:
  * samples_from_residuals: bootstrap from empirical residual distribution
  * samples_from_gaussian: parametric Gaussian draws
  * samples_from_quantiles: piecewise-linear inverse-CDF
  * score_density: CRPS + PIT KS + coverage + sharpness on a sample matrix
  * stack_samples: assembly from a list of Forecasts
"""
from __future__ import annotations

import numpy as np
import pytest

from thales.evaluation.density import (
    DEFAULT_N_SAMPLES,
    DensityBlock,
    samples_from_gaussian,
    samples_from_quantiles,
    samples_from_residuals,
    score_density,
    stack_samples,
)
from thales.evaluation.harness import Forecast


# ─── samples_from_residuals ──────────────────────────────────────────────


def test_residuals_returns_correct_shape():
    errors = np.array([-0.3, -0.1, 0.0, 0.1, 0.2, 0.4])
    s = samples_from_residuals(point=2.0, errors=errors, n_samples=300, seed=0)
    assert s.shape == (300,)


def test_residuals_centered_on_point_when_errors_centered():
    rng = np.random.default_rng(42)
    errors = rng.standard_normal(500) * 0.3
    s = samples_from_residuals(point=2.5, errors=errors, n_samples=2000, seed=1)
    # bootstrap mean ≈ point + errors.mean() ≈ point
    assert abs(np.mean(s) - 2.5) < 0.05


def test_residuals_preserves_residual_scale():
    """Bootstrap σ should match input σ within sampling noise."""
    rng = np.random.default_rng(7)
    errors = rng.standard_normal(400) * 0.4
    s = samples_from_residuals(point=0.0, errors=errors, n_samples=4000, seed=2)
    # 4000 bootstrap samples → tight σ estimate. Tolerance generous for safety.
    assert abs(np.std(s) - np.std(errors)) < 0.05


def test_residuals_too_few_returns_nan():
    s = samples_from_residuals(point=1.0, errors=np.array([0.1]), n_samples=100)
    assert np.all(np.isnan(s))
    assert s.shape == (100,)


def test_residuals_drops_nan_inputs():
    errors = np.array([0.1, np.nan, 0.2, np.nan, 0.3])
    s = samples_from_residuals(point=0.0, errors=errors, n_samples=200, seed=0)
    assert not np.any(np.isnan(s))
    # Only three valid residuals; bootstrap should resample them.
    unique_vals = np.unique(np.round(s, 6))
    assert set(unique_vals).issubset({0.1, 0.2, 0.3})


def test_residuals_deterministic_under_same_seed():
    errors = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    s1 = samples_from_residuals(point=1.0, errors=errors, n_samples=50, seed=11)
    s2 = samples_from_residuals(point=1.0, errors=errors, n_samples=50, seed=11)
    np.testing.assert_array_equal(s1, s2)


# ─── samples_from_gaussian ───────────────────────────────────────────────


def test_gaussian_correct_shape():
    s = samples_from_gaussian(mu=2.0, sigma=0.3, n_samples=400, seed=0)
    assert s.shape == (400,)


def test_gaussian_recovers_moments():
    s = samples_from_gaussian(mu=1.5, sigma=0.4, n_samples=20_000, seed=0)
    assert abs(np.mean(s) - 1.5) < 0.02
    assert abs(np.std(s) - 0.4) < 0.02


def test_gaussian_invalid_sigma_returns_nan():
    s = samples_from_gaussian(mu=1.0, sigma=0.0, n_samples=50)
    assert np.all(np.isnan(s))
    s = samples_from_gaussian(mu=1.0, sigma=-0.5, n_samples=50)
    assert np.all(np.isnan(s))
    s = samples_from_gaussian(mu=1.0, sigma=float("nan"), n_samples=50)
    assert np.all(np.isnan(s))


# ─── samples_from_quantiles ──────────────────────────────────────────────


def test_quantiles_identity_on_uniform_grid():
    """Quantile-vector with linear inverse CDF reproduces a uniform draw."""
    levels = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
    quantiles = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # linear inverse CDF
    s = samples_from_quantiles(quantiles, levels, n_samples=10_000, seed=0)
    # Mean ~= 0, slight nonzero from tail clamping.
    assert abs(np.mean(s)) < 0.05


def test_quantiles_too_few_returns_nan():
    s = samples_from_quantiles(np.array([0.5]), np.array([0.5]),
                                n_samples=50, seed=0)
    assert np.all(np.isnan(s))


# ─── score_density ───────────────────────────────────────────────────────


def test_score_density_calibrated_gaussian():
    """Gaussian forecasts on Gaussian truth — calibrated. PIT should pass."""
    rng = np.random.default_rng(12)
    n = 200
    actual = rng.standard_normal(n) * 0.5 + 1.0
    samples = np.array([
        rng.normal(loc=1.0, scale=0.5, size=500) for _ in range(n)
    ])  # forecaster predicts the truth distribution (calibrated, n=200)
    block = score_density(samples, actual)
    assert block.n == n
    assert block.cov80 > 0.74 and block.cov80 < 0.86
    assert block.cov95 > 0.91 and block.cov95 < 0.99
    # PIT KS test is a hypothesis test — when calibrated p > 0.05.
    assert block.pit_ks_pvalue > 0.05


def test_score_density_under_dispersed_fails_calibration():
    """Bands too narrow → realizations land in tails → undercoverage."""
    rng = np.random.default_rng(3)
    n = 200
    actual = rng.standard_normal(n) * 0.5
    samples = np.array([
        rng.normal(loc=0.0, scale=0.05, size=500) for _ in range(n)
    ])  # σ_pred = 0.05, σ_actual = 0.5 → severe under-dispersion
    block = score_density(samples, actual)
    assert block.cov80 < 0.5         # realized far below 80% nominal
    assert block.pit_ks_pvalue < 0.05  # PIT non-uniform


def test_score_density_drops_nan_rows():
    n = 100
    actual = np.zeros(n)
    samples = np.zeros((n, 200))
    # First 30 rows: NaN samples
    samples[:30] = np.nan
    block = score_density(samples, actual)
    assert block.n == 70


def test_score_density_empty_returns_nan_block():
    samples = np.full((10, 50), np.nan)
    actual = np.zeros(10)
    block = score_density(samples, actual)
    assert block.n == 0
    assert np.isnan(block.crps)


def test_score_density_shape_mismatch_raises():
    with pytest.raises(ValueError):
        score_density(np.zeros((10, 50)), np.zeros(11))


def test_score_density_block_summary_renders():
    rng = np.random.default_rng(0)
    n = 100
    actual = rng.standard_normal(n)
    samples = np.array([rng.standard_normal(200) for _ in range(n)])
    block = score_density(samples, actual)
    txt = block.summary()
    assert "CRPS" in txt
    assert "PIT-KS" in txt
    assert "cov80" in txt


# ─── stack_samples ───────────────────────────────────────────────────────


def test_stack_samples_assembles_aligned_matrix():
    import pandas as pd
    forecasts = []
    for i in range(5):
        s = np.full(DEFAULT_N_SAMPLES, float(i))
        forecasts.append(Forecast(
            origin=pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            target=pd.Timestamp("2024-01-02") + pd.Timedelta(days=i),
            point=float(i), samples=s,
        ))
    M = stack_samples(forecasts)
    assert M.shape == (5, DEFAULT_N_SAMPLES)
    for i in range(5):
        assert np.all(M[i] == float(i))


def test_stack_samples_handles_missing_samples():
    import pandas as pd
    f1 = Forecast(origin=pd.Timestamp("2024-01-01"),
                   target=pd.Timestamp("2024-01-02"), point=1.0,
                   samples=np.full(DEFAULT_N_SAMPLES, 1.0))
    f2 = Forecast(origin=pd.Timestamp("2024-01-02"),
                   target=pd.Timestamp("2024-01-03"), point=2.0,
                   samples=None)
    M = stack_samples([f1, f2])
    assert M.shape == (2, DEFAULT_N_SAMPLES)
    assert np.all(M[0] == 1.0)
    assert np.all(np.isnan(M[1]))
