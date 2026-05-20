"""Tests for evaluation.metrics — point, density, calibration, classification."""

from __future__ import annotations

import numpy as np
import pytest

from thales.evaluation import metrics as M


def test_rmse_mae_on_known_values() -> None:
    pred = np.array([1.0, 2.0, 3.0])
    actual = np.array([1.5, 2.5, 2.5])
    assert M.rmse(pred, actual) == 0.5
    assert M.mae(pred, actual) == 0.5


def test_rmse_drops_nans() -> None:
    pred = np.array([1.0, np.nan, 3.0])
    actual = np.array([1.0, 2.5, 2.0])
    # Only (1,1) and (3,2): errors 0 and 1, RMSE = sqrt(0.5)
    assert abs(M.rmse(pred, actual) - (0.5 ** 0.5)) < 1e-10


def test_rmse_empty_returns_nan() -> None:
    assert np.isnan(M.rmse(np.array([]), np.array([])))


def test_mase_equals_one_for_naive() -> None:
    naive = np.array([1.0, 2.0, 3.0])
    actual = np.array([1.5, 2.5, 2.5])
    # MASE of the naive itself is 1 by construction
    assert abs(M.mase(naive, actual, naive) - 1.0) < 1e-10


def test_directional_accuracy() -> None:
    # pred up vs previous, actual up: hit. down/down: hit. up/down: miss.
    actual = np.array([1.0, 2.0, 3.0, 2.5])
    pred   = np.array([0.0, 2.5, 3.5, 2.0])  # pred is what we forecast for each t
    # Reference (previous actual): [nan, 1, 2, 3]
    # t=1: pred 2.5 > 1 → up. actual 2 > 1 → up. hit.
    # t=2: pred 3.5 > 2 → up. actual 3 > 2 → up. hit.
    # t=3: pred 2.0 < 3 → down. actual 2.5 < 3 → down. hit.
    assert M.directional_accuracy(pred, actual) == 1.0


@pytest.mark.slow
def test_crps_samples_calibrated_gaussian() -> None:
    """Sample-based CRPS on a perfect Gaussian predictive vs its own draw.

    When samples come from N(μ, σ²) and the outcome is drawn from the same
    distribution, CRPS should equal the analytic Gaussian CRPS within MC error.
    """
    rng = np.random.default_rng(42)
    n = 500
    mu = np.zeros(n)
    sigma = np.ones(n)
    # Draw 1000 samples per origin
    samples = rng.normal(mu[:, None], sigma[:, None], size=(n, 1000))
    actual = rng.normal(mu, sigma)
    crps_s = M.crps_samples(samples, actual)
    crps_g = M.crps_gaussian(mu, sigma, actual)
    # Should agree within ~5% (Monte Carlo noise)
    assert abs(crps_s - crps_g) / crps_g < 0.05


def test_log_score_gaussian_matches_scipy() -> None:
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    mu, sigma = np.zeros(100), np.ones(100)
    y = rng.normal(0, 1, 100)
    ls = M.log_score_gaussian(mu, sigma, y)
    expected = norm.logpdf(y, mu, sigma).mean()
    assert abs(ls - expected) < 1e-10


def test_quantile_loss_median_equals_half_mae() -> None:
    pred = np.array([1.0, 2.0, 3.0])
    actual = np.array([1.5, 2.5, 2.5])
    # Pinball at τ=0.5 is 0.5 × MAE
    assert abs(M.quantile_loss(pred, actual, 0.5) - 0.5 * M.mae(pred, actual)) < 1e-10


@pytest.mark.slow
def test_pit_uniform_for_calibrated_gaussian() -> None:
    rng = np.random.default_rng(0)
    n, S = 400, 2000
    mu = np.zeros(n)
    samples = rng.normal(0.0, 1.0, size=(n, S))
    actual = rng.normal(0.0, 1.0, size=n)
    pit = M.pit_samples(samples, actual)
    # KS test p-value should not reject uniformity
    pval = M.pit_ks_pvalue(pit)
    assert pval > 0.05, f"PIT failed KS uniformity at p={pval}"


@pytest.mark.slow
def test_pit_nonuniform_for_biased_forecast() -> None:
    rng = np.random.default_rng(1)
    n, S = 400, 2000
    samples = rng.normal(0.0, 1.0, size=(n, S))
    # Outcome comes from a shifted distribution
    actual = rng.normal(2.0, 1.0, size=n)
    pit = M.pit_samples(samples, actual)
    pval = M.pit_ks_pvalue(pit)
    assert pval < 0.01, "KS should reject uniformity for biased forecast"


@pytest.mark.slow
def test_interval_coverage_near_nominal() -> None:
    rng = np.random.default_rng(0)
    n, S = 500, 2000
    samples = rng.normal(0.0, 1.0, size=(n, S))
    actual = rng.normal(0.0, 1.0, size=n)
    cov80 = M.interval_coverage(samples, actual, level=0.8)
    cov95 = M.interval_coverage(samples, actual, level=0.95)
    assert abs(cov80 - 0.8) < 0.05
    assert abs(cov95 - 0.95) < 0.03


def test_brier_and_log_loss_perfect() -> None:
    prob = np.array([1.0, 0.0, 1.0, 0.0])
    outcome = np.array([1.0, 0.0, 1.0, 0.0])
    assert M.brier_score(prob, outcome) == 0.0
    assert M.log_loss(prob, outcome) < 1e-6


def test_roc_auc_perfect_separator() -> None:
    prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    outcome = np.array([0, 0, 0, 1, 1])
    assert M.roc_auc(prob, outcome) == 1.0


@pytest.mark.slow
def test_bootstrap_ci_contains_truth() -> None:
    rng = np.random.default_rng(123)
    n = 200
    pred = rng.normal(0, 1, n)
    actual = pred + rng.normal(0, 0.5, n)
    point, lo, hi = M.bootstrap_ci(M.rmse, pred, actual, n_boot=500, seed=7)
    assert lo <= point <= hi
    # Known std of residual is 0.5, so RMSE ≈ 0.5
    assert lo < 0.5 < hi
