"""Tests for the statistical-tests module — DM, CW, GW, KS."""

from __future__ import annotations

import numpy as np
import pytest

from thales.evaluation import tests as T


def test_newey_west_reduces_to_variance_at_lag_zero() -> None:
    rng = np.random.default_rng(0)
    d = rng.normal(0, 1, 200)
    v0 = T.newey_west_var(d, lag=0)
    # Sample variance / n
    expected = float(np.var(d, ddof=0)) / len(d)
    assert abs(v0 - expected) < 1e-10


def test_diebold_mariano_identical_errors_near_zero() -> None:
    rng = np.random.default_rng(0)
    e = rng.normal(0, 1, 200)
    res = T.diebold_mariano(e, e.copy())
    # Degenerate — DM stat should be NaN or p ≈ 1
    assert np.isnan(res.statistic) or res.pvalue > 0.5


def test_diebold_mariano_detects_better_forecast() -> None:
    rng = np.random.default_rng(0)
    n = 500
    actual = rng.normal(0, 1, n)
    # A is noisy, B is accurate
    err_a = rng.normal(0, 2, n)
    err_b = rng.normal(0, 0.5, n)
    res = T.diebold_mariano(err_a, err_b, two_sided=True, loss="squared")
    assert res.statistic > 0  # A has larger MSE
    assert res.pvalue < 0.01


def test_clark_west_nested_rejects_when_large_helps() -> None:
    rng = np.random.default_rng(0)
    n = 500
    actual = rng.normal(0, 1, n)
    # Small model: predicts mean 0 (dumb)
    pred_small = np.zeros(n)
    # Large model: predicts actual with a bit of noise
    pred_large = actual + rng.normal(0, 0.3, n)
    err_small = pred_small - actual
    err_large = pred_large - actual
    res = T.clark_west(err_small, err_large, pred_small, pred_large)
    assert res.statistic > 0
    assert res.pvalue < 0.01


def test_clark_west_near_zero_when_predictions_identical() -> None:
    rng = np.random.default_rng(0)
    n = 500
    actual = rng.normal(0, 1, n)
    pred_small = rng.normal(0, 1, n)
    # Large == small exactly: nested with zero added info
    pred_large = pred_small.copy()
    err_small = pred_small - actual
    err_large = pred_large - actual
    res = T.clark_west(err_small, err_large, pred_small, pred_large)
    # Statistic should be NaN or exactly zero (d_cw = 0 for all rows)
    assert np.isnan(res.statistic) or abs(res.statistic) < 1e-10


def test_giacomini_white_unconditional_equals_dm() -> None:
    rng = np.random.default_rng(0)
    n = 300
    err_a = rng.normal(0, 1.5, n)
    err_b = rng.normal(0, 1.0, n)
    gw = T.giacomini_white(err_a, err_b)
    dm = T.diebold_mariano(err_a, err_b)
    # GW with no test function is DM
    assert abs(gw.statistic - dm.statistic) < 1e-10
    assert abs(gw.pvalue - dm.pvalue) < 1e-10


def test_giacomini_white_conditional_detects_state_dependence() -> None:
    rng = np.random.default_rng(0)
    n = 400
    actual = rng.normal(0, 1, n)
    # State variable (e.g., regime indicator)
    state = (rng.random(n) > 0.5).astype(float)
    # Model A is bad only in state=1, tied in state=0
    err_a = rng.normal(0, 1, n) + state * rng.normal(0, 1.5, n)
    err_b = rng.normal(0, 1, n)
    res = T.giacomini_white(err_a, err_b, test_function=state)
    assert res.pvalue < 0.05


@pytest.mark.slow
def test_ks_uniform_accepts_uniform() -> None:
    rng = np.random.default_rng(42)  # seed chosen so KS p > 0.1 for uniform
    pit = rng.uniform(0, 1, 2000)
    res = T.ks_uniform(pit)
    # With n=2000 and truly uniform data, KS rejects < 5% of the time.
    # Using seed 42 is far from the rejection region.
    assert res.pvalue > 0.1, f"p={res.pvalue}"


@pytest.mark.slow
def test_ks_uniform_rejects_biased_pit() -> None:
    # Beta(2, 5) is clearly non-uniform
    rng = np.random.default_rng(0)
    pit = rng.beta(2, 5, 500)
    res = T.ks_uniform(pit)
    assert res.pvalue < 0.01
