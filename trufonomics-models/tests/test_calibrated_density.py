"""Tests for the calibrated (time-varying-scale) density layer.

The expanding-window residual bootstrap in :mod:`thales.evaluation.density` is
correct but produces over-wide bands (it keeps high-volatility residuals from
old regimes forever, so coverage runs hot — 85-98 % at nominal 80 %). This
module's job is the calibration fix: a *recent-window* / EWMA scale so the band
width tracks the current volatility regime, while preserving fat tails via a
standardized-residual bootstrap.

These tests pin the behavioural contract before implementation (TDD).
"""
from __future__ import annotations

import numpy as np

from thales.evaluation.calibrated_density import (
    calibrated_samples,
    predictive_quantiles,
    rolling_scale,
)

RNG = np.random.default_rng(12345)


# ── rolling_scale ──────────────────────────────────────────────────────────

def test_rolling_scale_uses_recent_window_not_full_history():
    """200 calm residuals then 24 volatile; the recent-window scale tracks the
    volatile regime and is far above the diluted full-history scale."""
    res = np.concatenate([RNG.normal(0, 0.1, 200), RNG.normal(0, 1.0, 24)])
    s_recent = rolling_scale(res, window=24)
    s_all = rolling_scale(res, window=10_000)
    assert 0.7 < s_recent < 1.3
    assert s_recent > 2 * s_all


def test_rolling_scale_ewma_weights_recent_more():
    """With a recent volatility burst, the EWMA scale exceeds the flat-window
    scale (it weights the last few residuals more)."""
    res = np.concatenate([RNG.normal(0, 0.1, 50), RNG.normal(0, 1.0, 10)])
    s_flat = rolling_scale(res, window=60)
    s_ewma = rolling_scale(res, window=60, halflife=5)
    assert s_ewma > s_flat


def test_rolling_scale_insufficient_returns_nan():
    assert np.isnan(rolling_scale(np.array([0.3])))
    assert np.isnan(rolling_scale(np.array([])))


# ── calibrated_samples ─────────────────────────────────────────────────────

def test_calibrated_samples_centered_and_scaled():
    """Band centers on point + recent residual bias and has scale = recent sd."""
    res = RNG.normal(0.0, 0.5, 100)
    rw = res[-60:]
    s = calibrated_samples(2.0, res, n_samples=20_000, window=60, seed=1)
    assert abs(np.mean(s) - (2.0 + rw.mean())) < 0.02
    assert abs(np.std(s) - rw.std()) < 0.02


def test_calibrated_samples_wider_when_recent_vol_higher():
    lo = calibrated_samples(0.0, RNG.normal(0, 0.2, 60), n_samples=20_000, window=24, seed=2)
    hi = calibrated_samples(0.0, RNG.normal(0, 1.0, 60), n_samples=20_000, window=24, seed=3)
    assert np.std(hi) > 2 * np.std(lo)


def test_calibrated_samples_preserve_fat_tails():
    """A fat-tailed residual distribution yields fat-tailed samples (excess
    kurtosis preserved through the standardized bootstrap)."""
    base = RNG.standard_t(3, size=300) * 0.3
    s = calibrated_samples(0.0, base, n_samples=40_000, window=300, seed=4)
    z = (s - s.mean()) / s.std()
    assert np.mean(z ** 4) > 4.0  # normal == 3


def test_calibrated_samples_insufficient_returns_nan():
    s = calibrated_samples(1.0, np.array([0.2]), n_samples=100)
    assert np.all(np.isnan(s))


# ── predictive_quantiles ───────────────────────────────────────────────────

def test_predictive_quantiles_monotone():
    s = calibrated_samples(1.0, RNG.normal(0, 0.4, 80), n_samples=5_000, window=60, seed=5)
    q = predictive_quantiles(s, [5, 10, 25, 50, 75, 90, 95])
    assert np.all(np.diff(q) >= 0)


def test_predictive_quantiles_nan_samples_return_nan():
    q = predictive_quantiles(np.full(100, np.nan), [10, 90])
    assert np.all(np.isnan(q))


# ── calibration (the whole point) ──────────────────────────────────────────

def test_coverage_calibrated_on_stationary_synthetic():
    """iid residuals -> the rolling-bootstrap 80 %/50 % bands cover ~80 %/50 %."""
    sigma = 0.7
    state = np.random.default_rng(7)
    hist = list(state.normal(0, sigma, 30))
    hits80 = hits50 = n = 0
    for _ in range(400):
        s = calibrated_samples(0.0, np.array(hist), n_samples=3_000, window=36,
                               seed=int(state.integers(1_000_000_000)))
        lo, hi = predictive_quantiles(s, [10, 90])
        q25, q75 = predictive_quantiles(s, [25, 75])
        y = state.normal(0, sigma)
        hits80 += lo <= y <= hi
        hits50 += q25 <= y <= q75
        n += 1
        hist.append(y)
    assert 0.74 <= hits80 / n <= 0.86
    assert 0.43 <= hits50 / n <= 0.57


def test_rolling_beats_expanding_after_vol_drop():
    """High-vol then low-vol regime: a short rolling window recovers ~80 %
    coverage; an expanding window stays over-wide (over-covers). Coverage is
    measured on the back half, after the window has had time to adapt."""
    def run(window):
        state = np.random.default_rng(11)
        hist = list(state.normal(0, 1.5, 60))   # high-vol history
        covers = []
        for _ in range(200):
            s = calibrated_samples(0.0, np.array(hist), n_samples=3_000, window=window,
                                   seed=int(state.integers(1_000_000_000)))
            lo, hi = predictive_quantiles(s, [10, 90])
            y = state.normal(0, 0.3)            # low-vol actuals now
            covers.append(lo <= y <= hi)
            hist.append(y)
        return float(np.mean(covers[-100:]))    # back half, post-adaptation

    cov_roll = run(24)
    cov_exp = run(10_000)
    assert abs(cov_roll - 0.80) < abs(cov_exp - 0.80)
    assert cov_exp > cov_roll
