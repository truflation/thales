"""Tests for finite-sample conformal quantile helpers."""
from __future__ import annotations

import math

import numpy as np
import pytest

from thales.evaluation.conformal import (
    conformal_band_offsets,
    conformal_quantile_pair,
    min_n_for_alpha,
)


# ── Basic sanity ─────────────────────────────────────────────────────────


def test_conformal_band_offsets_returns_two_floats():
    errors = np.linspace(-1, 1, 100)
    lo, hi = conformal_band_offsets(errors, alpha=0.20)
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo < hi


def test_conformal_band_widens_with_smaller_alpha():
    """95% band must be wider than 80% band on the same residuals."""
    rng = np.random.default_rng(0)
    errors = rng.normal(0, 1, 200)
    lo80, hi80 = conformal_band_offsets(errors, alpha=0.20)
    lo95, hi95 = conformal_band_offsets(errors, alpha=0.05)
    assert lo95 < lo80
    assert hi95 > hi80


def test_conformal_band_centered_on_symmetric_residuals():
    rng = np.random.default_rng(1)
    errors = rng.normal(0, 1, 1000)
    lo, hi = conformal_band_offsets(errors, alpha=0.20)
    assert abs(abs(lo) - hi) < 0.2  # roughly symmetric


def test_conformal_band_offsets_invalid_alpha():
    errors = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        conformal_band_offsets(errors, alpha=0.0)
    with pytest.raises(ValueError):
        conformal_band_offsets(errors, alpha=1.0)
    with pytest.raises(ValueError):
        conformal_band_offsets(errors, alpha=-0.1)


def test_conformal_band_offsets_too_few_residuals():
    with pytest.raises(ValueError):
        conformal_band_offsets(np.array([0.0]), alpha=0.20)


# ── Finite-sample correction is conservative ─────────────────────────────


def test_finite_sample_conservative_vs_np_percentile():
    """For small n, conformal rank should be MORE conservative (wider
    band) than np.percentile interpolation."""
    rng = np.random.default_rng(42)
    n = 24    # small calibration window — exactly the one we use in PathA
    errors = rng.normal(0, 1, n)
    lo_c, hi_c = conformal_band_offsets(errors, alpha=0.20)
    lo_p = float(np.percentile(errors, 10))
    hi_p = float(np.percentile(errors, 90))
    # Conformal band must be at least as wide as np.percentile
    assert hi_c - lo_c >= hi_p - lo_p - 1e-9


# ── Coverage guarantee on synthetic data ─────────────────────────────────


def test_marginal_coverage_at_least_target_on_iid_normal():
    """Empirical coverage check: simulate many splits, each fitting a
    constant predictor on a calibration set and predicting a new point.
    Marginal coverage should be at least 1 − α."""
    rng = np.random.default_rng(7)
    alpha = 0.20
    n_cal = 50
    n_trials = 5_000
    hits = 0
    for _ in range(n_trials):
        cal = rng.normal(0, 1, n_cal)
        test = rng.normal(0, 1)
        lo, hi = conformal_band_offsets(cal, alpha=alpha)
        # Predictor is the constant 0; band is [0+lo, 0+hi]
        if lo <= test <= hi:
            hits += 1
    coverage = hits / n_trials
    # Theoretical guarantee: coverage ≥ 1 − α = 0.80. Allow a small
    # Monte-Carlo slack (binomial std for n=5000 ~0.005).
    assert coverage >= 0.79


# ── min_n_for_alpha sanity ───────────────────────────────────────────────


def test_min_n_for_alpha_at_standard_levels():
    # At α=0.20 (80% band): ⌈(2-0.20)/0.20⌉ = ⌈9⌉ = 9
    assert min_n_for_alpha(0.20) == 9
    # At α=0.05 (95% band): ⌈(2-0.05)/0.05⌉ = ⌈39⌉ = 39
    assert min_n_for_alpha(0.05) == 39
    # At α=0.01 (99% band): ⌈(2-0.01)/0.01⌉ = ⌈199⌉ = 199
    assert min_n_for_alpha(0.01) == 199


# ── Pair convenience ─────────────────────────────────────────────────────


def test_conformal_quantile_pair_default_keys():
    rng = np.random.default_rng(0)
    pairs = conformal_quantile_pair(rng.normal(0, 1, 100))
    assert set(pairs.keys()) == {0.20, 0.05}
    for k, (lo, hi) in pairs.items():
        assert lo < hi
