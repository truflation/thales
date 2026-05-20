"""Tests for regime-transition buffer in RegimeConditionalBridgeNowcaster — Fix #6."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from thales.models.same_month_nowcaster import (
    RegimeConditionalBridgeNowcaster,
    _markov_one_step_p_high,
    _regime_sigma,
)


# ── Closed-form Markov projection ────────────────────────────────────────


def test_markov_one_step_recovers_stationary_under_constant_p():
    """If p_h is at the stationary distribution, one-step-ahead should
    return the same value (definition of stationarity)."""
    p_stay_low = 0.95
    p_stay_high = 0.85
    pi_high = (1.0 - p_stay_low) / (2.0 - p_stay_low - p_stay_high)
    p_next = _markov_one_step_p_high(pi_high, p_stay_low, p_stay_high)
    assert p_next == pytest.approx(pi_high, abs=1e-12)


def test_markov_one_step_drifts_toward_stationary():
    """From p_h=0 (definitely low), one step should lift p_h toward
    the stationary distribution by exactly P(low → high)."""
    p_stay_low = 0.95
    p_stay_high = 0.85
    p_next = _markov_one_step_p_high(0.0, p_stay_low, p_stay_high)
    assert p_next == pytest.approx(1.0 - p_stay_low, abs=1e-12)
    p_next2 = _markov_one_step_p_high(1.0, p_stay_low, p_stay_high)
    assert p_next2 == pytest.approx(p_stay_high, abs=1e-12)


# ── Buffer-method semantics ──────────────────────────────────────────────


def test_filtered_buffer_matches_legacy_blend():
    """With ``filtered`` buffer, σ̂ is the original P(now)-weighted blend."""
    sigma_t, p_eff = _regime_sigma(
        p_high=0.30, sigma_low=0.5, sigma_high=2.0,
        buffer_method="filtered",
        p_stay_low=0.95, p_stay_high=0.85,
    )
    assert sigma_t == pytest.approx(0.7 * 0.5 + 0.3 * 2.0, abs=1e-12)
    assert p_eff == pytest.approx(0.30, abs=1e-12)


def test_transition_buffer_uses_one_step_p_high():
    """``transition`` buffer projects p_h one step ahead before blending."""
    sigma_t, p_eff = _regime_sigma(
        p_high=0.30, sigma_low=0.5, sigma_high=2.0,
        buffer_method="transition",
        p_stay_low=0.95, p_stay_high=0.85,
    )
    expected_p = 0.7 * 0.05 + 0.3 * 0.85
    expected_sigma = (1 - expected_p) * 0.5 + expected_p * 2.0
    assert p_eff == pytest.approx(expected_p, abs=1e-12)
    assert sigma_t == pytest.approx(expected_sigma, abs=1e-12)


def test_transition_buffer_widens_bands_before_flip():
    """When the system is leaving low-vol (p_h rising from 0), the
    transition-buffered σ̂ must be larger than the filtered σ̂ — that's
    the whole point of Fix #6."""
    p_h_now = 0.15    # filter says still mostly low, but rising
    s_filt, _ = _regime_sigma(
        p_h_now, sigma_low=0.5, sigma_high=2.0,
        buffer_method="filtered",
        p_stay_low=0.90, p_stay_high=0.80,
    )
    s_trans, _ = _regime_sigma(
        p_h_now, sigma_low=0.5, sigma_high=2.0,
        buffer_method="transition",
        p_stay_low=0.90, p_stay_high=0.80,
    )
    # Markov projection from p_h=0.15, P(L→H)=0.10, P(H→H)=0.80:
    #   p_next = 0.85·0.10 + 0.15·0.80 = 0.085 + 0.12 = 0.205
    # Higher than 0.15 → wider band.
    assert s_trans > s_filt


def test_transition_max_pins_to_high_sigma_near_boundary():
    """Within the transition_threshold, σ̂ should be max(σ_low, σ_high)."""
    sigma_t, _ = _regime_sigma(
        p_high=0.40, sigma_low=0.5, sigma_high=2.0,
        buffer_method="transition_max",
        p_stay_low=0.95, p_stay_high=0.85,
        transition_threshold=0.20,
    )
    # p_eff after one-step ≈ 0.6·0.05 + 0.4·0.85 = 0.37 → min(0.37, 0.63) = 0.37
    # 0.37 ≥ 0.20 (threshold) → σ̂ = max(0.5, 2.0) = 2.0
    assert sigma_t == pytest.approx(2.0, abs=1e-9)


def test_transition_max_falls_back_to_blend_when_certain():
    """Far from the boundary (p_h ≈ 0 or 1), transition_max should
    blend like ``transition``."""
    sigma_max, _ = _regime_sigma(
        p_high=0.02, sigma_low=0.5, sigma_high=2.0,
        buffer_method="transition_max",
        p_stay_low=0.95, p_stay_high=0.85,
        transition_threshold=0.20,
    )
    sigma_trans, _ = _regime_sigma(
        p_high=0.02, sigma_low=0.5, sigma_high=2.0,
        buffer_method="transition",
        p_stay_low=0.95, p_stay_high=0.85,
        transition_threshold=0.20,
    )
    assert sigma_max == pytest.approx(sigma_trans, abs=1e-9)


# ── End-to-end forecaster ────────────────────────────────────────────────


def _bridge_panel(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-31", periods=n, freq="ME")
    truf = rng.normal(0, 0.4, n)
    bls = np.empty(n)
    bls[0] = 2.5
    # Inject a high-vol burst from t=40..50 to give the MS fit something
    # to discover.
    for t in range(1, n):
        scale = 0.40 if 40 <= t <= 50 else 0.10
        bls[t] = 0.8 * bls[t - 1] + 0.1 * truf[t - 1] + rng.normal(0, scale)
    return pd.DataFrame({"bls_yoy": bls, "truf_yoy": truf}, index=idx)


def test_default_buffer_is_transition():
    fc = RegimeConditionalBridgeNowcaster()
    assert fc.buffer_method == "transition"


def test_each_buffer_method_runs_end_to_end():
    panel = _bridge_panel()
    for bm in ("filtered", "transition", "transition_max"):
        fc = RegimeConditionalBridgeNowcaster(
            train_window_months=60, train_min=24, buffer_method=bm)
        f = fc.fit_predict(panel, panel.index[60], panel.index[60])
        assert f.has_bands
        assert f.metadata["buffer_method"] == bm
        assert np.isfinite(f.metadata["sigma_conditional"])


def test_transition_widens_band_vs_filtered_on_real_panel():
    """End-to-end: on a panel with discoverable regimes, the transition
    buffer should produce ≥ as wide bands as filtered, on average over
    many origins."""
    panel = _bridge_panel(n=120)
    origins = panel.index[60:90]
    widths_filt: list[float] = []
    widths_trans: list[float] = []
    for o in origins:
        fc_f = RegimeConditionalBridgeNowcaster(
            train_window_months=60, train_min=24, buffer_method="filtered")
        fc_t = RegimeConditionalBridgeNowcaster(
            train_window_months=60, train_min=24, buffer_method="transition")
        try:
            f_f = fc_f.fit_predict(panel, o, o)
            f_t = fc_t.fit_predict(panel, o, o)
        except Exception:
            continue
        widths_filt.append(f_f.hi80 - f_f.lo80)
        widths_trans.append(f_t.hi80 - f_t.lo80)
    assert len(widths_filt) > 0
    # Mean width under transition should be ≥ filtered (allowing small
    # equality margin since some origins may have p_h ≈ stationary).
    assert np.mean(widths_trans) >= np.mean(widths_filt) - 1e-9
