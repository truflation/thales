"""Tests for JohansenGatedVECM — Fix #4."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thales.evaluation.harness import attach_actuals, score, walk_forward
from thales.models.archetypes.johansen_gated_vecm import (
    JohansenGatedVECM,
    johansen_test,
)


# ── Synthetic panels ─────────────────────────────────────────────────────


def _cointegrated_panel(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """y2 ~ y1 + stationary noise → cointegrated with β = (1, -1)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-31", periods=n, freq="ME")
    y1 = np.cumsum(rng.normal(0, 1, n)) + 5
    y2 = y1 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y1": y1, "y2": y2}, index=idx)


def _random_walk_panel(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Two independent random walks → no cointegration."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-31", periods=n, freq="ME")
    y1 = np.cumsum(rng.normal(0, 1, n))
    y2 = np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"y1": y1, "y2": y2}, index=idx)


# ── johansen_test helper ─────────────────────────────────────────────────


def test_johansen_detects_cointegration_on_cointegrated_pair():
    panel = _cointegrated_panel()
    j = johansen_test(panel["y1"].values, panel["y2"].values)
    assert j["cointegrated"] is True
    assert j["rank"] >= 1
    assert j["trace_stat"][0] > j["cv"][0]


def test_johansen_rejects_cointegration_on_random_walks():
    panel = _random_walk_panel()
    j = johansen_test(panel["y1"].values, panel["y2"].values)
    assert j["cointegrated"] is False
    assert j["rank"] == 0


def test_johansen_invalid_significance_raises():
    panel = _cointegrated_panel()
    with pytest.raises(ValueError):
        johansen_test(panel["y1"].values, panel["y2"].values,
                          significance_level=0.07)


# ── Gating ────────────────────────────────────────────────────────────────


def test_vecm_branch_taken_on_cointegrated_panel():
    panel = _cointegrated_panel()
    fc = JohansenGatedVECM(target_col="y1", paired_col="y2",
                                  train_window_months=60, train_min=36,
                                  fallback="ardl",
                                  band_method="rolling_conformal",
                                  calib_months=24)
    f = fc.fit_predict(panel, panel.index[80], panel.index[81])
    assert f.metadata["branch"] == "vecm"
    assert f.metadata["cointegrated"] is True
    assert np.isfinite(f.point)


def test_fallback_branch_taken_on_random_walks():
    panel = _random_walk_panel()
    fc = JohansenGatedVECM(target_col="y1", paired_col="y2",
                                  train_window_months=60, train_min=36,
                                  fallback="ardl",
                                  band_method="rolling_conformal",
                                  calib_months=24)
    f = fc.fit_predict(panel, panel.index[80], panel.index[81])
    assert f.metadata["branch"] == "ardl"
    assert f.metadata["cointegrated"] is False


# ── Each fallback runs and is selectable ────────────────────────────────


@pytest.mark.parametrize("fb", ["ardl", "bridge", "ar1"])
def test_each_fallback_runs(fb):
    panel = _random_walk_panel()
    fc = JohansenGatedVECM(target_col="y1", paired_col="y2",
                                  train_window_months=60, train_min=36,
                                  fallback=fb,
                                  band_method="rolling_conformal",
                                  calib_months=24)
    f = fc.fit_predict(panel, panel.index[80], panel.index[81])
    assert f.metadata["branch"] == fb
    assert np.isfinite(f.point)
    assert f.has_bands


# ── Forecaster Protocol compatibility ───────────────────────────────────


def test_walk_forward_compatibility_cointegrated():
    panel = _cointegrated_panel()
    fc = JohansenGatedVECM(target_col="y1", paired_col="y2",
                                  train_window_months=60, train_min=36,
                                  band_method="rolling_conformal",
                                  calib_months=24)
    origins = panel.index[60:90]
    forecasts = walk_forward(fc, panel, "y1", origins, horizon=1)
    assert len(forecasts) > 0
    df = attach_actuals(forecasts, panel["y1"])
    block = score(df)
    assert np.isfinite(block.rmse)
    # On cointegrated data, VECM should be at least competitive vs RW
    # (we don't test "beats" because the gate may flicker on small
    # training windows). Assert just that all-VECM branches were taken
    # most of the time.
    branches = [f.metadata["branch"] for f in forecasts]
    n_vecm = sum(b == "vecm" for b in branches)
    assert n_vecm >= 0.8 * len(branches)


def test_default_fallback_is_ardl():
    fc = JohansenGatedVECM()
    assert fc.fallback == "ardl"


def test_default_band_method_is_rolling_conformal():
    fc = JohansenGatedVECM()
    assert fc.band_method == "rolling_conformal"
