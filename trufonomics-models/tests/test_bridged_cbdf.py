"""Tests for BridgedCBDFForecaster — wired Forecaster for the
Truflation-scale → BLS-scale bridge.

Covers the structural correctness:
  * coefficient recovery on a synthetic bridge DGP
  * walk-forward + density emission through the harness
  * back-compat for forecasters without inner_pred at target
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thales.evaluation.harness import attach_actuals, score, walk_forward
from thales.models.composition.bridged_cbdf import BridgedCBDFForecaster


def _bridge_panel(n: int = 60, alpha: float = 0.5, beta: float = 0.4,
                    gamma: float = 0.5, sigma: float = 0.1, seed: int = 0
                    ) -> pd.DataFrame:
    """Synthetic panel with a known bridge DGP.

    BLS[t] = alpha + beta · BLS[t-1] + gamma · cbdf_pred[t-1] + ε

    where ``cbdf_pred[t]`` is the inner forecast made at time ``t``
    for ``t+1`` (origin-indexed prediction-for-next-period).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="MS")
    cbdf = 2.0 + rng.standard_normal(n) * 0.3
    bls = np.zeros(n)
    bls[0] = 2.0
    for t in range(1, n):
        bls[t] = (alpha + beta * bls[t - 1] + gamma * cbdf[t - 1]
                    + rng.standard_normal() * sigma)
    return pd.DataFrame({"bls_yoy": bls, "cbdf_pred": cbdf}, index=idx)


def test_recovers_known_coefs_on_synthetic_dgp():
    """Bridge should recover (alpha, beta, gamma) within tolerance."""
    panel = _bridge_panel(n=120, alpha=0.5, beta=0.3, gamma=0.5, sigma=0.05)
    fc = BridgedCBDFForecaster(calib_window=80, train_min=24)
    forecasts = walk_forward(fc, panel, target_col="bls_yoy",
                              origins=panel.index[80:90], horizon=1)
    # Average fitted coefficients across origins should be close to truth.
    metas = [f.metadata for f in forecasts]
    assert len(metas) >= 5
    avg_beta = np.mean([m["beta_lag"] for m in metas])
    avg_gamma = np.mean([m["gamma_inner"] for m in metas])
    assert abs(avg_beta - 0.3) < 0.10
    assert abs(avg_gamma - 0.5) < 0.10


def test_emits_density_samples():
    panel = _bridge_panel(n=60)
    fc = BridgedCBDFForecaster(calib_window=24, train_min=12,
                                  band_method="rolling_conformal")
    forecasts = walk_forward(fc, panel, target_col="bls_yoy",
                              origins=panel.index[35:50], horizon=1)
    for f in forecasts:
        assert f.has_density
        assert f.has_bands
        assert len(f.samples) > 100


def test_walk_forward_score_returns_crps():
    panel = _bridge_panel(n=80)
    fc = BridgedCBDFForecaster(calib_window=24, train_min=12,
                                  band_method="rolling_conformal")
    forecasts = walk_forward(fc, panel, target_col="bls_yoy",
                              origins=panel.index[40:65], horizon=1)
    df = attach_actuals(forecasts, panel["bls_yoy"])
    block = score(df)
    assert block.crps is not None
    assert block.cov80_density is not None
    assert block.pit_ks_pvalue is not None


def test_skips_when_inner_pred_missing_at_origin():
    """An origin with NaN inner_pred is skipped (walk_forward catches the
    ValueError) rather than emitting a NaN forecast."""
    panel = _bridge_panel(n=60)
    panel.loc[panel.index[40], "cbdf_pred"] = np.nan
    fc = BridgedCBDFForecaster(calib_window=24, train_min=12)
    forecasts = walk_forward(fc, panel, target_col="bls_yoy",
                              origins=[panel.index[40]], horizon=1)
    assert len(forecasts) == 0


def test_gaussian_band_method_works():
    panel = _bridge_panel(n=60)
    fc = BridgedCBDFForecaster(calib_window=24, train_min=12,
                                  band_method="gaussian")
    forecasts = walk_forward(fc, panel, target_col="bls_yoy",
                              origins=panel.index[40:50], horizon=1)
    assert all(f.has_bands for f in forecasts)
    assert all(f.metadata["band_source"] == "gaussian" for f in forecasts)


def test_raises_on_missing_columns():
    panel = pd.DataFrame({
        "wrong_target": [1.0] * 30,
        "wrong_pred": [1.0] * 30,
    }, index=pd.date_range("2020-01-01", periods=30, freq="MS"))
    fc = BridgedCBDFForecaster()
    with pytest.raises(ValueError, match="target_bls_col"):
        fc.fit_predict(panel, panel.index[20], panel.index[21])
