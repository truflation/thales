"""Rolling-conformal band tests for the same-month bridge family."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thales.models.same_month_nowcaster import (
    CompressedMultiComponentBridge,
    MultiComponentBridgeNowcaster,
    SameMonthBridgeNowcaster,
)


def _make_panel(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-31", periods=n, freq="ME")
    truf = rng.normal(0, 0.4, n)
    bls = np.empty(n)
    bls[0] = 2.0
    for t in range(1, n):
        bls[t] = 0.85 * bls[t - 1] + 0.10 * truf[t - 1] + rng.normal(0, 0.10)
    components = np.column_stack([rng.normal(0, 1, n) + 0.3 * truf
                                          for _ in range(12)])
    df = pd.DataFrame({"bls_yoy": bls, "truf_yoy": truf}, index=idx)
    for i in range(12):
        df[f"truf_c{i}"] = components[:, i]
    return df


COMP_COLS = [f"truf_c{i}" for i in range(12)]


# ── SameMonthBridgeNowcaster ─────────────────────────────────────────────


def test_same_month_bridge_default_is_gaussian():
    fc = SameMonthBridgeNowcaster()
    assert fc.band_method == "gaussian"


def test_same_month_bridge_rolling_conformal_runs():
    panel = _make_panel()
    fc = SameMonthBridgeNowcaster(
        train_window_months=36, train_min=12,
        band_method="rolling_conformal", calib_months=24)
    f = fc.fit_predict(panel, panel.index[60], panel.index[60])
    assert f.has_bands
    assert f.metadata["band_source"] == "rolling_conformal"
    assert f.metadata["n_calib"] == 24


def test_same_month_bridge_in_sample_conformal_runs():
    panel = _make_panel()
    fc = SameMonthBridgeNowcaster(
        train_window_months=36, train_min=12,
        band_method="in_sample")
    f = fc.fit_predict(panel, panel.index[60], panel.index[60])
    assert f.has_bands
    assert f.metadata["band_source"] == "in_sample_conformal"


def test_rolling_conformal_falls_back_when_calib_too_small():
    """If calib_months < min_n_for_alpha(0.20)=9, the band falls back to
    Gaussian instead of producing under-conservative conformal bands."""
    panel = _make_panel()
    fc = SameMonthBridgeNowcaster(
        train_window_months=36, train_min=12,
        band_method="rolling_conformal", calib_months=4)
    f = fc.fit_predict(panel, panel.index[60], panel.index[60])
    assert f.metadata["band_source"] == "gaussian_fallback_n_too_small"


def test_point_forecast_invariant_under_band_method():
    """Changing band_method must NOT change the point forecast — the
    point model is fit on all training data regardless."""
    panel = _make_panel()
    origin = panel.index[60]
    fcs = {
        "gaussian": SameMonthBridgeNowcaster(band_method="gaussian"),
        "in_sample": SameMonthBridgeNowcaster(band_method="in_sample"),
        "rolling": SameMonthBridgeNowcaster(
            band_method="rolling_conformal", calib_months=24),
    }
    points = {k: v.fit_predict(panel, origin, origin).point
                for k, v in fcs.items()}
    assert points["gaussian"] == pytest.approx(points["in_sample"], rel=1e-12)
    assert points["gaussian"] == pytest.approx(points["rolling"], rel=1e-12)


# ── MultiComponentBridgeNowcaster ────────────────────────────────────────


def test_multi_component_rolling_conformal_runs():
    panel = _make_panel()
    fc = MultiComponentBridgeNowcaster(
        truf_component_cols=COMP_COLS,
        train_window_months=36, train_min=24, ridge_alpha=1.0,
        band_method="rolling_conformal", calib_months=24)
    f = fc.fit_predict(panel, panel.index[60], panel.index[60])
    assert f.has_bands
    assert f.metadata["band_source"] == "rolling_conformal"


# ── CompressedMultiComponentBridge ───────────────────────────────────────


def test_compressed_pca_rolling_conformal_runs():
    panel = _make_panel()
    fc = CompressedMultiComponentBridge(
        truf_component_cols=COMP_COLS,
        feature_compression="pca", n_components=3,
        train_window_months=48, train_min=12,
        band_method="rolling_conformal", calib_months=24)
    f = fc.fit_predict(panel, panel.index[60], panel.index[60])
    assert f.has_bands
    assert f.metadata["band_source"] == "rolling_conformal"
    assert f.metadata["n_calib"] == 24
    assert f.metadata["n_components"] == 3


def test_compressed_pca_default_band_is_gaussian():
    fc = CompressedMultiComponentBridge(
        truf_component_cols=COMP_COLS, feature_compression="pca",
        n_components=3)
    assert fc.band_method == "gaussian"
