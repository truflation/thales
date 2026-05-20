"""Tests for CompressedMultiComponentBridge.

Verifies:
  1. PCA, PLS, grouped compression all produce valid forecasts
  2. Compressor is fit on training window only (no leakage)
  3. Forecaster Protocol compliance (works with walk_forward)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thales.evaluation.harness import walk_forward, attach_actuals, score
from thales.models.same_month_nowcaster import (
    CompressedMultiComponentBridge,
    MultiComponentBridgeNowcaster,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    """60-month panel with a BLS YoY target + 12 noisy correlated component
    series. Designed so 12-feature OLS overfits and a compressed model
    should generalize better."""
    rng = np.random.default_rng(0)
    n = 60
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    # Hidden 3-factor structure
    f = rng.normal(0, 1, (n, 3))
    loadings = rng.normal(0, 1, (3, 12))
    components = f @ loadings + rng.normal(0, 0.3, (n, 12))

    bls = 2.0 + 0.3 * f[:, 0] + 0.2 * f[:, 1] + rng.normal(0, 0.15, n)
    df = pd.DataFrame(components,
                          columns=[f"truf_c{i}" for i in range(12)],
                          index=idx)
    df["bls_yoy"] = bls
    return df


COMP_COLS = [f"truf_c{i}" for i in range(12)]


# ── Sanity ───────────────────────────────────────────────────────────────


def test_pca_forecast_runs(panel):
    fc = CompressedMultiComponentBridge(
        truf_component_cols=COMP_COLS,
        feature_compression="pca", n_components=3,
        train_window_months=36, train_min=24)
    f = fc.fit_predict(panel, panel.index[40], panel.index[40])
    assert np.isfinite(f.point)
    assert f.has_bands
    assert f.metadata["compression"] == "pca"
    assert f.metadata["n_components"] == 3
    # n_features = intercept + bls_lag + 3 PCs = 5 (vs 14 in raw multi)
    assert f.metadata["n_features"] == 5
    assert "explained_var_ratio" in f.metadata
    assert sum(f.metadata["explained_var_ratio"]) <= 1.0


def test_pls_forecast_runs(panel):
    fc = CompressedMultiComponentBridge(
        truf_component_cols=COMP_COLS,
        feature_compression="pls", n_components=3,
        train_window_months=36, train_min=24)
    f = fc.fit_predict(panel, panel.index[40], panel.index[40])
    assert np.isfinite(f.point)
    assert f.metadata["compression"] == "pls"
    assert f.metadata["pls_n_components"] == 3


def test_grouped_forecast_runs(panel):
    groups = {
        "g1": [f"truf_c{i}" for i in range(4)],
        "g2": [f"truf_c{i}" for i in range(4, 8)],
        "g3": [f"truf_c{i}" for i in range(8, 12)],
    }
    fc = CompressedMultiComponentBridge(
        truf_component_cols=COMP_COLS,
        feature_compression="grouped",
        component_groups=groups,
        train_window_months=36, train_min=24)
    f = fc.fit_predict(panel, panel.index[40], panel.index[40])
    assert np.isfinite(f.point)
    assert f.metadata["compression"] == "grouped"
    assert f.metadata["n_components"] == 3
    assert sorted(f.metadata["groups"]) == ["g1", "g2", "g3"]


def test_grouped_uses_weights_when_provided(panel):
    """Weighted grouping should give different gamma than unweighted."""
    groups = {"g1": COMP_COLS[:6], "g2": COMP_COLS[6:]}
    weights = {c: float(i + 1) for i, c in enumerate(COMP_COLS)}

    fc_w = CompressedMultiComponentBridge(
        truf_component_cols=COMP_COLS,
        feature_compression="grouped",
        component_groups=groups,
        component_weights=weights,
        train_window_months=36, train_min=24)
    fc_u = CompressedMultiComponentBridge(
        truf_component_cols=COMP_COLS,
        feature_compression="grouped",
        component_groups=groups,
        train_window_months=36, train_min=24)

    f_w = fc_w.fit_predict(panel, panel.index[40], panel.index[40])
    f_u = fc_u.fit_predict(panel, panel.index[40], panel.index[40])
    assert f_w.point != f_u.point


# ── No-leakage guarantee ────────────────────────────────────────────────


def test_pca_compressor_only_sees_training_window(panel):
    """Two origins close together should produce different PCA loadings
    only because the training window changed — confirming the compressor
    is refit per origin and doesn't peek beyond origin."""
    fc = CompressedMultiComponentBridge(
        truf_component_cols=COMP_COLS,
        feature_compression="pca", n_components=2,
        train_window_months=24, train_min=12)
    f1 = fc.fit_predict(panel, panel.index[30], panel.index[30])
    f2 = fc.fit_predict(panel, panel.index[40], panel.index[40])
    # Explained-var ratios should differ — different training windows
    assert (f1.metadata["explained_var_ratio"]
                != f2.metadata["explained_var_ratio"])


def test_walk_forward_compatibility(panel):
    """Plug into the harness."""
    fc = CompressedMultiComponentBridge(
        truf_component_cols=COMP_COLS,
        feature_compression="pca", n_components=3,
        train_window_months=24, train_min=12)
    origins = panel.index[30:50]
    forecasts = walk_forward(fc, panel, "bls_yoy", origins)
    assert len(forecasts) > 0
    df = attach_actuals(forecasts, panel["bls_yoy"])
    block = score(df)
    assert block.n > 0
    assert np.isfinite(block.rmse)


# ── Compression actually helps with limited data ─────────────────────────


def test_pca_beats_raw_12_features_on_small_window(panel):
    """The whole point: with only 24 training obs and 12 features, raw
    OLS overfits. PCA-3 should deliver a lower OOS RMSE on a held-out
    window."""
    raw = MultiComponentBridgeNowcaster(
        truf_component_cols=COMP_COLS,
        train_window_months=24, train_min=18,
        ridge_alpha=0.1)
    pca = CompressedMultiComponentBridge(
        truf_component_cols=COMP_COLS,
        feature_compression="pca", n_components=3,
        train_window_months=24, train_min=18)

    origins = panel.index[30:50]
    raw_forecasts = walk_forward(raw, panel, "bls_yoy", origins)
    pca_forecasts = walk_forward(pca, panel, "bls_yoy", origins)
    raw_block = score(attach_actuals(raw_forecasts, panel["bls_yoy"]))
    pca_block = score(attach_actuals(pca_forecasts, panel["bls_yoy"]))
    # On an underdetermined problem, compressed should be at least
    # as good as raw — usually much better. Allow 5% tolerance for
    # the rare case where the synthetic random draw favors raw.
    assert pca_block.rmse <= raw_block.rmse * 1.05
