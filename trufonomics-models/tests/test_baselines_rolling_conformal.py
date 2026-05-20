"""Rolling-conformal band tests for AR1Baseline and PathAForecaster.

The user-facing contract under rolling-conformal:

  1. Point coefficients are fit on ALL training data available at origin —
     never less than the in-sample baseline. (Split-conformal hurt RMSE
     because it discarded the trailing calib_months — this is what we
     fixed.)
  2. Band widths come from rolling-origin OOS residuals, not in-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thales.models.baselines import AR1Baseline, PathAForecaster


def _make_panel(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="MS")
    y = np.empty(n)
    y[0] = 2.0
    truf = rng.normal(0, 0.2, n)
    for t in range(1, n):
        y[t] = 0.85 * y[t - 1] + 0.10 * truf[t - 1] + rng.normal(0, 0.10)
    return pd.DataFrame({"y": y, "truf_yoy": truf}, index=idx)


# ── AR(1) ────────────────────────────────────────────────────────────────


def test_ar1_rolling_uses_full_training_for_point():
    """Rolling conformal must use ALL data for the point fit; the implied
    α/φ should match an AR(1) fit on the full available history (whereas
    split conformal would fit on history minus the last 24)."""
    panel = _make_panel()
    origin = panel.index[100]

    full = AR1Baseline(target_col="y", calib_months=24,
                         band_method="rolling_conformal")
    split = AR1Baseline(target_col="y", calib_months=24,
                          band_method="split_conformal")

    f_full = full.fit_predict(panel, origin, panel.index[101])
    f_split = split.fit_predict(panel, origin, panel.index[101])

    # Different point predictions: rolling uses 100 obs, split uses 76.
    assert f_full.metadata["n_train"] == 100
    assert f_split.metadata["n_train"] == 76
    assert f_full.metadata["band_source"] == "rolling_conformal"
    assert f_split.metadata["band_source"] == "split_conformal"

    # Independently fit AR(1) on the full window — coefficients must
    # match the rolling-conformal forecaster's stored α/φ.
    s = panel["y"].loc[panel.index <= origin].values
    X = np.column_stack([np.ones(len(s) - 1), s[:-1]])
    coef, *_ = np.linalg.lstsq(X, s[1:], rcond=None)
    assert f_full.metadata["alpha"] == pytest.approx(float(coef[0]))
    assert f_full.metadata["phi"] == pytest.approx(float(coef[1]))


def test_ar1_rolling_residuals_count_matches_calib_window():
    panel = _make_panel()
    origin = panel.index[100]
    fc = AR1Baseline(target_col="y", calib_months=24,
                       band_method="rolling_conformal")
    f = fc.fit_predict(panel, origin, panel.index[101])
    assert f.metadata["n_calib"] == 24


def test_ar1_in_sample_mode_collapses_calib():
    panel = _make_panel()
    origin = panel.index[100]
    fc = AR1Baseline(target_col="y", calib_months=24,
                       band_method="in_sample")
    f = fc.fit_predict(panel, origin, panel.index[101])
    assert f.metadata["band_source"] == "in_sample"
    assert f.metadata["n_calib"] == 0


# ── Path A ───────────────────────────────────────────────────────────────


def test_patha_rolling_uses_full_training_for_point():
    panel = _make_panel()
    origin = panel.index[100]

    full = PathAForecaster(target_col="y", truflation_col="truf_yoy",
                              calib_months=24,
                              band_method="rolling_conformal")
    split = PathAForecaster(target_col="y", truflation_col="truf_yoy",
                               calib_months=24,
                               band_method="split_conformal")

    f_full = full.fit_predict(panel, origin, panel.index[101])
    f_split = split.fit_predict(panel, origin, panel.index[101])

    assert f_full.metadata["n_train"] > f_split.metadata["n_train"]
    assert f_full.metadata["band_source"] == "rolling_conformal"
    assert f_split.metadata["band_source"] == "split_conformal"


def test_patha_rolling_residuals_count_matches_calib_window():
    panel = _make_panel()
    origin = panel.index[100]
    fc = PathAForecaster(target_col="y", truflation_col="truf_yoy",
                            calib_months=24,
                            band_method="rolling_conformal")
    f = fc.fit_predict(panel, origin, panel.index[101])
    assert f.metadata["n_calib"] == 24
    assert f.has_bands


def test_patha_default_is_rolling_conformal():
    """The default band_method should be rolling_conformal — the user's
    explicit production preference."""
    fc = PathAForecaster()
    assert fc.band_method == "rolling_conformal"
    fc_ar1 = AR1Baseline()
    assert fc_ar1.band_method == "rolling_conformal"
