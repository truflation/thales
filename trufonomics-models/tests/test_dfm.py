"""Stock-Watson DFM tests for the O'Keeffe head-to-head."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thales.evaluation.harness import attach_actuals, score, walk_forward
from thales.models.dfm import StockWatsonDFMForecaster, fit_dfm


def _simulate_factor_panel(T: int = 200, k: int = 12,
                              phi_f: float = 0.7, beta_z: float = 1.5,
                              seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic single-factor panel + target series."""
    rng = np.random.default_rng(seed)
    f = np.zeros(T)
    eta = rng.normal(0, 1, T)
    f[0] = eta[0]
    for t in range(1, T):
        f[t] = phi_f * f[t - 1] + eta[t]
    Λ = rng.uniform(0.5, 1.5, k)
    Y = (np.outer(f, Λ)
            + rng.normal(0, 0.5, (T, k)))   # idiosyncratic
    z = beta_z * f + rng.normal(0, 0.3, T)
    return Y, z


def test_fit_dfm_recovers_factor_ar_coefficient():
    Y, z = _simulate_factor_panel(T=300, phi_f=0.7, seed=1)
    fit = fit_dfm(Y, z)
    # PCA-extracted factor inherits idiosyncratic noise, so the recovered
    # φ is attenuated vs the true 0.7. Recovery within 0.2 is good for
    # idiosyncratic noise SD = 0.5 vs factor signal SD = 1.
    assert fit.phi_f == pytest.approx(0.7, abs=0.2)
    assert fit.phi_f > 0    # sign should be preserved


def test_fit_dfm_recovers_target_loading():
    Y, z = _simulate_factor_panel(T=400, phi_f=0.5, beta_z=1.5, seed=2)
    fit = fit_dfm(Y, z)
    # |β_z| should be ≈ 1.5 (sign depends on factor sign convention)
    assert abs(fit.beta_z) == pytest.approx(1.5, abs=0.3)


def test_fit_dfm_factor_variance_is_one():
    """Factor scaled to unit SD by construction."""
    Y, z = _simulate_factor_panel()
    fit = fit_dfm(Y, z)
    assert fit.factor.std(ddof=1) == pytest.approx(1.0, abs=1e-9)


def test_dfm_walk_forward_compat():
    Y, z = _simulate_factor_panel(T=300, seed=4)
    idx = pd.date_range("2018-01-31", periods=300, freq="ME")
    panel = pd.DataFrame(Y, columns=[f"c{i}" for i in range(12)], index=idx)
    panel["headline"] = z
    fc = StockWatsonDFMForecaster(
        component_cols=[f"c{i}" for i in range(12)],
        target_col="headline",
        horizon=1, train_min=60)
    origins = panel.index[60: -1]
    forecasts = walk_forward(fc, panel, "headline", origins, horizon=1)
    df = attach_actuals(forecasts, panel["headline"])
    block = score(df)
    assert block.n > 0
    assert np.isfinite(block.rmse)


def test_dfm_beats_persistence_on_factor_dgp():
    """On a single-factor DGP with strong loading, DFM should beat
    persistence by a meaningful margin."""
    Y, z = _simulate_factor_panel(T=300, phi_f=0.5, beta_z=2.0, seed=5)
    idx = pd.date_range("2018-01-31", periods=300, freq="ME")
    panel = pd.DataFrame(Y, columns=[f"c{i}" for i in range(12)], index=idx)
    panel["headline"] = z

    fc = StockWatsonDFMForecaster(
        component_cols=[f"c{i}" for i in range(12)],
        target_col="headline", horizon=1, train_min=60)
    origins = panel.index[60: -1]
    forecasts = walk_forward(fc, panel, "headline", origins, horizon=1)
    df = attach_actuals(forecasts, panel["headline"])
    block = score(df)
    assert block.rmse_naive is not None
    # DFM should reduce RMSE by ≥ 10% on this clean DGP. AR(1) with
    # φ=0.5 gives a strong persistence baseline, so the DFM has to do
    # real work to beat it.
    assert block.rmse_reduction_pct > 10.0


def test_dfm_band_widens_with_horizon():
    Y, z = _simulate_factor_panel(T=300, phi_f=0.7, seed=6)
    idx = pd.date_range("2018-01-31", periods=300, freq="ME")
    panel = pd.DataFrame(Y, columns=[f"c{i}" for i in range(12)], index=idx)
    panel["headline"] = z
    fc1 = StockWatsonDFMForecaster(
        component_cols=[f"c{i}" for i in range(12)],
        target_col="headline", horizon=1, train_min=60)
    fc6 = StockWatsonDFMForecaster(
        component_cols=[f"c{i}" for i in range(12)],
        target_col="headline", horizon=6, train_min=60)
    f1 = fc1.fit_predict(panel, panel.index[150], panel.index[151])
    f6 = fc6.fit_predict(panel, panel.index[150], panel.index[156])
    assert (f6.hi80 - f6.lo80) >= (f1.hi80 - f1.lo80)
