"""Tests for MoMComposedForecaster — Fix #5."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thales.evaluation.harness import Forecast, attach_actuals, score, walk_forward
from thales.models.mom_composed import (
    MoMComposedForecaster,
    compose_yoy_one_step,
    mom_from_level,
    yoy_from_level,
)


# ── Identity proofs ──────────────────────────────────────────────────────


def test_yoy_equals_sum_of_12_moms():
    """Algebraic identity: log YoY[t] = Σ_{k=t-11..t} log MoM[k]."""
    rng = np.random.default_rng(0)
    n = 36
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    level = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.005, n))) * 250.0,
                          index=idx)
    mom = mom_from_level(level, log=True)
    yoy = yoy_from_level(level, log=True)
    # yoy is defined from index[12] onward; verify the identity for
    # every available yoy date.
    for date_t in yoy.index:
        # Sum mom values from idx[T-11] through idx[T] inclusive.
        pos_t = idx.get_loc(date_t)
        window = idx[pos_t - 11: pos_t + 1]
        expected = mom.loc[window].sum()
        assert yoy.loc[date_t] == pytest.approx(expected, abs=1e-9)


def test_compose_yoy_one_step_matches_realized():
    """Synthetic level series → forecast mom perfectly → composed YoY
    must match realized YoY exactly."""
    rng = np.random.default_rng(1)
    n = 36
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    level = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.005, n))) * 250.0,
                          index=idx)
    mom = mom_from_level(level, log=True)
    yoy = yoy_from_level(level, log=True)
    # Pick origin such that yoy[origin] and yoy[origin+1] both exist.
    origin_label = idx[18]
    target_label = idx[19]
    mom_T_minus_11_label = idx[18 - 11]    # 11 months before origin
    # Cheat: forecast mom[T+1] perfectly
    mom_pred = float(mom.loc[target_label])
    mom_T_minus_11 = float(mom.loc[mom_T_minus_11_label])
    yoy_T = float(yoy.loc[origin_label])
    composed = compose_yoy_one_step(yoy_T, mom_pred, mom_T_minus_11)
    realized = float(yoy.loc[target_label])
    assert composed == pytest.approx(realized, abs=1e-9)


# ── Wrapper integration ─────────────────────────────────────────────────


class _ConstantMomForecaster:
    """Forecaster that always predicts mom = 0 (no MoM change)."""
    model_id = "const_mom"

    def fit_predict(self, panel, origin, target):
        return Forecast(origin=origin, target=target, point=0.0,
                          lo80=-1.0, hi80=1.0, lo95=-2.0, hi95=2.0)


@pytest.fixture
def cpi_panel() -> pd.DataFrame:
    """Synthetic CPI-like level + YoY for end-to-end tests."""
    rng = np.random.default_rng(7)
    n = 60
    idx = pd.date_range("2018-01-31", periods=n, freq="ME")
    level = np.exp(np.cumsum(rng.normal(0.002, 0.005, n))) * 250.0
    df = pd.DataFrame({"bls_level": level}, index=idx)
    df["bls_yoy"] = yoy_from_level(df["bls_level"], log=True)
    return df


def test_wrapper_runs_end_to_end(cpi_panel):
    fc = MoMComposedForecaster(inner=_ConstantMomForecaster(),
                                       bls_level_col="bls_level",
                                       bls_yoy_col="bls_yoy")
    origin_pos = 30
    f = fc.fit_predict(cpi_panel, cpi_panel.index[origin_pos],
                              cpi_panel.index[origin_pos + 1])
    assert np.isfinite(f.point)
    assert f.has_bands
    assert f.metadata["inner_model"] == "const_mom"
    # With mom_pred=0, composed yoy should equal yoy[T] - mom[T-11]
    expected = (f.metadata["yoy_T"] - f.metadata["mom_T_minus_11"])
    assert f.point == pytest.approx(expected, abs=1e-9)


def test_band_translation_preserves_widths(cpi_panel):
    """Bands in MoM space should translate 1:1 to YoY space (linear shift)."""
    fc = MoMComposedForecaster(inner=_ConstantMomForecaster(),
                                       bls_level_col="bls_level",
                                       bls_yoy_col="bls_yoy")
    f = fc.fit_predict(cpi_panel, cpi_panel.index[30],
                              cpi_panel.index[31])
    # Inner returns: point=0.0, lo80=-1.0, hi80=+1.0  → width80 = 2.0
    assert f.hi80 - f.lo80 == pytest.approx(2.0, abs=1e-9)
    assert f.hi95 - f.lo95 == pytest.approx(4.0, abs=1e-9)


def test_walk_forward_integration(cpi_panel):
    fc = MoMComposedForecaster(inner=_ConstantMomForecaster(),
                                       bls_level_col="bls_level",
                                       bls_yoy_col="bls_yoy")
    origins = cpi_panel.index[20:50]
    forecasts = walk_forward(fc, cpi_panel, "bls_yoy", origins, horizon=1)
    assert len(forecasts) > 0
    df = attach_actuals(forecasts, cpi_panel["bls_yoy"])
    block = score(df)
    assert np.isfinite(block.rmse)


def test_perfect_inner_recovers_actual_yoy_exactly(cpi_panel):
    """If the inner predicts mom[T+1] perfectly, the composed YoY must
    equal the realized YoY exactly. The point of the closed-form
    identity."""
    realized_mom = mom_from_level(cpi_panel["bls_level"], log=True)

    class OracleMom:
        model_id = "oracle"
        def fit_predict(self, panel, origin, target):
            return Forecast(origin=origin, target=target,
                                point=float(realized_mom.loc[target]),
                                lo80=0.0, hi80=0.0, lo95=0.0, hi95=0.0)

    fc = MoMComposedForecaster(inner=OracleMom(),
                                       bls_level_col="bls_level",
                                       bls_yoy_col="bls_yoy")
    origin_pos = 30
    target = cpi_panel.index[origin_pos + 1]
    f = fc.fit_predict(cpi_panel, cpi_panel.index[origin_pos], target)
    realized_yoy = float(cpi_panel.loc[target, "bls_yoy"])
    assert f.point == pytest.approx(realized_yoy, abs=1e-6)


def test_horizon_gt_1_requires_alpha_phi_metadata(cpi_panel):
    """An inner forecaster that doesn't expose AR(1) coefficients must
    fail loudly at h>1 — we can't iterate the chain without them."""
    fc = MoMComposedForecaster(inner=_ConstantMomForecaster(),
                                       bls_level_col="bls_level",
                                       bls_yoy_col="bls_yoy",
                                       horizon=3)
    with pytest.raises(NotImplementedError, match="alpha"):
        fc.fit_predict(cpi_panel, cpi_panel.index[30],
                            cpi_panel.index[33])


# ── Multi-horizon (h>1) extension ─────────────────────────────────────────


def test_compose_yoy_multi_step_matches_realized():
    """Synthetic level series → forecast every mom in the chain perfectly
    → composed YoY[T+h] must match realized YoY[T+h] exactly."""
    from thales.models.mom_composed import compose_yoy_multi_step
    rng = np.random.default_rng(11)
    n = 48
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    level = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.005, n))) * 250.0,
                          index=idx)
    mom = mom_from_level(level, log=True)
    yoy = yoy_from_level(level, log=True)
    origin_pos = 25
    h = 6
    yoy_T = float(yoy.iloc[origin_pos - 12])  # yoy is offset because of dropna
    # Use level-indexed positions consistently — find origin in mom/yoy index.
    origin_label = idx[origin_pos]
    target_label = idx[origin_pos + h]
    yoy_T = float(yoy.loc[origin_label])
    chain = [float(mom.loc[idx[origin_pos + k]]) for k in range(1, h + 1)]
    dropped = [float(mom.loc[idx[origin_pos - 11 + k]]) for k in range(h)]
    composed = compose_yoy_multi_step(yoy_T, chain, dropped)
    realized = float(yoy.loc[target_label])
    assert composed == pytest.approx(realized, abs=1e-9)


def test_compose_yoy_multi_step_validates_lengths():
    from thales.models.mom_composed import compose_yoy_multi_step
    with pytest.raises(ValueError, match="same length"):
        compose_yoy_multi_step(2.0, [0.1, 0.2, 0.3], [0.05, 0.05])


def test_horizon_3_returns_finite_with_ar1_inner(cpi_panel):
    """With AR1Baseline as the inner (which exposes alpha/phi in
    metadata), h=3 should produce a finite forecast and bands."""
    from thales.models.baselines import AR1Baseline

    # Add MoM column to panel
    panel = cpi_panel.copy()
    panel["bls_mom"] = mom_from_level(panel["bls_level"], log=True)

    fc = MoMComposedForecaster(
        inner=AR1Baseline(target_col="bls_mom",
                                  train_min=24, calib_months=18,
                                  band_method="rolling_conformal",
                                  model_id="ar1_mom_inner"),
        bls_level_col="bls_level",
        bls_yoy_col="bls_yoy",
        mom_col="bls_mom",
        log_mom=True,
        horizon=3,
        n_samples=200,
    )
    f = fc.fit_predict(panel, panel.index[50], panel.index[53])
    assert np.isfinite(f.point)
    assert f.has_bands
    assert f.has_density
    assert len(f.samples) == 200
    # Multi-step metadata exposes the deterministic chain + dropped MoMs
    assert "mom_chain_deterministic" in f.metadata
    assert len(f.metadata["mom_chain_deterministic"]) == 3
    assert len(f.metadata["mom_dropped"]) == 3


def test_horizon_6_walk_forward_runs(cpi_panel):
    """End-to-end: walk-forward at h=6 with samples emission produces a
    valid scoring frame."""
    from thales.models.baselines import AR1Baseline

    panel = cpi_panel.copy()
    panel["bls_mom"] = mom_from_level(panel["bls_level"], log=True)

    fc = MoMComposedForecaster(
        inner=AR1Baseline(target_col="bls_mom",
                                  train_min=24, calib_months=18,
                                  band_method="rolling_conformal",
                                  model_id="ar1_mom_inner"),
        bls_level_col="bls_level",
        bls_yoy_col="bls_yoy",
        mom_col="bls_mom",
        log_mom=True,
        horizon=6,
        n_samples=200,
    )
    forecasts = walk_forward(fc, panel, "bls_yoy",
                              panel.index[44:54], horizon=6)
    assert len(forecasts) > 0
    df = attach_actuals(forecasts, panel["bls_yoy"])
    block = score(df)
    assert np.isfinite(block.rmse)
    assert block.crps is not None


def test_fails_loudly_when_too_little_history(cpi_panel):
    fc = MoMComposedForecaster(inner=_ConstantMomForecaster(),
                                       bls_level_col="bls_level",
                                       bls_yoy_col="bls_yoy")
    # Origin too early — fewer than 12 months → yoy[origin] is NaN
    # (yoy_from_level needs ≥12 priors), and mom[origin-11] is also
    # NaN. Both error paths are valid; the wrapper raises ValueError
    # with one of two messages.
    with pytest.raises(ValueError):
        fc.fit_predict(cpi_panel, cpi_panel.index[5],
                            cpi_panel.index[6])
