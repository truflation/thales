"""Tests for thales.evaluation.harness — the walk-forward spine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thales.evaluation.harness import (
    Forecast,
    ScoreBlock,
    attach_actuals,
    evaluate,
    score,
    walk_forward,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def panel() -> pd.DataFrame:
    """A 60-day daily panel with a target series 'y' and one driver 'x'."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2026-01-01", periods=60, freq="D")
    # AR(1) target with x as a noisy driver
    y = np.empty(60)
    y[0] = 2.0
    x = rng.normal(0, 0.05, 60)
    for t in range(1, 60):
        y[t] = 0.95 * y[t - 1] + 0.05 * x[t] + rng.normal(0, 0.01)
    return pd.DataFrame({"y": y, "x": x}, index=idx)


class IdentityForecaster:
    """Trivial forecaster: always predicts target = panel.iloc[-1]['y'].
    Bands are ±0.05 / ±0.10 — fixed, calibrated to the AR(1) noise.
    """
    model_id = "identity_v0"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        last = float(panel["y"].iloc[-1])
        return Forecast(
            origin=origin, target=target, point=last,
            lo80=last - 0.05, hi80=last + 0.05,
            lo95=last - 0.10, hi95=last + 0.10,
            metadata={"strategy": "persistence"},
        )


class BiasedForecaster:
    """Always predicts a fixed offset above the last observed value — used
    to test that error metrics react correctly to systematic bias."""
    model_id = "biased_v0"

    def __init__(self, bias: float = 0.5):
        self.bias = bias

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        last = float(panel["y"].iloc[-1])
        return Forecast(
            origin=origin, target=target, point=last + self.bias,
            lo80=last + self.bias - 0.05, hi80=last + self.bias + 0.05,
            lo95=last + self.bias - 0.10, hi95=last + self.bias + 0.10,
        )


# ─── Forecast dataclass ───────────────────────────────────────────────────


def test_forecast_band_flag():
    f = Forecast(origin=pd.Timestamp("2026-01-01"),
                  target=pd.Timestamp("2026-01-02"), point=1.0)
    assert not f.has_bands
    assert not f.has_density

    f2 = Forecast(origin=pd.Timestamp("2026-01-01"),
                   target=pd.Timestamp("2026-01-02"), point=1.0,
                   lo80=0.9, hi80=1.1, lo95=0.8, hi95=1.2)
    assert f2.has_bands
    assert not f2.has_density

    f3 = Forecast(origin=pd.Timestamp("2026-01-01"),
                   target=pd.Timestamp("2026-01-02"), point=1.0,
                   samples=np.array([0.9, 1.0, 1.1]))
    assert f3.has_density


# ─── walk_forward ─────────────────────────────────────────────────────────


def test_walk_forward_no_peeking(panel):
    """The forecaster must only see panel rows up to and including origin."""
    seen_lengths = []

    class SpyForecaster:
        model_id = "spy"

        def fit_predict(self, panel_slice, origin, target):
            seen_lengths.append(len(panel_slice))
            assert panel_slice.index[-1] == origin, \
                "last row of slice must equal origin"
            return Forecast(origin=origin, target=target, point=0.0)

    origins = panel.index[10:15]  # 5 origins
    walk_forward(SpyForecaster(), panel, "y", origins)
    # First origin = panel.index[10] → slice has 11 rows (index 0..10)
    assert seen_lengths == [11, 12, 13, 14, 15]


def test_walk_forward_skips_targets_past_panel_end(panel):
    """Origins whose target falls past the panel end are skipped silently."""
    forecaster = IdentityForecaster()
    # Origin = last index → target would be one step past end → skip
    last = panel.index[-1]
    forecasts = walk_forward(forecaster, panel, "y", [last], horizon=1)
    assert forecasts == []


def test_walk_forward_handles_horizon(panel):
    forecaster = IdentityForecaster()
    origins = panel.index[10:15]
    forecasts = walk_forward(forecaster, panel, "y", origins, horizon=3)
    for fc, origin in zip(forecasts, origins):
        target_pos = panel.index.get_loc(origin) + 3
        assert fc.target == panel.index[target_pos]


def test_walk_forward_continues_past_failure(panel):
    """A forecaster exception on one origin should not abort the whole run."""
    call_count = {"n": 0}

    class FlakyForecaster:
        model_id = "flaky"

        def fit_predict(self, panel_slice, origin, target):
            call_count["n"] += 1
            if call_count["n"] == 3:
                raise RuntimeError("simulated failure")
            return Forecast(origin=origin, target=target,
                              point=float(panel_slice["y"].iloc[-1]))

    origins = panel.index[10:15]
    forecasts = walk_forward(FlakyForecaster(), panel, "y", origins)
    assert len(forecasts) == 4  # 5 origins, one failed


# ─── attach_actuals ───────────────────────────────────────────────────────


def test_attach_actuals_basic(panel):
    forecasts = walk_forward(IdentityForecaster(), panel, "y",
                              panel.index[10:15])
    df = attach_actuals(forecasts, panel["y"])
    assert len(df) == 5
    assert {"point", "actual", "error", "abs_error", "today",
             "naive_error", "hit_80", "hit_95",
             "pred_up", "actual_up", "direction_hit"}.issubset(df.columns)
    # Identity forecaster ⇒ point == today
    np.testing.assert_array_almost_equal(df["point"].values,
                                            df["today"].values)


def test_attach_actuals_uses_external_today_baseline(panel):
    """When a separate today series is passed, it should be used in place
    of actuals[origin]."""
    forecasts = walk_forward(IdentityForecaster(), panel, "y",
                              panel.index[10:15])
    fake_today = pd.Series(np.zeros(60), index=panel.index, name="today")
    df = attach_actuals(forecasts, panel["y"], today_baseline=fake_today)
    assert (df["today"] == 0.0).all()


def test_attach_actuals_drops_missing_actuals(panel):
    forecasts = walk_forward(IdentityForecaster(), panel, "y",
                              panel.index[10:15])
    actuals = panel["y"].copy()
    actuals.loc[panel.index[12]] = np.nan
    df = attach_actuals(forecasts, actuals)
    assert len(df) == 4


# ─── score ────────────────────────────────────────────────────────────────


def test_score_identity_forecaster_is_perfect_on_naive(panel):
    """Identity forecaster IS the naive baseline → naive RMSE == model RMSE
    → reduction 0%."""
    df, block = evaluate(IdentityForecaster(), panel, "y",
                          panel.index[10:55])
    assert isinstance(block, ScoreBlock)
    assert block.n == 45
    assert block.rmse == pytest.approx(block.rmse_naive, abs=1e-9)
    assert block.rmse_reduction_pct == pytest.approx(0.0, abs=1e-9)


def test_score_biased_forecaster_loses_to_naive(panel):
    df, block = evaluate(BiasedForecaster(bias=0.5), panel, "y",
                          panel.index[10:55])
    assert block.rmse > block.rmse_naive
    assert block.rmse_reduction_pct < 0  # worse than naive
    # 80% bands of width 0.10 around point=last+0.5 won't cover an actual
    # close to last → coverage near zero
    assert block.cov80 < 0.5


def test_score_directional_accuracy_unbiased(panel):
    """An identity forecaster predicts no change ⇒ pred_up always False
    ⇒ direction matches only when actual_up is also False (i.e., y went
    down or stayed flat)."""
    df, block = evaluate(IdentityForecaster(), panel, "y",
                          panel.index[10:55])
    expected = float((~df["actual_up"]).mean())
    assert block.dir_hit == pytest.approx(expected)


def test_score_empty_returns_insufficient():
    block = score(pd.DataFrame())
    assert block.n == 0
    assert block.ship_verdict == "INSUFFICIENT-DATA"


def test_score_ship_gate_on_calibrated_bands():
    """A frame where 80% coverage is in spec and RMSE roughly matches naive
    should yield SHIP."""
    rng = np.random.default_rng(0)
    n = 200
    actual = rng.normal(0, 1, n)
    point = actual + rng.normal(0, 0.1, n)  # tiny model error
    today = actual + rng.normal(0, 1.0, n)  # pretend the naive is comparable
    err = point - actual
    df = pd.DataFrame({
        "origin": pd.date_range("2026-01-01", periods=n, freq="D"),
        "target": pd.date_range("2026-01-02", periods=n, freq="D"),
        "point": point, "actual": actual, "today": today,
        "lo80": point + np.percentile(err, 10),
        "hi80": point + np.percentile(err, 90),
        "lo95": point + np.percentile(err, 2.5),
        "hi95": point + np.percentile(err, 97.5),
        "error": err, "abs_error": np.abs(err),
        "naive_error": today - actual,
        "hit_80": ((point + np.percentile(err, 10) <= actual) &
                    (actual <= point + np.percentile(err, 90))),
        "hit_95": ((point + np.percentile(err, 2.5) <= actual) &
                    (actual <= point + np.percentile(err, 97.5))),
        "width_80": np.percentile(err, 90) - np.percentile(err, 10),
        "width_95": np.percentile(err, 97.5) - np.percentile(err, 2.5),
        "pred_up": point > today,
        "actual_up": actual > today,
        "direction_hit": (point > today) == (actual > today),
    })
    block = score(df)
    assert block.ship_verdict == "SHIP"


def test_summary_string_renders():
    block = ScoreBlock(
        n=10, window_start="2026-01-01", window_end="2026-01-10",
        rmse=0.1, mae=0.08, rmse_naive=0.12, rmse_reduction_pct=16.7,
        mse_reduction_pct=30.6,
        cov80=0.80, cov95=0.95, width80=0.2, width95=0.4,
        dir_hit=0.6, base_rate_up=0.5, crps=None, ship_verdict="SHIP",
    )
    s = block.summary()
    assert "n=10" in s
    assert "RMSE: 0.1000" in s
    assert "Verdict: SHIP" in s
