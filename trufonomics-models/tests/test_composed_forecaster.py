"""Tests for ComposedForecaster — integration of per-component
sub-forecasters with the CBDF composer through the harness Forecaster
Protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thales.evaluation.harness import (
    Forecast, attach_actuals, score, walk_forward,
)
from thales.models.baselines import PersistenceBaseline
from thales.models.composition.composed_forecaster import ComposedForecaster
from thales.models.composition.cbdf import CBDFComposer
from thales.models.composition.weighted import WeightedComposer


def _make_panel(T: int = 80, R: int = 3, seed: int = 0) -> pd.DataFrame:
    """Make a panel with R component series + a 'headline' = weighted sum."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-31", periods=T, freq="ME")
    weights = np.array([0.5, 0.3, 0.2])
    components = {}
    for r in range(R):
        # AR(1) plus trend
        x = np.empty(T)
        x[0] = 2.0
        for t in range(1, T):
            x[t] = 0.95 * x[t - 1] + rng.normal(0, 0.1)
        components[f"c{r}"] = x
    headline = sum(weights[r] * components[f"c{r}"] for r in range(R))
    df = pd.DataFrame(components, index=idx)
    df["headline"] = headline
    return df


# ─── Basic API ────────────────────────────────────────────────────────────


def test_composed_forecaster_implements_protocol():
    panel = _make_panel(50)
    weights = {"c0": 0.5, "c1": 0.3, "c2": 0.2}
    composer = WeightedComposer(weights=weights, n_mc_samples=200)
    components = {k: PersistenceBaseline(target_col=k) for k in weights}
    composed = ComposedForecaster(components=components, composer=composer)
    assert composed.model_id == "composed_v1"
    # Has fit_predict
    origin = panel.index[30]
    target = panel.index[31]
    fc = composed.fit_predict(panel.loc[: origin], origin, target)
    assert isinstance(fc, Forecast)


def test_composed_forecaster_through_harness():
    """ComposedForecaster should drive walk_forward → attach_actuals → score
    end-to-end without errors."""
    panel = _make_panel(80)
    weights = {"c0": 0.5, "c1": 0.3, "c2": 0.2}
    composer = WeightedComposer(weights=weights, n_mc_samples=200)
    components = {k: PersistenceBaseline(target_col=k) for k in weights}
    composed = ComposedForecaster(components=components, composer=composer)

    origins = panel.index[30:75]
    forecasts = walk_forward(composed, panel, "headline", origins, horizon=1)
    df = attach_actuals(forecasts, panel["headline"])
    block = score(df)
    assert block.n > 30
    assert block.rmse > 0
    # Composed persistence should approximately match direct persistence on
    # weighted sum (by accounting identity)
    assert block.rmse < 1.0   # sanity: not blowing up


def test_composed_persistence_matches_direct_persistence():
    """For persistence forecasters, composed should equal direct headline
    persistence (identity preserved by construction)."""
    panel = _make_panel(60)
    weights = {"c0": 0.5, "c1": 0.3, "c2": 0.2}
    composer = WeightedComposer(weights=weights, n_mc_samples=100)
    components = {k: PersistenceBaseline(target_col=k) for k in weights}
    composed = ComposedForecaster(components=components, composer=composer)

    origins = panel.index[25:55]
    forecasts = walk_forward(composed, panel, "headline", origins, horizon=1)

    # Direct: persistence on the headline directly
    direct = PersistenceBaseline(target_col="headline")
    direct_forecasts = walk_forward(direct, panel, "headline", origins,
                                          horizon=1)

    # Their points should match within numerical tolerance
    for c, d in zip(forecasts, direct_forecasts):
        assert abs(c.point - d.point) < 1e-9, (
            f"composed {c.point:.6f} vs direct {d.point:.6f}")


def test_composed_with_cbdf_composer():
    """CBDFComposer also works as the composer."""
    panel = _make_panel(60)
    weights = {"c0": 0.5, "c1": 0.3, "c2": 0.2}
    composer = CBDFComposer(weights=weights, n_mc_samples=200)
    # No covariance fitted → falls back to independent draws
    components = {k: PersistenceBaseline(target_col=k) for k in weights}
    composed = ComposedForecaster(components=components, composer=composer,
                                       model_id="cbdf_v1")
    origin = panel.index[30]
    target = panel.index[31]
    fc = composed.fit_predict(panel.loc[: origin], origin, target)
    assert fc.metadata["composer"] in {"weighted", "cbdf_multivariate_gaussian"}


def test_metadata_preserves_per_component_contributions():
    panel = _make_panel(50)
    weights = {"c0": 0.5, "c1": 0.5}
    composer = WeightedComposer(weights=weights, n_mc_samples=100)
    components = {k: PersistenceBaseline(target_col=k) for k in weights}
    composed = ComposedForecaster(components=components, composer=composer)
    origin = panel.index[30]
    target = panel.index[31]
    fc = composed.fit_predict(panel.loc[: origin], origin, target)
    assert "contributions" in fc.metadata
    assert set(fc.metadata["contributions"].keys()) == {"c0", "c1"}
