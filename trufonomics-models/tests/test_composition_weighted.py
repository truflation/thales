"""Tests for the Phase 2.1 weighted composition layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thales.evaluation.harness import Forecast
from thales.models.composition.weighted import WeightedComposer


def _fc(point: float, sigma: float | None = None,
         samples: np.ndarray | None = None) -> Forecast:
    """Build a synthetic component Forecast with optional bands/samples."""
    f = Forecast(
        origin=pd.Timestamp("2026-01-31"),
        target=pd.Timestamp("2026-02-28"),
        point=point,
        samples=samples,
    )
    if sigma is not None:
        # 80% band: ±1.2816σ
        f.lo80 = point - 1.2816 * sigma
        f.hi80 = point + 1.2816 * sigma
        f.lo95 = point - 1.96 * sigma
        f.hi95 = point + 1.96 * sigma
    return f


# ─── Construction validation ──────────────────────────────────────────────


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="must be within"):
        WeightedComposer(weights={"a": 0.5, "b": 0.4})  # sums to 0.9


def test_weights_at_one_within_tolerance():
    # Tiny rounding
    WeightedComposer(weights={"a": 0.500001, "b": 0.499999})


# ─── Point composition ────────────────────────────────────────────────────


def test_point_is_weighted_sum():
    comp = WeightedComposer(weights={"a": 0.6, "b": 0.4})
    fcs = {"a": _fc(2.0), "b": _fc(3.0)}
    out = comp.compose(fcs,
                          origin=pd.Timestamp("2026-01-31"),
                          target=pd.Timestamp("2026-02-28"))
    assert out.point == pytest.approx(0.6 * 2.0 + 0.4 * 3.0)


def test_zero_weight_components_dropped():
    """Components with zero weight don't need to be passed in."""
    comp = WeightedComposer(weights={"a": 1.0, "b": 0.0})
    fcs = {"a": _fc(2.5)}
    out = comp.compose(fcs, pd.Timestamp("2026-01-31"),
                          pd.Timestamp("2026-02-28"))
    assert out.point == pytest.approx(2.5)


def test_missing_active_component_raises():
    comp = WeightedComposer(weights={"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError, match="missing"):
        comp.compose({"a": _fc(2.0)},
                       pd.Timestamp("2026-01-31"),
                       pd.Timestamp("2026-02-28"))


# ─── Density composition ──────────────────────────────────────────────────


def test_bands_from_gaussian_components():
    """Two independent Gaussian components should compose to a Gaussian
    with weighted mean and √(Σ w² σ²) SD."""
    sigma_a, sigma_b = 0.4, 0.3
    w_a, w_b = 0.6, 0.4
    comp = WeightedComposer(weights={"a": w_a, "b": w_b}, n_mc_samples=20000)
    fcs = {"a": _fc(2.0, sigma=sigma_a), "b": _fc(3.0, sigma=sigma_b)}
    out = comp.compose(fcs, pd.Timestamp("2026-01-31"),
                          pd.Timestamp("2026-02-28"))

    expected_mean = w_a * 2.0 + w_b * 3.0
    expected_sd = np.sqrt((w_a * sigma_a) ** 2 + (w_b * sigma_b) ** 2)

    # 80% band ≈ ±1.2816 σ; 95% ≈ ±1.96 σ
    band_sd_80 = (out.hi80 - out.lo80) / (2.0 * 1.2816)
    assert abs(out.point - expected_mean) < 0.02
    assert abs(band_sd_80 - expected_sd) / expected_sd < 0.10


def test_bands_use_explicit_samples_if_provided():
    """If a component supplies explicit samples (e.g. MCMC posterior), the
    composer uses those rather than reconstructing from the band."""
    rng = np.random.default_rng(0)
    skewed = rng.gamma(2.0, 1.0, size=5000) - 2.0   # right-skewed
    # Single-component composition trivially returns the same shape
    comp = WeightedComposer(weights={"a": 1.0}, n_mc_samples=5000)
    out = comp.compose({"a": _fc(skewed.mean(), samples=skewed)},
                          pd.Timestamp("2026-01-31"),
                          pd.Timestamp("2026-02-28"))
    # Band asymmetry preserved
    upper_tail = out.hi95 - out.point
    lower_tail = out.point - out.lo95
    assert upper_tail > lower_tail   # right-skew survives


def test_samples_returned_for_downstream_use():
    """Composed Forecast should expose its MC sample for further use."""
    comp = WeightedComposer(weights={"a": 0.5, "b": 0.5})
    fcs = {"a": _fc(2.0, sigma=0.3), "b": _fc(3.0, sigma=0.3)}
    out = comp.compose(fcs, pd.Timestamp("2026-01-31"),
                          pd.Timestamp("2026-02-28"))
    assert out.samples is not None
    assert len(out.samples) == comp.n_mc_samples


# ─── Identity preservation ────────────────────────────────────────────────


def test_identity_preserved_with_point_forecasts():
    """Σ w_r · component_r = headline by construction."""
    weights = {f"c{i}": 1 / 12 for i in range(12)}
    comp = WeightedComposer(weights=weights, n_mc_samples=1000)
    points = {f"c{i}": float(2.0 + 0.1 * i) for i in range(12)}
    fcs = {k: _fc(v) for k, v in points.items()}
    out = comp.compose(fcs, pd.Timestamp("2026-01-31"),
                          pd.Timestamp("2026-02-28"))
    expected = sum(weights[k] * points[k] for k in weights)
    assert out.point == pytest.approx(expected)


def test_attribution_orders_by_absolute_contribution():
    weights = {"a": 0.4, "b": 0.3, "c": 0.3}
    comp = WeightedComposer(weights=weights)
    fcs = {"a": _fc(2.0, sigma=0.1),
            "b": _fc(2.5, sigma=0.1),
            "c": _fc(2.0, sigma=0.1)}
    today = {"a": 2.5, "b": 2.0, "c": 2.0}   # a went down, b went up, c flat
    df = comp.attribution(fcs, today)
    assert len(df) == 3
    # Largest |contribution| first; b's +0.5 × 0.3 = +0.15 vs a's −0.5 × 0.4 = −0.20
    assert df.iloc[0]["component_id"] == "a"   # |0.20| > |0.15|
    assert df.iloc[1]["component_id"] == "b"
    # c has zero delta → smallest |contribution|
    assert df.iloc[2]["component_id"] == "c"
    assert df.iloc[2]["contribution_pp"] == pytest.approx(0.0)


# ─── Smoke ────────────────────────────────────────────────────────────────


def test_metadata_records_per_component_contributions():
    comp = WeightedComposer(weights={"a": 0.5, "b": 0.5})
    fcs = {"a": _fc(2.0), "b": _fc(4.0)}
    out = comp.compose(fcs, pd.Timestamp("2026-01-31"),
                          pd.Timestamp("2026-02-28"))
    contribs = out.metadata["contributions"]
    assert contribs["a"] == pytest.approx(1.0)
    assert contribs["b"] == pytest.approx(2.0)


def test_truflation_top_level_weights_compose():
    """Smoke: load actual top-level v2 weights, check they sum to ~1.0,
    then compose 12 dummy forecasts."""
    from thales.weights import get_top_level_weights
    w_df = get_top_level_weights("2026-04-25")
    # Truflation 'weight' column is in percentage points (0-100), normalize
    weights = {str(int(row.category_id)): float(row.weight) / 100.0
                 for row in w_df.itertuples()}
    # Filter zero-weight categories (some are zero in v2)
    weights = {k: v for k, v in weights.items() if v > 0}

    comp = WeightedComposer(weights=weights, n_mc_samples=500,
                                weight_sum_tol=5e-3)
    # Dummy forecasts at 2.5% YoY for each top-level category
    fcs = {k: _fc(2.5, sigma=0.3) for k in weights}
    out = comp.compose(fcs, pd.Timestamp("2026-01-31"),
                          pd.Timestamp("2026-02-28"))
    assert out.point == pytest.approx(2.5, abs=0.05)
