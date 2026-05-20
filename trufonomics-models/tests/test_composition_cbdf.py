"""Tests for the CBDF cross-component-correlation composer (Phase 2.1b)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thales.evaluation.harness import Forecast
from thales.models.composition.cbdf import CBDFComposer


def _fc(point: float, sigma: float) -> Forecast:
    return Forecast(
        origin=pd.Timestamp("2026-01-31"),
        target=pd.Timestamp("2026-02-28"),
        point=point,
        lo80=point - 1.2816 * sigma,
        hi80=point + 1.2816 * sigma,
        lo95=point - 1.96 * sigma,
        hi95=point + 1.96 * sigma,
    )


def _residual_panel_from_cov(cov: np.ndarray, n_origins: int,
                              component_ids: list[str],
                              seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.zeros(len(component_ids)),
                                       cov, size=n_origins)
    return pd.DataFrame(draws, columns=component_ids)


# ─── Basic API ────────────────────────────────────────────────────────────


def test_falls_back_when_covariance_not_fitted():
    """Without fitting, CBDFComposer behaves like WeightedComposer."""
    comp = CBDFComposer(weights={"a": 0.5, "b": 0.5})
    fcs = {"a": _fc(2.0, 0.3), "b": _fc(3.0, 0.3)}
    out = comp.compose(fcs, pd.Timestamp("2026-01-31"),
                          pd.Timestamp("2026-02-28"))
    assert out.point == pytest.approx(2.5)
    assert out.metadata["composer"] == "weighted"


def test_compose_after_fit_uses_joint_sampling():
    comp = CBDFComposer(weights={"a": 0.5, "b": 0.5}, n_mc_samples=5000)
    cov = np.array([[0.1, 0.05], [0.05, 0.1]])
    panel = _residual_panel_from_cov(cov, 100, ["a", "b"])
    comp.fit_residual_covariance(panel)
    fcs = {"a": _fc(2.0, 0.3), "b": _fc(3.0, 0.3)}
    out = comp.compose(fcs, pd.Timestamp("2026-01-31"),
                          pd.Timestamp("2026-02-28"))
    assert out.metadata["composer"] == "cbdf_multivariate_gaussian"
    # Point should still be the weighted sum
    assert out.point == pytest.approx(2.5, abs=1e-6)


# ─── Bands tighten with positive correlation, widen with negative ─────────


def test_positive_correlation_tightens_bands_vs_independence():
    """When two components are positively correlated, the weighted-sum
    variance is HIGHER than under independence (corr terms add to var
    of sum). Band should be WIDER than the independent baseline."""
    # Two components, strong positive correlation
    var = 0.10
    cov_independent = np.array([[var, 0.0], [0.0, var]])
    cov_correlated = np.array([[var, 0.08], [0.08, var]])

    panel_indep = _residual_panel_from_cov(cov_independent, 200, ["a", "b"],
                                               seed=0)
    panel_corr = _residual_panel_from_cov(cov_correlated, 200, ["a", "b"],
                                              seed=0)

    fcs = {"a": _fc(2.0, np.sqrt(var)), "b": _fc(3.0, np.sqrt(var))}

    comp_indep = CBDFComposer(weights={"a": 0.5, "b": 0.5}, n_mc_samples=10000,
                                  seed=1)
    comp_indep.fit_residual_covariance(panel_indep)
    out_indep = comp_indep.compose(fcs, pd.Timestamp("2026-01-31"),
                                        pd.Timestamp("2026-02-28"))

    comp_corr = CBDFComposer(weights={"a": 0.5, "b": 0.5}, n_mc_samples=10000,
                                 seed=1)
    comp_corr.fit_residual_covariance(panel_corr)
    out_corr = comp_corr.compose(fcs, pd.Timestamp("2026-01-31"),
                                      pd.Timestamp("2026-02-28"))

    # Band width comparison
    width_indep = out_indep.hi80 - out_indep.lo80
    width_corr = out_corr.hi80 - out_corr.lo80
    assert width_corr > width_indep * 1.10, (
        f"correlated band width {width_corr:.3f} should be > "
        f"independent band {width_indep:.3f} × 1.10")


def test_negative_correlation_tightens_bands():
    var = 0.10
    cov_negative = np.array([[var, -0.08], [-0.08, var]])
    panel_neg = _residual_panel_from_cov(cov_negative, 200, ["a", "b"], seed=2)

    fcs = {"a": _fc(2.0, np.sqrt(var)), "b": _fc(3.0, np.sqrt(var))}
    comp = CBDFComposer(weights={"a": 0.5, "b": 0.5},
                            n_mc_samples=10000, seed=3)
    comp.fit_residual_covariance(panel_neg)
    out = comp.compose(fcs, pd.Timestamp("2026-01-31"),
                          pd.Timestamp("2026-02-28"))

    # Compare to independent baseline width
    expected_indep_sd = np.sqrt(2 * (0.5 ** 2) * var)
    expected_indep_width_80 = 2 * 1.2816 * expected_indep_sd
    actual_width_80 = out.hi80 - out.lo80
    assert actual_width_80 < expected_indep_width_80, (
        f"negative-corr width {actual_width_80:.4f} should be < "
        f"independent width {expected_indep_width_80:.4f}")


# ─── Identity preservation ────────────────────────────────────────────────


def test_point_matches_weighted_sum_after_fit():
    """Point forecast is unchanged by fitting covariance — it's still
    just the weighted sum of component points."""
    comp = CBDFComposer(weights={"a": 0.4, "b": 0.6}, n_mc_samples=2000)
    cov = np.array([[0.1, 0.03], [0.03, 0.1]])
    panel = _residual_panel_from_cov(cov, 80, ["a", "b"])
    comp.fit_residual_covariance(panel)
    fcs = {"a": _fc(1.5, 0.2), "b": _fc(2.5, 0.2)}
    out = comp.compose(fcs, pd.Timestamp("2026-01-31"),
                          pd.Timestamp("2026-02-28"))
    assert out.point == pytest.approx(0.4 * 1.5 + 0.6 * 2.5)


# ─── Validation ───────────────────────────────────────────────────────────


def test_residual_panel_missing_columns_raises():
    comp = CBDFComposer(weights={"a": 0.5, "b": 0.5})
    panel = _residual_panel_from_cov(np.eye(2) * 0.1, 50, ["a", "x"])
    with pytest.raises(ValueError, match="missing columns"):
        comp.fit_residual_covariance(panel)


def test_residual_panel_too_short_raises():
    comp = CBDFComposer(weights={"a": 0.5, "b": 0.5})
    panel = _residual_panel_from_cov(np.eye(2) * 0.1, 5, ["a", "b"])
    with pytest.raises(ValueError, match="≥12"):
        comp.fit_residual_covariance(panel)


def test_compose_after_fit_requires_all_active():
    comp = CBDFComposer(weights={"a": 0.5, "b": 0.5})
    panel = _residual_panel_from_cov(np.eye(2) * 0.1, 50, ["a", "b"])
    comp.fit_residual_covariance(panel)
    with pytest.raises(ValueError, match="missing"):
        comp.compose({"a": _fc(1.0, 0.1)},
                       pd.Timestamp("2026-01-31"),
                       pd.Timestamp("2026-02-28"))


# ─── PSD safety ───────────────────────────────────────────────────────────


def test_short_panel_still_psd():
    """Even with very short panels the shrinkage keeps cov PSD."""
    comp = CBDFComposer(weights={f"c{i}": 1 / 6 for i in range(6)})
    cov = np.eye(6) * 0.1
    # n_origins (15) < 2 * n_comp (12) → shrinkage kicks in
    panel = _residual_panel_from_cov(cov, 15,
                                          [f"c{i}" for i in range(6)])
    comp.fit_residual_covariance(panel)
    # Compose should not blow up
    fcs = {f"c{i}": _fc(2.0 + 0.1 * i, 0.1) for i in range(6)}
    out = comp.compose(fcs, pd.Timestamp("2026-01-31"),
                          pd.Timestamp("2026-02-28"))
    assert np.isfinite(out.lo80)
    assert np.isfinite(out.hi80)
