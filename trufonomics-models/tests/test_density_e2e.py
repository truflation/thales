"""End-to-end test: forecaster → walk_forward → score returns a ScoreBlock
with CRPS, PIT-KS p-value, and density coverage populated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from thales.evaluation.harness import (
    attach_actuals,
    score,
    walk_forward,
)
from thales.models.baselines import (
    AR1Baseline,
    PathAForecaster,
    PersistenceBaseline,
)


def _synthetic_panel(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="MS")
    y = 2.0 + np.cumsum(rng.standard_normal(n) * 0.15)
    truf = y + rng.standard_normal(n) * 0.2
    return pd.DataFrame({"y": y, "truf_yoy": truf}, index=idx)


def test_persistence_walk_forward_emits_density():
    panel = _synthetic_panel(80)
    fc = PersistenceBaseline(target_col="y", train_min=24)
    forecasts = walk_forward(fc, panel, target_col="y",
                              origins=panel.index[40:60], horizon=1)
    assert len(forecasts) == 20
    assert all(f.has_density for f in forecasts)

    df = attach_actuals(forecasts, panel["y"])
    assert "samples" in df.columns
    block = score(df)
    assert block.crps is not None
    assert block.pit_ks_pvalue is not None
    assert block.cov80_density is not None
    assert block.sharp80_density is not None
    assert block.n_density and block.n_density > 0


def test_ar1_walk_forward_emits_density():
    panel = _synthetic_panel(80)
    fc = AR1Baseline(target_col="y", train_min=24, calib_months=18,
                      band_method="rolling_conformal")
    forecasts = walk_forward(fc, panel, target_col="y",
                              origins=panel.index[40:60], horizon=1)
    df = attach_actuals(forecasts, panel["y"])
    block = score(df)
    assert block.crps is not None
    assert block.cov80_density is not None
    assert 0.5 < block.cov80_density < 1.0


def test_patha_walk_forward_emits_density():
    panel = _synthetic_panel(80)
    fc = PathAForecaster(target_col="y", truflation_col="truf_yoy",
                          train_min=24, calib_months=18,
                          band_method="rolling_conformal")
    forecasts = walk_forward(fc, panel, target_col="y",
                              origins=panel.index[44:60], horizon=1)
    df = attach_actuals(forecasts, panel["y"])
    block = score(df)
    assert block.crps is not None
    assert block.pit_ks_pvalue is not None


def test_score_works_without_samples_when_forecaster_doesnt_emit():
    """Backward-compat: if a forecaster doesn't emit samples, score still
    returns a valid block with crps=None.
    """
    panel = _synthetic_panel(80)
    fc = PersistenceBaseline(target_col="y", train_min=24, n_samples=0)
    forecasts = walk_forward(fc, panel, target_col="y",
                              origins=panel.index[40:60], horizon=1)
    df = attach_actuals(forecasts, panel["y"])
    # No samples column → density block should be all None.
    block = score(df)
    assert block.crps is None
    assert block.pit_ks_pvalue is None
    assert block.cov80_density is None


def test_score_summary_renders_density_lines():
    panel = _synthetic_panel(80)
    fc = AR1Baseline(target_col="y", train_min=24, calib_months=18,
                      band_method="rolling_conformal")
    forecasts = walk_forward(fc, panel, target_col="y",
                              origins=panel.index[40:60], horizon=1)
    df = attach_actuals(forecasts, panel["y"])
    block = score(df)
    txt = block.summary()
    assert "CRPS" in txt
    assert "Density 80%/95% cov" in txt
