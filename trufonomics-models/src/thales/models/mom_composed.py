"""MoM-first YoY forecasting via composition — Fix #5.

Resolves user feedback: "Model MoM first, then compose YoY."

The argument: BLS / BEA YoY series have 12-month autocorrelation built
into the construction (yoy[t] = log P[t] - log P[t-12]). When you fit
a level-layer model (UC, BSTS LLT, AR(1)-on-YoY) directly to YoY, the
trend layer absorbs all variance; a regime-switching variance model
finds no regimes because the YoY innovation is a smoothed mixture of
12 months of MoM innovations.

The fix is exact, not approximate. For log returns:

    yoy[t]  =  log P[t] - log P[t-12]  =  Σ_{k=0..11} mom[t-k]
    mom[t]  =  log P[t] - log P[t-1]

So forecasting ``mom`` and rolling it into ``yoy`` is mathematically
equivalent to forecasting ``yoy`` directly, but with two benefits:

  1. ``mom`` is approximately stationary (no induced unit root from
     the YoY differencing).
  2. Variance regimes show up cleanly in ``mom`` (Hamilton-MS, MS-SV
     archetypes find them) where they were invisible in ``yoy``.

This module provides:

  * ``mom_from_level(level)`` — utility for log-MoM (in pp, ×100).
  * ``compose_yoy_one_step(yoy_T, mom_pred_T_plus_1, mom_T_minus_11)``
    — the closed-form roll-out for h=1.
  * ``MoMComposedForecaster`` — wraps any inner Forecaster trained on
    a MoM column, exposes a YoY forecast through composition.

The inner forecaster is unchanged; the wrapper just translates the
panel columns so the inner model sees `mom_col` and the outer
``Forecast`` reports YoY.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from thales.evaluation.density import (
    DEFAULT_N_SAMPLES,
    samples_from_residuals,
)
from thales.evaluation.harness import Forecast


def mom_from_level(level: pd.Series, log: bool = True) -> pd.Series:
    """Monthly log-return (in pp, ×100) from a level series.

    With ``log=True`` (default): ``mom[t] = 100·(log P[t] − log P[t-1])``.
    With ``log=False``: ``mom[t] = 100·(P[t]/P[t-1] − 1)``.

    The first observation is dropped (no lag-1).
    """
    if log:
        return (100.0 * (np.log(level) - np.log(level.shift(1)))).dropna()
    return (100.0 * (level / level.shift(1) - 1.0)).dropna()


def yoy_from_level(level: pd.Series, log: bool = True) -> pd.Series:
    """Year-over-year (in pp, ×100) from a level series."""
    if log:
        return (100.0 * (np.log(level) - np.log(level.shift(12)))).dropna()
    return (100.0 * (level / level.shift(12) - 1.0)).dropna()


def compose_yoy_one_step(yoy_T: float, mom_pred_T_plus_1: float,
                            mom_T_minus_11: float) -> float:
    """Closed-form one-step YoY composition (log form):

        yoy[T+1] = yoy[T] + mom[T+1] - mom[T-11]

    which is the exact identity log P[T+1] − log P[T−11] − (log P[T] − log P[T−12]).
    Returns the forecast YoY in the same units as ``yoy_T`` (pp).
    """
    return float(yoy_T + mom_pred_T_plus_1 - mom_T_minus_11)


def compose_yoy_multi_step(yoy_T: float,
                              mom_pred_chain: list[float] | np.ndarray,
                              mom_dropped: list[float] | np.ndarray) -> float:
    """Closed-form h-step YoY composition (log form):

        yoy[T+h]  =  yoy[T]  +  Σ_{k=1..h} mom[T+k]  −  Σ_{j=0..h-1} mom[T-11+j]

    Generalizes :func:`compose_yoy_one_step`. The first sum is the
    chain of predicted month-over-month moves arriving in months
    T+1 through T+h; the second sum is the equally-many month-over-
    month values that drop out of the trailing-12 window as the
    horizon advances. Both lists must have length ``h``.

    Returns the YoY forecast at horizon h in the same units as
    ``yoy_T`` (percentage points).
    """
    chain = np.asarray(mom_pred_chain, dtype=float)
    dropped = np.asarray(mom_dropped, dtype=float)
    if len(chain) != len(dropped):
        raise ValueError(
            f"mom_pred_chain (n={len(chain)}) and mom_dropped "
            f"(n={len(dropped)}) must be the same length")
    return float(yoy_T + chain.sum() - dropped.sum())


def _ar1_iterate(start: float, alpha: float, phi: float,
                  h: int) -> np.ndarray:
    """Deterministic h-step AR(1) chain starting from ``start``.

    Returns ``np.ndarray`` of length ``h``:
    ``[α + φ·start, α + φ·(α + φ·start), ...]``.
    """
    out = np.empty(h, dtype=float)
    last = start
    for k in range(h):
        last = alpha + phi * last
        out[k] = last
    return out


def _ar1_chain_samples(start: float, alpha: float, phi: float, h: int,
                        residuals: np.ndarray, n_samples: int,
                        seed: int) -> np.ndarray:
    """AR(1) bootstrap samples of the h-step MoM chain SUM.

    For each sample s in 1..n_samples, draw h residuals iid from the
    empirical AR(1) calibration distribution, build the noisy AR(1)
    chain, and return the sum of the h chain values. The result is
    a ``(n_samples,)`` array of samples of ``Σ_{k=1..h} mom[T+k]``.

    This is the right object to add to ``yoy_T - sum(mom_dropped)``
    to produce calibrated samples of the YoY forecast at horizon h.
    """
    if h <= 0:
        return np.zeros(n_samples)
    if len(residuals) < 2:
        return np.full(n_samples, np.nan)
    rng = np.random.default_rng(seed)
    # Draw h × n_samples residuals at once for efficiency.
    eta = residuals[rng.integers(0, len(residuals), size=(n_samples, h))]
    # Iterate the chain with the bootstrap residuals layered on the AR(1)
    # mean dynamics. Vectorized: track a running last value across samples.
    chains = np.empty((n_samples, h), dtype=float)
    last = np.full(n_samples, start, dtype=float)
    for k in range(h):
        last = alpha + phi * last + eta[:, k]
        chains[:, k] = last
    return chains.sum(axis=1)


class _InnerForecaster(Protocol):
    model_id: str
    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast: ...


@dataclass
class MoMComposedForecaster:
    """Compose YoY forecast from MoM forecast via the closed-form identity.

    Wraps an ``inner`` forecaster that operates on a MoM column. The
    wrapper:

      1. Reads the panel, derives MoM from the level column at construction-
         time of each call.
      2. Builds a sub-panel exposing only the MoM column for the inner.
      3. Calls ``inner.fit_predict(...)`` to get a MoM forecast at horizon=1.
      4. Composes: ``yoy[T+1] = yoy[T] + mom_pred[T+1] - mom[T-11]``.
      5. Returns a Forecast with the YoY point and bands rescaled
         appropriately (a pp shift in mom_pred is a pp shift in yoy_pred).

    Notes:
      * Only ``horizon=1`` is supported. Multi-step would need to predict
        a sequence of MoMs and roll them out; out of scope for Fix #5.
      * Inner Forecast bands (in MoM space) are translated 1-to-1 into
        YoY bands. That's correct because the composition is a linear
        shift; the predictive uncertainty around mom[T+1] equals the
        predictive uncertainty around yoy[T+1] (the other terms are
        constants known at origin).
      * The inner forecaster's ``today_baseline`` for direction tests
        will be in MoM space — it does NOT correspond to YoY direction.
        Use the harness's external ``today_baseline`` (yoy) at scoring
        time for correct YoY direction-hit reporting.
    """
    inner: _InnerForecaster
    bls_level_col: str = "bls_level"
    bls_yoy_col: str = "bls_yoy"
    mom_col: str = "bls_mom"
    log_mom: bool = True
    horizon: int = 1
    n_samples: int = DEFAULT_N_SAMPLES
    seed: int = 0
    model_id: str = "mom_composed_v1"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        if self.horizon < 1:
            raise ValueError(f"horizon must be ≥1; got {self.horizon}")
        if self.bls_level_col not in panel.columns:
            raise ValueError(
                f"bls_level_col '{self.bls_level_col}' not in panel")

        level = panel[self.bls_level_col]
        mom = mom_from_level(level, log=self.log_mom)
        mom = mom.reindex(panel.index)

        # Build sub-panel for the inner forecaster
        sub = panel.copy()
        sub[self.mom_col] = mom

        # Origin must have a non-NaN mom value (i.e. need level[T] and
        # level[T-1]). The harness pre-truncates the panel to [:origin],
        # so origin should be the last valid row.
        if origin not in sub.index or pd.isna(sub.loc[origin, self.mom_col]):
            raise ValueError(f"MoM not available at origin {origin}")
        origin_pos = sub.index.get_loc(origin)

        # Inner forecast at horizon=1 — we use it as a fitted-AR(1)
        # source. The inner model is responsible for fitting on MoM
        # history; we extract its alpha/phi (and residual SD if present)
        # from the metadata and iterate manually for h>1.
        inner_fc = self.inner.fit_predict(sub, origin, target)

        # YoY at origin — known
        if self.bls_yoy_col in panel.columns and pd.notna(
                panel.loc[origin, self.bls_yoy_col]):
            yoy_T = float(panel.loc[origin, self.bls_yoy_col])
        else:
            # Fallback: derive yoy from level if missing
            yoy_at_origin = yoy_from_level(level, log=self.log_mom)
            if origin not in yoy_at_origin.index:
                raise ValueError(f"yoy not available at origin {origin}")
            yoy_T = float(yoy_at_origin.loc[origin])

        h = self.horizon

        # Need 11 + h months of history for the multi-step composition
        # — the dropped-out MoMs at horizon h are mom[T-11..T-12+h].
        if origin_pos < 11 + (h - 1):
            raise ValueError(
                f"need ≥{11 + h - 1} months of history before {origin} "
                f"to compose YoY at horizon {h}")

        # The h MoMs that drop out as the YoY window slides forward.
        mom_dropped = np.array([
            float(sub.iloc[origin_pos - 11 + k][self.mom_col])
            for k in range(h)
        ])

        if h == 1:
            # Closed-form one-step path — preserves the original behavior
            # exactly, including band/samples translation through the inner.
            mom_pred = float(inner_fc.point)
            point = compose_yoy_one_step(yoy_T, mom_pred, float(mom_dropped[0]))
            if inner_fc.has_bands:
                d_lo80 = inner_fc.lo80 - inner_fc.point
                d_hi80 = inner_fc.hi80 - inner_fc.point
                d_lo95 = inner_fc.lo95 - inner_fc.point
                d_hi95 = inner_fc.hi95 - inner_fc.point
                lo80 = point + d_lo80
                hi80 = point + d_hi80
                lo95 = point + d_lo95
                hi95 = point + d_hi95
            else:
                lo80 = hi80 = lo95 = hi95 = None
            yoy_samples = None
            if inner_fc.has_density:
                yoy_samples = inner_fc.samples + (point - mom_pred)
            return Forecast(
                origin=origin, target=target, point=point,
                lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
                samples=yoy_samples,
                metadata={
                    "model": "mom_composed",
                    "inner_model": getattr(self.inner, "model_id", "unknown"),
                    "yoy_T": yoy_T,
                    "mom_pred": mom_pred,
                    "mom_T_minus_11": float(mom_dropped[0]),
                    "horizon": self.horizon,
                    **({"inner_metadata": inner_fc.metadata}
                         if inner_fc.metadata else {}),
                },
            )

        # ── Multi-step path (h > 1) ───────────────────────────────────
        meta = inner_fc.metadata or {}
        if "alpha" not in meta or "phi" not in meta:
            raise NotImplementedError(
                "MoMComposedForecaster horizon>1 requires inner forecaster "
                "to expose AR(1) coefficients 'alpha' and 'phi' in metadata "
                "(use AR1Baseline as inner)")
        alpha = float(meta["alpha"])
        phi = float(meta["phi"])

        mom_T = float(sub.loc[origin, self.mom_col])
        mom_chain_det = _ar1_iterate(mom_T, alpha, phi, h)
        point = compose_yoy_multi_step(
            yoy_T, mom_chain_det.tolist(), mom_dropped.tolist())

        # Density via AR(1) bootstrap of the chain SUM. Reconstruct the
        # inner's calibration residuals from its samples (samples are
        # point + bootstrapped residuals), so resampling works the same
        # way at h-step.
        chain_sum_samples: np.ndarray | None = None
        if inner_fc.has_density:
            inner_residuals = (np.asarray(inner_fc.samples)
                                  - inner_fc.point)
            chain_sum_samples = _ar1_chain_samples(
                start=mom_T, alpha=alpha, phi=phi, h=h,
                residuals=inner_residuals,
                n_samples=self.n_samples,
                seed=self.seed + hash(origin) % 10_000,
            )
        # YoY samples: yoy_T + chain_sum - sum(dropped)  →  shift the
        # chain-sum samples by (yoy_T - sum(dropped)).
        yoy_samples = None
        if chain_sum_samples is not None and not np.all(
                np.isnan(chain_sum_samples)):
            yoy_samples = chain_sum_samples + (yoy_T - mom_dropped.sum())

        # Bands from the empirical sample distribution.
        if yoy_samples is not None:
            lo80 = float(np.quantile(yoy_samples, 0.10))
            hi80 = float(np.quantile(yoy_samples, 0.90))
            lo95 = float(np.quantile(yoy_samples, 0.025))
            hi95 = float(np.quantile(yoy_samples, 0.975))
        else:
            lo80 = hi80 = lo95 = hi95 = None

        return Forecast(
            origin=origin, target=target, point=point,
            lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
            samples=yoy_samples,
            metadata={
                "model": "mom_composed",
                "inner_model": getattr(self.inner, "model_id", "unknown"),
                "yoy_T": yoy_T,
                "alpha": alpha,
                "phi": phi,
                "mom_chain_deterministic": mom_chain_det.tolist(),
                "mom_dropped": mom_dropped.tolist(),
                "horizon": self.horizon,
                "band_source": ("ar1_bootstrap_chain"
                                  if yoy_samples is not None else "none"),
            },
        )
