"""BridgedCBDFForecaster — wired Forecaster for the CBDF→BLS bridge.

The O'Keeffe-Petrova 2025 CBDF architecture composes per-component
forecasts into a *Truflation-scale* headline — fine when components
sum to the same target by construction (their original GDP setup), but
on inflation our 12-component Truflation panel doesn't exactly equal
BLS Headline CPI YoY (different surveys, weights, ~50 bp structural
gap). Direct CBDF lost to Stock-Watson DFM by 74-80 % RMSE.

The fix is a rolling-OLS bridge layer on top of CBDF::

    BLS_yoy[T+1]  =  α  +  β · BLS_yoy[T]  +  γ · CBDF_pred[T+1]  +  ε

where (α, β, γ) are estimated each origin on the trailing
``calib_window`` rows of (BLS_actual, BLS_lag, CBDF_pred) tuples.
Bridged-CBDF beats DFM by +25.6-30.6 % RMSE, p < 0.0001 — see
``OKEEFE_HEADTOHEAD_FINDINGS.md``.

This module promotes that inline logic from the head-to-head script
into a Forecaster-Protocol class. The caller assembles a panel with
both the BLS target column and a pre-computed ``inner_pred_col``
(typically: per-origin CBDF predictions joined back as a series), and
the bridge takes over from there. The class doesn't care whether the
inner signal is CBDF, a TSFM, or any other Truflation-scale
forecaster — it's a generic Truflation→BLS bridge.

Bands and density samples come from rolling-origin OOS residuals on
the bridge regression itself, so the predictive uncertainty reflects
the actual bridge calibration error, not the inner model's variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from thales.evaluation.conformal import (
    conformal_band_offsets,
    min_n_for_alpha,
)
from thales.evaluation.density import (
    DEFAULT_N_SAMPLES,
    samples_from_gaussian,
    samples_from_residuals,
)
from thales.evaluation.harness import Forecast


BandMethod = Literal["gaussian", "rolling_conformal"]


def _bridge_bands(point: float, errors: np.ndarray, residual_sd: float
                    ) -> tuple[float, float, float, float]:
    """80 % / 95 % bands with conformal-or-Gaussian fallback."""
    n = len(errors)
    if n >= min_n_for_alpha(0.20):
        a, b = conformal_band_offsets(errors, alpha=0.20)
        lo80, hi80 = point + a, point + b
    else:
        lo80, hi80 = point - 1.2816 * residual_sd, point + 1.2816 * residual_sd
    if n >= min_n_for_alpha(0.05):
        a, b = conformal_band_offsets(errors, alpha=0.05)
        lo95, hi95 = point + a, point + b
    else:
        lo95, hi95 = point - 1.96 * residual_sd, point + 1.96 * residual_sd
    return lo80, hi80, lo95, hi95


@dataclass
class BridgedCBDFForecaster:
    """Rolling-OLS bridge from a Truflation-scale signal to BLS YoY.

    Walk-forward semantics: at origin T, fit the bridge on the trailing
    ``calib_window`` of completed (target, BLS_actual[t], BLS[t-1],
    inner_pred[t-1]) tuples, then project ahead using
    ``inner_pred[origin]`` as the next-period inner forecast::

        BLS_yoy[t]      ~ α + β · BLS_yoy[t-1] + γ · inner_pred[t-1] + ε
        BLS_yoy[T+1]    = α̂ + β̂ · BLS_yoy[T]   + γ̂ · inner_pred[T]

    Convention: ``panel.loc[t, inner_pred_col]`` must hold the inner
    forecaster's prediction *made at time t* for the *next-period*
    target (i.e. it's an origin-indexed prediction-for-tomorrow). The
    caller assembles this column once before calling walk_forward —
    typically by running the inner forecaster across all origins and
    writing each prediction back to the origin's row.

    Parameters
    ----------
    target_bls_col : str
        Column name of the BLS target YoY series in the panel.
    inner_pred_col : str
        Column name of pre-computed inner predictions. ``panel[col][t]``
        is the prediction *made at t* for *t+1*.
    calib_window : int
        Trailing months on which to fit the bridge OLS each origin.
    band_method : "gaussian" or "rolling_conformal"
        How bands and samples are constructed. Rolling-conformal uses
        the bridge residuals as a finite-sample residual distribution.
    n_samples, seed : sample emission knobs.
    """
    target_bls_col: str = "bls_yoy"
    inner_pred_col: str = "cbdf_pred"
    calib_window: int = 24
    train_min: int = 12
    band_method: BandMethod = "rolling_conformal"
    n_samples: int = DEFAULT_N_SAMPLES
    seed: int = 0
    model_id: str = "bridged_cbdf_v1"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        if self.target_bls_col not in panel.columns:
            raise ValueError(
                f"target_bls_col '{self.target_bls_col}' not in panel")
        if self.inner_pred_col not in panel.columns:
            raise ValueError(
                f"inner_pred_col '{self.inner_pred_col}' not in panel")
        if origin not in panel.index:
            raise ValueError(f"BridgedCBDF: origin {origin} not in panel")

        # Construct supervised tuples with the prediction-for-next-period
        # convention: the regressor column ``inner_lag`` at row t is the
        # inner forecast made at t-1, i.e. ``panel[inner_pred_col][t-1]``.
        df = panel[[self.target_bls_col, self.inner_pred_col]].copy()
        df["bls_lag"] = df[self.target_bls_col].shift(1)
        df["inner_lag"] = df[self.inner_pred_col].shift(1)

        train_full = df.dropna()
        train_full = train_full.loc[train_full.index <= origin]
        if self.calib_window and len(train_full) > self.calib_window:
            train_full = train_full.iloc[-self.calib_window:]
        if len(train_full) < self.train_min:
            raise ValueError(
                f"BridgedCBDF: need ≥{self.train_min} rows; "
                f"have {len(train_full)}")

        X = np.column_stack([
            np.ones(len(train_full)),
            train_full["bls_lag"].values,
            train_full["inner_lag"].values,
        ])
        y = train_full[self.target_bls_col].values
        cf, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta_lag, gamma = (float(cf[0]), float(cf[1]), float(cf[2]))

        residuals = y - X @ cf
        residual_sd = (float(np.std(residuals, ddof=3))
                          if len(residuals) > 3 else float("nan"))

        # Predict at origin → target:
        #   BLS_lag = BLS[origin] (most recent published value)
        #   inner_at_target = inner forecast made at origin for target
        bls_lag = float(panel.loc[origin, self.target_bls_col])
        inner_at_target = panel.loc[origin, self.inner_pred_col]
        if pd.isna(bls_lag):
            raise ValueError(
                f"BridgedCBDF: target_bls_col NaN at origin {origin}")
        if pd.isna(inner_at_target):
            raise ValueError(
                f"BridgedCBDF: inner_pred_col NaN at origin {origin}")
        inner_at_target = float(inner_at_target)
        point = alpha + beta_lag * bls_lag + gamma * inner_at_target

        # ── Bands and samples ─────────────────────────────────────────
        if self.band_method == "rolling_conformal" and len(residuals) >= 9:
            lo80, hi80, lo95, hi95 = _bridge_bands(
                point, residuals, residual_sd)
            band_source = "rolling_conformal"
            samples = samples_from_residuals(
                point, residuals, n_samples=self.n_samples,
                seed=self.seed + hash(origin) % 10_000,
            ) if self.n_samples > 0 else None
        else:
            sigma = residual_sd if np.isfinite(residual_sd) else 0.0
            lo80 = point - 1.2816 * sigma
            hi80 = point + 1.2816 * sigma
            lo95 = point - 1.96 * sigma
            hi95 = point + 1.96 * sigma
            band_source = "gaussian"
            samples = (samples_from_gaussian(
                point, sigma, n_samples=self.n_samples,
                seed=self.seed + hash(origin) % 10_000)
                if (self.n_samples > 0 and sigma > 0) else None)

        return Forecast(
            origin=origin, target=target, point=point,
            lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
            samples=samples,
            metadata={
                "model": "bridged_cbdf",
                "alpha": alpha, "beta_lag": beta_lag, "gamma_inner": gamma,
                "n_train": len(train_full),
                "bls_lag_at_origin": bls_lag,
                "inner_pred_at_target": inner_at_target,
                "residual_sd": residual_sd,
                "band_source": band_source,
            },
        )
