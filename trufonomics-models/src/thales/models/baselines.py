"""Baseline forecasters for the official-target evaluation track.

Each baseline implements the :class:`thales.evaluation.harness.Forecaster`
Protocol — drops straight into ``walk_forward`` and ``ScoringDB`` without
glue code. These baselines are the floor every Thales archetype must beat
on the official BLS CPI / BEA PCE targets.

Provided:

  * :class:`PersistenceBaseline` — predict ``y[T+h] = y[T]``. Hard to beat
    on near-term inflation YoY (Stock-Watson 2007). The reference floor.
  * :class:`AR1Baseline` — fit a univariate AR(1) on training history.
    Closer to the "no information" autoregressive baseline used in the
    forecast-evaluation literature.
  * :class:`PathAForecaster` — kairos Path A retargeted: 2-feature OLS
    stacker with persistence (current target YoY) + Truflation YoY as a
    monthly exogenous signal. The first model that actually carries
    information beyond persistence.

Band methods (``band_method`` parameter on AR1 / PathA):

  * ``"in_sample"`` — empirical quantiles of in-sample residuals.
    Biased: residuals after fit are tighter than out-of-sample errors,
    so bands undercover. Sanity baseline only.
  * ``"split_conformal"`` — Vovk-Lei-Tibshirani 2018 style. Fit on the
    first ``len(train) - calib_months`` rows; use OOS errors on the
    held-out tail for bands. Calibrated, but **point model loses the
    most-recent ``calib_months`` of training data** — hurts RMSE.
  * ``"rolling_conformal"`` (default) — fit point model on ALL data
    available at origin. Bands come from rolling-origin OOS residuals:
    for each calibration position c in the trailing ``calib_months``
    window, refit on data strictly before c, predict y[c]. The band
    width therefore reflects realistic recent forecast errors **without
    discarding training data**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from thales.evaluation.conformal import (
    conformal_band_offsets,
    min_n_for_alpha,
)
from thales.evaluation.density import (
    DEFAULT_N_SAMPLES,
    samples_from_residuals,
)
from thales.evaluation.harness import Forecast


BandMethod = Literal["in_sample", "split_conformal", "rolling_conformal"]


def _samples(point: float, errors: np.ndarray, n_samples: int, seed: int
              ) -> np.ndarray | None:
    """Bootstrap samples from calibration residuals; None if too few."""
    if n_samples <= 0 or len(errors) < 2:
        return None
    return samples_from_residuals(point, errors, n_samples=n_samples,
                                    seed=seed)


def _bands_from_residuals(point: float, errors: np.ndarray
                            ) -> tuple[float, float, float, float]:
    """Helper: turn calibration residuals into 80% and 95% bands.

    Per-α fallback: uses finite-sample conformal quantiles (Lei et al.
    2018) when n ≥ ``min_n_for_alpha(α)``, else falls back to Gaussian
    bands (z·σ) for that α only. This prevents the rank-clamp artifact
    where insufficient calibration data silently undercovers.

    Returns ``(lo80, hi80, lo95, hi95)``.
    """
    n = len(errors)
    sigma = float(np.std(errors)) if n > 1 else 0.0
    if n >= min_n_for_alpha(0.20):
        a, b = conformal_band_offsets(errors, alpha=0.20)
        lo80, hi80 = point + a, point + b
    else:
        lo80, hi80 = point - 1.2816 * sigma, point + 1.2816 * sigma
    if n >= min_n_for_alpha(0.05):
        a, b = conformal_band_offsets(errors, alpha=0.05)
        lo95, hi95 = point + a, point + b
    else:
        lo95, hi95 = point - 1.96 * sigma, point + 1.96 * sigma
    return lo80, hi80, lo95, hi95


@dataclass
class PersistenceBaseline:
    """Trivial: predict next-period target equals current target.

    Bands come from the empirical distribution of in-sample first
    differences ``y[t] - y[t-h]`` over the training window — captures the
    realized period-over-period volatility of the target.

    ``target_col`` is the column in ``panel`` containing the target series.
    """
    target_col: str = "y"
    horizon: int = 1
    train_min: int = 24      # require at least 24 obs of history before forecasting
    n_samples: int = DEFAULT_N_SAMPLES
    seed: int = 0
    model_id: str = "persistence_v1"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        s = panel[self.target_col].dropna()
        s = s.loc[s.index <= origin]
        if len(s) < self.train_min:
            raise ValueError(
                f"persistence: need ≥{self.train_min} obs before "
                f"{origin:%Y-%m-%d}, have {len(s)}")

        point = float(s.iloc[-1])

        diffs = s.diff(self.horizon).dropna().values
        if len(diffs) < 10:
            return Forecast(origin=origin, target=target, point=point,
                              metadata={"baseline": "persistence",
                                          "n_train": len(s)})
        lo80, hi80, lo95, hi95 = _bands_from_residuals(point, diffs)
        return Forecast(
            origin=origin, target=target, point=point,
            lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
            samples=_samples(point, diffs, self.n_samples,
                              self.seed + hash(origin) % 10_000),
            metadata={"baseline": "persistence", "n_train": len(s),
                       "diff_sd": float(np.std(diffs)),
                       "band_source": "conformal_diff_distribution"},
        )


def _patha_design(df: pd.DataFrame, target_col: str,
                    truflation_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) design matrices from a Path-A training frame.
    ``df`` must already contain ``__y_target`` (the lead-shifted target).
    """
    X = np.column_stack([
        np.ones(len(df)),
        df[target_col].values,
        df[truflation_col].values,
    ])
    y = df["__y_target"].values
    return X, y


@dataclass
class PathAForecaster:
    """Kairos Path A retargeted to the new harness.

    Two-feature regression of the official BLS CPI / BEA PCE YoY:

        y[T+h] ~ α + β_y · y[T] + β_t · truf_yoy[T] + ε

    with α/β estimated by OLS on the training window.

    Bands controlled by ``band_method`` (see module docstring). Default is
    ``"rolling_conformal"`` — point model uses ALL training data; bands
    are derived from rolling-origin OOS residuals over the trailing
    ``calib_months`` positions. ``calib_months`` is ignored when
    ``band_method="in_sample"``.

    Original Path A was a *same-month* nowcast (+25 days into month M).
    This is the ``+h`` forecast retargeting — informative about whether
    Truflation's daily-updating signal carries information **beyond
    persistence** at the +1m horizon.

    ``truflation_col`` is the column in ``panel`` containing the
    Truflation YoY signal aligned to the monthly index.
    """
    target_col: str = "y"
    truflation_col: str = "truf_yoy"
    horizon: int = 1
    train_min: int = 36     # 3 yrs minimum to have meaningful regression
    calib_months: int = 0   # 0 ⇒ in-sample bands regardless of band_method
    band_method: BandMethod = "rolling_conformal"
    n_samples: int = DEFAULT_N_SAMPLES
    seed: int = 0
    model_id: str = "patha_v1"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        df = panel[[self.target_col, self.truflation_col]].copy()
        df["__y_target"] = df[self.target_col].shift(-self.horizon)

        all_train = df.loc[df.index < origin].dropna()
        # Effective calibration window — 0 collapses to in-sample bands.
        eff_calib = (self.calib_months
                       if self.band_method != "in_sample" else 0)
        if len(all_train) < self.train_min + eff_calib:
            raise ValueError(
                f"PathA: need ≥{self.train_min + eff_calib} obs "
                f"of training data before {origin:%Y-%m-%d}, "
                f"have {len(all_train)}")

        # Decide point-model training set + calibration residual source.
        if eff_calib > 0 and self.band_method == "split_conformal":
            train = all_train.iloc[: -eff_calib]
            calib = all_train.iloc[-eff_calib:]
            X, y = _patha_design(train, self.target_col, self.truflation_col)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            X_cal, y_cal = _patha_design(calib, self.target_col,
                                            self.truflation_col)
            errors = y_cal - X_cal @ coef
            band_source = "split_conformal"
            n_train = len(train)
            n_calib = len(calib)
        elif eff_calib > 0 and self.band_method == "rolling_conformal":
            # Point model: fit on ALL training data — no holdout penalty.
            X, y = _patha_design(all_train, self.target_col,
                                  self.truflation_col)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            # Bands: rolling-origin OOS residuals over trailing window.
            errors = _rolling_oos_residuals_patha(
                all_train, self.target_col, self.truflation_col,
                calib_months=eff_calib, train_min=self.train_min)
            band_source = "rolling_conformal"
            n_train = len(all_train)
            n_calib = len(errors)
        else:
            X, y = _patha_design(all_train, self.target_col,
                                  self.truflation_col)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            errors = y - X @ coef
            band_source = "in_sample"
            n_train = len(all_train)
            n_calib = 0

        alpha, beta_y, beta_t = (float(coef[0]), float(coef[1]),
                                    float(coef[2]))

        # Predict at origin
        if origin not in panel.index:
            raise ValueError(f"PathA: origin {origin} not in panel index")
        y_origin = float(panel.loc[origin, self.target_col])
        truf_origin = float(panel.loc[origin, self.truflation_col])
        if np.isnan(y_origin) or np.isnan(truf_origin):
            raise ValueError(
                f"PathA: feature missing at origin {origin}: "
                f"y={y_origin}, truf={truf_origin}")
        point = alpha + beta_y * y_origin + beta_t * truf_origin

        meta = {"baseline": "patha", "alpha": alpha, "beta_y": beta_y,
                  "beta_t": beta_t, "n_train": n_train,
                  "n_calib": n_calib, "band_source": band_source}

        if len(errors) < 10:
            return Forecast(origin=origin, target=target, point=point,
                              metadata=meta)
        lo80, hi80, lo95, hi95 = _bands_from_residuals(point, errors)
        meta["residual_sd"] = float(np.std(errors))
        return Forecast(
            origin=origin, target=target, point=point,
            lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
            samples=_samples(point, errors, self.n_samples,
                              self.seed + hash(origin) % 10_000),
            metadata=meta,
        )


def _rolling_oos_residuals_patha(all_train: pd.DataFrame,
                                    target_col: str,
                                    truflation_col: str,
                                    calib_months: int,
                                    train_min: int) -> np.ndarray:
    """Rolling-origin OOS residuals for Path A.

    For each calibration position ``c`` in the trailing ``calib_months``
    rows of ``all_train``, refit OLS on rows ``[0:c]`` and predict the
    target at row ``c``. Residual = actual - predicted. Returns the array
    of those residuals (one per position, may be shorter than
    ``calib_months`` if the warm-up requirement bites).
    """
    n_total = len(all_train)
    cal_start = max(n_total - calib_months, train_min)
    if cal_start >= n_total:
        return np.array([])

    residuals: list[float] = []
    for c in range(cal_start, n_total):
        tr = all_train.iloc[:c]
        Xtr, ytr = _patha_design(tr, target_col, truflation_col)
        cf, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        row = all_train.iloc[c]
        pred_c = (cf[0]
                    + cf[1] * row[target_col]
                    + cf[2] * row[truflation_col])
        residuals.append(float(row["__y_target"] - pred_c))
    return np.asarray(residuals)


@dataclass
class AR1Baseline:
    """Univariate AR(1) on ``target_col``.

    Estimated by OLS on lag-1: ``y[t] = α + φ y[t-1] + ε``. One-step
    prediction is ``α + φ y[origin]``.

    Bands controlled by ``band_method`` (see module docstring). Default
    is ``"rolling_conformal"`` — point coefficients fit on ALL lag-1
    pairs available at origin, bands derived from rolling-origin OOS
    residuals over the trailing ``calib_months`` positions.
    """
    target_col: str = "y"
    horizon: int = 1
    train_min: int = 24
    calib_months: int = 0
    band_method: BandMethod = "rolling_conformal"
    n_samples: int = DEFAULT_N_SAMPLES
    seed: int = 0
    model_id: str = "ar1_v1"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        s = panel[self.target_col].dropna()
        s = s.loc[s.index <= origin]
        eff_calib = (self.calib_months
                       if self.band_method != "in_sample" else 0)
        if len(s) < self.train_min + eff_calib:
            raise ValueError(
                f"AR(1): need ≥{self.train_min + eff_calib} obs "
                f"before {origin:%Y-%m-%d}, have {len(s)}")

        # AR(1) needs lag-1 pairs.
        y_arr = s.values
        x_lag_all = y_arr[:-1]
        y_target_all = y_arr[1:]

        if eff_calib > 0 and self.band_method == "split_conformal":
            split = len(x_lag_all) - eff_calib
            x_tr, y_tr = x_lag_all[:split], y_target_all[:split]
            X = np.column_stack([np.ones_like(x_tr), x_tr])
            coef, *_ = np.linalg.lstsq(X, y_tr, rcond=None)
            alpha, phi = float(coef[0]), float(coef[1])
            x_cal = x_lag_all[split:]
            y_cal = y_target_all[split:]
            errors = y_cal - (alpha + phi * x_cal)
            band_source = "split_conformal"
            n_train = len(x_tr)
            n_calib = len(x_cal)
        elif eff_calib > 0 and self.band_method == "rolling_conformal":
            # Final coefficients on ALL lag-1 pairs.
            X = np.column_stack([np.ones_like(x_lag_all), x_lag_all])
            coef, *_ = np.linalg.lstsq(X, y_target_all, rcond=None)
            alpha, phi = float(coef[0]), float(coef[1])
            errors = _rolling_oos_residuals_ar1(
                x_lag_all, y_target_all,
                calib_months=eff_calib, train_min=self.train_min)
            band_source = "rolling_conformal"
            n_train = len(x_lag_all)
            n_calib = len(errors)
        else:
            X = np.column_stack([np.ones_like(x_lag_all), x_lag_all])
            coef, *_ = np.linalg.lstsq(X, y_target_all, rcond=None)
            alpha, phi = float(coef[0]), float(coef[1])
            errors = y_target_all - (alpha + phi * x_lag_all)
            band_source = "in_sample"
            n_train = len(x_lag_all)
            n_calib = 0

        # Iterate horizon steps from the last observation
        last = float(s.iloc[-1])
        for _ in range(self.horizon):
            last = alpha + phi * last
        point = last

        meta = {"baseline": "ar1", "alpha": alpha, "phi": phi,
                  "n_train": n_train, "n_calib": n_calib,
                  "band_source": band_source}

        if len(errors) < 10:
            return Forecast(origin=origin, target=target, point=point,
                              metadata=meta)
        lo80, hi80, lo95, hi95 = _bands_from_residuals(point, errors)
        meta["residual_sd"] = float(np.std(errors))
        return Forecast(
            origin=origin, target=target, point=point,
            lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
            samples=_samples(point, errors, self.n_samples,
                              self.seed + hash(origin) % 10_000),
            metadata=meta,
        )


def _rolling_oos_residuals_ar1(x_lag: np.ndarray,
                                 y_target: np.ndarray,
                                 calib_months: int,
                                 train_min: int) -> np.ndarray:
    """Rolling-origin OOS residuals for AR(1).

    For each calibration position ``c`` in the trailing ``calib_months``
    rows, refit AR(1) on ``[0:c]``, predict ``y_target[c]`` from
    ``x_lag[c]``, record the residual.
    """
    n_total = len(x_lag)
    cal_start = max(n_total - calib_months, train_min)
    if cal_start >= n_total:
        return np.array([])

    residuals: list[float] = []
    for c in range(cal_start, n_total):
        Xtr = np.column_stack([np.ones(c), x_lag[:c]])
        cf, *_ = np.linalg.lstsq(Xtr, y_target[:c], rcond=None)
        pred_c = cf[0] + cf[1] * x_lag[c]
        residuals.append(float(y_target[c] - pred_c))
    return np.asarray(residuals)
