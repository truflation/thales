"""Johansen-gated bivariate VECM with fallback — Fix #4.

Resolves user feedback: "Implement Johansen as a VECM gate, with
fallback. The premise: only run VECM if cointegration is detected;
otherwise fall back to ARDL / bridge / TVP."

The original Phase 1.4 ``vecm.py`` archetype assumed β = (1, −1) was
known and fit by per-equation OLS. That's correct when theory says the
spread is stationary (Truflation Clothing × BLS Apparel) — but when
the cointegrating relationship breaks (regime shift, methodology
divergence, structural break), forcing a VECM produces biased
forecasts that drift toward the assumed equilibrium that no longer
exists.

This module wraps the VECM in a Johansen pre-test:

  * Run ``coint_johansen`` (Johansen 1991, MacKinnon-Haug-Michelis 1999
    critical values) on the bivariate series at each origin.
  * If trace_stat[r=0] > CV(α): cointegration detected, fit VECM.
  * Otherwise: fall back to ``fallback ∈ {"ardl", "bridge", "ar1"}``.

The fallback is selected once per construction (not per origin) — the
gate just decides "VECM vs not." This keeps the prediction pipeline
deterministic and the bands well-defined per band-source category.

Bands use the production rolling-conformal pipeline (Fix #1/1b/1c).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from thales.evaluation.conformal import conformal_band_offsets, min_n_for_alpha
from thales.evaluation.harness import Forecast


FallbackMethod = Literal["ardl", "bridge", "ar1"]
BandMethod = Literal["gaussian", "in_sample", "rolling_conformal"]


def johansen_test(y1: np.ndarray, y2: np.ndarray,
                     k_ar_diff: int = 1, det_order: int = 0,
                     significance_level: float = 0.05) -> dict:
    """Bivariate Johansen trace test for cointegration rank.

    Returns a dict with:
      * ``trace_stat`` — array of two trace statistics for H_0:r=0 and r≤1
      * ``cv`` — array of two critical values at the chosen significance
      * ``eigenvalues`` — array of two ordered eigenvalues
      * ``cointegrated`` — bool, True iff trace_stat[0] > cv[0] (reject r=0)
      * ``rank`` — best rank estimate (0, 1, or 2) based on the trace
        sequence

    ``det_order=0`` includes constant only; ``k_ar_diff=1`` uses one
    difference lag (the most common choice for monthly economic data).
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    df = np.column_stack([y1, y2])
    if not (significance_level in (0.10, 0.05, 0.01)):
        raise ValueError("significance_level must be 0.10, 0.05, or 0.01")
    cv_idx = {0.10: 0, 0.05: 1, 0.01: 2}[significance_level]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res = coint_johansen(df, det_order=det_order, k_ar_diff=k_ar_diff)

    trace = np.asarray(res.lr1)
    cv = np.asarray(res.cvt[:, cv_idx])
    eigs = np.asarray(res.eig)

    # Sequential test: r=0 vs r=1, then r=1 vs r=2.
    rank = 0
    for r in range(len(trace)):
        if trace[r] > cv[r]:
            rank = r + 1
        else:
            break

    return {
        "trace_stat": trace,
        "cv": cv,
        "eigenvalues": eigs,
        "cointegrated": bool(trace[0] > cv[0]),
        "rank": rank,
    }


def _bands_from_residuals(point: float, errors: np.ndarray
                            ) -> tuple[float, float, float, float]:
    """Per-α conformal-or-Gaussian fallback. Mirror of helper in
    ``baselines.py``/``same_month_nowcaster.py`` — kept local to avoid
    a cross-module import for one helper."""
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
class JohansenGatedVECM:
    """Johansen-gated bivariate VECM with three-way fallback.

    Forecasts ``target_col[T+h]`` from a bivariate panel containing
    ``target_col`` and ``paired_col``. Both are assumed to be level
    series (e.g. log-prices). At each origin:

      1. Run Johansen on the trailing training window.
      2. If cointegrated: fit VECM with β=(1,-1) (known) and predict
         via the error-correction equation.
      3. Else: fall back to one of:

         * ``"ardl"`` — Δy_target[t] = c + α y_target[t-1] + β y_paired[t-1]
                          + γ Δy_paired[t] + ε. The unconstrained ARDL
                          form (no error-correction).
         * ``"bridge"`` — y_target[t] = α + β y_paired[t] + ε.
                          Pure contemporaneous regression.
         * ``"ar1"`` — y_target[t] = α + φ y_target[t-1] + ε.
                          Univariate AR(1) — most conservative.

    Bands from rolling-conformal OOS residuals (Fix #1c).
    """
    target_col: str = "y1"
    paired_col: str = "y2"
    horizon: int = 1
    train_window_months: int = 60
    train_min: int = 36
    significance_level: float = 0.05
    fallback: FallbackMethod = "ardl"
    band_method: BandMethod = "rolling_conformal"
    calib_months: int = 24
    model_id: str = "johansen_gated_vecm_v1"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        if self.target_col not in panel.columns:
            raise ValueError(f"target_col '{self.target_col}' not in panel")
        if self.paired_col not in panel.columns:
            raise ValueError(f"paired_col '{self.paired_col}' not in panel")

        data = panel[[self.target_col, self.paired_col]].copy().dropna()
        train = data.loc[data.index <= origin]
        if self.train_window_months and len(train) > self.train_window_months:
            train = train.iloc[-self.train_window_months:]
        if len(train) < self.train_min:
            raise ValueError(
                f"johansen-vecm: need ≥{self.train_min} obs at origin "
                f"{origin}; have {len(train)}")

        y1 = train[self.target_col].values
        y2 = train[self.paired_col].values

        # ── Step 1: Johansen test on training window ─────────────────
        try:
            j = johansen_test(y1, y2,
                                  significance_level=self.significance_level)
        except Exception as e:    # noqa: BLE001
            j = {"cointegrated": False, "rank": 0,
                 "trace_stat": np.array([np.nan, np.nan]),
                 "cv": np.array([np.nan, np.nan]),
                 "eigenvalues": np.array([np.nan, np.nan]),
                 "test_error": f"{type(e).__name__}: {e}"}

        # ── Step 2: Branch on cointegration result ───────────────────
        if j["cointegrated"]:
            point, predict_one_h, branch = self._fit_vecm_branch(
                train, y1, y2)
        else:
            point, predict_one_h, branch = self._fit_fallback_branch(
                train, y1, y2)

        # ── Step 3: Rolling-conformal calibration ────────────────────
        meta_band: dict = {}
        if self.band_method == "rolling_conformal":
            cal_residuals = self._rolling_oos_residuals(train, branch)
            if len(cal_residuals) >= 9:
                lo80, hi80, lo95, hi95 = _bands_from_residuals(
                    point, cal_residuals)
                meta_band["band_source"] = "rolling_conformal"
                meta_band["n_calib"] = int(len(cal_residuals))
            else:
                fitted_resid = self._in_sample_residuals(train, branch)
                lo80, hi80, lo95, hi95 = _bands_from_residuals(
                    point, fitted_resid)
                meta_band["band_source"] = "in_sample_fallback_calib_too_small"
        elif self.band_method == "in_sample":
            fitted_resid = self._in_sample_residuals(train, branch)
            lo80, hi80, lo95, hi95 = _bands_from_residuals(
                point, fitted_resid)
            meta_band["band_source"] = "in_sample_conformal"
        else:    # gaussian
            fitted_resid = self._in_sample_residuals(train, branch)
            sigma = float(np.std(fitted_resid))
            lo80, hi80 = point - 1.2816 * sigma, point + 1.2816 * sigma
            lo95, hi95 = point - 1.96 * sigma, point + 1.96 * sigma
            meta_band["band_source"] = "gaussian"

        return Forecast(
            origin=origin, target=target, point=point,
            lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
            metadata={
                "model": "johansen_gated_vecm",
                "branch": branch,
                "n_train": len(train),
                "trace_stat_r0": float(j["trace_stat"][0])
                                       if not np.isnan(j["trace_stat"][0]) else None,
                "trace_cv_r0": float(j["cv"][0])
                                    if not np.isnan(j["cv"][0]) else None,
                "cointegrated": bool(j["cointegrated"]),
                "rank": int(j["rank"]),
                **meta_band,
            },
        )

    # ── VECM branch ──────────────────────────────────────────────────

    def _fit_vecm_branch(self, train, y1, y2) -> tuple[float, callable, str]:
        """Fit VECM with known β=(1,-1). Returns (point, _predict_h, branch)."""
        z = y1 - y2
        dy1 = np.diff(y1)
        z_lag = z[:-1]
        X = np.column_stack([z_lag, np.ones_like(z_lag)])
        coefs, *_ = np.linalg.lstsq(X, dy1, rcond=None)
        alpha_1, c_1 = float(coefs[0]), float(coefs[1])

        # Single-step prediction; iterate ``horizon`` times if h > 1.
        y1_T = float(y1[-1])
        y2_T = float(y2[-1])
        # We don't have a model for y2 here — assume y2 stays at last value
        # over the horizon (random-walk approx). This is the standard VECM
        # forecast for the partner under the bivariate spec.
        y1_h, y2_h = y1_T, y2_T
        for _ in range(self.horizon):
            z_t = y1_h - y2_h
            dy1_pred = alpha_1 * z_t + c_1
            y1_h = y1_h + dy1_pred
        point = y1_h

        def _predict_for_calib(prev_y1, prev_y2):
            """One-step VECM prediction given lagged values."""
            return prev_y1 + alpha_1 * (prev_y1 - prev_y2) + c_1

        # Stash the predictor for residual computation
        self._vecm_predictor = _predict_for_calib
        self._vecm_alpha = alpha_1
        self._vecm_c = c_1
        return point, _predict_for_calib, "vecm"

    # ── Fallback branches ────────────────────────────────────────────

    def _fit_fallback_branch(self, train, y1, y2) -> tuple[float, callable, str]:
        if self.fallback == "ardl":
            return self._fit_ardl(y1, y2)
        elif self.fallback == "bridge":
            return self._fit_bridge(train, y1, y2)
        else:    # ar1
            return self._fit_ar1(y1)

    def _fit_ardl(self, y1, y2) -> tuple[float, callable, str]:
        """ARDL: Δy_1[t] = c + α y_1[t-1] + β y_2[t-1] + γ Δy_2[t] + ε."""
        dy1 = np.diff(y1)
        dy2 = np.diff(y2)
        y1_lag = y1[:-1]
        y2_lag = y2[:-1]
        X = np.column_stack([y1_lag, y2_lag, dy2, np.ones_like(y1_lag)])
        coefs, *_ = np.linalg.lstsq(X, dy1, rcond=None)
        a, b, c, intercept = (float(coefs[0]), float(coefs[1]),
                                  float(coefs[2]), float(coefs[3]))

        y1_T = float(y1[-1])
        y2_T = float(y2[-1])
        # Assume Δy_2 over horizon = 0 (no model for y2 dynamics)
        y1_h = y1_T
        for _ in range(self.horizon):
            dy1_pred = intercept + a * y1_h + b * y2_T + c * 0.0
            y1_h = y1_h + dy1_pred
        point = y1_h

        def _predict_for_calib(prev_y1, prev_y2, dy2_t):
            dy1_pred = intercept + a * prev_y1 + b * prev_y2 + c * dy2_t
            return prev_y1 + dy1_pred

        self._ardl = (a, b, c, intercept)
        return point, _predict_for_calib, "ardl"

    def _fit_bridge(self, train, y1, y2) -> tuple[float, callable, str]:
        """Bridge: y_1[t] = α + β y_2[t] + ε. Contemporaneous regression."""
        X = np.column_stack([np.ones_like(y2), y2])
        coefs, *_ = np.linalg.lstsq(X, y1, rcond=None)
        intercept, beta = float(coefs[0]), float(coefs[1])

        # For h-step ahead prediction, we need y_2[T+h]. Without a y_2
        # model, assume y_2[T+h] = y_2[T] (random-walk approx).
        y2_T = float(y2[-1])
        point = intercept + beta * y2_T

        def _predict_for_calib(y2_t):
            return intercept + beta * y2_t

        self._bridge = (intercept, beta)
        return point, _predict_for_calib, "bridge"

    def _fit_ar1(self, y1) -> tuple[float, callable, str]:
        """AR(1) on target only: y_1[t] = α + φ y_1[t-1] + ε."""
        y_lag = y1[:-1]
        y_target = y1[1:]
        X = np.column_stack([np.ones_like(y_lag), y_lag])
        coefs, *_ = np.linalg.lstsq(X, y_target, rcond=None)
        alpha, phi = float(coefs[0]), float(coefs[1])
        y1_h = float(y1[-1])
        for _ in range(self.horizon):
            y1_h = alpha + phi * y1_h
        point = y1_h

        def _predict_for_calib(prev_y1):
            return alpha + phi * prev_y1

        self._ar1 = (alpha, phi)
        return point, _predict_for_calib, "ar1"

    # ── Residual computation helpers ────────────────────────────────

    def _in_sample_residuals(self, train, branch) -> np.ndarray:
        """One-step in-sample residuals for the chosen branch."""
        y1 = train[self.target_col].values
        y2 = train[self.paired_col].values
        if branch == "vecm":
            preds = np.array([self._vecm_predictor(y1[t-1], y2[t-1])
                                  for t in range(1, len(y1))])
            actuals = y1[1:]
        elif branch == "ardl":
            a, b, c, intercept = self._ardl
            preds, actuals = [], []
            for t in range(1, len(y1)):
                dy2 = y2[t] - y2[t-1]
                preds.append(y1[t-1] + intercept + a*y1[t-1] + b*y2[t-1] + c*dy2)
                actuals.append(y1[t])
            preds = np.array(preds)
            actuals = np.array(actuals)
        elif branch == "bridge":
            intercept, beta = self._bridge
            preds = intercept + beta * y2
            actuals = y1
        else:    # ar1
            alpha, phi = self._ar1
            preds = alpha + phi * y1[:-1]
            actuals = y1[1:]
        return np.asarray(actuals) - np.asarray(preds)

    def _rolling_oos_residuals(self, train, branch) -> np.ndarray:
        """Rolling-origin OOS residuals for the chosen branch.

        For each calibration position c in the trailing ``calib_months``
        rows, refit the same branch on rows [0:c], predict y_target[c]
        from y[c-1], record signed residual.
        """
        n_total = len(train)
        cal_start = max(n_total - self.calib_months, self.train_min)
        if cal_start >= n_total:
            return np.array([])
        residuals: list[float] = []
        for c in range(cal_start, n_total):
            tr = train.iloc[:c]
            y1_tr = tr[self.target_col].values
            y2_tr = tr[self.paired_col].values

            # Refit the branch (without re-running Johansen — we trust
            # the origin-time gate decision and ask "given this branch,
            # what would the residuals look like rolling forward?")
            if branch == "vecm":
                _, _, _ = self._fit_vecm_branch(tr, y1_tr, y2_tr)
                pred_c = self._vecm_predictor(y1_tr[-1], y2_tr[-1])
            elif branch == "ardl":
                _, _, _ = self._fit_ardl(y1_tr, y2_tr)
                a, b, cgam, intercept = self._ardl
                dy2_c = train[self.paired_col].iloc[c] - y2_tr[-1]
                pred_c = (y1_tr[-1] + intercept + a * y1_tr[-1]
                            + b * y2_tr[-1] + cgam * dy2_c)
            elif branch == "bridge":
                _, _, _ = self._fit_bridge(tr, y1_tr, y2_tr)
                intercept, beta = self._bridge
                pred_c = intercept + beta * train[self.paired_col].iloc[c]
            else:    # ar1
                _, _, _ = self._fit_ar1(y1_tr)
                alpha, phi = self._ar1
                pred_c = alpha + phi * y1_tr[-1]

            actual_c = float(train[self.target_col].iloc[c])
            residuals.append(actual_c - pred_c)

        # Restore final-train branch state (we mutated self._* above)
        if branch == "vecm":
            self._fit_vecm_branch(train, train[self.target_col].values,
                                          train[self.paired_col].values)
        elif branch == "ardl":
            self._fit_ardl(train[self.target_col].values,
                              train[self.paired_col].values)
        elif branch == "bridge":
            self._fit_bridge(train, train[self.target_col].values,
                                 train[self.paired_col].values)
        else:
            self._fit_ar1(train[self.target_col].values)

        return np.asarray(residuals)
