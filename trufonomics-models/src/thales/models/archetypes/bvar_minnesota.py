"""Phase 3.1 archetype — Bayesian VAR(p) with Minnesota prior.

The transmission-VAR backbone for the per-industry cost-structure
products. A small ``k``-dimensional VAR(p) on industry-relevant
variables (e.g. fuel, labor, maintenance, freight rate, margin,
volume) shrunk toward a univariate random-walk prior à la Litterman
1986. The Minnesota prior is a closed-form *conjugate* solution: the
posterior is multivariate-normal-Wishart with explicit moments, so no
MCMC is required and the model fits in milliseconds.

This module exposes:

  * ``minnesota_prior_diag(...)`` — build the prior precision diagonal
    given the standard hyperparameters (overall tightness λ, cross-
    variable shrinkage λ_cross, lag-decay λ_lag).
  * ``fit_bvar_minnesota(Y, p, hyperparams)`` — closed-form posterior
    mean of the AR coefficient matrix and Σ.
  * ``BVARFit`` — container with coefficients, residual covariance, IRFs.
  * ``cholesky_irf(coefs, sigma, h)`` — orthogonalized impulse responses
    via lower-Cholesky identification.
  * ``fevd(coefs, sigma, h)`` — forecast-error variance decomposition.
  * ``BVARForecaster`` — Forecaster-protocol wrapper for ``walk_forward``.

References:

  * Litterman 1986 — *Forecasting With Bayesian Vector Autoregressions—
    Five Years of Experience.* JBES.
  * Doan-Litterman-Sims 1984 — *Forecasting and Conditional Projection
    Using Realistic Prior Distributions.*
  * Bańbura-Giannone-Reichlin 2010 — *Large Bayesian Vector Auto
    Regressions.* JAE. (The "BGR" formulation we follow.)
  * Lütkepohl 2005 — *New Introduction to Multiple Time Series
    Analysis.* (For Cholesky IRF / FEVD definitions.)
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
from thales.evaluation.harness import Forecast


def _bands_from_residuals(point: float, errors: np.ndarray
                            ) -> tuple[float, float, float, float]:
    """Per-α conformal-or-Gaussian fallback (mirrors helper in baselines)."""
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


# ─── Prior construction ──────────────────────────────────────────────────


def minnesota_prior_diag(k: int, p: int, sigma_diag: np.ndarray,
                            overall_tightness: float = 0.2,
                            cross_tightness: float = 0.5,
                            lag_decay: float = 1.0,
                            ) -> np.ndarray:
    """Diagonal of the Minnesota prior precision for VAR(p) coefficients.

    For variable ``i`` and lag ``l`` of variable ``j`` in equation ``i``,
    the prior SD on the coefficient β_{ij,l} is:

        σ_{ij,l}  =  λ · (1/l^d) · σ_i^{1/2} / σ_j^{1/2}     if i ≠ j
        σ_{ii,l}  =  λ · (1/l^d)                              if i = j

    where ``λ = overall_tightness``, ``d = lag_decay``, and the
    cross-variable scaling uses ``cross_tightness < 1`` to shrink
    cross-variable effects more aggressively.

    Returns a flattened diagonal of length ``k · (k·p + 1)`` ordered as:
    ``[α_1_1, β_11_1, β_12_1, ..., β_1k_p, α_2_1, ...]`` (intercept
    first per equation, then lags).

    ``sigma_diag`` is ``(k,)`` — typically the diagonal of an OLS
    AR(1) residual covariance, used to scale coefficients across
    variables of different magnitudes.
    """
    if not (0 < overall_tightness < 5):
        raise ValueError("overall_tightness should be in (0, 5)")
    if not (0 < cross_tightness <= 1):
        raise ValueError("cross_tightness should be in (0, 1]")
    if lag_decay < 0:
        raise ValueError("lag_decay must be ≥ 0")

    # Intercept gets a very loose prior (huge SD ⇒ small precision)
    intercept_sd = 1e3
    sds: list[float] = []
    for i in range(k):
        sds.append(intercept_sd)    # intercept for equation i
        for l in range(1, p + 1):
            for j in range(k):
                lag_factor = (1.0 / l) ** lag_decay
                if i == j:
                    sd = overall_tightness * lag_factor
                else:
                    sd = (overall_tightness * cross_tightness * lag_factor
                            * np.sqrt(sigma_diag[i] / sigma_diag[j]))
                sds.append(sd)
    return np.asarray(sds, dtype=float)


def _sigma_diag_from_ar1(Y: np.ndarray) -> np.ndarray:
    """Per-variable AR(1) residual variance — used for prior scaling."""
    k = Y.shape[1]
    out = np.empty(k)
    for j in range(k):
        y = Y[:, j]
        x = np.column_stack([np.ones(len(y) - 1), y[:-1]])
        coef, *_ = np.linalg.lstsq(x, y[1:], rcond=None)
        resid = y[1:] - x @ coef
        out[j] = float(np.var(resid, ddof=2))
    return out


# ─── Posterior fit ───────────────────────────────────────────────────────


@dataclass
class BVARFit:
    """Posterior summary of a Bayesian VAR(p) with Minnesota prior."""
    p: int                                      # lag order
    k: int                                      # number of variables
    coefs: np.ndarray                           # (k, k*p + 1) — equation-by-row, [intercept, A_1, ..., A_p]
    sigma: np.ndarray                           # (k, k) residual covariance
    n_train: int                                # number of obs used after lagging
    log_marginal: float | None = None           # log marginal likelihood (optional)


def _build_design(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack lags into a design matrix.

    Returns ``(X, Y_target)`` where:
      * ``X`` is ``(T-p, k*p + 1)``: each row is [1, Y[t-1], ..., Y[t-p]]
      * ``Y_target`` is ``(T-p, k)``: rows are Y[t] for t = p..T-1
    """
    T, k = Y.shape
    if T <= p + 1:
        raise ValueError(f"need T > p+1; have T={T}, p={p}")
    rows = []
    for t in range(p, T):
        lags = [Y[t - l] for l in range(1, p + 1)]
        rows.append(np.concatenate([[1.0], np.concatenate(lags)]))
    X = np.asarray(rows)
    Y_target = Y[p:]
    return X, Y_target


def fit_bvar_minnesota(Y: np.ndarray, p: int = 1,
                          overall_tightness: float = 0.2,
                          cross_tightness: float = 0.5,
                          lag_decay: float = 1.0) -> BVARFit:
    """Fit a VAR(p) by ridge with Minnesota prior diagonal.

    Implementation: per-equation generalized ridge with the Minnesota
    diagonal as the regularization. Equivalent to the conjugate
    Normal-inverse-Wishart posterior mean when the prior covariance
    matrix is diagonal — the standard Bańbura-Giannone-Reichlin form.

    For each equation ``i`` (target = column ``i`` of Y_target):
        β̂_i  =  (X' X + Λ_i)^{-1}  (X' y_i + Λ_i β̄_i)

    where ``β̄_i`` is the prior mean: a random-walk prior, so β̄_i has
    a 1 on the own-lag-1 coefficient and 0 elsewhere. ``Λ_i`` is
    diag(1 / σ_{ij,l}^2) — the Minnesota prior precision diagonal.

    Σ is estimated as the empirical residual covariance of the
    posterior-mean coefficients.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be 2-D (T, k)")
    T, k = Y.shape
    if T < 4 * k:
        raise ValueError(
            f"insufficient observations: T={T} < 4·k={4*k}; "
            f"BVAR posterior is data-poor")

    X, Y_target = _build_design(Y, p)
    n_eff = X.shape[0]
    n_params = k * p + 1

    sigma_diag = _sigma_diag_from_ar1(Y)
    prior_sd = minnesota_prior_diag(
        k, p, sigma_diag,
        overall_tightness=overall_tightness,
        cross_tightness=cross_tightness,
        lag_decay=lag_decay)
    # prior_sd has length k * n_params; reshape to (k, n_params)
    prior_sd = prior_sd.reshape(k, n_params)

    # Random-walk prior mean: own-lag-1 = 1, all else = 0
    # Layout per equation: [intercept, lag-1 of var-0, ..., lag-1 of var-(k-1),
    #                         lag-2 of var-0, ..., lag-p of var-(k-1)]
    # So own-lag-1 coefficient for equation i is at position 1 + i.
    coefs = np.empty((k, n_params))
    XtX = X.T @ X
    for i in range(k):
        prior_prec = np.diag(1.0 / (prior_sd[i] ** 2))
        prior_mean = np.zeros(n_params)
        prior_mean[1 + i] = 1.0    # random-walk on own series
        rhs = X.T @ Y_target[:, i] + prior_prec @ prior_mean
        coefs[i] = np.linalg.solve(XtX + prior_prec, rhs)

    # Σ from posterior-mean residuals (BGR style: simple plug-in)
    resid = Y_target - X @ coefs.T
    sigma = (resid.T @ resid) / max(n_eff - n_params, 1)

    return BVARFit(p=p, k=k, coefs=coefs, sigma=sigma, n_train=n_eff)


# ─── IRF + FEVD ──────────────────────────────────────────────────────────


def _companion_matrix(A_list: list[np.ndarray]) -> np.ndarray:
    """Build the (k·p, k·p) companion matrix from a list of A_l matrices."""
    p = len(A_list)
    k = A_list[0].shape[0]
    F = np.zeros((k * p, k * p))
    F[:k] = np.hstack(A_list)
    if p > 1:
        F[k:, : k * (p - 1)] = np.eye(k * (p - 1))
    return F


def _ar_matrices(coefs: np.ndarray, k: int, p: int) -> list[np.ndarray]:
    """Extract list of A_l matrices from the equation-stacked coefficients.

    coefs is (k, k*p + 1); column 0 is the intercept, columns 1..k are
    A_1, columns k+1..2k are A_2, etc. A_l is (k, k) with A_l[i, j] =
    effect of lag-l of variable j on variable i.
    """
    A_list: list[np.ndarray] = []
    for l in range(p):
        cols = slice(1 + l * k, 1 + (l + 1) * k)
        A_list.append(coefs[:, cols])
    return A_list


def cholesky_irf(fit: BVARFit, h: int = 24,
                    cholesky_order: list[int] | None = None) -> np.ndarray:
    """Orthogonalized impulse responses via lower-Cholesky identification.

    Returns ``(h+1, k, k)`` array: ``irf[s, i, j]`` is the response of
    variable ``i`` at horizon ``s`` to a one-SD shock to variable ``j``.

    ``cholesky_order`` is a permutation of ``range(k)`` specifying the
    causal ordering: variable at position 0 is most exogenous. If
    None, uses identity ordering (variable 0 most exogenous).

    The Cholesky factorization gives the lower-triangular ``B`` such
    that ``B B' = Σ``. A unit shock to component ``j`` of ε_t is then
    a structural shock with covariance B B' = Σ.
    """
    k, p = fit.k, fit.p
    A_list = _ar_matrices(fit.coefs, k, p)
    F = _companion_matrix(A_list)

    sigma = fit.sigma
    if cholesky_order is not None:
        if sorted(cholesky_order) != list(range(k)):
            raise ValueError("cholesky_order must be a permutation of range(k)")
        P = np.eye(k)[cholesky_order]
        sigma_p = P @ sigma @ P.T
        L_p = np.linalg.cholesky(sigma_p)
        L = P.T @ L_p @ P
    else:
        L = np.linalg.cholesky(sigma)

    # Pad shock matrix into companion-state space
    L_pad = np.zeros((k * p, k))
    L_pad[:k] = L

    irf = np.empty((h + 1, k, k))
    state = L_pad
    irf[0] = state[:k]
    for s in range(1, h + 1):
        state = F @ state
        irf[s] = state[:k]
    return irf


def shock_scenario(fit: BVARFit, baseline: np.ndarray,
                       shock_var_idx: int, shock_size: float, h: int,
                       cholesky_order: list[int] | None = None,
                       ) -> np.ndarray:
    """Project the trajectory of all variables h steps ahead given a
    one-time **structural** shock to ``shock_var_idx`` of size
    ``shock_size`` (in the same units as the variable, after Cholesky
    decomposition).

    Different from ``conditional_forecast``: this engages the
    **contemporaneous** transmission via Σ (the Cholesky factor),
    which is the dominant channel on monthly data. Use this for
    "+20% diesel shock — what happens to freight, labor, maintenance,
    volume?" scenarios.

    Returns ``(h+1, k)`` — the deterministic deviation from baseline
    at each horizon, where row 0 is the impact at the moment of the
    shock and row h is at horizon h.

    The deviation comes from the IRF re-scaled to match the requested
    shock magnitude. For a stable VAR the deviations decay to 0 with h.

    ``baseline`` is unused for the deviation calculation — the IRF
    is already a deviation from a no-shock counterfactual — but it's
    accepted for symmetry with `conditional_forecast` and to support
    an absolute-level convenience output by the caller.

    A shock of size ``s`` (e.g. s = log(1.20) = 0.182 for +20%) maps
    to a unit-vector structural shock of magnitude ``s / L[i, i]``
    where ``L`` is the Cholesky factor; the IRF column is then
    multiplied by this scalar.
    """
    irf = cholesky_irf(fit, h=h, cholesky_order=cholesky_order)
    # IRF is (h+1, k, k); irf[s, i, j] = response of i at horizon s to
    # 1-SD shock to j. Need to scale the SD-1 shock to produce ``shock_size``
    # of contemporaneous response in variable ``shock_var_idx``.
    # By construction, irf[0, j, j] = L[j, j] = sqrt(Σ[j, j]) (under
    # default ordering), so the scale factor is shock_size / irf[0, j, j].
    own_h0 = irf[0, shock_var_idx, shock_var_idx]
    if abs(own_h0) < 1e-12:
        raise ValueError(
            f"variable {shock_var_idx} has zero own-shock response at h=0; "
            f"check Cholesky ordering")
    scale = shock_size / own_h0
    return irf[:, :, shock_var_idx] * scale


def conditional_forecast(fit: BVARFit, history: np.ndarray,
                              forced_paths: dict[int, np.ndarray],
                              h: int,
                              n_samples: int = 1000,
                              seed: int | None = None,
                              ) -> dict[str, np.ndarray]:
    """Conditional forecast: project the BVAR ``h`` steps forward, holding
    a subset of variables on a forced path each step.

    Use case: "given this oil-futures curve over the next 12 months,
    what's the distribution of freight rates / labor costs?"

    Implementation: Monte-Carlo. For each of ``n_samples`` draws,
    iterate the VAR forward ``h`` steps. At each step:
      1. Compute the unconditional one-step forecast from the previous
         state.
      2. Add a Cholesky-decomposed Gaussian shock.
      3. **Override** the forced variables to their prescribed values
         (treats those values as known, not as model output).
    Aggregating over draws yields the conditional distribution of the
    free variables.

    This is the simple-and-honest variant — no Doan-Litterman-Sims
    "tunes" weighting. For the product use case (futures curves), the
    forced path is exact and DLS tunes are not needed.

    Parameters
    ----------
    fit : BVARFit
    history : (k*p, ) or (T_hist, k) — ``p`` most recent observations.
        If 2-D, the last ``p`` rows are used.
    forced_paths : ``{var_index: np.ndarray of shape (h,)}``
        Forced path for each conditioned variable. The forecast horizon
        for free variables is also ``h``.
    h : forecast horizon
    n_samples : number of Monte-Carlo draws
    seed : optional RNG seed

    Returns
    -------
    dict with:
      * ``"mean"`` — (h, k) — conditional mean trajectory
      * ``"q05"``, ``"q25"``, ``"q50"``, ``"q75"``, ``"q95"`` —
        per-variable, per-horizon quantiles, each (h, k)
      * ``"samples"`` — (n_samples, h, k) — full sample paths
    """
    rng = np.random.default_rng(seed)
    k, p = fit.k, fit.p

    if history.ndim == 1:
        if history.size != k * p:
            raise ValueError(
                f"flat history must have size k*p={k*p}; got {history.size}")
        last_p = history.reshape(p, k)
    else:
        if history.shape[1] != k:
            raise ValueError(
                f"history must have {k} columns; got {history.shape[1]}")
        if history.shape[0] < p:
            raise ValueError(f"need ≥{p} rows of history; got {history.shape[0]}")
        last_p = history[-p:]

    # Validate forced_paths
    for var_idx, path in forced_paths.items():
        if not (0 <= var_idx < k):
            raise ValueError(f"forced var_index {var_idx} out of range [0, {k})")
        if len(path) != h:
            raise ValueError(
                f"forced path for var {var_idx} must have length h={h}; "
                f"got {len(path)}")
    forced_idx = sorted(forced_paths.keys())
    free_idx = [i for i in range(k) if i not in forced_paths]

    A_list = _ar_matrices(fit.coefs, k, p)
    intercept = fit.coefs[:, 0]
    L = np.linalg.cholesky(fit.sigma)

    # Deterministic conditional projection: iterate with no shocks,
    # forcing the conditioned variables. This is the closed-form
    # conditional mean trajectory (Doan-Litterman-Sims 1984 style).
    deterministic = np.empty((h, k))
    state_d = [last_p[-1 - i].copy() for i in range(p)]
    for t in range(h):
        y_next = intercept.copy()
        for l in range(p):
            y_next = y_next + A_list[l] @ state_d[l]
        for var_idx, path in forced_paths.items():
            y_next[var_idx] = path[t]
        deterministic[t] = y_next
        state_d = [y_next] + state_d[:-1]

    # Stochastic conditional projection: same loop, with Cholesky shocks.
    # Distribution around the deterministic trajectory; quantiles come
    # from this.
    samples = np.empty((n_samples, h, k))
    for s in range(n_samples):
        state = [last_p[-1 - i].copy() for i in range(p)]
        for t in range(h):
            y_next = intercept.copy()
            for l in range(p):
                y_next = y_next + A_list[l] @ state[l]
            eps = L @ rng.normal(0, 1, k)
            y_next = y_next + eps
            for var_idx, path in forced_paths.items():
                y_next[var_idx] = path[t]
            samples[s, t] = y_next
            state = [y_next] + state[:-1]

    return {
        "mean": deterministic,                              # no MC noise
        "stochastic_mean": samples.mean(axis=0),            # MC mean (≈ deterministic at large n)
        "q05": np.quantile(samples, 0.05, axis=0),
        "q25": np.quantile(samples, 0.25, axis=0),
        "q50": np.quantile(samples, 0.50, axis=0),
        "q75": np.quantile(samples, 0.75, axis=0),
        "q95": np.quantile(samples, 0.95, axis=0),
        "samples": samples,
    }


def fevd(fit: BVARFit, h: int = 24,
            cholesky_order: list[int] | None = None) -> np.ndarray:
    """Forecast-error variance decomposition.

    Returns ``(h+1, k, k)`` array: ``fevd[s, i, j]`` is the share of
    forecast-error variance in variable ``i`` at horizon ``s``
    attributable to (Cholesky) shock ``j``. Each row sums to 1.
    """
    irf = cholesky_irf(fit, h=h, cholesky_order=cholesky_order)
    sq = irf ** 2
    cum = np.cumsum(sq, axis=0)
    total = cum.sum(axis=2, keepdims=True)
    return cum / np.where(total > 0, total, 1.0)


# ─── Forecaster protocol wrapper ────────────────────────────────────────


@dataclass
class BVARForecaster:
    """BVAR(p) wrapped as a single-target Forecaster.

    Operates on a panel containing ``var_cols`` (the k-dim vector) and
    forecasts ``target_col`` at ``horizon`` steps ahead. Iterates the
    BVAR forward by ``horizon`` periods using the posterior-mean
    coefficients (no posterior sampling — point forecast only).

    Bands controlled by ``band_method``:

      * ``"gaussian"`` (legacy default for back-compat) — h-step
        Gaussian forecast SD from MA representation. Fast and standard,
        but assumes Gaussian residuals; can systematically miscover.
      * ``"rolling_conformal"`` — rolling-origin OOS residuals on the
        target variable over the trailing ``calib_months`` positions,
        then finite-sample conformal quantiles (Vovk-Lei-Tibshirani).
        Per-α fallback to Gaussian if calibration set is too small for
        the requested coverage.
    """
    var_cols: list[str]
    target_col: str
    horizon: int = 1
    p: int = 1
    overall_tightness: float = 0.2
    cross_tightness: float = 0.5
    lag_decay: float = 1.0
    train_min: int = 60
    train_window: int | None = None
    band_method: Literal["gaussian", "rolling_conformal"] = "gaussian"
    calib_months: int = 24
    model_id: str = "bvar_minnesota_v1"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        if self.target_col not in self.var_cols:
            raise ValueError(
                f"target_col '{self.target_col}' must be in var_cols")
        target_idx = self.var_cols.index(self.target_col)

        data = panel[self.var_cols].copy()
        data = data.loc[data.index <= origin].dropna()
        if self.train_window:
            data = data.iloc[-self.train_window:]
        if len(data) < self.train_min:
            raise ValueError(
                f"BVAR: need ≥{self.train_min} obs; have {len(data)}")

        Y = data.values
        fit = fit_bvar_minnesota(
            Y, p=self.p,
            overall_tightness=self.overall_tightness,
            cross_tightness=self.cross_tightness,
            lag_decay=self.lag_decay)

        # Iterate forward `horizon` periods
        A_list = _ar_matrices(fit.coefs, fit.k, fit.p)
        intercept = fit.coefs[:, 0]
        last_p = Y[-fit.p:][::-1]    # most recent first
        state = list(last_p)
        for _ in range(self.horizon):
            y_next = intercept.copy()
            for l in range(fit.p):
                y_next = y_next + A_list[l] @ state[l]
            state = [y_next] + state[:-1]
        point = float(state[0][target_idx])

        # ── Bands ───────────────────────────────────────────────
        meta_band: dict = {}
        if self.band_method == "rolling_conformal":
            cal_residuals = self._rolling_oos_target_residuals(
                Y, target_idx)
            if len(cal_residuals) >= 9:
                lo80, hi80, lo95, hi95 = _bands_from_residuals(
                    point, cal_residuals)
                meta_band["band_source"] = "rolling_conformal"
                meta_band["n_calib"] = int(len(cal_residuals))
            else:
                lo80, hi80, lo95, hi95 = self._gaussian_bands(
                    fit, A_list, target_idx, point)
                meta_band["band_source"] = "gaussian_fallback_n_too_small"
        else:
            lo80, hi80, lo95, hi95 = self._gaussian_bands(
                fit, A_list, target_idx, point)
            meta_band["band_source"] = "gaussian"

        return Forecast(
            origin=origin, target=target, point=point,
            lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
            metadata={
                "model": "bvar_minnesota",
                "k": fit.k, "p": fit.p,
                "n_train": fit.n_train,
                "horizon": self.horizon,
                "intercept": intercept.tolist(),
                **meta_band,
            },
        )

    def _gaussian_bands(self, fit, A_list, target_idx: int, point: float
                            ) -> tuple[float, float, float, float]:
        """h-step Gaussian forecast SD from MA representation."""
        F = _companion_matrix(A_list)
        L = np.linalg.cholesky(fit.sigma)
        L_pad = np.zeros((fit.k * fit.p, fit.k))
        L_pad[:fit.k] = L
        var_acc = np.zeros((fit.k, fit.k))
        psi = L_pad
        for s in range(self.horizon):
            top = psi[:fit.k]
            var_acc = var_acc + top @ top.T
            psi = F @ psi
        sigma_h = float(np.sqrt(var_acc[target_idx, target_idx]))
        return (point - 1.2816 * sigma_h, point + 1.2816 * sigma_h,
                point - 1.96 * sigma_h,   point + 1.96 * sigma_h)

    def _rolling_oos_target_residuals(self, Y: np.ndarray,
                                          target_idx: int) -> np.ndarray:
        """Rolling-origin OOS residuals on the target column.

        For each calibration position c in the trailing ``calib_months``
        rows, refit the BVAR on rows [0:c], iterate `horizon` periods
        forward, and record (actual - predicted) for the target column
        at position c+horizon-1 (one-step iterated for h>1).
        Used for conformal calibration of the band on the target.
        """
        T_obs = Y.shape[0]
        cal_start = max(T_obs - self.calib_months, self.train_min)
        h = self.horizon
        if cal_start + h > T_obs:
            return np.array([])

        residuals: list[float] = []
        for c in range(cal_start, T_obs - h + 1):
            tr = Y[:c]
            try:
                fit_c = fit_bvar_minnesota(
                    tr, p=self.p,
                    overall_tightness=self.overall_tightness,
                    cross_tightness=self.cross_tightness,
                    lag_decay=self.lag_decay)
            except ValueError:
                continue
            A_list_c = _ar_matrices(fit_c.coefs, fit_c.k, fit_c.p)
            intercept_c = fit_c.coefs[:, 0]
            last_p_c = tr[-fit_c.p:][::-1]
            state_c = list(last_p_c)
            for _ in range(h):
                y_next = intercept_c.copy()
                for l in range(fit_c.p):
                    y_next = y_next + A_list_c[l] @ state_c[l]
                state_c = [y_next] + state_c[:-1]
            pred = float(state_c[0][target_idx])
            actual = float(Y[c + h - 1, target_idx])
            residuals.append(actual - pred)
        return np.asarray(residuals)
