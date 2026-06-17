"""Calibrated, time-varying-scale density layer.

The expanding-window residual bootstrap in :mod:`thales.evaluation.density`
characterizes the predictive distribution by resampling *all* prior out-of-
sample errors. That is correct but over-wide: a volatile old regime (COVID, the
2022 surge) keeps inflating the band long after volatility has subsided, so
empirical coverage runs hot (85-98 % at a nominal 80 %).

This module is the calibration fix. The predictive distribution is built as::

    y_hat ~ point + mu_w + scale_t * z

where

  * ``mu_w``    — mean of the recent residual window (a bias correction; the
                  band centers on ``point + mu_w``);
  * ``scale_t`` — a *recent-window* or EWMA estimate of residual dispersion, so
                  the width tracks the current volatility regime;
  * ``z``       — unit-variance draws from the **standardized** recent residuals
                  (bootstrap, fat tails preserved) or a Student-t fallback.

When ``halflife`` is ``None`` and the window spans the whole history this
reduces exactly to ``point + bootstrap(residuals)`` — the same object as
:func:`thales.evaluation.density.samples_from_residuals`. The value-add is the
recent-window / EWMA scale, which is what restores calibration.

Score the resulting samples with :func:`thales.evaluation.density.score_density`
(CRPS, PIT-KS, coverage, sharpness) — this module only *builds* the samples.

References:
    Engle 1982 / Bollerslev 1986 — time-varying conditional variance.
    Stock, Watson 2007 — unobserved-components SV for inflation.
    Lei et al. 2018 — distribution-free predictive inference (the residual-
        bootstrap band this generalizes).
"""
from __future__ import annotations

import numpy as np

DEFAULT_N_SAMPLES = 2000
DEFAULT_WINDOW = 36
MIN_RESID = 8  # need enough residuals to characterize a distribution


def _clean(residuals: np.ndarray) -> np.ndarray:
    r = np.asarray(residuals, dtype=float)
    return r[np.isfinite(r)]


def rolling_scale(residuals: np.ndarray,
                  window: int = DEFAULT_WINDOW,
                  halflife: float | None = None) -> float:
    """Recent-window dispersion of signed residuals (the predictive scale).

    Uses the de-meaned residuals over the last ``window`` observations. With
    ``halflife`` set, weights are ``0.5 ** (age / halflife)`` (recent residuals
    count more) — an EWMA standard deviation; otherwise a flat-window
    population standard deviation.

    Returns ``nan`` if fewer than two usable residuals are available.
    """
    r = _clean(residuals)
    if len(r) < 2:
        return float("nan")
    if window and len(r) > window:
        r = r[-window:]
    d2 = (r - r.mean()) ** 2
    if halflife:
        age = np.arange(len(r))[::-1]          # 0 == most recent (residuals are chronological)
        w = 0.5 ** (age / halflife)
        w = w / w.sum()
        return float(np.sqrt(np.sum(w * d2)))
    return float(np.sqrt(np.mean(d2)))


def calibrated_samples(point: float,
                       residuals: np.ndarray,
                       *,
                       n_samples: int = DEFAULT_N_SAMPLES,
                       window: int = DEFAULT_WINDOW,
                       halflife: float | None = None,
                       df: float | None = None,
                       seed: int = 0) -> np.ndarray:
    """Predictive samples ``point + mu_w + scale_t * z``.

    Parameters
    ----------
    point : central forecast.
    residuals : chronological signed OOS errors (``actual - forecast``); only
        prior residuals should be passed (leak-free is the caller's job).
    window : recent-window length for the bias/scale estimate.
    halflife : if set, EWMA-weight the scale toward recent residuals.
    df : if set (``> 2``), draw the shape from a unit-variance Student-t with
        ``df`` degrees of freedom instead of bootstrapping (short-history /
        explicit-fat-tail fallback).

    Returns an ``(n_samples,)`` array, or all-NaN if there are fewer than
    :data:`MIN_RESID` usable residuals or the scale is degenerate.
    """
    r = _clean(residuals)
    if len(r) < MIN_RESID:
        return np.full(n_samples, np.nan)
    rw = r[-window:] if window and len(r) > window else r
    mu = float(rw.mean())
    scale = rolling_scale(r, window=window, halflife=halflife)
    if not np.isfinite(scale) or scale <= 0:
        return np.full(n_samples, np.nan)

    rng = np.random.default_rng(seed)
    if df is not None and df > 2:
        z = rng.standard_t(df, size=n_samples) * np.sqrt((df - 2) / df)  # unit variance
    else:
        sd = rw.std()
        z = (rng.choice(rw, size=n_samples, replace=True) - mu) / (sd if sd > 0 else 1.0)
    return point + mu + scale * z


def predictive_quantiles(samples: np.ndarray, levels) -> np.ndarray:
    """Predictive quantiles at ``levels`` (percentages in [0, 100]).

    Monotone by construction (``np.percentile``). Returns NaNs if the sample
    vector is empty or all-NaN.
    """
    s = np.asarray(samples, dtype=float)
    s = s[~np.isnan(s)]
    levels = np.atleast_1d(levels)
    if len(s) == 0:
        return np.full(len(levels), np.nan)
    return np.percentile(s, levels)
