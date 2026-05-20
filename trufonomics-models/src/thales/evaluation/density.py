"""Density forecasting helpers — sample emission + density-score block.

A point forecast tells you the central tendency of the next observation.
A density forecast tells you the full predictive distribution. The
institutional buyer for inflation forecasts (TIPS desks, options market-
makers, central-bank watchers) wants the second, not the first.

This module bridges the gap between forecasters that emit point + bands
(via the rolling-conformal residual machinery in
:mod:`thales.evaluation.conformal`) and the proper-scoring metrics in
:mod:`thales.evaluation.metrics` (CRPS, PIT, sharpness, log-score).

Three sample-emission paths are provided to cover the model zoo:

  * :func:`samples_from_residuals` — the workhorse. Given a point
    estimate and an array of calibration residuals, draw ``n_samples``
    via bootstrap with replacement: ``samples = point + bootstrap(errors)``.
    This is the empirical-residual-distribution-as-density approach
    that's the natural counterpart of split-conformal banding (Lei et
    al. 2018).
  * :func:`samples_from_gaussian` — for closed-form forecasters with a
    posterior σ (Stock-Watson DFM, BVAR-Minnesota), sample from
    ``N(mu, sigma²)``. Reduces to the Lehmann-Casella shortcut.
  * :func:`samples_from_quantiles` — for forecasters whose native output
    is a quantile vector (CQR, gradient-boosted quantile regressors),
    invert via piecewise-linear interpolation. Not yet used in Thales
    but here for symmetry with the literature.

The :func:`score_density` function takes a sample matrix ``(n, S)``
plus actuals ``(n,)`` and returns a :class:`DensityBlock` with CRPS,
PIT KS p-value, empirical interval coverage at 80 % / 95 %, and
sharpness — every density-forecast quality summary an enterprise
customer or a journal referee will ask about.

References:
    Gneiting, Raftery 2007 — *Strictly Proper Scoring Rules,
        Prediction, and Estimation.* JASA 102(477).
    Lei, G'Sell, Rinaldo, Tibshirani, Wasserman 2018 — *Distribution-
        Free Predictive Inference for Regression.* JASA 113(523).
    Hersbach 2000 — *Decomposition of the Continuous Ranked Probability
        Score for Ensemble Prediction Systems.* Weather and Forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from thales.evaluation import metrics as M


DEFAULT_N_SAMPLES = 500
DEFAULT_SEED = 0


# ─── Sample emission ──────────────────────────────────────────────────────


def samples_from_residuals(point: float,
                            errors: np.ndarray,
                            n_samples: int = DEFAULT_N_SAMPLES,
                            seed: int = DEFAULT_SEED,
                            ) -> np.ndarray:
    """Bootstrap samples from the empirical residual distribution.

    Given a point forecast and a vector of calibration residuals (signed
    OOS errors from rolling-origin or split-conformal calibration), draw
    ``n_samples`` predictive samples by resampling residuals with
    replacement:

        sample[i] = point + errors[idx[i]],   idx ~ Uniform{0..n-1}

    Returns ``np.ndarray`` of shape ``(n_samples,)``. Returns an array
    of NaNs if ``errors`` has fewer than 2 entries (band/density
    cannot be characterized).

    This is the natural density counterpart of the rolling-conformal
    band construction in :mod:`thales.evaluation.conformal` — the same
    residual distribution defines both the [lo, hi] interval (via
    quantiles) and the predictive density (via bootstrap).
    """
    errors = np.asarray(errors, dtype=float)
    errors = errors[~np.isnan(errors)]
    if len(errors) < 2:
        return np.full(n_samples, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(errors), size=n_samples)
    return point + errors[idx]


def samples_from_gaussian(mu: float,
                            sigma: float,
                            n_samples: int = DEFAULT_N_SAMPLES,
                            seed: int = DEFAULT_SEED,
                            ) -> np.ndarray:
    """Predictive samples from ``N(mu, sigma²)``.

    For closed-form forecasters (Stock-Watson DFM, BVAR-Minnesota in the
    Gaussian-posterior regime) where the predictive distribution is
    parametric. Returns ``np.ndarray`` of shape ``(n_samples,)``.

    Returns NaN samples if ``sigma`` is non-positive or non-finite.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        return np.full(n_samples, np.nan)
    rng = np.random.default_rng(seed)
    return float(mu) + float(sigma) * rng.standard_normal(n_samples)


def samples_from_quantiles(quantiles: np.ndarray,
                            levels: np.ndarray,
                            n_samples: int = DEFAULT_N_SAMPLES,
                            seed: int = DEFAULT_SEED,
                            ) -> np.ndarray:
    """Predictive samples from a quantile-defined forecast.

    Given a vector of predictive quantiles (``quantiles[i]`` at level
    ``levels[i]`` ∈ (0, 1)), invert via piecewise-linear interpolation
    of the inverse CDF and draw ``n_samples`` uniform-on-(0,1) inverses.

    Tail extrapolation: uniform-uniform draws below ``min(levels)``
    map to ``quantiles[0]``; draws above ``max(levels)`` map to
    ``quantiles[-1]``. So the supplied quantile vector is a hard
    truncation; pass extreme tails (e.g. 0.025 and 0.975) to capture
    them.
    """
    quantiles = np.asarray(quantiles, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if len(quantiles) != len(levels) or len(quantiles) < 2:
        return np.full(n_samples, np.nan)
    order = np.argsort(levels)
    levels = levels[order]
    quantiles = quantiles[order]
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n_samples)
    return np.interp(u, levels, quantiles)


# ─── Density score block ──────────────────────────────────────────────────


@dataclass
class DensityBlock:
    """Summary of density-forecast quality over a window.

    Reports five numbers that together characterize a density forecast:

      * ``crps`` — Continuous Ranked Probability Score, lower is better.
        The proper-scoring-rule analog of MAE for densities. CRPS of a
        deterministic forecast (zero-width density) reduces to MAE of
        the point.
      * ``pit_ks_pvalue`` — KS test that PIT values are Uniform(0,1).
        High p ⇒ density is calibrated. Low p (< 0.05) ⇒ miscalibrated.
      * ``cov80`` / ``cov95`` — empirical coverage of the central 80 %
        and 95 % credible intervals derived from the sample matrix.
      * ``sharpness80`` / ``sharpness95`` — mean width of the central
        80 % / 95 % credible intervals. Calibration without sharpness is
        useless (a band as wide as the historical range covers
        everything but says nothing).

    All fields are NaN if ``n == 0`` or every sample row is degenerate.
    """
    n: int
    crps: float
    pit_ks_pvalue: float
    cov80: float
    cov95: float
    sharpness80: float
    sharpness95: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def summary(self) -> str:
        return (
            f"n={self.n}  CRPS={self.crps:.4f}  "
            f"PIT-KS p={self.pit_ks_pvalue:.3f}  "
            f"cov80={self.cov80:.1%} (w {self.sharpness80:.3f})  "
            f"cov95={self.cov95:.1%} (w {self.sharpness95:.3f})"
        )


def score_density(samples: np.ndarray,
                   actual: np.ndarray) -> DensityBlock:
    """Score a sample-based density forecast.

    Parameters
    ----------
    samples : ``(n, S)`` array, row i = ``S`` predictive samples for
              observation i.
    actual  : ``(n,)`` array of realized values.

    Rows with all-NaN samples or NaN actuals are dropped pairwise.
    Empty input returns NaN-filled block.
    """
    samples = np.asarray(samples, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if samples.ndim != 2:
        raise ValueError(
            f"samples must be 2-D (n, S); got shape {samples.shape}")
    if len(samples) != len(actual):
        raise ValueError(
            f"samples ({len(samples)}) and actual ({len(actual)}) "
            f"length mismatch")

    # Drop rows with no usable samples or NaN actuals.
    row_ok = ~np.all(np.isnan(samples), axis=1) & ~np.isnan(actual)
    s = samples[row_ok]
    a = actual[row_ok]
    n = len(a)

    if n == 0:
        return DensityBlock(n=0, crps=float("nan"),
                              pit_ks_pvalue=float("nan"),
                              cov80=float("nan"), cov95=float("nan"),
                              sharpness80=float("nan"),
                              sharpness95=float("nan"))

    crps = M.crps_samples(s, a)
    pit = M.pit_samples(s, a)
    pit_ks = M.pit_ks_pvalue(pit)
    cov80 = M.interval_coverage(s, a, level=0.80)
    cov95 = M.interval_coverage(s, a, level=0.95)
    width80 = M.sharpness(s, level=0.80)
    width95 = M.sharpness(s, level=0.95)

    return DensityBlock(
        n=n, crps=crps, pit_ks_pvalue=pit_ks,
        cov80=cov80, cov95=cov95,
        sharpness80=width80, sharpness95=width95,
    )


# ─── Convenience — sample matrix from a list of Forecasts ────────────────


def stack_samples(forecasts: list, n_samples: int = DEFAULT_N_SAMPLES
                    ) -> np.ndarray:
    """Stack the ``samples`` arrays from a list of Forecast objects into
    an ``(n, n_samples)`` matrix. Forecasts with missing samples
    contribute a NaN row.

    Useful when ``walk_forward`` returns a heterogeneous list of
    forecasts (some with samples, some without) and you want a single
    aligned density-score input.
    """
    n = len(forecasts)
    out = np.full((n, n_samples), np.nan)
    for i, f in enumerate(forecasts):
        s = getattr(f, "samples", None)
        if s is None or len(s) == 0:
            continue
        s = np.asarray(s, dtype=float)
        if len(s) >= n_samples:
            out[i, :] = s[:n_samples]
        else:
            # Bootstrap up to n_samples for length parity.
            rng = np.random.default_rng(i)
            idx = rng.integers(0, len(s), size=n_samples)
            out[i, :] = s[idx]
    return out
