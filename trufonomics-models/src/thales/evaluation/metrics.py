"""Evaluation metrics — point, density, calibration, classification.

Forecast representation conventions used throughout:

  * **Point forecast**: ``pred: np.ndarray`` of shape ``(n,)``.
  * **Sample forecast**: ``samples: np.ndarray`` of shape ``(n, S)`` where
    each row is a draw of ``S`` samples from the predictive distribution for
    origin ``i``. This is the workhorse form for Bayesian posteriors.
  * **Quantile forecast**: ``quantiles: np.ndarray`` of shape ``(n, Q)``
    plus a 1D ``levels: np.ndarray`` of shape ``(Q,)`` with values in (0, 1).
  * **Actual**: ``actual: np.ndarray`` of shape ``(n,)``.

NaNs are dropped pairwise before scoring. Empty inputs return ``np.nan``.
"""

from __future__ import annotations

import numpy as np
import properscoring as ps

# ─── Point metrics ──────────────────────────────────────────────────────────

def _pairwise_valid(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Drop rows where any array has NaN. Returns tuple of clean arrays."""
    masks = [~np.isnan(a) for a in arrays]
    keep = masks[0]
    for m in masks[1:]:
        keep &= m
    return tuple(a[keep] for a in arrays)


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    p, y = _pairwise_valid(np.asarray(pred, float), np.asarray(actual, float))
    if len(y) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    p, y = _pairwise_valid(np.asarray(pred, float), np.asarray(actual, float))
    if len(y) == 0:
        return float("nan")
    return float(np.mean(np.abs(p - y)))


def mase(pred: np.ndarray, actual: np.ndarray,
          naive_pred: np.ndarray) -> float:
    """Mean absolute scaled error — Hyndman-Koehler 2006.

    ``naive_pred`` is the naive baseline's prediction (e.g., persistence).
    Values < 1 mean the candidate beats naive on MAE.
    """
    p, y, n = _pairwise_valid(
        np.asarray(pred, float), np.asarray(actual, float),
        np.asarray(naive_pred, float))
    if len(y) == 0:
        return float("nan")
    num = np.mean(np.abs(p - y))
    den = np.mean(np.abs(n - y))
    return float(num / den) if den > 0 else float("nan")


def directional_accuracy(pred: np.ndarray, actual: np.ndarray,
                          reference: np.ndarray | None = None) -> float:
    """Fraction of origins where sign(pred - reference) == sign(actual - reference).

    If ``reference`` is None, uses the previous-period actual (row-shift by 1).
    Returns value in [0, 1]; 0.5 = no directional skill.
    """
    p = np.asarray(pred, float)
    y = np.asarray(actual, float)
    if reference is None:
        reference = np.roll(y, 1)
        reference[0] = np.nan
    r = np.asarray(reference, float)
    mask = ~(np.isnan(p) | np.isnan(y) | np.isnan(r))
    if mask.sum() == 0:
        return float("nan")
    hits = np.sign(p[mask] - r[mask]) == np.sign(y[mask] - r[mask])
    return float(hits.mean())


# ─── Density metrics ────────────────────────────────────────────────────────

def crps_samples(samples: np.ndarray, actual: np.ndarray) -> float:
    """Mean CRPS from sample-based predictive distributions.

    samples: (n, S), actual: (n,). Returns scalar mean CRPS.
    """
    samples = np.asarray(samples, float)
    actual = np.asarray(actual, float)
    # properscoring's crps_ensemble accepts (observations, forecasts) where
    # forecasts is (n, members). Returns per-observation CRPS.
    per_obs = ps.crps_ensemble(actual, samples)
    per_obs = per_obs[~np.isnan(per_obs)]
    return float(per_obs.mean()) if len(per_obs) else float("nan")


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray,
                   actual: np.ndarray) -> float:
    """Mean CRPS under Gaussian predictive distribution."""
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    actual = np.asarray(actual, float)
    per_obs = ps.crps_gaussian(actual, mu=mu, sig=sigma)
    per_obs = per_obs[~np.isnan(per_obs)]
    return float(per_obs.mean()) if len(per_obs) else float("nan")


def log_score_gaussian(mu: np.ndarray, sigma: np.ndarray,
                        actual: np.ndarray) -> float:
    """Mean log predictive score under Gaussian predictive distribution.

    Higher is better. Returns log p(y|μ, σ) averaged over observations.
    """
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    actual = np.asarray(actual, float)
    mask = ~(np.isnan(mu) | np.isnan(sigma) | np.isnan(actual))
    if mask.sum() == 0:
        return float("nan")
    mu, sigma, y = mu[mask], sigma[mask], actual[mask]
    z = (y - mu) / sigma
    log_p = -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * z ** 2
    return float(log_p.mean())


def quantile_loss(pred: np.ndarray, actual: np.ndarray,
                    quantile: float) -> float:
    """Pinball loss at a single quantile level τ ∈ (0, 1). Lower is better."""
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    p, y = _pairwise_valid(np.asarray(pred, float), np.asarray(actual, float))
    if len(y) == 0:
        return float("nan")
    e = y - p
    return float(np.mean(np.maximum(quantile * e, (quantile - 1) * e)))


# ─── Calibration ────────────────────────────────────────────────────────────

def pit_samples(samples: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Probability Integral Transform from sample-based predictive distributions.

    For each observation, returns the empirical CDF of the samples evaluated
    at the realized value. Calibrated densities produce PIT values uniform on
    [0, 1]. Shape (n,).
    """
    samples = np.asarray(samples, float)
    actual = np.asarray(actual, float)
    n = len(actual)
    pit = np.full(n, np.nan)
    for i in range(n):
        row = samples[i]
        row = row[~np.isnan(row)]
        if len(row) == 0 or np.isnan(actual[i]):
            continue
        pit[i] = float((row <= actual[i]).mean())
    return pit


def pit_ks_pvalue(pit: np.ndarray) -> float:
    """Kolmogorov-Smirnov p-value for uniformity of PIT values.

    Low p-value → PIT non-uniform → density miscalibrated.
    """
    from scipy.stats import kstest
    pit = pit[~np.isnan(pit)]
    if len(pit) < 3:
        return float("nan")
    return float(kstest(pit, "uniform").pvalue)


def interval_coverage(samples: np.ndarray, actual: np.ndarray,
                       level: float = 0.8) -> float:
    """Empirical coverage of the central ``level`` credible interval."""
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    samples = np.asarray(samples, float)
    actual = np.asarray(actual, float)
    alpha = (1 - level) / 2
    lo = np.nanpercentile(samples, 100 * alpha, axis=1)
    hi = np.nanpercentile(samples, 100 * (1 - alpha), axis=1)
    mask = ~np.isnan(actual) & ~np.isnan(lo) & ~np.isnan(hi)
    if mask.sum() == 0:
        return float("nan")
    in_band = (actual[mask] >= lo[mask]) & (actual[mask] <= hi[mask])
    return float(in_band.mean())


def sharpness(samples: np.ndarray, level: float = 0.8) -> float:
    """Mean width of the central ``level`` credible interval."""
    samples = np.asarray(samples, float)
    alpha = (1 - level) / 2
    lo = np.nanpercentile(samples, 100 * alpha, axis=1)
    hi = np.nanpercentile(samples, 100 * (1 - alpha), axis=1)
    widths = hi - lo
    widths = widths[~np.isnan(widths)]
    return float(widths.mean()) if len(widths) else float("nan")


# ─── Classification (for regime models) ─────────────────────────────────────

def brier_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Mean squared error of probabilistic binary predictions. Lower is better."""
    p, y = _pairwise_valid(np.asarray(prob, float), np.asarray(outcome, float))
    if len(y) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def log_loss(prob: np.ndarray, outcome: np.ndarray,
              eps: float = 1e-12) -> float:
    """Binary log loss. Lower is better."""
    p, y = _pairwise_valid(np.asarray(prob, float), np.asarray(outcome, float))
    if len(y) == 0:
        return float("nan")
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(prob: np.ndarray, outcome: np.ndarray) -> float:
    """ROC AUC for binary outcomes. Higher is better."""
    from sklearn.metrics import roc_auc_score
    p, y = _pairwise_valid(np.asarray(prob, float), np.asarray(outcome, float))
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


# ─── Bootstrap uncertainty ──────────────────────────────────────────────────

def bootstrap_ci(
    statistic_fn,
    *arrays: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired-row bootstrap CI for any statistic over aligned arrays.

    Returns (point_estimate, lo, hi) at the 1-alpha coverage level.
    """
    arrays = [np.asarray(a, float) for a in arrays]
    mask = ~np.any(np.stack([np.isnan(a) for a in arrays]), axis=0)
    clean = [a[mask] for a in arrays]
    n = len(clean[0])
    if n < 10:
        return (float("nan"),) * 3
    point = statistic_fn(*clean)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = statistic_fn(*[a[idx] for a in clean])
    return (
        float(point),
        float(np.percentile(draws, 100 * alpha / 2)),
        float(np.percentile(draws, 100 * (1 - alpha / 2))),
    )
