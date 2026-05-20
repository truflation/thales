"""Statistical tests for forecast comparison.

All tests use Newey-West HAC-adjusted variance with a configurable lag
(default: 3, matching the nowcast pre-reg convention).

Tests provided:

  * ``diebold_mariano`` — non-nested comparison of two forecasts' squared-error
    sequences. Two-sided by default.
  * ``clark_west`` — nested comparison where model B nests A. One-sided
    (tests whether the nested B adds information over A).
  * ``giacomini_white`` — conditional predictive ability test. Handles nested
    models and parameter-estimation uncertainty under rolling windows.
  * ``ks_uniform`` — Kolmogorov-Smirnov test that PIT values are Uniform(0,1).

References:
    Diebold & Mariano (1995). "Comparing Predictive Accuracy." JBES.
    Clark & West (2007). "Approximately Normal Tests for Equal Predictive
        Accuracy in Nested Models." J. Econometrics.
    Giacomini & White (2006). "Tests of Conditional Predictive Ability."
        Econometrica.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, kstest

DEFAULT_NW_LAG = 3


# ─── Newey-West HAC variance ────────────────────────────────────────────────

def newey_west_var(d: np.ndarray, lag: int = DEFAULT_NW_LAG) -> float:
    """HAC estimator of Var(mean(d))."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return float("nan")
    dc = d - d.mean()
    gamma0 = float(np.mean(dc ** 2))
    acc = gamma0
    for l_ in range(1, min(lag + 1, n)):
        w = 1 - l_ / (lag + 1)
        gl = float(np.mean(dc[l_:] * dc[:-l_]))
        acc += 2 * w * gl
    return acc / n


# ─── Result container ───────────────────────────────────────────────────────

@dataclass
class TestResult:
    statistic: float
    pvalue: float
    n: int

    def __repr__(self) -> str:
        return (f"TestResult(statistic={self.statistic:+.4f}, "
                f"p={self.pvalue:.4f}, n={self.n})")


# ─── Diebold-Mariano (non-nested) ───────────────────────────────────────────

def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    lag: int = DEFAULT_NW_LAG,
    two_sided: bool = True,
    loss: str = "squared",
) -> TestResult:
    """Test H0: E[L(err_a) − L(err_b)] = 0.

    Positive statistic → A has larger loss → B is more accurate.

    Parameters
    ----------
    errors_a, errors_b : arrays of forecast errors (pred − actual).
    lag : Newey-West lag.
    two_sided : True for two-sided p; False returns one-sided P(B beats A).
    loss : 'squared' (default), 'absolute', or 'abs'.
    """
    ea = np.asarray(errors_a, float)
    eb = np.asarray(errors_b, float)
    if loss in ("squared", "se", "mse"):
        la = ea ** 2
        lb = eb ** 2
    elif loss in ("absolute", "abs", "mae"):
        la = np.abs(ea)
        lb = np.abs(eb)
    else:
        raise ValueError(f"unknown loss {loss!r}")
    d = la - lb
    mask = ~np.isnan(d)
    d = d[mask]
    n = len(d)
    if n < 3:
        return TestResult(float("nan"), float("nan"), n)
    var = newey_west_var(d, lag)
    if var <= 0 or np.isnan(var):
        return TestResult(float("nan"), float("nan"), n)
    t = d.mean() / np.sqrt(var)
    if two_sided:
        p = 2 * (1 - norm.cdf(abs(t)))
    else:
        p = 1 - norm.cdf(t)
    return TestResult(float(t), float(p), n)


# ─── Clark-West (nested) ────────────────────────────────────────────────────

def clark_west(
    errors_small: np.ndarray,
    errors_large: np.ndarray,
    pred_small: np.ndarray,
    pred_large: np.ndarray,
    lag: int = DEFAULT_NW_LAG,
) -> TestResult:
    """CW test for nested models. H0: large model adds nothing over small.

    Positive statistic, low one-sided p → large model adds predictive info.

    d_cw = err_small² − err_large² + (pred_small − pred_large)²
    """
    es = np.asarray(errors_small, float)
    el = np.asarray(errors_large, float)
    ps = np.asarray(pred_small, float)
    pl = np.asarray(pred_large, float)
    d = es ** 2 - el ** 2 + (ps - pl) ** 2
    mask = ~np.isnan(d)
    d = d[mask]
    n = len(d)
    if n < 3:
        return TestResult(float("nan"), float("nan"), n)
    var = newey_west_var(d, lag)
    if var <= 0 or np.isnan(var):
        return TestResult(float("nan"), float("nan"), n)
    t = d.mean() / np.sqrt(var)
    p = 1 - norm.cdf(t)  # one-sided
    return TestResult(float(t), float(p), n)


# ─── Giacomini-White (conditional predictive ability) ───────────────────────

def giacomini_white(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    test_function: np.ndarray | None = None,
    lag: int = DEFAULT_NW_LAG,
    loss: str = "squared",
) -> TestResult:
    """GW test of conditional predictive ability.

    If ``test_function`` is None, reduces to the unconditional version, which
    is equivalent to a two-sided DM test on the specified loss.

    Otherwise ``test_function`` is a 1D or 2D array of test instruments
    ``h_t`` (shape (n,) or (n, k)), and we test:

        H0: E[h_t · (L(err_a) − L(err_b))] = 0

    Statistic distributed as χ²_k under the null. Low p → B has conditionally
    different loss given the test instruments.
    """
    ea = np.asarray(errors_a, float)
    eb = np.asarray(errors_b, float)
    if loss in ("squared", "se", "mse"):
        dL = ea ** 2 - eb ** 2
    elif loss in ("absolute", "abs", "mae"):
        dL = np.abs(ea) - np.abs(eb)
    else:
        raise ValueError(f"unknown loss {loss!r}")

    if test_function is None:
        # Unconditional GW → two-sided DM
        return diebold_mariano(ea, eb, lag=lag, two_sided=True, loss=loss)

    h = np.asarray(test_function, float)
    if h.ndim == 1:
        h = h[:, None]
    if len(h) != len(dL):
        raise ValueError(
            f"test_function length {len(h)} != errors length {len(dL)}")
    mask = ~(np.isnan(dL) | np.isnan(h).any(axis=1))
    z = h[mask] * dL[mask, None]  # (n, k)
    n, k = z.shape
    if n < max(10, k + 2):
        return TestResult(float("nan"), float("nan"), n)

    z_bar = z.mean(axis=0)
    # HAC variance of z_bar
    zc = z - z_bar
    acc = (zc.T @ zc) / n
    for l_ in range(1, min(lag + 1, n)):
        w = 1 - l_ / (lag + 1)
        gamma = (zc[l_:].T @ zc[:-l_]) / n
        acc += w * (gamma + gamma.T)
    omega = acc / n  # var of z_bar
    try:
        inv_omega = np.linalg.inv(omega)
    except np.linalg.LinAlgError:
        return TestResult(float("nan"), float("nan"), n)
    stat = float(z_bar @ inv_omega @ z_bar)
    from scipy.stats import chi2
    p = float(1 - chi2.cdf(stat, df=k))
    return TestResult(stat, p, n)


# ─── KS uniformity (PIT calibration) ────────────────────────────────────────

def ks_uniform(pit: np.ndarray) -> TestResult:
    """Kolmogorov-Smirnov test that PIT values are Uniform(0, 1).

    Low p → PIT non-uniform → density miscalibrated.
    """
    pit = np.asarray(pit, float)
    pit = pit[~np.isnan(pit)]
    n = len(pit)
    if n < 3:
        return TestResult(float("nan"), float("nan"), n)
    res = kstest(pit, "uniform")
    return TestResult(float(res.statistic), float(res.pvalue), n)
