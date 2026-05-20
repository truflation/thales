"""Finite-sample conformal quantile helpers.

For a calibration set of n exchangeable residuals, classical conformal
prediction (Vovk, Gammerman, Shafer 2005; Lei et al. 2018) constructs
prediction intervals with **provable marginal coverage**:

    P( y_test ∈ Ĉ(x_test) )  ≥  1 − α

where the lower bound is achieved with equality up to a 1/(n+1) gap.
The construction uses an empirical quantile of the calibration
residuals at rank ``⌈(n+1)·(1−α)⌉``, which is **strictly more
conservative than ``np.percentile(errors, 100·(1−α))``** for small n.

This module provides the rank-based quantile so band code in
``thales.models.baselines`` and ``thales.models.same_month_nowcaster``
can claim conformal-style coverage guarantees rather than just
"empirical quantile of residuals".

The two-sided band form is the workhorse:

    band_offsets(errors, alpha=0.20)  →  (lo_offset, hi_offset)
    [point + lo_offset, point + hi_offset] covers actual w.p. ≥ 0.80

References:

  * Vovk, Gammerman, Shafer 2005 — *Algorithmic Learning in a Random
    World* (the textbook).
  * Lei, G'Sell, Rinaldo, Tibshirani, Wasserman 2018 — *Distribution-
    Free Predictive Inference for Regression*. JASA. (The split-
    conformal regression paper Path A v2 cites.)
  * Romano, Patterson, Candes 2019 — Conformalized Quantile Regression
    (CQR). Asymmetric extension; not used here.
"""

from __future__ import annotations

import math
import numpy as np


def conformal_band_offsets(errors: np.ndarray, alpha: float,
                              strict: bool = False
                              ) -> tuple[float, float]:
    """Two-sided finite-sample conformal band offsets at miscoverage α.

    Returns ``(lo_offset, hi_offset)`` such that

        [point + lo_offset, point + hi_offset]

    has marginal coverage at least ``1 − α`` over exchangeable test
    points (Lei et al. 2018, eqn. 2.1, two-sided form). For ``α=0.20``
    this is an 80% band; for ``α=0.05`` a 95% band.

    Uses **signed** residuals (asymmetric tail). Splits α equally
    between the two tails.

    Implementation:
      * Lower offset is the rank-``k_lo`` empirical quantile of
        signed residuals where ``k_lo = ⌊(n+1)·α/2⌋``.
      * Upper offset is the rank-``k_hi`` empirical quantile where
        ``k_hi = ⌈(n+1)·(1 − α/2)⌉``.
      * Ranks are clamped to ``[1, n]``: if ``(n+1)·α/2 < 1``, the
        sample is too small for the requested band and we return
        the most extreme available residual (with a warning suggested
        upstream — see ``min_n_for_alpha``).

    Parameters
    ----------
    errors : 1-D array of signed calibration residuals (actual − pred)
    alpha  : miscoverage level in (0, 1) — e.g. 0.20 for 80% coverage

    Returns
    -------
    (lo_offset, hi_offset) — both floats; lo is typically negative, hi
    typically positive but no sign guarantee imposed.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    n = len(errors)
    if n < 2:
        raise ValueError(f"need ≥2 calibration residuals; got {n}")

    # Conformal coverage of (1 − α) requires
    # n ≥ ⌈(2 − α) / α⌉ — see ``min_n_for_alpha``. Below this n the
    # upper rank ⌈(n+1)(1 − α/2)⌉ exceeds n and clamps to the sample
    # max, producing a band that may undercover. ``strict=True`` raises
    # in that regime so callers can fall back to Gaussian (or a wider
    # method) instead of silently mis-calibrating.
    if strict and n < min_n_for_alpha(alpha):
        raise ValueError(
            f"conformal_band_offsets: n={n} < min_n_for_alpha({alpha})"
            f"={min_n_for_alpha(alpha)}; cannot deliver requested coverage")

    sorted_e = np.sort(np.asarray(errors, dtype=float))

    # Rank positions (1-indexed) for the two tails.
    k_lo = max(1, int(math.floor((n + 1) * alpha / 2)))
    k_hi = min(n, int(math.ceil((n + 1) * (1 - alpha / 2))))

    # Convert to 0-indexed array positions.
    return float(sorted_e[k_lo - 1]), float(sorted_e[k_hi - 1])


def min_n_for_alpha(alpha: float) -> int:
    """Smallest n such that the rank ``⌈(n+1)·(1−α/2)⌉`` is ≤ n,
    i.e. the calibration set is large enough to *deliver* the
    requested two-sided coverage.

    Below this n, the conformal upper rank pins to ``n`` (the
    sample max), which understates the true tail and undercovers.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    # We need ⌈(n+1)(1-α/2)⌉ ≤ n  ⇔  (n+1)(1-α/2) ≤ n  ⇔
    # 1 - α/2 ≤ n·(α/2) / 1 etc. Simpler: solve for the smallest
    # integer n with (n+1)(1 - α/2) ≤ n.
    # → n − (n+1)(1−α/2) ≥ 0 → n·α/2 − 1 + α/2 ≥ 0 → n ≥ (2−α)/α.
    return int(math.ceil((2 - alpha) / alpha))


def conformal_quantile_pair(errors: np.ndarray,
                              alphas: tuple[float, ...] = (0.20, 0.05),
                              ) -> dict[float, tuple[float, float]]:
    """Convenience: return ``{α: (lo_offset, hi_offset)}`` for several α.

    Default alphas give 80% and 95% bands.
    """
    return {a: conformal_band_offsets(errors, a) for a in alphas}
