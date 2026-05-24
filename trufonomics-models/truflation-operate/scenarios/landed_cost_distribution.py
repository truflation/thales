"""Operator Landed-Cost — joint-distribution benchmark.

Companion to ``landed_cost_forecast.py``. Where that one evaluates
**point forecasts** of landed cost, this one evaluates **predictive
distributions**: does the BVAR's joint covariance structure produce
better-calibrated, sharper, lower-CRPS forecasts than naive baselines
that ignore cross-input correlations?

This is the test where the BVAR should genuinely win. Naive AR(1)
per input assumes independence across inputs. When inputs co-move
(diesel + freight under an oil shock; FX + import cost under a
crisis), the naive aggregation under-counts the basket variance and
produces overconfident bands. The BVAR has the correct covariance
and should be better-calibrated.

Three methods, all producing distributional forecasts:

  * **bvar**       — sample residual-bootstrap paths from the fitted
                     BVAR; preserves the joint Σ.
  * **naive_ar1**  — per-input AR(1), bootstrap each input's residuals
                     INDEPENDENTLY. Wrong joint distribution.
  * **naive_rw**   — per-input random walk, bootstrap per-input
                     residuals INDEPENDENTLY. Same independence flaw.

Metrics per method (lower-is-better for CRPS):

  * CRPS on the landed-cost log-deviation aggregate (proper scoring)
  * Coverage at 80% and 95% (empirical fraction of actuals inside the
    band — closer to nominal is better calibrated)
  * Sharpness — width of the 80% band (narrower is better IF calibration
    is right; narrower at the cost of coverage is overconfidence)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation.metrics import crps_samples    # noqa: E402
from thales.models.archetypes.bvar_minnesota import (    # noqa: E402
    _ar_matrices,
    fit_bvar_minnesota,
)

# Re-use CostShare from the point-forecast module to keep one definition.
sys.path.insert(0, str(ROOT / "truflation-operate" / "scenarios"))
import landed_cost_forecast as lcf    # noqa: E402

CostShare = lcf.CostShare


# ─── BVAR distributional sampler ──────────────────────────────────────────


def _sample_bvar_paths(Y: np.ndarray, p: int, h: int,
                            n_samples: int = 500,
                            rng: np.random.Generator | None = None,
                            ) -> np.ndarray:
    """Sample-path bootstrap from a fitted BVAR.

    Bootstrap residuals jointly across variables (preserving Σ-driven
    correlation), then iterate the BVAR forward h steps under each
    bootstrapped innovation path.

    Returns (n_samples, h, k) array of log-level deviations from
    Y[-1, :].
    """
    if rng is None:
        rng = np.random.default_rng()
    fit = fit_bvar_minnesota(Y, p=p)
    A_list = _ar_matrices(fit.coefs, fit.k, fit.p)
    intercept = fit.coefs[:, 0]

    # In-sample residuals — preserve the joint covariance.
    # _build_design returns Z, Y_target; we reconstruct fitted values
    # from coefs and design.
    from thales.models.archetypes.bvar_minnesota import _build_design
    Z, Y_target = _build_design(Y, fit.p)
    fitted = (Z @ fit.coefs.T)    # (T-p, k)
    resid = Y_target - fitted    # (T-p, k)  — joint residuals per period

    base = Y[-1].copy()
    out = np.zeros((n_samples, h, fit.k))
    for s in range(n_samples):
        # Bootstrap an h-length residual block (with replacement)
        eps_idx = rng.integers(0, resid.shape[0], size=h)
        eps_path = resid[eps_idx]    # (h, k)  — preserves joint Σ
        state = [Y[-1 - l].copy() for l in range(fit.p)]
        for step in range(h):
            y_next = intercept.copy()
            for l in range(fit.p):
                y_next = y_next + A_list[l] @ state[l]
            y_next = y_next + eps_path[step]
            out[s, step] = y_next - base
            state = [y_next] + state[:-1]
    return out


def _sample_naive_ar1_independent(Y: np.ndarray, h: int,
                                          n_samples: int = 500,
                                          rng: np.random.Generator | None = None,
                                          ) -> np.ndarray:
    """Per-input AR(1) bootstrap, residuals sampled INDEPENDENTLY across
    inputs. Wrong joint distribution — the test."""
    if rng is None:
        rng = np.random.default_rng()
    k = Y.shape[1]
    out = np.zeros((n_samples, h, k))
    for j in range(k):
        x = Y[:, j]
        d = np.diff(x)
        if len(d) < 4:
            continue
        a, b = d[:-1], d[1:]
        X = np.column_stack([np.ones_like(a), a])
        coef, *_ = np.linalg.lstsq(X, b, rcond=None)
        alpha, phi = float(coef[0]), float(coef[1])
        res_j = b - (alpha + phi * a)
        # Bootstrap per-input residuals INDEPENDENTLY across paths and
        # across draws — drops the cross-input covariance.
        last_d = float(d[-1])
        for s in range(n_samples):
            eps_idx = rng.integers(0, len(res_j), size=h)
            eps = res_j[eps_idx]
            cum = 0.0
            ld = last_d
            for step in range(h):
                next_d = alpha + phi * ld + eps[step]
                cum += next_d
                out[s, step, j] = cum
                ld = next_d
    return out


def _sample_naive_rw_independent(Y: np.ndarray, h: int,
                                          n_samples: int = 500,
                                          rng: np.random.Generator | None = None,
                                          ) -> np.ndarray:
    """Per-input random walk: cumulative-sum of bootstrapped first-
    differences, independently across inputs."""
    if rng is None:
        rng = np.random.default_rng()
    k = Y.shape[1]
    out = np.zeros((n_samples, h, k))
    for j in range(k):
        d = np.diff(Y[:, j])
        if len(d) < 4:
            continue
        for s in range(n_samples):
            eps_idx = rng.integers(0, len(d), size=h)
            out[s, :, j] = np.cumsum(d[eps_idx])
    return out


# ─── Aggregate samples to landed-cost log-deviation ───────────────────────


def _aggregate_landed_cost_samples(samples: np.ndarray,
                                          var_cols: list[str],
                                          cost_shares: list[CostShare],
                                          ) -> np.ndarray:
    """samples shape (n_samples, h, k) → returns (n_samples, h)
    weighted landed-cost log-dev per sample path."""
    weights = np.zeros(len(var_cols))
    for cs in cost_shares:
        if cs.var_name in var_cols:
            j = var_cols.index(cs.var_name)
            weights[j] = cs.share
    return samples @ weights


# ─── Walk-forward CRPS / coverage / sharpness benchmark ───────────────────


def walk_forward_distribution_benchmark(panel: pd.DataFrame,
                                                  cost_shares: list[CostShare],
                                                  h: int = 1,
                                                  train_min: int = 60,
                                                  p: int = 1,
                                                  n_samples: int = 500,
                                                  seed: int = 0,
                                                  ) -> pd.DataFrame:
    """Walk-forward h-step-ahead distributional comparison.

    For each origin t in [train_min, T-h]:
      * sample n_samples paths from each method
      * aggregate via cost shares to landed-cost samples at horizon h
      * compute CRPS vs the actual landed-cost realisation
      * record band edges for coverage / sharpness

    Returns DataFrame with one row per method:
      n, crps_log, cov80, cov95, sharpness80_log,
      crps_vs_naive_ar1_red_pct (only set on bvar row)
    """
    var_cols = list(panel.columns)
    Y_full = panel.values
    methods = ["bvar", "naive_ar1", "naive_rw"]
    crps_obs: dict[str, list[float]] = {m: [] for m in methods}
    cov80: dict[str, list[bool]] = {m: [] for m in methods}
    cov95: dict[str, list[bool]] = {m: [] for m in methods}
    width80: dict[str, list[float]] = {m: [] for m in methods}

    rng = np.random.default_rng(seed)

    for t in range(train_min, len(Y_full) - h):
        Y_train = Y_full[: t + 1]
        # Actual landed-cost log-deviation at horizon h
        actual_dev = Y_full[t + h] - Y_train[-1]
        weights = np.zeros(len(var_cols))
        for cs in cost_shares:
            if cs.var_name in var_cols:
                j = var_cols.index(cs.var_name)
                weights[j] = cs.share
        actual_landed = float(actual_dev @ weights)

        for m in methods:
            if m == "bvar":
                S = _sample_bvar_paths(Y_train, p, h, n_samples, rng)
            elif m == "naive_ar1":
                S = _sample_naive_ar1_independent(Y_train, h, n_samples, rng)
            elif m == "naive_rw":
                S = _sample_naive_rw_independent(Y_train, h, n_samples, rng)
            landed_samples = _aggregate_landed_cost_samples(
                S, var_cols, cost_shares)
            samples_at_h = landed_samples[:, h - 1]
            # CRPS for this one observation
            crps_obs[m].append(crps_samples(
                samples_at_h.reshape(1, -1),
                np.array([actual_landed])))
            # 80% / 95% bands and coverage
            lo80, hi80 = np.quantile(samples_at_h, [0.10, 0.90])
            lo95, hi95 = np.quantile(samples_at_h, [0.025, 0.975])
            cov80[m].append(bool(lo80 <= actual_landed <= hi80))
            cov95[m].append(bool(lo95 <= actual_landed <= hi95))
            width80[m].append(float(hi80 - lo80))

    rows = []
    for m in methods:
        rows.append({
            "method":               m,
            "n":                    len(crps_obs[m]),
            "crps_log":             float(np.mean(crps_obs[m])) if crps_obs[m] else float("nan"),
            "cov80":                float(np.mean(cov80[m])) if cov80[m] else float("nan"),
            "cov95":                float(np.mean(cov95[m])) if cov95[m] else float("nan"),
            "width80_log":          float(np.mean(width80[m])) if width80[m] else float("nan"),
        })
    df = pd.DataFrame(rows)

    # Improvement vs naive_ar1 on CRPS
    ar1 = df.loc[df["method"] == "naive_ar1", "crps_log"].values[0]
    bvar = df.loc[df["method"] == "bvar", "crps_log"].values[0]
    df["crps_vs_naive_ar1_red_pct"] = float("nan")
    df.loc[df["method"] == "bvar", "crps_vs_naive_ar1_red_pct"] = (
        (1 - bvar / ar1) * 100 if ar1 > 0 else float("nan"))
    return df
