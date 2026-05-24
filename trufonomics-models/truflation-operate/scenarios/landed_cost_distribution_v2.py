"""Operator Landed-Cost — joint-distribution benchmark v2.

Two improvements over v1:

  1. **Fit BVAR on log-DIFFERENCES** (log returns) rather than log-
     LEVELS. The log-level series are near-integrated (max|eig|
     close to 1.0) which makes the BVAR's forecast variance grow
     unboundedly. Fitting on log-returns is stationary and gives a
     proper predictive distribution. h-step landed-cost log-deviation
     is then the cumulative sum of forecasted log-returns.

  2. **Regime-stratified CRPS.** The BVAR's joint structure helps most
     when input correlations are high — i.e. during crisis periods
     (COVID 2020, energy spike 2022). In stable periods naive_ar1's
     per-input independence is approximately right. We slice the OOS
     into regime sub-periods and score CRPS per sub-period.

This is consistent with the strategic reframe: we shouldn't expect
to win on point forecasts at all times, but we should win on
distributional calibration when it matters most.
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
    _build_design,
    fit_bvar_minnesota,
)

sys.path.insert(0, str(ROOT / "truflation-operate" / "scenarios"))
import landed_cost_forecast as lcf    # noqa: E402

CostShare = lcf.CostShare


# ─── BVAR on log-returns: sample paths of h-step log-level deviation ─────


def _sample_bvar_logreturn_paths(Y_level: np.ndarray, p: int, h: int,
                                          n_samples: int = 500,
                                          rng: np.random.Generator | None = None,
                                          ) -> np.ndarray:
    """Fit BVAR(p) on log-RETURNS (Y_level differenced), bootstrap-
    sample h-step log-return paths, cumsum to log-level deviation.

    Returns (n_samples, h, k) log-level deviations from Y_level[-1].
    """
    if rng is None:
        rng = np.random.default_rng()
    R = np.diff(Y_level, axis=0)        # log-returns, (T-1, k)
    if R.shape[0] < p + 4:
        return np.zeros((n_samples, h, Y_level.shape[1]))

    fit = fit_bvar_minnesota(R, p=p)
    A_list = _ar_matrices(fit.coefs, fit.k, fit.p)
    intercept = fit.coefs[:, 0]

    # In-sample joint residuals — preserves Σ.
    Z, Y_target = _build_design(R, fit.p)
    fitted = Z @ fit.coefs.T
    resid = Y_target - fitted

    last_p = R[-fit.p:][::-1]
    out = np.zeros((n_samples, h, fit.k))
    for s in range(n_samples):
        eps_idx = rng.integers(0, resid.shape[0], size=h)
        eps_path = resid[eps_idx]
        state = [s.copy() for s in last_p]
        cum = np.zeros(fit.k)
        for step in range(h):
            r_next = intercept.copy()
            for l in range(fit.p):
                r_next = r_next + A_list[l] @ state[l]
            r_next = r_next + eps_path[step]
            cum = cum + r_next
            out[s, step] = cum
            state = [r_next] + state[:-1]
    return out


def _sample_naive_ar1_returns_independent(Y_level: np.ndarray, h: int,
                                                    n_samples: int = 500,
                                                    rng: np.random.Generator | None = None,
                                                    ) -> np.ndarray:
    """Per-input AR(1) on log-returns, independent residual bootstrap.

    Symmetric to the BVAR variant but drops the cross-input Σ. h-step
    log-level deviation is the cumsum of forecasted log-returns.
    """
    if rng is None:
        rng = np.random.default_rng()
    R = np.diff(Y_level, axis=0)
    k = R.shape[1]
    out = np.zeros((n_samples, h, k))
    for j in range(k):
        x = R[:, j]
        if len(x) < 4:
            continue
        a, b = x[:-1], x[1:]
        X = np.column_stack([np.ones_like(a), a])
        coef, *_ = np.linalg.lstsq(X, b, rcond=None)
        alpha, phi = float(coef[0]), float(coef[1])
        res_j = b - (alpha + phi * a)
        for s in range(n_samples):
            eps_idx = rng.integers(0, len(res_j), size=h)
            eps = res_j[eps_idx]
            cum = 0.0
            last_r = x[-1]
            for step in range(h):
                next_r = alpha + phi * last_r + eps[step]
                cum += next_r
                out[s, step, j] = cum
                last_r = next_r
    return out


def _sample_naive_rw_returns_independent(Y_level: np.ndarray, h: int,
                                                    n_samples: int = 500,
                                                    rng: np.random.Generator | None = None,
                                                    ) -> np.ndarray:
    """Per-input random walk on log-RETURNS — zero-mean iid bootstrap
    of historical log-returns, cumsum to log-level deviation."""
    if rng is None:
        rng = np.random.default_rng()
    R = np.diff(Y_level, axis=0)
    k = R.shape[1]
    out = np.zeros((n_samples, h, k))
    for j in range(k):
        x = R[:, j]
        if len(x) < 4:
            continue
        for s in range(n_samples):
            eps_idx = rng.integers(0, len(x), size=h)
            out[s, :, j] = np.cumsum(x[eps_idx])
    return out


def _aggregate_samples(samples: np.ndarray,
                            var_cols: list[str],
                            cost_shares: list[CostShare]) -> np.ndarray:
    weights = np.zeros(len(var_cols))
    for cs in cost_shares:
        if cs.var_name in var_cols:
            j = var_cols.index(cs.var_name)
            weights[j] = cs.share
    return samples @ weights


def walk_forward_v2(panel: pd.DataFrame,
                          cost_shares: list[CostShare],
                          h: int = 1,
                          train_min: int = 60,
                          p: int = 1,
                          n_samples: int = 500,
                          seed: int = 0,
                          ) -> pd.DataFrame:
    """V2 benchmark: BVAR-on-returns vs naive_ar1-on-returns vs naive_rw.

    Returns one row per (method, regime). Regime is one of:
      * "all"          all OOS origins
      * "stable"       origins in 2014-01 → 2019-12 (pre-COVID stable)
      * "covid"        origins in 2020-01 → 2021-12
      * "ukraine_post" origins in 2022-01 → 2023-12
      * "recent"       origins in 2024-01 → end
    """
    var_cols = list(panel.columns)
    dates_full = panel.index
    Y_full = panel.values
    methods = ["bvar_returns", "naive_ar1_returns", "naive_rw_returns"]
    weights = np.zeros(len(var_cols))
    for cs in cost_shares:
        if cs.var_name in var_cols:
            j = var_cols.index(cs.var_name)
            weights[j] = cs.share

    obs: dict[str, list] = {m: [] for m in methods}
    cov80: dict[str, list[bool]] = {m: [] for m in methods}
    width80: dict[str, list[float]] = {m: [] for m in methods}
    target_dates: list[pd.Timestamp] = []
    actuals: list[float] = []

    rng = np.random.default_rng(seed)

    for t in range(train_min, len(Y_full) - h):
        Y_train = Y_full[: t + 1]
        actual_dev = Y_full[t + h] - Y_train[-1]
        actual_landed = float(actual_dev @ weights)
        target_dates.append(dates_full[t + h])
        actuals.append(actual_landed)

        for m in methods:
            if m == "bvar_returns":
                S = _sample_bvar_logreturn_paths(Y_train, p, h, n_samples, rng)
            elif m == "naive_ar1_returns":
                S = _sample_naive_ar1_returns_independent(Y_train, h, n_samples, rng)
            else:
                S = _sample_naive_rw_returns_independent(Y_train, h, n_samples, rng)
            land = _aggregate_samples(S, var_cols, cost_shares)
            sah = land[:, h - 1]
            obs[m].append(crps_samples(sah.reshape(1, -1),
                                                np.array([actual_landed])))
            lo80, hi80 = np.quantile(sah, [0.10, 0.90])
            cov80[m].append(bool(lo80 <= actual_landed <= hi80))
            width80[m].append(float(hi80 - lo80))

    def _slice(dates, mask):
        return [i for i, d in enumerate(dates) if mask(d)]

    regimes = {
        "all":           lambda d: True,
        "stable":        lambda d: pd.Timestamp("2014-01-01") <= d <= pd.Timestamp("2019-12-31"),
        "covid":         lambda d: pd.Timestamp("2020-01-01") <= d <= pd.Timestamp("2021-12-31"),
        "ukraine_post":  lambda d: pd.Timestamp("2022-01-01") <= d <= pd.Timestamp("2023-12-31"),
        "recent":        lambda d: pd.Timestamp("2024-01-01") <= d,
    }

    rows = []
    for rname, mask in regimes.items():
        idx = _slice(target_dates, mask)
        n = len(idx)
        for m in methods:
            if n == 0:
                rows.append({"regime": rname, "method": m, "n": 0,
                                 "crps": float("nan"), "cov80": float("nan"),
                                 "width80": float("nan")})
                continue
            crps = float(np.mean([obs[m][i] for i in idx]))
            cov = float(np.mean([cov80[m][i] for i in idx]))
            w = float(np.mean([width80[m][i] for i in idx]))
            rows.append({
                "regime":  rname, "method": m, "n": n,
                "crps":    crps,
                "cov80":   cov,
                "width80": w,
            })
    df = pd.DataFrame(rows)

    # vs-naive_ar1 reduction per regime
    df["crps_vs_naive_ar1_red_pct"] = float("nan")
    for rname in regimes:
        sub = df[df["regime"] == rname]
        if sub.empty: continue
        ar1 = sub.loc[sub["method"] == "naive_ar1_returns", "crps"].values
        bv = sub.loc[sub["method"] == "bvar_returns", "crps"].values
        if len(ar1) and len(bv) and ar1[0] > 0:
            red = (1 - bv[0] / ar1[0]) * 100
            mask = (df["regime"] == rname) & (df["method"] == "bvar_returns")
            df.loc[mask, "crps_vs_naive_ar1_red_pct"] = red
    return df
