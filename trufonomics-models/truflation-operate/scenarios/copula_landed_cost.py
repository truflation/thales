"""Copula + AR(1) marginals — alternative to BVAR for landed-cost.

The BVAR loses to naive_ar1 on point forecasts because:
  * Per-input AR(1) marginals are empirically strong at monthly grid.
  * The BVAR's k×k parameter estimation adds noise without adding
    enough cross-input forecasting signal in our data.

This module implements a copula-based alternative that keeps the
strong AR(1) marginals AND adds correct joint structure via an
empirical copula on standardised residuals. The decomposition is
canonical (Sklar 1959): any joint distribution = marginals + copula.

For each input i:
  * Fit AR(1) on log-returns
  * Get standardised residuals u_i = F_i(eps_i) where F_i is the
    empirical CDF
For the joint:
  * Empirical Gaussian copula on (u_1, ..., u_k) — fit correlation
    matrix on Φ⁻¹(u_i) (the inverse normal of the uniformised
    residuals)

Sampling a landed-cost distribution at horizon h:
  * Draw n_samples joint uniforms (u_1*, ..., u_k*) from the copula
  * Invert each through F_i to get residual draw eps_i*
  * Project each input forward h steps using its AR(1) + bootstrapped
    eps_i* path
  * Aggregate to landed cost via cost shares

Why it's expected to beat BVAR on joint distribution:
  * Marginal accuracy = AR(1) (strongest baseline)
  * Joint accuracy = empirical copula (captures the actual dependence
    structure including tail co-movement, no parametric mis-spec)
  * Total parameters: k AR(1) (3k) + k×k correlation matrix (k²) vs
    BVAR's k + k² + k(k+1)/2 — fewer parameters, less estimation noise
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation.metrics import crps_samples    # noqa: E402


@dataclass(frozen=True)
class CostShare:
    var_name: str
    share: float


@dataclass
class CopulaFit:
    var_names: list[str]
    ar1_alphas: np.ndarray           # (k,)
    ar1_phis: np.ndarray             # (k,)
    residuals: np.ndarray            # (T, k) — empirical residual matrix
    corr_matrix: np.ndarray          # (k, k) — Gaussian/t copula correlation
    family: str = "gaussian"         # "gaussian" or "t"
    t_df: float | None = None        # degrees of freedom for t-copula


def fit_t_copula_df(z: np.ndarray) -> float:
    """Fit Student-t copula degrees of freedom via profile MLE.

    Given pre-uniformised, normal-quantile-mapped residuals z (n × k),
    the t-copula log-likelihood as a function of df ν is maximised on
    a coarse grid. df < 30 indicates heavy tails (more co-movement in
    extremes); df > 30 ≈ Gaussian.
    """
    n, k = z.shape
    # Cap df at 50 (effectively Gaussian) and floor at 3 (heavy tails)
    grid = [3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    R = np.corrcoef(z, rowvar=False)
    # Numerical PD guard
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.clip(eigvals, 1e-6, None)
    R = eigvecs @ np.diag(eigvals) @ eigvecs.T

    best_ll, best_df = -np.inf, 30.0
    R_inv = np.linalg.inv(R)
    R_det = float(np.linalg.det(R))
    if R_det <= 0:
        return 30.0    # fallback
    log_det_R = float(np.log(R_det))
    from scipy.special import gammaln
    for df in grid:
        # t-copula log-density (per Demarta & McNeil 2005)
        # log c(u) = log Γ((ν+k)/2) + (k-1)log Γ(ν/2) - k log Γ((ν+1)/2)
        #          - 0.5 log|R| - ((ν+k)/2) log(1 + (z'R⁻¹z)/ν)
        #          + ((ν+1)/2) Σ log(1 + z_i²/ν)
        # Sum over observations
        zRz = np.einsum("ij,jk,ik->i", z, R_inv, z)
        log_c = (gammaln((df + k) / 2)
                    + (k - 1) * gammaln(df / 2)
                    - k * gammaln((df + 1) / 2)
                    - 0.5 * log_det_R
                    - ((df + k) / 2) * np.log(1 + zRz / df)
                    + ((df + 1) / 2) * np.log(1 + z ** 2 / df).sum(axis=1))
        ll = float(log_c.sum())
        if ll > best_ll:
            best_ll, best_df = ll, df
    return float(best_df)


def fit_copula_ar1(Y_level: np.ndarray, family: str = "gaussian") -> CopulaFit:
    """Fit per-input AR(1) on log-returns and Gaussian copula on
    standardised residuals."""
    R = np.diff(Y_level, axis=0)
    T, k = R.shape
    alphas = np.zeros(k)
    phis = np.zeros(k)
    resid = np.zeros((T - 1, k))
    for j in range(k):
        x = R[:, j]
        a, b = x[:-1], x[1:]
        X = np.column_stack([np.ones_like(a), a])
        coef, *_ = np.linalg.lstsq(X, b, rcond=None)
        alphas[j] = float(coef[0])
        phis[j] = float(coef[1])
        resid[:, j] = b - (alphas[j] + phis[j] * a)

    # Gaussian copula correlation: rank-transform each column to
    # uniforms u_j = rank(eps_j) / (n+1), then map to standard normal
    # via Φ⁻¹, then compute correlation matrix.
    n = resid.shape[0]
    u = np.zeros_like(resid)
    for j in range(k):
        ranks = stats.rankdata(resid[:, j])
        u[:, j] = ranks / (n + 1)
    z = stats.norm.ppf(u)
    corr = np.corrcoef(z, rowvar=False)
    # Numerical safety: clip eigenvalues slightly positive
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-6, None)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    t_df = None
    if family == "t":
        t_df = fit_t_copula_df(z)
    return CopulaFit(
        var_names=[f"y{i}" for i in range(k)],
        ar1_alphas=alphas, ar1_phis=phis,
        residuals=resid, corr_matrix=corr,
        family=family, t_df=t_df)


def sample_copula_paths(fit: CopulaFit, Y_level: np.ndarray, h: int,
                              n_samples: int = 500,
                              rng: np.random.Generator | None = None,
                              ) -> np.ndarray:
    """Sample (n_samples, h, k) joint paths of log-level deviations
    from Y_level[-1]."""
    if rng is None:
        rng = np.random.default_rng()
    R = np.diff(Y_level, axis=0)
    last_r = R[-1].copy()
    k = R.shape[1]

    out = np.zeros((n_samples, h, k))
    # Draw n_samples × h joint standardised residuals from the Gaussian
    # copula, then invert through each variable's empirical CDF to get
    # marginal-correct AR(1) innovations.
    L = np.linalg.cholesky(fit.corr_matrix)
    n_resid = fit.residuals.shape[0]

    for s in range(n_samples):
        # Draw h joint z-vectors with the copula correlation. For
        # t-copula scale each draw by sqrt(df / chi²(df)) so the joint
        # has Student-t heavy tails with the requested df.
        z_normal = rng.standard_normal((h, k)) @ L.T
        if fit.family == "t" and fit.t_df is not None:
            # Per-row chi² gives proper t-multivariate
            chi2 = rng.chisquare(fit.t_df, size=h) / fit.t_df
            z = z_normal / np.sqrt(chi2[:, None])
            u = stats.t.cdf(z, df=fit.t_df)
        else:
            z = z_normal
            u = stats.norm.cdf(z)    # uniforms with Gaussian copula structure
        # Invert through empirical residual CDF per input
        eps_path = np.zeros((h, k))
        for j in range(k):
            sorted_r = np.sort(fit.residuals[:, j])
            # Empirical quantile via linear interp
            idx = u[:, j] * (n_resid - 1)
            lo = np.floor(idx).astype(int)
            hi = np.minimum(lo + 1, n_resid - 1)
            frac = idx - lo
            eps_path[:, j] = sorted_r[lo] * (1 - frac) + sorted_r[hi] * frac
        # Iterate AR(1) per input with the joint innovations
        cum = np.zeros(k)
        prev_r = last_r.copy()
        for step in range(h):
            r_next = fit.ar1_alphas + fit.ar1_phis * prev_r + eps_path[step]
            cum = cum + r_next
            out[s, step] = cum
            prev_r = r_next
    return out


def _aggregate(samples: np.ndarray, var_cols: list[str],
                    shares: list[CostShare]) -> np.ndarray:
    weights = np.zeros(len(var_cols))
    for cs in shares:
        if cs.var_name in var_cols:
            weights[var_cols.index(cs.var_name)] = cs.share
    return samples @ weights


def walk_forward_copula_vs_bvar(panel: pd.DataFrame,
                                          cost_shares: list[CostShare],
                                          h: int = 1,
                                          train_min: int = 60,
                                          n_samples: int = 500,
                                          seed: int = 0) -> pd.DataFrame:
    """Walk-forward distributional benchmark: copula+AR(1) vs BVAR vs
    naive_ar1 independent. CRPS, coverage, sharpness on landed cost."""
    from thales.models.archetypes.bvar_minnesota import (
        _ar_matrices, _build_design, fit_bvar_minnesota)

    var_cols = list(panel.columns)
    Y_full = panel.values
    methods = ["copula_ar1", "bvar_returns", "naive_ar1_independent"]

    crps_o = {m: [] for m in methods}
    cov80 = {m: [] for m in methods}
    width80 = {m: [] for m in methods}
    rng = np.random.default_rng(seed)
    weights = np.zeros(len(var_cols))
    for cs in cost_shares:
        if cs.var_name in var_cols:
            weights[var_cols.index(cs.var_name)] = cs.share

    for t in range(train_min, len(Y_full) - h):
        Y_train = Y_full[: t + 1]
        actual_dev = Y_full[t + h] - Y_train[-1]
        actual_landed = float(actual_dev @ weights)

        # Copula + AR(1) marginals
        cfit = fit_copula_ar1(Y_train)
        S_cop = sample_copula_paths(cfit, Y_train, h, n_samples, rng)
        # BVAR on returns
        R = np.diff(Y_train, axis=0)
        bvarfit = fit_bvar_minnesota(R, p=1)
        A_list = _ar_matrices(bvarfit.coefs, bvarfit.k, bvarfit.p)
        intercept = bvarfit.coefs[:, 0]
        Z, Yt = _build_design(R, bvarfit.p)
        bvarresid = Yt - Z @ bvarfit.coefs.T
        last_p = R[-bvarfit.p:][::-1]
        S_bvar = np.zeros((n_samples, h, bvarfit.k))
        for s in range(n_samples):
            eps_idx = rng.integers(0, bvarresid.shape[0], size=h)
            eps = bvarresid[eps_idx]
            state = [r.copy() for r in last_p]
            cum = np.zeros(bvarfit.k)
            for step in range(h):
                rn = intercept.copy()
                for l in range(bvarfit.p):
                    rn = rn + A_list[l] @ state[l]
                rn = rn + eps[step]
                cum = cum + rn
                S_bvar[s, step] = cum
                state = [rn] + state[:-1]
        # Naive AR(1) per input, independent
        S_naive = np.zeros((n_samples, h, len(var_cols)))
        for j in range(len(var_cols)):
            r_j = R[:, j]
            a, b = r_j[:-1], r_j[1:]
            X = np.column_stack([np.ones_like(a), a])
            coef, *_ = np.linalg.lstsq(X, b, rcond=None)
            alpha_j, phi_j = float(coef[0]), float(coef[1])
            res_j = b - (alpha_j + phi_j * a)
            for s in range(n_samples):
                eps_idx = rng.integers(0, len(res_j), size=h)
                eps = res_j[eps_idx]
                cum = 0.0
                last_r = r_j[-1]
                for step in range(h):
                    next_r = alpha_j + phi_j * last_r + eps[step]
                    cum += next_r
                    S_naive[s, step, j] = cum
                    last_r = next_r

        for m, S in [("copula_ar1", S_cop), ("bvar_returns", S_bvar),
                          ("naive_ar1_independent", S_naive)]:
            landed = _aggregate(S, var_cols, cost_shares)
            sah = landed[:, h - 1]
            crps_o[m].append(crps_samples(sah.reshape(1, -1),
                                                  np.array([actual_landed])))
            lo80, hi80 = np.quantile(sah, [0.10, 0.90])
            cov80[m].append(bool(lo80 <= actual_landed <= hi80))
            width80[m].append(float(hi80 - lo80))

    rows = []
    for m in methods:
        rows.append({
            "method":   m,
            "n":        len(crps_o[m]),
            "crps":     float(np.mean(crps_o[m])),
            "cov80":    float(np.mean(cov80[m])),
            "width80":  float(np.mean(width80[m])),
        })
    df = pd.DataFrame(rows)
    naive_crps = df.loc[df["method"] == "naive_ar1_independent", "crps"].values[0]
    df["crps_vs_naive_red_pct"] = float("nan")
    for m in ["copula_ar1", "bvar_returns"]:
        v = df.loc[df["method"] == m, "crps"].values[0]
        red = (1 - v / naive_crps) * 100 if naive_crps > 0 else float("nan")
        df.loc[df["method"] == m, "crps_vs_naive_red_pct"] = red
    return df
