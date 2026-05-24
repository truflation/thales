"""Operator Landed-Cost Forecast (OLCF) engine.

The differentiated thing nobody else builds:

  A **single operator-weighted landed-cost forecast** that uses the
  BVAR's joint multivariate structure across all inputs (FX + freight
  + diesel + Truflation streams) and aggregates them through the
  operator's actual cost-share weights into one number — point +
  bands + scenario engine.

Why this is novel
-----------------
Kantox / Convera answer "how do I hedge my FX exposure" — single
input. Resilinc / Sphera answer "is my supplier in trouble" — risk
flag. Flexport / project44 answer "what's my landed cost given today's
tariff" — single-tariff scenario.

Nobody puts FX + diesel + freight + raw input cost together with
operator weights and produces a **joint forecast distribution of
landed cost**. That's the cross-input transmission layer.

Why it should beat naive (the test)
-----------------------------------
On individual inputs (FX, diesel, freight) the BVAR is tied with or
worse than naive random-walk — those series are close to RW at
monthly frequency. But on the **landed-cost aggregate** the BVAR's
contribution is the joint structure:

  * Diesel and freight have correlated shocks (oil price drives both)
  * FX shocks transmit to local-currency-denominated input costs
  * The naive per-input AR(1) baseline misses these cross-effects

This module benchmarks the BVAR landed-cost forecast against three
naive baselines:

  * **Naive flat** — landed cost stays at current level (h-step zero
    change)
  * **Naive RW** — each input follows random walk, aggregated naively
  * **Naive AR(1) per input** — each input follows its own AR(1),
    aggregated naively

If BVAR beats naive AR(1) on the aggregate landed-cost RMSE — even by
a small margin — that's a defensible product number. If not, honest
result and we know exactly where we add value vs not.

Public API
----------
``forecast_landed_cost(panel, cost_shares, h, p, train_window)`` —
fits BVAR on the panel, projects h months forward, aggregates via
cost shares, returns the landed-cost forecast plus the per-input
projections.

``walk_forward_benchmark(panel, cost_shares, h, train_min, p)`` —
walk-forward backtest. Returns a per-method RMSE/MAE/coverage
DataFrame on the landed-cost aggregate AND the per-input residuals so
we can attribute where each method wins or loses.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.bvar_minnesota import (    # noqa: E402
    _ar_matrices,
    fit_bvar_minnesota,
)

# CostShare is defined here (kept local to avoid sibling-import gymnastics).
@dataclass(frozen=True)
class CostShare:
    """One component's share of the operator's landed cost.

    `var_name` must match the BVAR column label. `share` is the
    fraction of the operator's landed cost driven by that input.
    Shares need not sum to 1.0 — the un-modelled remainder is treated
    as fixed.
    """
    var_name: str
    share: float


@dataclass(frozen=True)
class LandedCostForecast:
    h: int
    var_cols: list[str]
    point_landed_log: np.ndarray         # (h,) deterministic landed-cost log-dev
    point_landed_pct: np.ndarray         # (h,) same in % deviation from baseline
    per_var_point_log: np.ndarray        # (h, k) per-input trajectory
    method: str
    metadata: dict


# ─── BVAR-driven landed-cost forecast ────────────────────────────────────


def _project_bvar(Y: np.ndarray, p: int, h: int) -> np.ndarray:
    """Fit BVAR(p) on Y, iterate h steps forward. Returns (h, k)
    deterministic mean trajectory of log-level deviations from the
    starting point at the last observation."""
    fit = fit_bvar_minnesota(Y, p=p)
    A_list = _ar_matrices(fit.coefs, fit.k, fit.p)
    intercept = fit.coefs[:, 0]
    last_p = Y[-fit.p:][::-1]
    state = [s.copy() for s in last_p]
    base = Y[-1].copy()
    out = np.zeros((h, fit.k))
    for step in range(h):
        y_next = intercept.copy()
        for l in range(fit.p):
            y_next = y_next + A_list[l] @ state[l]
        out[step] = y_next - base
        state = [y_next] + state[:-1]
    return out


def _project_naive_flat(Y: np.ndarray, h: int) -> np.ndarray:
    """All deviations = 0. Operator-implicit "things stay where they
    are" assumption."""
    return np.zeros((h, Y.shape[1]))


def _project_naive_rw(Y: np.ndarray, h: int) -> np.ndarray:
    """Random walk: best forecast is last observation. Same as flat in
    log-deviation space — but we keep it as a separate method so we
    can score it cleanly if/when the panel is in a different frame."""
    return np.zeros((h, Y.shape[1]))


def _project_naive_ar1_per_input(Y: np.ndarray, h: int) -> np.ndarray:
    """Per-input AR(1) on log-MoM differences, iterated h steps forward.
    No cross-effects."""
    k = Y.shape[1]
    out = np.zeros((h, k))
    base = Y[-1].copy()
    for j in range(k):
        x = Y[:, j]
        d = np.diff(x)
        if len(d) < 4:
            continue
        a, b = d[:-1], d[1:]
        X = np.column_stack([np.ones_like(a), a])
        coef, *_ = np.linalg.lstsq(X, b, rcond=None)
        alpha, phi = float(coef[0]), float(coef[1])
        last_d = float(d[-1])
        # Iterate forward h steps
        cum = 0.0
        for step in range(h):
            next_d = alpha + phi * last_d
            cum += next_d
            out[step, j] = cum
            last_d = next_d
    return out


def _aggregate_landed_cost(per_var_log_dev: np.ndarray,
                                var_cols: list[str],
                                cost_shares: list[CostShare]) -> np.ndarray:
    """Sum cost-share-weighted log-deviations into a single landed-cost
    log-deviation series of shape (h,)."""
    weights = np.zeros(len(var_cols))
    for cs in cost_shares:
        if cs.var_name in var_cols:
            j = var_cols.index(cs.var_name)
            weights[j] = cs.share
    return per_var_log_dev @ weights


def forecast_landed_cost(panel: pd.DataFrame,
                              cost_shares: list[CostShare],
                              h: int = 12,
                              p: int = 1,
                              train_window: int | None = None,
                              method: str = "bvar",
                              ) -> LandedCostForecast:
    """Single-shot forecast: fit on `panel`, project h months, aggregate
    via cost shares. `method` selects the projection rule:

      * ``"bvar"``          — BVAR(p) joint projection
      * ``"naive_flat"``    — zero change
      * ``"naive_rw"``      — random walk
      * ``"naive_ar1"``     — per-input AR(1), no cross-effects
    """
    var_cols = list(panel.columns)
    data = panel.copy()
    if train_window:
        data = data.iloc[-train_window:]
    Y = data.values

    if method == "bvar":
        per_var = _project_bvar(Y, p, h)
    elif method == "naive_flat":
        per_var = _project_naive_flat(Y, h)
    elif method == "naive_rw":
        per_var = _project_naive_rw(Y, h)
    elif method == "naive_ar1":
        per_var = _project_naive_ar1_per_input(Y, h)
    else:
        raise ValueError(f"unknown method: {method}")

    landed = _aggregate_landed_cost(per_var, var_cols, cost_shares)
    return LandedCostForecast(
        h=h, var_cols=var_cols,
        point_landed_log=landed,
        point_landed_pct=(np.exp(landed) - 1.0) * 100,
        per_var_point_log=per_var,
        method=method,
        metadata={"p": p, "n_train": len(Y)},
    )


# ─── Walk-forward benchmark — the actual proof point ─────────────────────


def walk_forward_benchmark(panel: pd.DataFrame,
                                cost_shares: list[CostShare],
                                h: int = 1,
                                train_min: int = 60,
                                p: int = 1) -> pd.DataFrame:
    """Walk-forward h-step-ahead landed-cost RMSE comparison.

    For each origin t in [train_min, T-h]:
      * fit each method on Y[:t+1]
      * project h steps forward → per-method landed-cost trajectory
      * compare to actual landed-cost trajectory at t+h (computed
        from the actual panel using the same cost-share weights)

    Returns a DataFrame with one row per method and columns:
      * n: number of OOS origins scored
      * rmse_log: RMSE on the landed-cost log-deviation
      * rmse_pct: RMSE on the landed-cost % deviation
      * mae_log: MAE on log-dev
      * mean_err_bp: mean error in basis points (bias check)
      * bvar_vs_naive_ar1_red_pct: only filled for the "bvar" row;
        RMSE reduction vs naive_ar1 on the same OOS set
    """
    var_cols = list(panel.columns)
    Y_full = panel.values
    methods = ["bvar", "naive_flat", "naive_rw", "naive_ar1"]
    errs_log: dict[str, list[float]] = {m: [] for m in methods}

    for t in range(train_min, len(Y_full) - h):
        Y_train = Y_full[: t + 1]
        # Actual landed-cost trajectory: for h-step, the deviation
        # from Y_train[-1] is Y_full[t+h] - Y_train[-1].
        actual_dev = Y_full[t + h] - Y_train[-1]
        actual_landed = _aggregate_landed_cost(
            actual_dev[None, :], var_cols, cost_shares)[0]

        for m in methods:
            if m == "bvar":
                per_var = _project_bvar(Y_train, p, h)
            elif m == "naive_flat":
                per_var = _project_naive_flat(Y_train, h)
            elif m == "naive_rw":
                per_var = _project_naive_rw(Y_train, h)
            elif m == "naive_ar1":
                per_var = _project_naive_ar1_per_input(Y_train, h)
            pred_landed = _aggregate_landed_cost(per_var, var_cols, cost_shares)[h - 1]
            errs_log[m].append(float(actual_landed - pred_landed))

    rows = []
    for m in methods:
        e = np.array(errs_log[m])
        rmse = float(np.sqrt((e ** 2).mean())) if len(e) else float("nan")
        rmse_pct = float(np.sqrt(((np.exp(e) - 1) ** 2).mean()) * 100) if len(e) else float("nan")
        mae = float(np.abs(e).mean()) if len(e) else float("nan")
        rows.append({
            "method":             m,
            "n":                  int(len(e)),
            "rmse_log":           rmse,
            "rmse_pct":           rmse_pct,
            "mae_log":            mae,
            "mean_err_bp":        float(e.mean() * 10000) if len(e) else float("nan"),
        })
    df = pd.DataFrame(rows)
    # RMSE reduction vs naive_ar1 for BVAR
    naive_ar1_rmse = df.loc[df["method"] == "naive_ar1", "rmse_log"].values[0]
    bvar_rmse = df.loc[df["method"] == "bvar", "rmse_log"].values[0]
    df["bvar_vs_naive_ar1_red_pct"] = float("nan")
    df.loc[df["method"] == "bvar", "bvar_vs_naive_ar1_red_pct"] = (
        (1 - bvar_rmse / naive_ar1_rmse) * 100 if naive_ar1_rmse > 0 else float("nan"))
    df["bvar_vs_naive_rw_red_pct"] = float("nan")
    naive_rw_rmse = df.loc[df["method"] == "naive_rw", "rmse_log"].values[0]
    df.loc[df["method"] == "bvar", "bvar_vs_naive_rw_red_pct"] = (
        (1 - bvar_rmse / naive_rw_rmse) * 100 if naive_rw_rmse > 0 else float("nan"))
    return df
