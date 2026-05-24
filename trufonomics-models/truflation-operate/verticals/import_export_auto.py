"""Import/export vertical — Paris auto importer.

Five-variable BVAR(1) for an EU-side importer of US-made light vehicles.
Frame: monthly log-levels, Minnesota prior, Cholesky ordering by
exogeneity.

Endogenous vector (most exogenous first):

  1. log_fx_eurusd     — DEXUSEU (US Dollars per Euro). Global FX, slow
                          adjuster on monthly grid; placed first.
  2. log_diesel        — GASDESW (US On-Highway Diesel retail). US-side
                          input but proxy for global fuel transmission;
                          the EU diesel ingest is on the upgrade list.
  3. log_freight       — PCU484121484121 (PPI Long-Distance Truckload
                          Trucking) as a freight-cost proxy; ocean ro-ro
                          spot index is a paid-tier upgrade.
  4. log_truf_vehicle  — Truflation `vehicle_purchases_net_outlay_cars_
                          and_trucks_new` daily index, monthly-aggregated.
                          Captures the wholesale-vehicle-cost component.
  5. log_truf_transport — Truflation `transport` top-level stream,
                          monthly-aggregated. The downstream demand-side
                          variable; lands at the end of the Cholesky.

The BVAR estimates how shocks propagate from FX / diesel / freight into
vehicle and transport costs. Forecasting these inputs is genuinely
hard at monthly horizons (FX and diesel both have near-random-walk
behavior); the product value is in the **transmission** — IRF, FEVD,
and conditional scenarios — not in the point forecasts.

Output:
  * IRF table per shock variable
  * FEVD table at h = 12
  * Walk-forward 1-step RMSE per variable vs naive AR(1) baseline
  * Saved to ``truflation-operate/results/import_export_auto_*.csv``

Run::

    uv run python truflation-operate/verticals/import_export_auto.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.bvar_minnesota import (    # noqa: E402
    cholesky_irf,
    fevd,
    fit_bvar_minnesota,
    _ar_matrices,
)

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "truflation-operate" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Variable specification: (label, source, series_id, transform)
# - source = vintage-store source tag
# - transform = how we put the series on the log-level monthly grid
VAR_SPEC = [
    ("log_fx_eurusd",      "operate_fred", "DEXUSEU",
        "daily_to_monthly_mean_then_log"),
    ("log_diesel",         "operate_fred", "GASDESW",
        "weekly_to_monthly_mean_then_log"),
    ("log_freight",        "operate_fred", "PCU484121484121",
        "monthly_log"),
    ("log_truf_vehicle",   "truf_network",
        "vehicle_purchases_net_outlay_cars_and_trucks_new",
        "daily_to_monthly_mean_then_log"),
    ("log_truf_transport", "truf_network", "transport",
        "daily_to_monthly_mean_then_log"),
]


def _load_series(con: duckdb.DuckDBPyConnection,
                     series_id: str, source: str) -> pd.Series:
    rows = con.execute(
        "SELECT reference_date, value FROM vintage "
        "WHERE series_id = ? AND source = ? "
        "AND (series_id, reference_date, as_of_date) IN ("
        "  SELECT series_id, reference_date, MAX(as_of_date) "
        "  FROM vintage WHERE series_id = ? AND source = ? "
        "  GROUP BY series_id, reference_date) "
        "ORDER BY reference_date",
        [series_id, source, series_id, source],
    ).fetchall()
    if not rows:
        raise RuntimeError(f"empty series {series_id} (source={source})")
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
    return pd.Series([r[1] for r in rows], index=idx, name=series_id)


def _to_monthly(s: pd.Series, transform: str) -> pd.Series:
    if transform == "monthly_log":
        # Already monthly; index already at month-start or month-end
        s = s.copy()
        s.index = s.index + pd.offsets.MonthEnd(0)
        return np.log(s)
    if transform in ("daily_to_monthly_mean_then_log",
                         "weekly_to_monthly_mean_then_log"):
        monthly = s.resample("ME").mean().dropna()
        return np.log(monthly)
    raise ValueError(f"unknown transform: {transform}")


def load_panel() -> pd.DataFrame:
    """5-var monthly log-level panel, intersected on common dates."""
    with duckdb.connect(str(VINTAGE_DB), read_only=True) as con:
        cols: dict[str, pd.Series] = {}
        for label, source, sid, transform in VAR_SPEC:
            raw = _load_series(con, sid, source)
            cols[label] = _to_monthly(raw, transform)
    df = pd.DataFrame(cols).dropna()
    return df


def _walk_forward_rmse(panel: pd.DataFrame,
                            var_cols: list[str],
                            train_min: int = 60,
                            p: int = 1) -> pd.DataFrame:
    """Walk-forward 1-step-ahead RMSE per variable vs naive AR(1)."""
    Y_full = panel.values
    k = len(var_cols)
    bvar_errs = {c: [] for c in var_cols}
    naive_errs = {c: [] for c in var_cols}
    targets_kept = []
    for t in range(train_min, len(Y_full) - 1):
        Y_train = Y_full[: t + 1]
        Y_target = Y_full[t + 1]
        # BVAR fit + 1-step iter
        fit = fit_bvar_minnesota(Y_train, p=p)
        A_list = _ar_matrices(fit.coefs, fit.k, fit.p)
        intercept = fit.coefs[:, 0]
        last_p = Y_train[-fit.p:][::-1]
        y_next = intercept.copy()
        for l in range(fit.p):
            y_next = y_next + A_list[l] @ last_p[l]
        # naive AR(1) per column (random-walk on log level)
        naive = Y_train[-1]
        for j, c in enumerate(var_cols):
            bvar_errs[c].append(float(Y_target[j] - y_next[j]))
            naive_errs[c].append(float(Y_target[j] - naive[j]))
        targets_kept.append(t + 1)
    rows = []
    for j, c in enumerate(var_cols):
        bvar_rmse = float(np.sqrt(np.mean(np.array(bvar_errs[c]) ** 2)))
        naive_rmse = float(np.sqrt(np.mean(np.array(naive_errs[c]) ** 2)))
        red = (1 - bvar_rmse / naive_rmse) * 100 if naive_rmse > 0 else float("nan")
        rows.append({
            "target":           c,
            "n":                len(targets_kept),
            "bvar_rmse_log":    bvar_rmse,
            "naive_rmse_log":   naive_rmse,
            "rmse_red_pct":     red,
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 78)
    print("Import/export vertical — Paris auto importer (5-var BVAR)")
    print("=" * 78)

    panel = load_panel()
    var_cols = list(panel.columns)
    print(f"\nPanel: n = {len(panel)} months, "
            f"{panel.index.min().date()} → {panel.index.max().date()}")
    print(f"Variables (Cholesky order, most-exogenous first):")
    for c in var_cols:
        print(f"  {c}")

    # ── Fit on full sample (for IRF / FEVD reporting) ───────────────
    Y = panel.values
    fit = fit_bvar_minnesota(Y, p=1)
    A_list = _ar_matrices(fit.coefs, fit.k, fit.p)

    # Stability check
    from thales.models.archetypes.bvar_minnesota import _companion_matrix
    F = _companion_matrix(A_list)
    eigs = np.linalg.eigvals(F)
    max_eig = float(np.max(np.abs(eigs)))
    stable = max_eig < 1.0
    print(f"\nStability: max|eig| = {max_eig:.4f}  ({'STABLE' if stable else 'UNSTABLE'})")

    # ── IRF (1-SD shock to FX, h=24) ────────────────────────────────
    H_IRF = 24
    irf = cholesky_irf(fit, h=H_IRF)
    irf_df = pd.DataFrame(
        {f"resp_{var_cols[j]}": irf[:, j, 0]    # response to shock #0 (FX)
            for j in range(len(var_cols))},
        index=range(H_IRF + 1),
    )
    irf_df.index.name = "h"
    print(f"\nIRF (1-SD shock to {var_cols[0]}, log responses, h=0..{H_IRF}):")
    print(irf_df.round(4).head(13))

    # ── FEVD at h = 12 ──────────────────────────────────────────────
    fevd_h = fevd(fit, h=12)
    fevd_df = pd.DataFrame(
        fevd_h[-1] * 100, index=var_cols,
        columns=[f"from_{c}" for c in var_cols],
    ).round(2)
    print(f"\nFEVD at h=12 (% of forecast-error variance attributable to each shock):")
    print(fevd_df)

    # ── Walk-forward 1-step RMSE vs naive ───────────────────────────
    print("\nWalk-forward 1-step-ahead RMSE (log-space), BVAR vs naive RW…")
    wf = _walk_forward_rmse(panel, var_cols, train_min=60, p=1)
    print(wf.round(4).to_string(index=False))

    # ── Persist results ─────────────────────────────────────────────
    today = date.today()
    irf_df.to_csv(OUT_DIR / f"import_export_auto_irf_{today}.csv")
    fevd_df.to_csv(OUT_DIR / f"import_export_auto_fevd_h12_{today}.csv")
    wf.to_csv(OUT_DIR / f"import_export_auto_walkforward_{today}.csv", index=False)
    panel.to_csv(OUT_DIR / f"import_export_auto_panel_{today}.csv")

    summary = {
        "client":            "paris_auto_importer",
        "as_of_date":        str(today),
        "n_obs":             int(len(panel)),
        "n_vars":            int(len(var_cols)),
        "window_start":      str(panel.index.min().date()),
        "window_end":        str(panel.index.max().date()),
        "max_abs_eig":       max_eig,
        "stable":            bool(stable),
        "walk_forward_rmse": wf.to_dict(orient="records"),
    }
    (OUT_DIR / f"import_export_auto_summary_{today}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\nSaved per-vertical artefacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
