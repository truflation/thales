"""Import/export vertical — US textile importer (Asia source).

Five-variable BVAR(1) for a US-side importer of finished textiles from
an Asian source. Frame: monthly log-levels, Minnesota prior, Cholesky
ordering by exogeneity.

Endogenous vector (most exogenous first):

  1. log_fx_cnyusd       — DEXCHUS (Chinese Yuan per USD). Primary
                           source-country FX exposure; secondary Asia
                           FX (INR, KRW) live in a future variant.
  2. log_freight          — PCU484121484121 (PPI Long-Distance Truckload
                           Trucking) as cross-border + inland freight
                           proxy; container ocean spot (Freightos FBX)
                           is a paid-tier upgrade.
  3. log_diesel           — GASDESW (US On-Highway Diesel retail).
                           Inland-trucking fuel cost.
  4. log_truf_clothing    — Truflation `clothing_and_footwear` daily
                           index, monthly-aggregated. Anchors to US
                           retail tradables inflation; closest available
                           Truflation proxy for the importer's revenue-
                           side price level.
  5. log_truf_transport   — Truflation `transport`, monthly-aggregated.
                           Downstream demand variable for outbound
                           shipping costs.

Same engine as the auto-importer variant (``bvar_minnesota``). The
product value sits in IRF / FEVD / conditional scenarios — not in
point forecasts of FX or diesel.

Run::

    uv run python truflation-operate/verticals/import_export_textile.py
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
    _ar_matrices,
    _companion_matrix,
    cholesky_irf,
    fevd,
    fit_bvar_minnesota,
)

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "truflation-operate" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAR_SPEC = [
    ("log_fx_cnyusd",      "operate_fred", "DEXCHUS",
        "daily_to_monthly_mean_then_log"),
    ("log_freight",        "operate_fred", "PCU484121484121",
        "monthly_log"),
    ("log_diesel",         "operate_fred", "GASDESW",
        "weekly_to_monthly_mean_then_log"),
    ("log_truf_clothing",  "truf_network", "clothing_and_footwear",
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
        s = s.copy()
        s.index = s.index + pd.offsets.MonthEnd(0)
        return np.log(s)
    if transform in ("daily_to_monthly_mean_then_log",
                         "weekly_to_monthly_mean_then_log"):
        monthly = s.resample("ME").mean().dropna()
        return np.log(monthly)
    raise ValueError(f"unknown transform: {transform}")


def load_panel() -> pd.DataFrame:
    with duckdb.connect(str(VINTAGE_DB), read_only=True) as con:
        cols: dict[str, pd.Series] = {}
        for label, source, sid, transform in VAR_SPEC:
            raw = _load_series(con, sid, source)
            cols[label] = _to_monthly(raw, transform)
    return pd.DataFrame(cols).dropna()


def _walk_forward_rmse(panel: pd.DataFrame, var_cols: list[str],
                            train_min: int = 60, p: int = 1) -> pd.DataFrame:
    Y_full = panel.values
    bvar_errs = {c: [] for c in var_cols}
    naive_errs = {c: [] for c in var_cols}
    n_targets = 0
    for t in range(train_min, len(Y_full) - 1):
        Y_train = Y_full[: t + 1]
        Y_target = Y_full[t + 1]
        fit = fit_bvar_minnesota(Y_train, p=p)
        A_list = _ar_matrices(fit.coefs, fit.k, fit.p)
        intercept = fit.coefs[:, 0]
        last_p = Y_train[-fit.p:][::-1]
        y_next = intercept.copy()
        for l in range(fit.p):
            y_next = y_next + A_list[l] @ last_p[l]
        naive = Y_train[-1]
        for j, c in enumerate(var_cols):
            bvar_errs[c].append(float(Y_target[j] - y_next[j]))
            naive_errs[c].append(float(Y_target[j] - naive[j]))
        n_targets += 1
    rows = []
    for c in var_cols:
        bvar_rmse = float(np.sqrt(np.mean(np.array(bvar_errs[c]) ** 2)))
        naive_rmse = float(np.sqrt(np.mean(np.array(naive_errs[c]) ** 2)))
        red = (1 - bvar_rmse / naive_rmse) * 100 if naive_rmse > 0 else float("nan")
        rows.append({
            "target":           c,
            "n":                n_targets,
            "bvar_rmse_log":    bvar_rmse,
            "naive_rmse_log":   naive_rmse,
            "rmse_red_pct":     red,
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 78)
    print("Import/export vertical — US textile importer (5-var BVAR)")
    print("=" * 78)

    panel = load_panel()
    var_cols = list(panel.columns)
    print(f"\nPanel: n = {len(panel)} months, "
            f"{panel.index.min().date()} → {panel.index.max().date()}")
    print(f"Variables (Cholesky order, most-exogenous first):")
    for c in var_cols:
        print(f"  {c}")

    Y = panel.values
    fit = fit_bvar_minnesota(Y, p=1)
    A_list = _ar_matrices(fit.coefs, fit.k, fit.p)
    F = _companion_matrix(A_list)
    max_eig = float(np.max(np.abs(np.linalg.eigvals(F))))
    print(f"\nStability: max|eig| = {max_eig:.4f}  "
            f"({'STABLE' if max_eig < 1.0 else 'UNSTABLE'})")

    # IRF on FX shock
    H_IRF = 24
    irf = cholesky_irf(fit, h=H_IRF)
    irf_df = pd.DataFrame(
        {f"resp_{var_cols[j]}": irf[:, j, 0] for j in range(len(var_cols))},
        index=range(H_IRF + 1),
    )
    irf_df.index.name = "h"
    print(f"\nIRF (1-SD shock to {var_cols[0]}, log responses, h=0..{H_IRF}):")
    print(irf_df.round(4).head(13))

    fevd_h = fevd(fit, h=12)
    fevd_df = pd.DataFrame(
        fevd_h[-1] * 100, index=var_cols,
        columns=[f"from_{c}" for c in var_cols],
    ).round(2)
    print(f"\nFEVD at h=12 (% of forecast-error variance attributable to each shock):")
    print(fevd_df)

    print("\nWalk-forward 1-step-ahead RMSE (log-space), BVAR vs naive RW…")
    wf = _walk_forward_rmse(panel, var_cols, train_min=60, p=1)
    print(wf.round(4).to_string(index=False))

    today = date.today()
    irf_df.to_csv(OUT_DIR / f"import_export_textile_irf_{today}.csv")
    fevd_df.to_csv(OUT_DIR / f"import_export_textile_fevd_h12_{today}.csv")
    wf.to_csv(OUT_DIR / f"import_export_textile_walkforward_{today}.csv", index=False)
    panel.to_csv(OUT_DIR / f"import_export_textile_panel_{today}.csv")

    summary = {
        "client":            "us_textile_importer",
        "as_of_date":        str(today),
        "n_obs":             int(len(panel)),
        "n_vars":            int(len(var_cols)),
        "window_start":      str(panel.index.min().date()),
        "window_end":        str(panel.index.max().date()),
        "max_abs_eig":       max_eig,
        "stable":            bool(max_eig < 1.0),
        "walk_forward_rmse": wf.to_dict(orient="records"),
    }
    (OUT_DIR / f"import_export_textile_summary_{today}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\nSaved per-vertical artefacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
