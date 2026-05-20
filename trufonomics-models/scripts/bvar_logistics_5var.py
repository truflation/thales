"""Phase 3.1 — 5-variable logistics BVAR fit on real data.

Adds labor / maintenance / freight / volume to the partial fuel chain.
Insurance series isn't on FRED at the right granularity (Cass insurance
isn't free; CPI auto insurance covers passenger vehicles, not commercial
fleet) — flagged as a stub.

Endogenous vector (Cholesky ordering, most exogenous first):

  1. log_diesel       — US retail diesel (GASDESW), monthly log
  2. log_freight      — PPI specialized trucking (PCU48414841), monthly log
  3. log_maintenance  — CPI vehicle maintenance & repair (CUSR0000SETD)
  4. log_labor        — Avg hourly earnings, T&W (CES4300000008), monthly log
  5. log_volume       — ATA Truck Tonnage Index (TRUCKD11), monthly log

Cholesky rationale:
  * Diesel is the most-exogenous "global price" input.
  * Freight rate adjusts to fuel + demand.
  * Maintenance moves with parts/labor cost trends.
  * Labor is sticky, slow-moving.
  * Volume is the demand-side endogenous outcome (responds to all
    upstream cost moves).

Reports IRF + FEVD + walk-forward forecasts of each variable.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.cost_structures import get_cost_structure  # noqa: E402
from thales.evaluation.harness import attach_actuals, score, walk_forward  # noqa: E402
from thales.models.archetypes.bvar_minnesota import (  # noqa: E402
    BVARForecaster,
    _ar_matrices,
    _companion_matrix,
    cholesky_irf,
    fevd,
    fit_bvar_minnesota,
)
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "real_data_archetypes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_panel() -> pd.DataFrame:
    """5-var monthly panel of log-levels."""
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        diesel = store.get_vintage("GASDESW", date.today()).dropna()
        freight = store.get_vintage("PCU48414841", date.today()).dropna()
        maint = store.get_vintage("CUSR0000SETD", date.today()).dropna()
        labor = store.get_vintage("CES4300000008", date.today()).dropna()
        volume = store.get_vintage("TRUCKD11", date.today()).dropna()

    # All to month-end log-level
    panel = pd.concat({
        "log_diesel":      np.log(diesel.resample("ME").last()),
        "log_freight":     np.log(freight.resample("ME").last()),
        "log_maintenance": np.log(maint.resample("ME").last()),
        "log_labor":       np.log(labor.resample("ME").last()),
        "log_volume":      np.log(volume.resample("ME").last()),
    }, axis=1).dropna()
    return panel


def main() -> None:
    print("=" * 78)
    print("Phase 3.1 — 5-variable logistics BVAR")
    print("=" * 78)

    cs = get_cost_structure("logistics")
    print(f"\nCost structure: {cs.industry} (source: {cs.source})")
    for k, v in cs.weights.items():
        print(f"  {k:<14s} {v:>5.0%}")

    panel = _load_panel()
    var_cols = list(panel.columns)
    print(f"\nPanel: n={len(panel)}  range "
          f"{panel.index.min():%Y-%m} → {panel.index.max():%Y-%m}")
    for c in var_cols:
        s = panel[c]
        print(f"  {c:<18s}  mean={s.mean():+.3f}  sd={s.std():.3f}  "
                f"AC1={s.autocorr(1):+.3f}")

    # ── Static fit on full panel ────────────────────────────────────
    print("\n── Static fit p=1 ──")
    Y = panel.values
    fit1 = fit_bvar_minnesota(Y, p=1, overall_tightness=0.5,
                                       cross_tightness=0.5, lag_decay=1.0)
    A_list = _ar_matrices(fit1.coefs, fit1.k, fit1.p)
    eig = np.abs(np.linalg.eigvals(_companion_matrix(A_list)))
    print(f"  n_train={fit1.n_train}  max|eig|={eig.max():.4f}  "
            f"({'STABLE' if eig.max() < 1.0 else 'NON-STATIONARY'})")
    print("  AR(1) matrix (rows: equation, cols: lag-1 of):")
    print("  " + "  ".join(f"{c[:8]:>8s}" for c in var_cols))
    for i, row_name in enumerate(var_cols):
        row = "  ".join(f"{A_list[0][i, j]:+8.4f}" for j in range(fit1.k))
        print(f"  {row}    ← {row_name}")

    # ── Cholesky IRF (24 months) ────────────────────────────────────
    print("\n── Cholesky IRF, 24-month horizon ──")
    print("  ordering:", " → ".join(var_cols))
    irf = cholesky_irf(fit1, h=24)
    # Print diesel-shock effect on each variable at key horizons
    print("\n  Effect of a 1-SD diesel shock on each variable (% log-points):")
    for h in [0, 1, 3, 6, 12, 24]:
        row = "  ".join(f"{100 * irf[h, i, 0]:+7.3f}"
                              for i in range(fit1.k))
        print(f"  h={h:>3d}:  {row}    [{', '.join(c[4:8] for c in var_cols)}]")

    # ── FEVD at h=12 ────────────────────────────────────────────────
    print("\n── FEVD at h=12 (variance share by shock, %) ──")
    f = fevd(fit1, h=24)
    fevd_df = pd.DataFrame((f[12] * 100).round(1),
                                  index=[c for c in var_cols],
                                  columns=[c for c in var_cols])
    fevd_df.index.name = "response"
    fevd_df.columns.name = "shock"
    print(fevd_df.to_string())

    # ── Walk-forward 1-month forecast for each variable ──────────────
    print("\n── Walk-forward 1-month forecasts (n_OOS per variable) ──")
    print(f"  {'target':<18s}  {'n':>4s}  {'RMSE':>8s}  {'naive':>8s}  "
            f"{'Δ%':>7s}  {'cov80':>6s}  {'cov95':>6s}")
    print("  " + "-" * 70)
    summary_rows = []
    for target in var_cols:
        fc = BVARForecaster(
            var_cols=var_cols, target_col=target,
            horizon=1, p=1, overall_tightness=0.5,
            train_min=60, model_id=f"bvar_logistics5_{target}")
        origins = panel.index[60:-1]
        forecasts = walk_forward(fc, panel, target, origins, horizon=1)
        if not forecasts:
            continue
        df = attach_actuals(forecasts, panel[target])
        block = score(df)
        summary_rows.append({
            "target": target, "n": block.n, "rmse": block.rmse,
            "rmse_naive": block.rmse_naive,
            "rmse_red_pct": block.rmse_reduction_pct,
            "cov80": block.cov80, "cov95": block.cov95,
            "dir_hit": block.dir_hit,
        })
        cov80 = f"{block.cov80:.1%}" if block.cov80 is not None else "—"
        cov95 = f"{block.cov95:.1%}" if block.cov95 is not None else "—"
        red = (f"{block.rmse_reduction_pct:+.2f}%"
                  if block.rmse_reduction_pct is not None else "—")
        print(f"  {target:<18s}  {block.n:>4d}  {block.rmse:>8.5f}  "
                f"{block.rmse_naive:>8.5f}  {red:>7s}  "
                f"{cov80:>6s}  {cov95:>6s}")

    # Persist
    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / "bvar_logistics_5var_summary.csv", index=False)
    fevd_df.to_csv(OUT_DIR / "bvar_logistics_5var_fevd_h12.csv")
    pd.DataFrame(
        {f"{var_cols[i]}<-{var_cols[j]}": irf[:, i, j]
            for i in range(fit1.k) for j in range(fit1.k)}
    ).to_csv(OUT_DIR / "bvar_logistics_5var_irf.csv",
                 index_label="horizon_months")
    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    main()
