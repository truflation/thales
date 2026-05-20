"""Phase 3.1 — partial fuel-chain BVAR on real data.

Three-variable transmission VAR using only the data we already have in
the vintage store:

  1. ``oil`` — log WTI crude oil spot (DCOILWTICO, daily)
  2. ``gas`` — log US regular gasoline retail price (GASREGW, weekly)
  3. ``truf_fuel`` — Truflation transport_gasoline_other_fuels_and_motor_oil
     YoY (daily, % YoY computed from level)

Cholesky ordering = [oil, gas, truf_fuel] reflects the structural
hypothesis: oil shocks → propagate to retail gasoline → propagate to
the consumer-facing fuel CPI proxy. Most exogenous first.

This is **not** the full Phase 3.1 logistics VAR (which needs labor,
maintenance, freight rates, margin, volume — see the Data Gaps
section in the findings). It's the pipeline-validation cut: confirm
the BVAR fits real data, IRFs have the right sign and decay, and
forecasts beat naive on the consumer-facing fuel target.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation.harness import attach_actuals, score, walk_forward  # noqa: E402
from thales.models.archetypes.bvar_minnesota import (  # noqa: E402
    BVARForecaster,
    cholesky_irf,
    fevd,
    fit_bvar_minnesota,
)
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "real_data_archetypes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_panel() -> pd.DataFrame:
    """Build a monthly panel of [log_oil, log_gas, truf_fuel_yoy].

    All three series resampled to month-end. log() of oil/gas levels
    so the VAR works on (approximately) stationary log-returns once
    differenced. truf_fuel already in YoY-pp form.
    """
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        oil = store.get_vintage("DCOILWTICO", date.today()).dropna()
        gas = store.get_vintage("GASREGW", date.today()).dropna()
        truf = store.get_vintage(
            "transport_gasoline_other_fuels_and_motor_oil",
            date.today()).dropna()

    # Monthly log-levels for oil/gas
    oil_m = np.log(oil.resample("ME").last())
    gas_m = np.log(gas.resample("ME").last())
    # Truflation: monthly index → YoY in pp
    truf_m = truf.resample("ME").last()
    truf_yoy = (100.0 * (truf_m / truf_m.shift(12) - 1.0)).dropna()

    panel = pd.concat({
        "log_oil":   oil_m,
        "log_gas":   gas_m,
        "truf_fuel": truf_yoy,
    }, axis=1).dropna()
    return panel


def main() -> None:
    print("=" * 78)
    print("Phase 3.1 — partial fuel-chain BVAR (oil → gas → truf_fuel)")
    print("=" * 78)

    panel = _load_panel()
    print(f"\nPanel: n={len(panel)}  range "
          f"{panel.index.min():%Y-%m} → {panel.index.max():%Y-%m}")
    for c in panel.columns:
        s = panel[c]
        print(f"  {c:<12s}  mean={s.mean():+.3f}  sd={s.std():.3f}  "
                f"AC1={s.autocorr(1):+.3f}")

    # ── Static fit on the full panel ────────────────────────────────
    Y = panel.values
    print("\n── Static fit on full panel ──")
    for p in (1, 2):
        fit = fit_bvar_minnesota(Y, p=p, overall_tightness=0.5,
                                        cross_tightness=0.5, lag_decay=1.0)
        # Eigenvalues of the companion matrix → stability check
        from thales.models.archetypes.bvar_minnesota import (
            _ar_matrices, _companion_matrix)
        A_list = _ar_matrices(fit.coefs, fit.k, fit.p)
        eigs = np.abs(np.linalg.eigvals(_companion_matrix(A_list)))
        print(f"\n  p={p}  n_train={fit.n_train}  "
                f"max|eig|={eigs.max():.4f}  "
                f"({'STABLE' if eigs.max() < 1.0 else 'NON-STATIONARY'})")
        print("  AR(1) matrix [oil → gas → fuel rows; oil/gas/fuel cols]:")
        print("    ", np.array_str(A_list[0], precision=3, suppress_small=True))
        print("  intercepts:", np.array_str(fit.coefs[:, 0],
                                                       precision=3,
                                                       suppress_small=True))
        print("  Σ:")
        print("    ", np.array_str(fit.sigma, precision=3,
                                              suppress_small=True))

    # ── Cholesky IRF + FEVD ──────────────────────────────────────────
    print("\n── Cholesky IRF (24 months, ordering [oil, gas, truf_fuel]) ──")
    fit = fit_bvar_minnesota(Y, p=2, overall_tightness=0.5)
    irf = cholesky_irf(fit, h=24)
    var_names = ["oil", "gas", "truf_fuel"]
    print()
    print(f"  {'horizon':>7s}  ", end="")
    for shock in var_names:
        for resp in var_names:
            print(f"{resp[:4]}<-{shock[:3]:>3s}", end="  ")
    print()
    for h in [0, 1, 3, 6, 12, 24]:
        print(f"  {h:>7d}  ", end="")
        for j in range(3):    # shock
            for i in range(3):    # response
                print(f"{irf[h, i, j]:+>8.4f}", end="  ")
        print()

    print("\n── FEVD at h=12 (variance share by shock) ──")
    f = fevd(fit, h=24)
    df_fevd = pd.DataFrame(f[12], index=var_names, columns=var_names)
    df_fevd.index.name = "response"
    df_fevd.columns.name = "shock"
    print(df_fevd.round(4))

    # ── Walk-forward forecast on truf_fuel (the consumer-facing target) ──
    print("\n── Walk-forward 1-month forecast of truf_fuel YoY ──")
    print("  (3-var BVAR; train_min=36, p=1 → keeps the walk-forward")
    print("   window viable on the limited 2021+ panel)")
    fc = BVARForecaster(
        var_cols=["log_oil", "log_gas", "truf_fuel"],
        target_col="truf_fuel",
        horizon=1, p=1, overall_tightness=0.5,
        cross_tightness=0.5, lag_decay=1.0,
        train_min=36, model_id="bvar_fuel_chain_v1")
    origins = panel.index[36:-1]
    forecasts = walk_forward(fc, panel, "truf_fuel", origins, horizon=1)
    df = attach_actuals(forecasts, panel["truf_fuel"])
    block = score(df)
    print()
    print("  " + block.summary().replace("\n", "\n  "))

    # ── Longer-history validation: oil → gas only (since 2010) ──────
    print("\n── Companion validation: oil → gas BVAR on 2010+ panel ──")
    print("  (Pipeline check using only the FRED series — no Truflation")
    print("   limitation, ~180 monthly obs available)")
    long_panel = panel[["log_oil", "log_gas"]].copy()
    # Reload from store to get pre-2021 history
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        oil_full = np.log(
            store.get_vintage("DCOILWTICO", date.today())
                  .dropna().resample("ME").last())
        gas_full = np.log(
            store.get_vintage("GASREGW", date.today())
                  .dropna().resample("ME").last())
    long_panel = pd.concat({"log_oil": oil_full, "log_gas": gas_full},
                                  axis=1).dropna()
    print(f"  long panel: n={len(long_panel)}  range "
            f"{long_panel.index.min():%Y-%m} → {long_panel.index.max():%Y-%m}")
    fc_long = BVARForecaster(
        var_cols=["log_oil", "log_gas"],
        target_col="log_gas",
        horizon=1, p=1, overall_tightness=0.5,
        train_min=60, model_id="bvar_oil_gas_v1")
    origins_long = long_panel.index[60:-1]
    forecasts_long = walk_forward(fc_long, long_panel, "log_gas",
                                          origins_long, horizon=1)
    df_long = attach_actuals(forecasts_long, long_panel["log_gas"])
    block_long = score(df_long)
    print()
    print("  " + block_long.summary().replace("\n", "\n  "))

    # Persist
    out_csv = OUT_DIR / "bvar_fuel_chain_predictions.csv"
    df.to_csv(out_csv, index=False)
    out_irf = OUT_DIR / "bvar_fuel_chain_irf.csv"
    pd.DataFrame(
        {f"{var_names[i]}<-{var_names[j]}": irf[:, i, j]
            for i in range(3) for j in range(3)}
    ).to_csv(out_irf, index_label="horizon_months")
    out_fevd = OUT_DIR / "bvar_fuel_chain_fevd.csv"
    pd.DataFrame(
        {f"{var_names[i]}<-{var_names[j]}": f[:, i, j]
            for i in range(3) for j in range(3)}
    ).to_csv(out_fevd, index_label="horizon_months")
    print(f"\nSaved:\n  {out_csv}\n  {out_irf}\n  {out_fevd}")


if __name__ == "__main__":
    main()
