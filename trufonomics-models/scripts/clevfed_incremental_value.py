"""Cleveland Fed incremental-value test — Fix #3.

Resolves user feedback: "For Cleveland Fed, test incremental value, not
just standalone RMSE." Standalone RMSE comparisons are misleading when
two forecasters are correlated — what matters is whether Thales adds
**independent information** to a forecast that already has access to
Cleveland Fed.

Two regressions, then two evaluation modes:

  Model A:  actual = α + β · clev + ε
  Model B:  actual = α + β · clev + γ · thales_signal + ε

Test 1 (in-sample, biased but standard textbook):
  - Joint regression on all (clev, thales, actual) rows
  - t-stat on γ — is Thales' coefficient significantly non-zero?
  - F-test of A vs B — does adding γ improve fit?
  - Adjusted R² gain

Test 2 (OOS, the honest test):
  - Walk-forward: at each origin T, fit Model A and Model B on training
    window of (clev[t], thales[t], actual[t]) for t < T. Predict
    actual[T] using clev[T] and thales[T] (both known at end-of-T).
  - Report ΔRMSE between A and B over the OOS window.

Three candidate Thales signals tested:
  1. ``truf_yoy[T]`` — raw Truflation headline (simplest, what Clev
     can't see)
  2. ``same_month_bridge_v1[T]`` — learned bridge prediction (a
     re-projection of truf_yoy through BLS persistence — may double-dip)
  3. ``compressed_pca_3[T]`` — PCA-3 of 12 per-component series
     (only available 2024-01 onward)

Frame: same-month nowcast (predict BLS_yoy[T] at end-of-T).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation.harness import attach_actuals, score, walk_forward  # noqa: E402
from thales.models.same_month_nowcaster import (  # noqa: E402
    CompressedMultiComponentBridge,
    SameMonthBridgeNowcaster,
)
from thales.weights import build_crosswalk, get_top_level_weights  # noqa: E402
from thales import targets as T  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
KAIROS_PARQUET = Path("/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
TRUFL_HEADLINE_COL = "truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy"
OUT_DIR = ROOT / "results" / "baseline_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (coefs, residuals, R²)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, resid, r2


def adj_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R²; p = number of regressors INCLUDING intercept."""
    if n - p <= 0:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p)


def f_test_nested(rss_full: float, rss_reduced: float,
                    n: int, p_full: int, p_reduced: int) -> tuple[float, float]:
    """F-stat for nested OLS comparison; p includes intercept."""
    df1 = p_full - p_reduced
    df2 = n - p_full
    if df1 <= 0 or df2 <= 0:
        return float("nan"), float("nan")
    f = ((rss_reduced - rss_full) / df1) / (rss_full / df2)
    p = float(stats.f.sf(f, df1, df2))
    return float(f), p


def in_sample_test(df: pd.DataFrame, signal_name: str) -> dict:
    """Run Model A vs Model B in-sample, return summary dict."""
    sub = df[["actual", "clev", signal_name]].dropna()
    n = len(sub)
    y = sub["actual"].values
    XA = np.column_stack([np.ones(n), sub["clev"].values])
    XB = np.column_stack([np.ones(n), sub["clev"].values,
                              sub[signal_name].values])

    coefA, residA, r2A = fit_ols(XA, y)
    coefB, residB, r2B = fit_ols(XB, y)
    rssA = float(np.sum(residA ** 2))
    rssB = float(np.sum(residB ** 2))

    # t-stat on γ (the thales coefficient in B)
    sigma_sq = rssB / (n - 3)
    XtX_inv = np.linalg.inv(XB.T @ XB)
    gamma_se = float(np.sqrt(sigma_sq * XtX_inv[2, 2]))
    gamma = float(coefB[2])
    t_gamma = gamma / gamma_se if gamma_se > 0 else float("nan")
    p_gamma = float(2 * stats.t.sf(abs(t_gamma), df=n - 3))

    f, p_f = f_test_nested(rssB, rssA, n, p_full=3, p_reduced=2)

    return {
        "signal": signal_name,
        "n": n,
        "r2_A": r2A,
        "r2_B": r2B,
        "adj_r2_A": adj_r2(r2A, n, 2),
        "adj_r2_B": adj_r2(r2B, n, 3),
        "delta_r2": r2B - r2A,
        "beta_clev_A": float(coefA[1]),
        "beta_clev_B": float(coefB[1]),
        "gamma": gamma,
        "gamma_se": gamma_se,
        "t_gamma": t_gamma,
        "p_gamma": p_gamma,
        "f_stat": f,
        "p_f": p_f,
    }


def oos_walkforward(df: pd.DataFrame, signal_name: str,
                       train_min: int = 36) -> dict:
    """Walk-forward Model A and Model B, return RMSEs."""
    sub = df[["actual", "clev", signal_name]].dropna().reset_index(drop=True)
    if len(sub) < train_min + 6:
        return {"signal": signal_name, "n_oos": 0,
                  "rmse_A": float("nan"), "rmse_B": float("nan"),
                  "rmse_red_pct": float("nan")}
    preds_A, preds_B, actuals = [], [], []
    for c in range(train_min, len(sub)):
        tr = sub.iloc[:c]
        te = sub.iloc[c]
        n_tr = len(tr)
        XA = np.column_stack([np.ones(n_tr), tr["clev"].values])
        XB = np.column_stack([np.ones(n_tr), tr["clev"].values,
                                  tr[signal_name].values])
        cA, *_ = np.linalg.lstsq(XA, tr["actual"].values, rcond=None)
        cB, *_ = np.linalg.lstsq(XB, tr["actual"].values, rcond=None)
        preds_A.append(cA[0] + cA[1] * te["clev"])
        preds_B.append(cB[0] + cB[1] * te["clev"]
                          + cB[2] * te[signal_name])
        actuals.append(te["actual"])

    preds_A = np.asarray(preds_A)
    preds_B = np.asarray(preds_B)
    actuals = np.asarray(actuals)
    rmseA = float(np.sqrt(np.mean((preds_A - actuals) ** 2)))
    rmseB = float(np.sqrt(np.mean((preds_B - actuals) ** 2)))
    rmse_red = (1 - rmseB / rmseA) * 100 if rmseA > 0 else float("nan")
    mse_red = (1 - (1 - rmse_red / 100) ** 2) * 100

    # Diebold-Mariano-style sign test on squared-error differences
    sq_err_A = (preds_A - actuals) ** 2
    sq_err_B = (preds_B - actuals) ** 2
    diffs = sq_err_A - sq_err_B
    dm_t = (diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)))
              if diffs.std(ddof=1) > 0 else float("nan"))
    dm_p = float(2 * stats.t.sf(abs(dm_t), df=len(diffs) - 1))

    return {
        "signal": signal_name,
        "n_oos": len(actuals),
        "rmse_A": rmseA,
        "rmse_B": rmseB,
        "rmse_red_pct": rmse_red,
        "mse_red_pct": mse_red,
        "dm_t": dm_t,
        "dm_p": dm_p,
    }


def build_panel() -> pd.DataFrame:
    """Build the merged (actual, clev, truf_yoy, bridge_pred, pca_pred)
    panel — one row per month-end."""
    print("Loading data ...")
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        bls_yoy = T.load_target_yoy(store, "cpi", as_of=date.today())
        clev_yoy = T.load_nowcast_comparator(store, "cpi", as_of=date.today())

    parq = pd.read_parquet(KAIROS_PARQUET)
    parq["date"] = pd.to_datetime(parq["date"])
    parq = parq.set_index("date").sort_index()
    truf_yoy = (parq[TRUFL_HEADLINE_COL].dropna()
                   .resample("ME").last())

    panel_long = pd.concat({
        "actual": bls_yoy,
        "clev":   clev_yoy,
        "truf_yoy": truf_yoy,
    }, axis=1).dropna()

    # Add same_month_bridge prediction at each origin (rolling-fit; OOS at
    # the level of the bridge — this is a fair "Thales signal" because it
    # never sees BLS_yoy[T] when predicting at T).
    print("Computing same_month_bridge predictions ...")
    bridge = SameMonthBridgeNowcaster(
        target_bls_col="actual", truf_col="truf_yoy",
        train_window_months=36, train_min=24)
    panel_long["bridge_pred"] = np.nan
    for origin in panel_long.index[36:]:
        try:
            f = bridge.fit_predict(panel_long, origin, origin)
            panel_long.loc[origin, "bridge_pred"] = f.point
        except Exception:
            continue

    # Add PCA-3 compressed predictions where 12 components exist
    print("Loading 12 component series + computing PCA-3 predictions ...")
    w_df = get_top_level_weights("2026-04-25")
    streams_df = pd.read_csv(ROOT / "data" / "truflation" / "streams_catalog.csv")
    cw = build_crosswalk(streams_df["raw_name"])
    cw = cw.dropna(subset=["category_id"]).copy()
    cid_to_raw = {str(int(r.category_id)): r.raw_name
                    for _, r in cw[cw["category_id"].astype(int).isin(
                        w_df["category_id"].astype(int).tolist())].iterrows()}

    truf_components: dict[str, pd.Series] = {}
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        for cid, raw in cid_to_raw.items():
            s = store.get_vintage(raw, date.today()).dropna()
            monthly = s.resample("ME").last()
            yoy = ((monthly / monthly.shift(12) - 1.0) * 100.0).dropna()
            truf_components[f"truf_c{cid}"] = yoy

    panel_short = pd.concat({
        "actual": bls_yoy,
        "clev":   clev_yoy,
        "truf_yoy": truf_yoy,
        **truf_components,
    }, axis=1).dropna()
    component_cols = sorted([c for c in panel_short.columns
                                if c.startswith("truf_c")])

    pca_bridge = CompressedMultiComponentBridge(
        target_bls_col="actual",
        truf_component_cols=component_cols,
        feature_compression="pca", n_components=3,
        train_window_months=36, train_min=24)
    panel_short["pca_pred"] = np.nan
    for origin in panel_short.index[36:]:
        try:
            f = pca_bridge.fit_predict(panel_short, origin, origin)
            panel_short.loc[origin, "pca_pred"] = f.point
        except Exception:
            continue

    panel_long["pca_pred"] = panel_short["pca_pred"]
    return panel_long


def main() -> None:
    print("=" * 78)
    print("Cleveland Fed incremental-value test — Fix #3")
    print("=" * 78)

    panel = build_panel()
    print(f"\nPanel range:    {panel.index.min():%Y-%m} → {panel.index.max():%Y-%m}")
    print(f"  actual notna:  {panel['actual'].notna().sum()}")
    print(f"  clev notna:    {panel['clev'].notna().sum()}")
    print(f"  truf_yoy:      {panel['truf_yoy'].notna().sum()}")
    print(f"  bridge_pred:   {panel['bridge_pred'].notna().sum()}")
    print(f"  pca_pred:      {panel['pca_pred'].notna().sum()}")

    # ── In-sample tests ──────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("Test 1: In-sample regression  actual ~ clev  vs  actual ~ clev + thales")
    print("=" * 78)
    is_rows = []
    for sig in ["truf_yoy", "bridge_pred", "pca_pred"]:
        r = in_sample_test(panel, sig)
        is_rows.append(r)

        print()
        print(f"  ── thales_signal = {sig}  (n={r['n']}) ──")
        print(f"     R²(A: clev only)         = {r['r2_A']:.4f}   "
                f"adj = {r['adj_r2_A']:.4f}")
        print(f"     R²(B: clev + thales)     = {r['r2_B']:.4f}   "
                f"adj = {r['adj_r2_B']:.4f}")
        print(f"     ΔR²                      = {r['delta_r2']:+.4f}")
        print(f"     β_clev (A)               = {r['beta_clev_A']:+.4f}")
        print(f"     β_clev (B, controlled)   = {r['beta_clev_B']:+.4f}")
        print(f"     γ_thales                 = {r['gamma']:+.4f}  "
                f"(SE {r['gamma_se']:.4f})")
        print(f"     t(γ)                     = {r['t_gamma']:+.3f}  "
                f"p = {r['p_gamma']:.4f}")
        print(f"     F(B vs A) nested         = {r['f_stat']:.3f}    "
                f"p = {r['p_f']:.4f}")
        verdict = ("ADDS INFO (in-sample)" if r["p_gamma"] < 0.05
                     else "no significant info (in-sample)")
        print(f"     verdict                  = {verdict}")
    pd.DataFrame(is_rows).to_csv(OUT_DIR / "clevfed_incremental_in_sample.csv",
                                       index=False)

    # ── OOS walk-forward tests ──────────────────────────────────────────
    print()
    print("=" * 78)
    print("Test 2: OOS rolling-origin walk-forward")
    print("=" * 78)
    oos_rows = []
    for sig in ["truf_yoy", "bridge_pred", "pca_pred"]:
        r = oos_walkforward(panel, sig, train_min=36)
        oos_rows.append(r)
        if r["n_oos"] == 0:
            print(f"\n  [{sig}] insufficient OOS data — skipped")
            continue
        print()
        print(f"  ── thales_signal = {sig}  (n_oos={r['n_oos']}) ──")
        print(f"     RMSE Model A (clev only)        = {r['rmse_A']:.4f}")
        print(f"     RMSE Model B (clev + thales)    = {r['rmse_B']:.4f}")
        print(f"     RMSE reduction                  = "
                f"{r['rmse_red_pct']:+.2f}%   "
                f"(MSE {r['mse_red_pct']:+.2f}%)")
        print(f"     DM-style t on Δsq-err           = "
                f"{r['dm_t']:+.3f}    p = {r['dm_p']:.4f}")
        verdict = ("ADDS INFO (OOS)" if r["dm_p"] < 0.05 and r["rmse_red_pct"] > 0
                     else ("trends-positive but n.s." if r["rmse_red_pct"] > 0
                              else "no OOS gain"))
        print(f"     verdict                          = {verdict}")
    pd.DataFrame(oos_rows).to_csv(OUT_DIR / "clevfed_incremental_oos.csv",
                                        index=False)

    # ── Verdict summary ─────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("Summary verdict")
    print("=" * 78)
    print()
    print(f"  {'signal':<14s}  {'in-sample p(γ)':>15s}  {'OOS RMSE Δ':>11s}  "
            f"{'OOS p(DM)':>10s}")
    print("  " + "-" * 60)
    for is_r, oos_r in zip(is_rows, oos_rows):
        oos_delta = (f"{oos_r['rmse_red_pct']:+.2f}%"
                       if oos_r["n_oos"] > 0 else "n/a")
        oos_p = f"{oos_r['dm_p']:.4f}" if oos_r["n_oos"] > 0 else "n/a"
        print(f"  {is_r['signal']:<14s}  {is_r['p_gamma']:>15.4f}  "
                f"{oos_delta:>11s}  {oos_p:>10s}")

    print()
    print(f"Per-row tables saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
