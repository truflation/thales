"""MoM-first vs YoY-direct forecasting — Fix #5 evaluation.

Compares two approaches on real Headline CPI YoY:

  1. **YoY-direct AR(1)**: fit AR(1) on the YoY series; forecast yoy[T+1].
  2. **MoM-first AR(1) composed**: fit AR(1) on the MoM series; forecast
     mom[T+1]; compose to yoy[T+1] via the closed-form identity.

These should be **mathematically equivalent** in expectation if the
underlying DGP is stationary in MoM. The MoM-first approach often
wins because:
  * AR(1) on MoM doesn't have to absorb the 12-month autocorrelation
    YoY induces.
  * Residuals are mean-zero and roughly stationary → conformal bands
    calibrate cleanly.
  * Variance regimes (Hamilton MS) become detectable on MoM where
    they were invisible on YoY.

Bonus: also report what regime detection finds on each series. The
hypothesis: MoM has discoverable regimes, YoY does not.
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
from thales.models.baselines import AR1Baseline  # noqa: E402
from thales.models.mom_composed import (  # noqa: E402
    MoMComposedForecaster,
    mom_from_level,
)
from thales import targets as T  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "baseline_eval"


def main() -> None:
    print("=" * 78)
    print("MoM-first vs YoY-direct AR(1) — Fix #5 evaluation")
    print("=" * 78)

    # ── Build the panel ──────────────────────────────────────────────
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        level = T.load_target_level(store, "cpi", as_of=date.today())
        yoy = T.load_target_yoy(store, "cpi", as_of=date.today())

    panel = pd.concat({"bls_level": level, "bls_yoy": yoy}, axis=1).dropna()
    panel["bls_mom"] = mom_from_level(panel["bls_level"], log=True)
    panel = panel.dropna()    # drop the row where mom is NaN
    print(f"\nPanel: n={len(panel)}  range "
          f"{panel.index.min():%Y-%m} → {panel.index.max():%Y-%m}")
    print(f"  YoY mean / SD:  {panel['bls_yoy'].mean():+.3f} / "
          f"{panel['bls_yoy'].std():.3f}")
    print(f"  MoM mean / SD:  {panel['bls_mom'].mean():+.3f} / "
          f"{panel['bls_mom'].std():.3f}")

    # Autocorrelation comparison
    yoy_ac1 = panel["bls_yoy"].autocorr(lag=1)
    mom_ac1 = panel["bls_mom"].autocorr(lag=1)
    print(f"  AR(1) coefficient (lag-1 autocorr):")
    print(f"    YoY: {yoy_ac1:+.4f}   MoM: {mom_ac1:+.4f}")

    # ── Forecasters ──────────────────────────────────────────────────
    yoy_direct = AR1Baseline(target_col="bls_yoy", horizon=1, train_min=24,
                                  calib_months=24,
                                  band_method="rolling_conformal",
                                  model_id="ar1_yoy_direct")
    inner_mom = AR1Baseline(target_col="bls_mom", horizon=1, train_min=24,
                                  calib_months=24,
                                  band_method="rolling_conformal",
                                  model_id="ar1_mom_inner")
    mom_composed = MoMComposedForecaster(
        inner=inner_mom,
        bls_level_col="bls_level",
        bls_yoy_col="bls_yoy",
        mom_col="bls_mom",
        log_mom=True,
        horizon=1,
        model_id="ar1_mom_composed_v1")

    # ── Walk-forward ─────────────────────────────────────────────────
    # Origins start where we have ≥train_min training months
    train_min = 36
    origins = panel.index[train_min:]
    today = panel["bls_yoy"].shift(1)

    print(f"\nOrigins: {len(origins)}  ({origins[0]:%Y-%m} → {origins[-1]:%Y-%m})")
    print()

    # YoY-direct
    yoy_fcs = walk_forward(yoy_direct, panel, "bls_yoy", origins, horizon=1)
    yoy_df = attach_actuals(yoy_fcs, panel["bls_yoy"], today_baseline=today)
    yoy_block = score(yoy_df)
    print("── ar1_yoy_direct ─────────────────────")
    print("  " + yoy_block.summary().replace("\n", "\n  "))

    # MoM-first composed
    mom_fcs = walk_forward(mom_composed, panel, "bls_yoy", origins, horizon=1)
    mom_df = attach_actuals(mom_fcs, panel["bls_yoy"], today_baseline=today)
    mom_block = score(mom_df)
    print()
    print("── ar1_mom_composed ─────────────────────")
    print("  " + mom_block.summary().replace("\n", "\n  "))

    # ── Side-by-side comparison ─────────────────────────────────────
    print()
    print("=" * 78)
    print("Comparison")
    print("=" * 78)
    print()
    print(f"  {'metric':<22s}  {'YoY-direct':>12s}  {'MoM-composed':>14s}  "
            f"{'Δ':>10s}")
    print("  " + "-" * 64)
    rmse_red = (1 - mom_block.rmse / yoy_block.rmse) * 100
    print(f"  {'RMSE':<22s}  {yoy_block.rmse:>12.4f}  "
            f"{mom_block.rmse:>14.4f}  {rmse_red:+>9.2f}%")
    print(f"  {'MAE':<22s}  {yoy_block.mae:>12.4f}  {mom_block.mae:>14.4f}")
    if yoy_block.cov80 is not None and mom_block.cov80 is not None:
        print(f"  {'cov80':<22s}  {yoy_block.cov80:>11.1%}  "
                f"{mom_block.cov80:>13.1%}")
    if yoy_block.cov95 is not None and mom_block.cov95 is not None:
        print(f"  {'cov95':<22s}  {yoy_block.cov95:>11.1%}  "
                f"{mom_block.cov95:>13.1%}")
    if yoy_block.dir_hit is not None and mom_block.dir_hit is not None:
        print(f"  {'direction hit':<22s}  {yoy_block.dir_hit:>11.1%}  "
                f"{mom_block.dir_hit:>13.1%}")

    # ── Regime-detection sanity check ────────────────────────────────
    # Hamilton MS-2 on YoY vs MoM. The hypothesis: regimes show up on
    # MoM but not YoY.
    print()
    print("=" * 78)
    print("Regime detectability — Hamilton MS-2")
    print("=" * 78)
    try:
        from thales.models.archetypes.regime_switching import fit_hamilton_2state
        for label, series in [("YoY", panel["bls_yoy"].values),
                                  ("MoM", panel["bls_mom"].values)]:
            try:
                fit = fit_hamilton_2state(series)
                print(f"\n  {label}:")
                print(f"    σ_low  = {fit.sigma_low:.4f}")
                print(f"    σ_high = {fit.sigma_high:.4f}")
                print(f"    σ_high / σ_low = {fit.sigma_high / fit.sigma_low:.2f}x")
                p_hi_smoothed = fit.smoothed_prob_high
                pct_high = 100 * np.mean(p_hi_smoothed > 0.5)
                print(f"    pct months in high regime: {pct_high:.1f}%")
            except Exception as e:    # noqa: BLE001
                print(f"  {label}: fit failed → {type(e).__name__}: {e}")
    except ImportError:
        print("  (regime_switching not importable — skipping)")

    # ── Persist ──────────────────────────────────────────────────────
    out = OUT_DIR / "mom_first_vs_yoy_direct.csv"
    summary_rows = []
    for label, block in [("ar1_yoy_direct", yoy_block),
                              ("ar1_mom_composed", mom_block)]:
        summary_rows.append({
            "model": label,
            "n": block.n,
            "rmse": block.rmse,
            "mae": block.mae,
            "cov80": block.cov80,
            "cov95": block.cov95,
            "dir_hit": block.dir_hit,
        })
    pd.DataFrame(summary_rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
