"""Gate-2 in the SAME-MONTH NOWCAST frame.

The proper Tier 1 product test. At end of month T, predict BLS_yoy[T]
before BLS publishes (~mid-T+1). This is the frame Path A v1
documented +42% MSE reduction in.

Three forecasters scored:

  1. **last_release_v1** — predict BLS_yoy[T] = BLS_yoy[T-1]
     (the natural baseline; what news headlines do)
  2. **same_month_bridge_v1** — α + β·BLS_yoy[T-1] + γ·truf_yoy[T]
     (Thales same-month nowcaster — the simplest version of Tier 1
     with persistence + Truflation lead value)
  3. **clevfed_native_h0** — Cleveland Fed nowcast at end-of-T as a
     forecast of BLS_yoy[T] (the institutional comparator in its
     native frame, +54.8% RMSE reduction vs last-release we documented)

Compares their RMSE / MAE / coverage / direction over the same window.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation.harness import Forecast, attach_actuals, score  # noqa: E402
from thales.evaluation.scoring_db import open_scoring_db  # noqa: E402
from thales.models.same_month_nowcaster import (  # noqa: E402
    LastReleaseBaseline,
    MultiComponentBridgeNowcaster,
    RegimeConditionalBridgeNowcaster,
    SameMonthBridgeNowcaster,
)
from thales.weights import build_crosswalk, get_top_level_weights  # noqa: E402
from thales import targets as T  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
KAIROS_PARQUET = Path("/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
TRUFL_HEADLINE_COL = "truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy"
SCORING_DB = ROOT / "results" / "baseline_eval" / "scoring.duckdb"


def _walk_h0(forecaster, panel: pd.DataFrame,
                origins: pd.DatetimeIndex,
                target_col: str = "bls_yoy",
                today_baseline: pd.Series | None = None,
                ) -> tuple[list[Forecast], pd.DataFrame]:
    """Walk-forward at h=0 (same-month nowcast).

    For each origin, slice panel up through origin, call forecaster with
    target=origin, collect forecasts. Then attach_actuals using a
    custom today_baseline (lag-1 BLS persistence by default — the
    natural h=0 baseline).
    """
    forecasts = []
    for origin in origins:
        if origin not in panel.index:
            continue
        slice_panel = panel.loc[: origin]
        try:
            fc = forecaster.fit_predict(slice_panel, origin, origin)
        except Exception as e:
            print(f"  [warn] {origin:%Y-%m-%d}: {type(e).__name__}: {e}")
            continue
        forecasts.append(fc)

    df = attach_actuals(forecasts, panel[target_col],
                            today_baseline=today_baseline)
    return forecasts, df


def main() -> None:
    print("=" * 78)
    print("Gate-2 same-month nowcast — Thales bridge vs last-release vs Clev Fed")
    print("=" * 78)

    # ── Load data ────────────────────────────────────────────────────────
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        bls_yoy = T.load_target_yoy(store, "cpi", as_of=date.today())
        clev_yoy = T.load_nowcast_comparator(store, "cpi", as_of=date.today())

    # Truflation headline YoY (monthly, end-of-month aggregated)
    parq = pd.read_parquet(KAIROS_PARQUET)
    parq["date"] = pd.to_datetime(parq["date"])
    parq = parq.set_index("date").sort_index()
    truf_yoy = (parq[TRUFL_HEADLINE_COL].dropna()
                   .resample("ME").last())

    # Per-component Truflation YoYs (12 top-level categories)
    w_df = get_top_level_weights("2026-04-25")
    streams_df = pd.read_csv(ROOT / "data" / "truflation" / "streams_catalog.csv")
    cw = build_crosswalk(streams_df["raw_name"])
    cid_to_raw = {str(int(r.category_id)): r.raw_name
                    for _, r in cw[cw["category_id"].astype(int).isin(
                        w_df["category_id"].astype(int).tolist())].iterrows()}
    truf_components = {}
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        for cid, raw in cid_to_raw.items():
            s = store.get_vintage(raw, date.today()).dropna()
            monthly = s.resample("ME").last()
            yoy = ((monthly / monthly.shift(12) - 1.0) * 100.0).dropna()
            truf_components[f"truf_c{cid}"] = yoy

    # Two panels:
    #   panel_long  — headline-only variants (bls + truf_yoy + clevfed)
    #                 starts mid-2013 once Cleveland Fed data is available
    #   panel_short — multi-component variant (panel_long + 12 components)
    #                 starts 2022-01 once per-component Truflation data covers
    panel_long = pd.concat({
        "bls_yoy": bls_yoy,
        "truf_yoy": truf_yoy,
        "clevfed_yoy": clev_yoy,
    }, axis=1).dropna()
    panel_short = pd.concat({
        **{k: panel_long[k] for k in panel_long.columns},
        **truf_components,
    }, axis=1).dropna()
    panel = panel_long  # default for headline-only forecasters

    print(f"\nLong panel:  n={len(panel_long)}  range "
          f"{panel_long.index.min():%Y-%m-%d} → {panel_long.index.max():%Y-%m-%d}")
    print(f"Short panel: n={len(panel_short)}  range "
          f"{panel_short.index.min():%Y-%m-%d} → {panel_short.index.max():%Y-%m-%d}")

    # Long origins: skip first 36 months (training window for bridge)
    long_origins = panel_long.index[36:]
    short_origins = panel_short.index[36:]
    print(f"Long origins:  {len(long_origins)}  "
          f"({long_origins[0]:%Y-%m} → {long_origins[-1]:%Y-%m})")
    print(f"Short origins: {len(short_origins)}  "
          f"({short_origins[0]:%Y-%m} → {short_origins[-1]:%Y-%m})")

    # Today baseline = lag-1 BLS (natural h=0 baseline)
    today_baseline_long = panel_long["bls_yoy"].shift(1)
    today_baseline_short = panel_short["bls_yoy"].shift(1)

    # ── Forecasters ──────────────────────────────────────────────────────
    last_rel = LastReleaseBaseline(target_col="bls_yoy")
    bridge = SameMonthBridgeNowcaster(
        target_bls_col="bls_yoy", truf_col="truf_yoy",
        train_window_months=36)
    truf_comp_cols = [c for c in panel_short.columns if c.startswith("truf_c")]
    multi = MultiComponentBridgeNowcaster(
        target_bls_col="bls_yoy",
        truf_component_cols=truf_comp_cols,
        train_window_months=36,
        ridge_alpha=10.0)
    regime_cond = RegimeConditionalBridgeNowcaster(
        target_bls_col="bls_yoy",
        truf_col="truf_yoy",
        train_window_months=60)

    # ── Run all variants ────────────────────────────────────────────────
    # Long-window variants (n=115 origins, 2016-08 → 2026-03)
    print(f"\n--- LONG-WINDOW variants (n_origins={len(long_origins)}) ---")
    print("Running last_release_v1...")
    last_rel_fcs, last_rel_df = _walk_h0(last_rel, panel_long, long_origins,
                                                  today_baseline=today_baseline_long)
    print(f"  scored {len(last_rel_df)} origins")

    print("Running same_month_bridge_v1 (headline only)...")
    bridge_fcs, bridge_df = _walk_h0(bridge, panel_long, long_origins,
                                            today_baseline=today_baseline_long)
    print(f"  scored {len(bridge_df)} origins")

    print("Running regime_conditional_bridge_v1...")
    regime_fcs, regime_df = _walk_h0(regime_cond, panel_long, long_origins,
                                            today_baseline=today_baseline_long)
    print(f"  scored {len(regime_df)} origins")

    # Short-window variants (n=26 origins, 2024-01 → 2026-03)
    print(f"\n--- SHORT-WINDOW variant (n_origins={len(short_origins)}) ---")
    print("Running multi_component_bridge_v1...")
    multi_fcs, multi_df = _walk_h0(multi, panel_short, short_origins,
                                          today_baseline=today_baseline_short)
    print(f"  scored {len(multi_df)} origins")

    # Cleveland Fed (long window)
    print("\nRunning clevfed_native_h0 (long window)...")
    clev_fcs = []
    for origin in long_origins:
        if origin not in panel_long.index:
            continue
        clev_pred = float(panel_long.loc[origin, "clevfed_yoy"])
        if np.isnan(clev_pred):
            continue
        clev_fcs.append(Forecast(origin=origin, target=origin,
                                       point=clev_pred,
                                       metadata={"source": "clevfed_h0"}))
    clev_df = attach_actuals(clev_fcs, panel_long["bls_yoy"],
                                  today_baseline=today_baseline_long)
    print(f"  scored {len(clev_df)} origins")

    # ── Score ────────────────────────────────────────────────────────────
    last_block = score(last_rel_df)
    bridge_block = score(bridge_df)
    multi_block = score(multi_df)
    regime_block = score(regime_df)
    clev_block = score(clev_df)

    print()
    print("─── last_release_v1 (the floor) ───")
    print("  " + last_block.summary().replace("\n", "\n  "))
    print()
    print("─── same_month_bridge_v1 (Thales — headline only) ───")
    print("  " + bridge_block.summary().replace("\n", "\n  "))
    print()
    print("─── multi_component_bridge_v1 (Thales — 12 components) ───")
    print("  " + multi_block.summary().replace("\n", "\n  "))
    print()
    print("─── regime_conditional_bridge_v1 (Thales — regime-cond bands) ───")
    print("  " + regime_block.summary().replace("\n", "\n  "))
    print()
    print("─── clevfed_native_h0 (institutional comparator) ───")
    print("  " + clev_block.summary().replace("\n", "\n  "))

    # ── Verdict ──────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("Comparison vs last-release floor")
    print("=" * 78)
    rmse_red_bridge = (1 - bridge_block.rmse / last_block.rmse) * 100
    rmse_red_multi = (1 - multi_block.rmse / last_block.rmse) * 100
    rmse_red_clev = (1 - clev_block.rmse / last_block.rmse) * 100

    print()
    print(f"  {'Model':<35s}  {'RMSE':>7s}  {'vs last-release':>16s}  "
          f"{'80% cov':>9s}  {'Direction':>10s}")
    print("  " + "-" * 85)
    print(f"  {'last_release_v1':<35s}  {last_block.rmse:>7.4f}  "
          f"{'(floor)':>16s}  {last_block.cov80:>9.1%}  "
          f"{last_block.dir_hit:>10.1%}")
    print(f"  {'same_month_bridge_v1 (headline)':<35s}  {bridge_block.rmse:>7.4f}  "
          f"{rmse_red_bridge:+>15.2f}%  {bridge_block.cov80:>9.1%}  "
          f"{bridge_block.dir_hit:>10.1%}")
    print(f"  {'multi_component_bridge_v1 (12 comp)':<35s}  {multi_block.rmse:>7.4f}  "
          f"{rmse_red_multi:+>15.2f}%  {multi_block.cov80:>9.1%}  "
          f"{multi_block.dir_hit:>10.1%}")
    print(f"  {'clevfed_native_h0':<35s}  {clev_block.rmse:>7.4f}  "
          f"{rmse_red_clev:+>15.2f}%  "
          f"{'n/a':>9s}  {clev_block.dir_hit:>10.1%}")

    print()
    if rmse_red_bridge > 0:
        print(f"  ✓ Thales bridge BEATS last-release by {rmse_red_bridge:.2f}% RMSE")
    if rmse_red_bridge > rmse_red_clev:
        print(f"  ✓ Thales bridge BEATS Cleveland Fed nowcast "
              f"({rmse_red_bridge:.2f}% vs {rmse_red_clev:.2f}%)")
    elif rmse_red_clev > rmse_red_bridge:
        gap = rmse_red_clev - rmse_red_bridge
        print(f"  ~ Cleveland Fed leads Thales by {gap:.2f}pp RMSE reduction")

    # Bridge coefficients
    if bridge_fcs:
        print()
        print("Bridge coefficients at last origin:")
        last_meta = bridge_fcs[-1].metadata
        print(f"  α = {last_meta['alpha']:+.4f}")
        print(f"  β (BLS persistence) = {last_meta['beta_lag']:+.4f}")
        print(f"  γ (Truflation lead) = {last_meta['gamma_truf']:+.4f}")
        print(f"  residual SD = {last_meta['residual_sd']:.4f}")
        print(f"  n_train = {last_meta['n_train']}")

    # Persist
    print()
    print(f"Persisting all 3 runs to {SCORING_DB}")
    with open_scoring_db(SCORING_DB) as db:
        for model_id, fcs, df in [
            ("last_release_v1", last_rel_fcs, last_rel_df),
            ("same_month_bridge_v1", bridge_fcs, bridge_df),
            ("multi_component_bridge_v1", multi_fcs, multi_df),
            ("regime_conditional_bridge_v1", regime_fcs, regime_df),
            ("clevfed_native_h0", clev_fcs, clev_df),
        ]:
            for fc in fcs:
                db.insert_forecast(model_id, "bls_cpi_yoy_h0", fc)
            for _, row in df.iterrows():
                db.attach_actual(model_id, "bls_cpi_yoy_h0",
                                    pd.Timestamp(row["target"]).date(),
                                    actual=row["actual"],
                                    today_baseline=row["today"])


if __name__ == "__main__":
    main()
