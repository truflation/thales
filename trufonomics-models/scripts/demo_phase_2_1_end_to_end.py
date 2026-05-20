"""End-to-end Phase 2.1 demo on real Truflation data.

For each of the 12 top-level Truflation categories:
  1. Load the daily index series from the vintage store
  2. Compute monthly YoY (12-month log change in %)
  3. Walk-forward 1-month-ahead persistence forecasts on each component

Then compose all 12 component forecasts via CBDFComposer (with real
2026 v2 weights) and produce a headline forecast. Compare to the direct
headline persistence forecast on the published Truflation YoY series.

By accounting identity:
  composed_headline_yoy[T+1]  =  Σ_r w_r · component_r_yoy[T+1]

Persistence is linear, so:
  composed_persistence  =  Σ_r w_r · y_r,T
                       =  weighted-sum-of-Ts
                       ≈  direct_persistence_on_(weighted-sum-of-Ts)

The two should match within the small composition residual that
`composition_check/FINDINGS.md` already reported (median 0.000pp, 94%
of days within 0.5pp). This script reproduces that result through the
CBDFComposer rather than a one-shot weighted average.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation.harness import (  # noqa: E402
    Forecast, attach_actuals, score, walk_forward,
)
from thales.models.baselines import PersistenceBaseline  # noqa: E402
from thales.models.composition.cbdf import CBDFComposer  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402
from thales.weights import build_crosswalk, get_top_level_weights  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
KAIROS_PARQUET = Path("/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
TRUFL_HEADLINE_COL = "truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy"
OUT_DIR = ROOT / "results" / "composition"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_monthly_yoy(daily_index: pd.Series) -> pd.Series:
    """Convert a daily index to monthly YoY (12-month log change in %)."""
    monthly = daily_index.resample("ME").last().dropna()
    yoy = (monthly / monthly.shift(12) - 1.0) * 100.0
    return yoy.dropna()


def main() -> None:
    print("=" * 72)
    print("Phase 2.1 End-to-End Demo — Real Truflation Composition")
    print("=" * 72)

    # ── Top-level weights ────────────────────────────────────────────────
    w_df = get_top_level_weights("2026-04-25")
    weights = {str(int(row.category_id)): float(row.weight) / 100.0
                 for row in w_df.itertuples()}
    weights = {k: v for k, v in weights.items() if v > 0}
    print(f"\n12 top-level weights (sum = {sum(weights.values()):.4f}):")
    for cid, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        cat_name = w_df[w_df["category_id"] == int(cid)]["category"].iloc[0]
        print(f"  {cid:<5s} {cat_name:<40s} {w * 100:>6.2f}%")

    # ── Component series ─────────────────────────────────────────────────
    streams_df = pd.read_csv(ROOT / "data" / "truflation" / "streams_catalog.csv")
    cw = build_crosswalk(streams_df["raw_name"])
    top_streams = cw[cw["category_id"].astype(str).isin(weights.keys())]
    cid_to_raw = {str(int(r.category_id)): r.raw_name
                    for _, r in top_streams.iterrows()}

    print(f"\nLoading 12 daily index series from vintage store...")
    yoy_series: dict[str, pd.Series] = {}
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        for cid, raw in cid_to_raw.items():
            s = store.get_vintage(raw, date.today())
            yoy_series[cid] = _to_monthly_yoy(s)

    # ── Common monthly index ─────────────────────────────────────────────
    common_idx = sorted(set.intersection(*[set(s.index)
                                                  for s in yoy_series.values()]))
    common_idx = pd.DatetimeIndex(common_idx)
    panel = pd.DataFrame({cid: s.reindex(common_idx)
                              for cid, s in yoy_series.items()})
    print(f"Component-YoY monthly panel: {panel.shape}  range "
          f"{panel.index.min():%Y-%m-%d} → {panel.index.max():%Y-%m-%d}")

    # ── Direct Truflation headline (published frozen YoY) ────────────────
    print("\nLoading published Truflation headline YoY...")
    parq = pd.read_parquet(KAIROS_PARQUET)
    parq["date"] = pd.to_datetime(parq["date"])
    parq = parq.set_index("date").sort_index()
    headline_daily = parq[TRUFL_HEADLINE_COL].dropna()
    headline_yoy = headline_daily.resample("ME").last().dropna()
    headline_yoy = headline_yoy.reindex(common_idx).dropna()
    print(f"  range: {headline_yoy.index.min():%Y-%m-%d} → "
          f"{headline_yoy.index.max():%Y-%m-%d}  "
          f"latest = {headline_yoy.iloc[-1]:.4f}%")

    # ── Walk-forward persistence per component ──────────────────────────
    panel = panel.loc[headline_yoy.index].dropna()
    headline_yoy = headline_yoy.loc[panel.index]
    origins = panel.index[24: -1]  # 1-month-ahead, train_min=24
    print(f"\nRunning per-component persistence walk-forward over "
          f"{len(origins)} origins...")

    # ── For each origin, build per-component forecasts → compose ────────
    composer = CBDFComposer(weights=weights, weight_sum_tol=5e-3,
                                n_mc_samples=500, seed=0)

    composed_rows = []
    direct_rows = []
    for origin in origins:
        target = panel.index[panel.index.get_loc(origin) + 1]
        # Per-component persistence forecast (point only — bands derived in composer)
        per_comp = {}
        for cid in weights:
            point = float(panel.loc[origin, cid])
            per_comp[cid] = Forecast(origin=origin, target=target,
                                          point=point)
        composed = composer.compose(per_comp, origin, target)
        actual = float(headline_yoy.loc[target])
        composed_rows.append({
            "origin": origin, "target": target,
            "composed_point": composed.point,
            "actual": actual,
            "error": composed.point - actual,
        })
        # Direct: just predict headline_yoy[target] = headline_yoy[origin]
        direct = float(headline_yoy.loc[origin])
        direct_rows.append({
            "origin": origin, "target": target,
            "direct_point": direct,
            "actual": actual,
            "error": direct - actual,
        })

    composed_df = pd.DataFrame(composed_rows)
    direct_df = pd.DataFrame(direct_rows)
    merged = composed_df.merge(direct_df, on=["origin", "target", "actual"])

    # ── Comparison ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Composed (per-component persistence + CBDF) vs Direct headline persistence")
    print("=" * 72)

    rmse_composed = np.sqrt(np.mean(merged["error_x"] ** 2))
    rmse_direct = np.sqrt(np.mean(merged["error_y"] ** 2))
    mae_composed = np.mean(np.abs(merged["error_x"]))
    mae_direct = np.mean(np.abs(merged["error_y"]))

    print(f"  n_origins         = {len(merged)}")
    print(f"  Composed RMSE     = {rmse_composed:.4f} pp")
    print(f"  Direct   RMSE     = {rmse_direct:.4f} pp")
    print(f"  Composed MAE      = {mae_composed:.4f} pp")
    print(f"  Direct   MAE      = {mae_direct:.4f} pp")

    # Composition residual: how close are composed vs direct points?
    # (Should be small — only differs by the published vs reconstructed
    # composition mismatch, which is < 0.5 pp 94% of the time per
    # composition_check/FINDINGS.md)
    comp_residual = merged["composed_point"] - merged["direct_point"]
    print()
    print("Per-origin: composed_point − direct_point (should be small):")
    print(f"  median  = {comp_residual.median():+.4f} pp")
    print(f"  p10/p90 = {comp_residual.quantile(0.10):+.4f} / "
          f"{comp_residual.quantile(0.90):+.4f} pp")
    print(f"  max abs = {comp_residual.abs().max():.4f} pp")

    # Save artifacts
    out_path = OUT_DIR / "phase_2_1_end_to_end_results.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
