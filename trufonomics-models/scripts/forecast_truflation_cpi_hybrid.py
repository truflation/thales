"""Truflation US CPI YoY — Phase 1.75 hybrid forecaster.

Phase 1 has the right point forecast (anchor-corrected to actual YoY at
origin, ~0 bias) but bands are too narrow (66% empirical 80% coverage
at h=30d). Phase 2 has the right *shape* density (regime-aware widths,
78.8% empirical 80% coverage at h=30d) but the wrong point (-1.02 pp
bias from random-walk-trend lag).

Hybrid: take Phase 2's sample distribution and *recenter* it on Phase
1's point. Quantile-based bands shift uniformly, so the hybrid bands
are simply::

    hybrid_lo = phase2_lo + (phase1_point - phase2_point)
    hybrid_hi = phase2_hi + (phase1_point - phase2_point)
    hybrid_point = phase1_point

This is equivalent to translating Phase 2's sample distribution to be
centered on Phase 1's point, preserving the regime-aware width.

Reads:
    results/truflation_cpi_forecast/walk_forward_summary_phase2.csv

Calls Phase 1's forecaster at each Phase 2 origin to get Phase 1 points
at h ∈ {30, 90} days.

Writes:
    results/truflation_cpi_forecast/walk_forward_summary_hybrid.csv
    results/truflation_cpi_forecast/walk_forward_aggregate_hybrid.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse Phase 1's wired pieces
from forecast_truflation_cpi_bottomup import (   # noqa: E402
    HORIZONS_DAYS,
    VINTAGE_DB,
    load_component_levels,
    load_truflation_headline_yoy,
    run_forecast_at_origin,
)

OUT_DIR = ROOT / "results" / "truflation_cpi_forecast"
P2_CSV = OUT_DIR / "walk_forward_summary_phase2.csv"

# Map Phase 2's monthly horizons to Phase 1's daily horizons
M_TO_D = {1: 30, 3: 90}


def main() -> None:
    if not P2_CSV.exists():
        raise SystemExit(
            f"Phase 2 results not found at {P2_CSV}. Run "
            "scripts/forecast_truflation_cpi_phase2.py first.")
    p2 = pd.read_csv(P2_CSV, parse_dates=["origin", "target_date"])
    p2_origins = sorted(p2["origin"].unique())
    print(f"Phase 2 has {len(p2_origins)} origins, "
            f"{p2_origins[0].date()} → {p2_origins[-1].date()}")

    # Load Phase 1 (top12) panel + weights
    print("\nLoading Phase 1 (top12) panel + weights…")
    con = duckdb.connect(str(VINTAGE_DB), read_only=True)
    component_levels, weights_pct = load_component_levels(
        con, crosswalk_level="top12")
    con.close()
    actual_yoy = load_truflation_headline_yoy()

    # Run Phase 1 at each Phase 2 origin, collect point forecasts at h ∈ {30, 90}
    rows = []
    print(f"\nRunning Phase 1 forecaster at {len(p2_origins)} origins…")
    for i, origin in enumerate(p2_origins, 1):
        # Phase 1 needs daily reference_date in component_levels.
        # The origin (last day of month) should be present.
        if origin not in component_levels.index:
            available = component_levels.index[
                component_levels.index <= origin]
            if len(available) == 0:
                continue
            origin_use = available[-1]
        else:
            origin_use = origin
        anchor = (float(actual_yoy.loc[origin_use])
                    if origin_use in actual_yoy.index else None)
        # Use Phase 1's run_forecast_at_origin
        headline = run_forecast_at_origin(
            component_levels, weights_pct, origin_use,
            horizons=[30, 90], anchor_yoy=anchor)
        for h_d, fc in headline.items():
            if fc is None:
                continue
            rows.append({
                "origin": origin,
                "horizon_days": h_d,
                "phase1_point": float(fc["point"]),
                "phase1_target_date": pd.to_datetime(fc["target_date"]),
            })
        if i % 5 == 0:
            print(f"  [{i}/{len(p2_origins)}]")
    p1 = pd.DataFrame(rows)
    print(f"  Got {len(p1)} Phase 1 points")

    # Join Phase 1 + Phase 2 on (origin, horizon_days_label)
    p2_join = p2.copy()
    p2_join["horizon_days"] = p2_join["horizon_days_label"].astype(int)
    merged = p2.merge(
        p1.rename(columns={"phase1_point": "p1_point"}),
        left_on=["origin", "horizon_days_label"],
        right_on=["origin", "horizon_days"],
        how="inner",
    )
    print(f"\nMerged: {len(merged)} (origin, horizon) rows")

    # Hybrid: shift Phase 2 bands to Phase 1 point
    merged["hybrid_point"] = merged["p1_point"]
    merged["shift"] = merged["p1_point"] - merged["point"]    # P1 - P2
    merged["lo80_hyb"] = merged["lo80"] + merged["shift"]
    merged["hi80_hyb"] = merged["hi80"] + merged["shift"]
    merged["lo95_hyb"] = merged["lo95"] + merged["shift"]
    merged["hi95_hyb"] = merged["hi95"] + merged["shift"]

    # Score the hybrid
    merged["hyb_error_pp"] = merged["hybrid_point"] - merged["actual"]
    merged["hyb_in_80"] = ((merged["lo80_hyb"] <= merged["actual"])
                                & (merged["actual"] <= merged["hi80_hyb"]))
    merged["hyb_in_95"] = ((merged["lo95_hyb"] <= merged["actual"])
                                & (merged["actual"] <= merged["hi95_hyb"]))
    merged["hyb_width80"] = merged["hi80_hyb"] - merged["lo80_hyb"]
    merged["hyb_width95"] = merged["hi95_hyb"] - merged["lo95_hyb"]

    scored = merged.dropna(subset=["actual"]).copy()
    print("\nHybrid walk-forward summary by horizon:")
    agg = scored.groupby("horizon_days_label").agg(
        n=("actual", "count"),
        rmse=("hyb_error_pp", lambda x: float(np.sqrt(np.mean(x ** 2)))),
        mae=("hyb_error_pp", lambda x: float(np.mean(np.abs(x)))),
        mean_err=("hyb_error_pp", "mean"),
        cov_80=("hyb_in_80", "mean"),
        cov_95=("hyb_in_95", "mean"),
        width80=("hyb_width80", "mean"),
        width95=("hyb_width95", "mean"),
    ).reset_index().rename(columns={"horizon_days_label": "horizon_days"})
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Side-by-side table
    print("\nHead-to-head at the Phase 2 schedule (33 origins):")
    print(f"{'horizon':>8s}  {'P1 RMSE':>9s}  {'P2 RMSE':>9s}  "
            f"{'Hyb RMSE':>9s}  {'P1 cov80':>9s}  {'P2 cov80':>9s}  "
            f"{'Hyb cov80':>9s}")
    for h_d in [30, 90]:
        sub_p2 = p2[p2["horizon_days_label"] == h_d].dropna(subset=["actual"])
        sub_hyb = scored[scored["horizon_days_label"] == h_d]
        # P1 only error (against same actual)
        p1_err = (sub_hyb["p1_point"] - sub_hyb["actual"]).values
        p1_in80 = ((sub_hyb["lo80"] - sub_hyb["shift"] <=  # noqa: F841
                       sub_hyb["actual"])).mean()  # not used
        p1_rmse = float(np.sqrt(np.mean(p1_err ** 2)))
        p2_rmse = float(np.sqrt(((sub_p2["point"] - sub_p2["actual"]) ** 2).mean()))
        hyb_rmse = float(np.sqrt((sub_hyb["hyb_error_pp"] ** 2).mean()))
        p2_cov80 = float(sub_p2["in_80"].mean())
        hyb_cov80 = float(sub_hyb["hyb_in_80"].mean())
        # P1 cov80 at this schedule = uses Phase 1's *original* bands; we
        # don't have those at these origins — flag as N/A
        p1_cov80_label = "n/a"
        print(f"{h_d:>6d}d  {p1_rmse:>9.4f}  {p2_rmse:>9.4f}  "
                f"{hyb_rmse:>9.4f}  {p1_cov80_label:>9s}  "
                f"{p2_cov80:>9.4f}  {hyb_cov80:>9.4f}")

    out_csv = OUT_DIR / "walk_forward_summary_hybrid.csv"
    keep_cols = ["origin", "horizon_days_label", "target_date", "actual",
                  "p1_point", "point", "hybrid_point", "shift",
                  "lo80_hyb", "hi80_hyb", "lo95_hyb", "hi95_hyb",
                  "hyb_error_pp", "hyb_in_80", "hyb_in_95",
                  "hyb_width80", "hyb_width95"]
    merged[keep_cols].to_csv(out_csv, index=False)
    agg_csv = OUT_DIR / "walk_forward_aggregate_hybrid.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {agg_csv}")


if __name__ == "__main__":
    main()
