"""Score Phase 2 (UC+SV+MS, monthly) vs Phase 1 (bottom-up AR(1), daily)
and persistence at h ∈ {30d, 90d}.

Phase 1 walk-forward steps every 30 days, Phase 2 every 3 months. They
share targets only when origin + h aligns. We score each independently
on its own walk-forward set, then compare aggregate stats.

Reads:
    results/truflation_cpi_forecast/walk_forward_summary_phase2.csv
    results/truflation_cpi_forecast/walk_forward_summary_top12.csv

Outputs:
    results/truflation_cpi_forecast/phase2_vs_phase1_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation.metrics import rmse, mae    # noqa: E402

OUT_DIR = ROOT / "results" / "truflation_cpi_forecast"


def _persistence_rmse_for_phase2(p2: pd.DataFrame) -> dict[int, dict]:
    """Persistence baseline at the Phase 2 target schedule.

    For each Phase 2 (origin, h) row, the persistence prediction is YoY
    at origin (use the actual at h=1m row of the same origin if present;
    otherwise use latest published value at origin date).
    """
    # Phase 2 has target rows only — origins lack a self-label. We
    # approximate persistence by looking up the actual Truflation value
    # at the origin date from the Phase 2 actuals series itself by
    # joining (origin) → (target where target == origin from a different
    # row).
    p2 = p2.copy()
    p2["origin"] = pd.to_datetime(p2["origin"])
    p2["target_date"] = pd.to_datetime(p2["target_date"])
    # Build a lookup of {date → actual} from Phase 2 (target_date, actual)
    # rows; fall back to whatever's available.
    actual_lookup = (p2.dropna(subset=["actual"])
                     .drop_duplicates("target_date")
                     .set_index("target_date")["actual"]
                     .to_dict())

    out: dict[int, dict] = {}
    for h_m in sorted(p2["horizon_months"].unique()):
        sub = p2[p2["horizon_months"] == h_m].dropna(subset=["actual"]).copy()
        # Persistence prediction = actual at origin (closest target_date
        # within the lookup; if origin is exactly someone's target_date,
        # we have it)
        sub["persistence"] = sub["origin"].map(actual_lookup)
        sub = sub.dropna(subset=["persistence"])
        bu_err = (sub["point"] - sub["actual"]).values
        pe_err = (sub["persistence"] - sub["actual"]).values
        out[int(h_m)] = {
            "n": len(sub),
            "rmse_p2": rmse(sub["point"].values, sub["actual"].values),
            "rmse_persist": rmse(sub["persistence"].values, sub["actual"].values),
            "mae_p2": mae(sub["point"].values, sub["actual"].values),
            "mae_persist": mae(sub["persistence"].values, sub["actual"].values),
            "bias_p2": float(np.mean(bu_err)),
        }
    return out


def main() -> None:
    p2_csv = OUT_DIR / "walk_forward_summary_phase2.csv"
    p1_csv = OUT_DIR / "walk_forward_summary_top12.csv"
    if not p2_csv.exists():
        raise SystemExit(
            f"Phase 2 results not found at {p2_csv}. Run "
            "scripts/forecast_truflation_cpi_phase2.py first.")

    p2 = pd.read_csv(p2_csv, parse_dates=["origin", "target_date"])
    p1 = pd.read_csv(p1_csv, parse_dates=["origin", "target_date"])

    p2_scored = p2.dropna(subset=["actual"])
    p1_30 = p1[(p1["horizon_days"] == 30)].dropna(subset=["actual"])
    p1_90 = p1[(p1["horizon_days"] == 90)].dropna(subset=["actual"])

    # Phase 2 own evaluation
    p2_agg = p2_scored.groupby("horizon_months").agg(
        n=("actual", "count"),
        rmse=("error_pp", lambda x: float(np.sqrt(np.mean(x ** 2)))),
        mae=("error_pp", lambda x: float(np.mean(np.abs(x)))),
        mean_err=("error_pp", "mean"),
        cov_80=("in_80", "mean"),
        cov_95=("in_95", "mean"),
        width80=("width80_pp", "mean"),
        width95=("width95_pp", "mean"),
    ).reset_index()
    print("\nPhase 2 walk-forward summary (UC+SV+MS, monthly):")
    print(p2_agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Persistence at the Phase 2 schedule
    p2_persist = _persistence_rmse_for_phase2(p2)
    rows = []
    for h_m, stats in p2_persist.items():
        h_d = 30 if h_m == 1 else 90
        red_rmse = (1 - stats["rmse_p2"] / stats["rmse_persist"]) * 100
        red_mae = (1 - stats["mae_p2"] / stats["mae_persist"]) * 100
        rows.append({
            "horizon_months": h_m,
            "horizon_days_label": h_d,
            "n": stats["n"],
            "rmse_phase2": round(stats["rmse_p2"], 4),
            "rmse_persistence": round(stats["rmse_persist"], 4),
            "rmse_reduction_pct": round(red_rmse, 2),
            "mae_phase2": round(stats["mae_p2"], 4),
            "mae_persistence": round(stats["mae_persist"], 4),
            "mae_reduction_pct": round(red_mae, 2),
            "bias_phase2": round(stats["bias_p2"], 4),
        })
    print("\nPhase 2 vs Persistence:")
    print(pd.DataFrame(rows).to_string(index=False))

    # Phase 1 (top12) reference numbers from the existing summary
    p1_30_rmse = float(np.sqrt((p1_30["error_pp"] ** 2).mean()))
    p1_90_rmse = float(np.sqrt((p1_90["error_pp"] ** 2).mean()))
    p1_30_cov80 = float(p1_30["in_80"].mean())
    p1_90_cov80 = float(p1_90["in_80"].mean())
    print("\nPhase 1 (top12) reference (different schedule, n different):")
    print(f"  h=30d  RMSE={p1_30_rmse:.4f}  cov80={p1_30_cov80:.3f}  "
            f"n={len(p1_30)}")
    print(f"  h=90d  RMSE={p1_90_rmse:.4f}  cov80={p1_90_cov80:.3f}  "
            f"n={len(p1_90)}")

    # Headline comparison
    print("\nHeadline comparison (RMSE pp; lower is better):")
    print(f"{'horizon':>8}  {'Phase 1':>9}  {'Phase 2':>9}  "
            f"{'Persist (P2 sched)':>20}")
    for r in rows:
        h_lbl = f"{r['horizon_days_label']}d"
        p1_rmse = p1_30_rmse if r["horizon_months"] == 1 else p1_90_rmse
        print(f"{h_lbl:>8}  {p1_rmse:>9.4f}  {r['rmse_phase2']:>9.4f}  "
                f"{r['rmse_persistence']:>20.4f}")

    # Save
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "phase2_vs_phase1_summary.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'phase2_vs_phase1_summary.csv'}")


if __name__ == "__main__":
    main()
