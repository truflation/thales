"""Score Phase 1 / 1.5 bottom-up forecasts vs naive persistence baseline.

Persistence at horizon h: predict YoY[origin] for YoY[origin + h]. The
RMSE-reduction vs persistence is the headline number Phase 2+ DL models
must beat.

Reads:
    results/truflation_cpi_forecast/walk_forward_summary_<label>.csv

Outputs:
    results/truflation_cpi_forecast/persistence_comparison_<label>.csv
    Console: per-horizon RMSE table for {bottom-up, persistence}, +DM test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation.metrics import rmse, mae    # noqa: E402
from thales.evaluation.tests import diebold_mariano    # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="top12",
                        help="Walk-forward output label "
                              "(top12 = Phase 1, leaves58 = Phase 1.5)")
    args = parser.parse_args()
    csv_path = (ROOT / "results" / "truflation_cpi_forecast"
                  / f"walk_forward_summary_{args.label}.csv")
    df = pd.read_csv(csv_path, parse_dates=["origin", "target_date"])
    df = df.dropna(subset=["actual"]).copy()

    # Persistence baseline = actual YoY at origin. The h=1 bottom-up point is
    # anchor-corrected to match actual YoY at origin within rounding, so use
    # it as a clean per-origin anchor source.
    anchor_by_origin = (df[df["horizon_days"] == 1]
                        .set_index("origin")["point"]
                        .to_dict())

    rows = []
    for h in sorted(df["horizon_days"].unique()):
        sub = df[df["horizon_days"] == h].copy()
        sub["persistence"] = sub["origin"].map(anchor_by_origin)
        sub = sub.dropna(subset=["persistence"])

        bu_err = (sub["point"] - sub["actual"]).values
        pe_err = (sub["persistence"] - sub["actual"]).values

        bu_rmse = rmse(sub["point"].values, sub["actual"].values)
        pe_rmse = rmse(sub["persistence"].values, sub["actual"].values)
        bu_mae = mae(sub["point"].values, sub["actual"].values)
        pe_mae = mae(sub["persistence"].values, sub["actual"].values)

        red_rmse = (1 - bu_rmse / pe_rmse) * 100 if pe_rmse > 0 else float("nan")
        red_mae = (1 - bu_mae / pe_mae) * 100 if pe_mae > 0 else float("nan")

        # Diebold-Mariano: positive stat → bottom-up beats persistence
        dm = diebold_mariano(pe_err, bu_err, lag=3, two_sided=True,
                              loss="squared")

        rows.append({
            "horizon_days": int(h),
            "n": len(sub),
            "rmse_bu": round(bu_rmse, 4),
            "rmse_persist": round(pe_rmse, 4),
            "rmse_reduction_pct": round(red_rmse, 2),
            "mae_bu": round(bu_mae, 4),
            "mae_persist": round(pe_mae, 4),
            "mae_reduction_pct": round(red_mae, 2),
            "dm_statistic": round(dm.statistic, 3) if not np.isnan(dm.statistic) else None,
            "dm_pvalue": round(dm.pvalue, 4) if not np.isnan(dm.pvalue) else None,
        })

    out = pd.DataFrame(rows)
    print(f"\nBottom-up ({args.label}) vs Persistence baseline:\n")
    print(out.to_string(index=False))

    out_csv = (ROOT / "results" / "truflation_cpi_forecast"
                  / f"persistence_comparison_{args.label}.csv")
    out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
