"""Phase 3.1d demo — diesel-shock scenarios on real logistics BVAR.

Customer-facing question: "If diesel jumps +20%, what happens to my
freight rates, labor costs, maintenance, and tonnage?"

Answer mechanism: ``shock_scenario()`` — runs a one-time structural
shock through the BVAR's IRF (which captures both the contemporaneous
Σ-correlation channel AND the AR cross-effects). Three scenarios:

  * +20% diesel shock
  * −20% diesel shock
  * +50% diesel shock (peak-2022-style)

Output: 12-month deviation trajectory for every variable, plus a
dollar-P&L translation using the cost-structure DB.
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
from thales.models.archetypes.bvar_minnesota import (  # noqa: E402
    fit_bvar_minnesota,
    shock_scenario,
)
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "real_data_archetypes"


def _load_panel() -> pd.DataFrame:
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        diesel = store.get_vintage("GASDESW", date.today()).dropna()
        freight = store.get_vintage("PCU48414841", date.today()).dropna()
        maint = store.get_vintage("CUSR0000SETD", date.today()).dropna()
        labor = store.get_vintage("CES4300000008", date.today()).dropna()
        volume = store.get_vintage("TRUCKD11", date.today()).dropna()
    return pd.concat({
        "log_diesel":      np.log(diesel.resample("ME").last()),
        "log_freight":     np.log(freight.resample("ME").last()),
        "log_maintenance": np.log(maint.resample("ME").last()),
        "log_labor":       np.log(labor.resample("ME").last()),
        "log_volume":      np.log(volume.resample("ME").last()),
    }, axis=1).dropna()


def main() -> None:
    print("=" * 78)
    print("Phase 3.1d — diesel shock scenarios on logistics BVAR")
    print("=" * 78)

    panel = _load_panel()
    var_cols = list(panel.columns)
    Y = panel.values
    print(f"\nFit panel: n={len(panel)}  range "
          f"{panel.index.min():%Y-%m} → {panel.index.max():%Y-%m}")

    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=0.5,
                                       cross_tightness=0.5, lag_decay=1.0)

    # Three shock magnitudes (in log-space; +20% = log(1.20) = 0.182)
    h = 12
    scenarios = {
        "+20% diesel": np.log(1.20),
        "+50% diesel (peak-2022 style)": np.log(1.50),
        "-20% diesel": np.log(0.80),
    }
    diesel_idx = 0    # log_diesel is column 0

    for scen_name, shock_size in scenarios.items():
        traj = shock_scenario(fit, baseline=Y[-1],
                                     shock_var_idx=diesel_idx,
                                     shock_size=shock_size, h=h)
        print(f"\n── scenario: {scen_name} ──")
        print(f"  Trajectory (deviation from no-shock counterfactual, %):")
        print(f"  {'horizon':>3s}  ", end="")
        for col in var_cols:
            print(f"{col[4:11]:>8s}  ", end="")
        print()
        for hh in [0, 1, 3, 6, 12]:
            print(f"  h={hh:>2d}     ", end="")
            for i in range(fit.k):
                pct = (np.exp(traj[hh, i]) - 1) * 100
                print(f"{pct:+7.2f}%  ", end="")
            print()

    # ── Dollar P&L for $100M-revenue shipper ────────────────────────
    cs = get_cost_structure("logistics")
    print("\n" + "=" * 78)
    print("Translation: 12-month cumulative cost-line impact")
    print("for a $100M-revenue logistics company")
    print("=" * 78)
    print(f"  cost structure (from {cs.source}):")
    for k, v in cs.weights.items():
        print(f"    {k:<14s} {v:>5.0%}")

    revenue = 100_000_000
    op_cost_share = 0.85
    cost_pool = revenue * op_cost_share
    print(f"\n  baseline opex pool: ${cost_pool/1e6:.1f}M "
            f"(85% of $100M revenue)")
    print()

    var_to_bucket = {
        "log_diesel":      "fuel",
        "log_freight":     None,             # output, not cost
        "log_maintenance": "maintenance",
        "log_labor":       "labor",
        "log_volume":      None,             # demand, not cost
    }

    print(f"  {'scenario':<32s}  ", end="")
    for bucket in ["fuel", "labor", "maintenance"]:
        print(f"{bucket+' Δ$':>14s}  ", end="")
    print(f"{'TOTAL Δ$':>14s}  {'as % opex':>10s}")
    print("  " + "-" * 100)

    for scen_name, shock_size in scenarios.items():
        traj = shock_scenario(fit, baseline=Y[-1],
                                     shock_var_idx=diesel_idx,
                                     shock_size=shock_size, h=h)
        print(f"  {scen_name:<32s}  ", end="")
        total = 0.0
        for bucket in ["fuel", "labor", "maintenance"]:
            var_idx = None
            for i, col in enumerate(var_cols):
                if var_to_bucket.get(col) == bucket:
                    var_idx = i
                    break
            if var_idx is None:
                continue
            # Average deviation over the 12-month horizon (cost is a
            # flow, integrated over the year)
            avg_dev = float(np.mean(np.exp(traj[1:h+1, var_idx]) - 1))
            bucket_dollars = cost_pool * cs.weights[bucket] * avg_dev
            total += bucket_dollars
            print(f"{bucket_dollars:+>13,.0f}  ", end="")
        print(f"{total:+>13,.0f}  {total/cost_pool:>9.2%}")

    print("\n  Note: insurance + 'other' cost lines are held at zero — those")
    print("  variables aren't in the BVAR (data gap). Real impact is larger.")


if __name__ == "__main__":
    main()
