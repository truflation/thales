"""Phase 3.2 — 6-variable restaurants BVAR.

Endogenous vector with Cholesky ordering (most exogenous → most endogenous):

  1. log_food_cogs    — PPI: Final demand foods (upstream commodity prices)
  2. log_utilities    — CPI: Energy services
  3. log_rent         — PPI: Lessors of nonresidential buildings
  4. log_labor        — L&H avg hourly earnings (restaurant wages)
  5. log_menu         — CPI: Food away from home (output / pricing side)
  6. log_traffic      — Retail sales: Food services & drinking places (volume)

Cost-structure attribution from the registry:
  food_cogs 30%, labor 30%, rent 8%, utilities 4%, other 28%

Customer scenario: "If food commodity prices spike +20%, what happens to
my menu pricing power, traffic, and downstream cost lines?"
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
    shock_scenario,
)
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "real_data_archetypes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_panel() -> pd.DataFrame:
    """Restaurant 6-var panel in **MoM log-differences** — stationary,
    avoids the shared-trend double-counting that inflated IRFs in the
    log-level version. Same lesson as Fix #5: model MoM, not levels.

    Traffic is real (deflated by menu CPI) before differencing so it
    doesn't carry inflation in its growth rate.
    """
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        food = store.get_vintage("WPSFD49207", date.today()).dropna()
        utils = store.get_vintage("CUSR0000SEHF", date.today()).dropna()
        rent = store.get_vintage("PCU531120531120", date.today()).dropna()
        labor = store.get_vintage("CES7000000008", date.today()).dropna()
        menu = store.get_vintage("CUSR0000SEFV", date.today()).dropna()
        traffic_nominal = store.get_vintage("RSFSDP", date.today()).dropna()
    food_m = food.resample("ME").last()
    utils_m = utils.resample("ME").last()
    rent_m = rent.resample("ME").last()
    labor_m = labor.resample("ME").last()
    menu_m = menu.resample("ME").last()
    traffic_m = traffic_nominal.resample("ME").last()
    real_traffic = (traffic_m / menu_m * 100.0).dropna()
    levels = pd.concat({
        "log_food_cogs": np.log(food_m),
        "log_utilities": np.log(utils_m),
        "log_rent":      np.log(rent_m),
        "log_labor":     np.log(labor_m),
        "log_menu":      np.log(menu_m),
        "log_traffic":   np.log(real_traffic),
    }, axis=1).dropna()
    # First-difference (×100 → MoM in pp for readability)
    return (levels.diff() * 100.0).dropna()


def main() -> None:
    print("=" * 78)
    print("Phase 3.2 — 6-variable restaurants BVAR")
    print("=" * 78)

    cs = get_cost_structure("restaurants")
    print(f"\nCost structure: {cs.industry} (source: {cs.source})")
    for k, v in cs.weights.items():
        print(f"  {k:<14s} {v:>5.0%}")

    panel = _load_panel()
    # Rename to mom_* so the rest of the script's interpretation is honest
    panel.columns = [c.replace("log_", "mom_") for c in panel.columns]
    var_cols = list(panel.columns)
    Y = panel.values
    print(f"\nPanel: n={len(panel)}  range "
          f"{panel.index.min():%Y-%m} → {panel.index.max():%Y-%m}")
    print("  (panel is MoM log-differences in pp — frame is first-differences)")
    for c in var_cols:
        s = panel[c]
        print(f"  {c:<16s}  mean={s.mean():+.3f}  sd={s.std():.3f}  "
                f"AC1={s.autocorr(1):+.3f}")

    # ── Static fit ──────────────────────────────────────────────────
    print("\n── Static fit p=1, Minnesota prior ──")
    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=0.5,
                                       cross_tightness=0.5, lag_decay=1.0)
    A_list = _ar_matrices(fit.coefs, fit.k, fit.p)
    eig = np.abs(np.linalg.eigvals(_companion_matrix(A_list)))
    print(f"  n_train={fit.n_train}  max|eig|={eig.max():.4f}  "
            f"({'STABLE' if eig.max() < 1.0 else 'NON-STATIONARY'})")
    print("  AR(1) matrix [rows: equation, cols: lag-1 of variable]:")
    print("  " + "  ".join(f"{c[4:11]:>9s}" for c in var_cols))
    for i, name in enumerate(var_cols):
        row = "  ".join(f"{A_list[0][i, j]:+9.4f}" for j in range(fit.k))
        print(f"  {row}    ← {name}")

    # ── IRF: food_cogs shock effect ─────────────────────────────────
    print("\n── Cholesky IRF, 24-month horizon (1-SD food_cogs shock) ──")
    print("  ordering:", " → ".join(c[4:11] for c in var_cols))
    print("  values are response in MoM growth rate (pp)")
    irf = cholesky_irf(fit, h=24)
    print(f"\n  {'h':>3s}  ", end="")
    for c in var_cols:
        print(f"{c[4:11]:>9s}  ", end="")
    print()
    for h in [0, 1, 3, 6, 12, 24]:
        print(f"  h={h:>2d} ", end="")
        for i in range(fit.k):
            # IRF is already in pp (panel is in pp)
            print(f"{irf[h, i, 0]:+8.3f}pp  ", end="")
        print()
    print("\n  Cumulative LEVEL effect at h=12 (sum of MoM responses, pp):")
    print(f"  {'level Δ pp':>11s}  ", end="")
    for i in range(fit.k):
        cum = float(np.sum(irf[:13, i, 0]))    # h=0..12
        print(f"{cum:+8.3f}pp  ", end="")
    print()

    # ── FEVD at h=12 ────────────────────────────────────────────────
    print("\n── FEVD at h=12 (variance share by shock, %) ──")
    f = fevd(fit, h=24)
    fevd_df = pd.DataFrame((f[12] * 100).round(1),
                                  index=[c for c in var_cols],
                                  columns=[c for c in var_cols])
    fevd_df.index.name = "response"
    fevd_df.columns.name = "shock"
    print(fevd_df.to_string())

    # ── Walk-forward forecasts ──────────────────────────────────────
    print("\n── Walk-forward 1-month forecasts ──")
    print(f"  {'target':<16s}  {'n':>4s}  {'RMSE':>8s}  {'naive':>8s}  "
            f"{'Δ%':>7s}  {'cov80':>6s}  {'cov95':>6s}")
    print("  " + "-" * 70)
    summary_rows = []
    for target in var_cols:
        fc = BVARForecaster(
            var_cols=var_cols, target_col=target,
            horizon=1, p=1, overall_tightness=0.5,
            train_min=60, model_id=f"bvar_restaurants_{target}")
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
        red = (f"{block.rmse_reduction_pct:+.2f}%"
                  if block.rmse_reduction_pct is not None else "—")
        cov80 = f"{block.cov80:.1%}" if block.cov80 is not None else "—"
        cov95 = f"{block.cov95:.1%}" if block.cov95 is not None else "—"
        print(f"  {target:<16s}  {block.n:>4d}  {block.rmse:>8.5f}  "
                f"{block.rmse_naive:>8.5f}  {red:>7s}  "
                f"{cov80:>6s}  {cov95:>6s}")

    # ── Customer scenario: +20% food commodity shock ─────────────────
    # In the MoM frame, "+20% food shock" = a one-time +20pp jump in
    # the food MoM growth rate that month. We translate the per-step
    # MoM responses into a CUMULATIVE level deviation (the right
    # customer-facing metric: "how much does my menu price level move
    # 12 months after the shock?").
    print("\n" + "=" * 78)
    print("Customer scenario: +20pp food commodity MoM shock — exposure map")
    print("=" * 78)
    print("  (one-time +20pp MoM jump in food_cogs; cumulative level")
    print("   effect on each variable shown over 12 months)")

    h = 12
    # Shock is in pp (matches panel units)
    shock_scenarios = {
        "+20pp food_cogs MoM":  20.0,
        "+50pp food_cogs MoM":  50.0,
        "-20pp food_cogs MoM":  -20.0,
    }
    food_idx = 0

    for scen_name, shock_size in shock_scenarios.items():
        traj = shock_scenario(fit, baseline=Y[-1],
                                     shock_var_idx=food_idx,
                                     shock_size=shock_size, h=h)
        print(f"\n── {scen_name} ──")
        print(f"  CUMULATIVE level deviation by horizon (pp):")
        print(f"  {'h':>3s}  ", end="")
        for c in var_cols:
            print(f"{c[4:11]:>9s}  ", end="")
        print()
        for hh in [0, 1, 3, 6, 12]:
            print(f"  h={hh:>2d} ", end="")
            for i in range(fit.k):
                # Cumulative sum of MoM responses up to horizon hh
                cum_pp = float(np.sum(traj[:hh + 1, i]))
                print(f"{cum_pp:+8.3f}pp  ", end="")
            print()

    # ── Cost-line exposure for $10M-revenue restaurant ──────────────
    print("\n" + "=" * 78)
    print("Cost-line exposure map for a $10M-revenue restaurant")
    print("=" * 78)
    print(f"  cost structure: {dict(cs.weights)}")
    revenue = 10_000_000
    op_cost_share = 0.95   # restaurants run higher opex than logistics
    cost_pool = revenue * op_cost_share

    var_to_bucket = {
        "mom_food_cogs": "food_cogs",
        "mom_utilities": "utilities",
        "mom_rent":      "rent",
        "mom_labor":     "labor",
        "mom_menu":      None,    # output, not cost
        "mom_traffic":   None,    # demand, not cost
    }

    print(f"\n  baseline annual opex pool: ${cost_pool/1e6:.1f}M "
            f"({op_cost_share*100:.0f}% of $10M revenue)")
    print("  Δ$ shown is the EXPOSURE: cost-line dollars under the scenario")
    print("  trajectory averaged over a 12-month horizon.\n")
    print(f"  {'scenario':<22s}  ", end="")
    for bucket in ["food_cogs", "labor", "rent", "utilities"]:
        print(f"{bucket+' ΔS':>14s}  ", end="")
    print(f"{'TOTAL Δ$':>14s}  {'menu Δpp':>9s}  {'traffic Δpp':>11s}")
    print("  " + "-" * 110)

    for scen_name, shock_size in shock_scenarios.items():
        traj = shock_scenario(fit, baseline=Y[-1],
                                     shock_var_idx=food_idx,
                                     shock_size=shock_size, h=h)
        print(f"  {scen_name:<22s}  ", end="")
        total = 0.0
        for bucket in ["food_cogs", "labor", "rent", "utilities"]:
            var_idx = None
            for i, col in enumerate(var_cols):
                if var_to_bucket.get(col) == bucket:
                    var_idx = i
                    break
            if var_idx is None:
                continue
            # Average cumulative level deviation over the horizon
            # (each cumulative path point is in pp; convert to fractional).
            avg_cum_pp = float(np.mean([np.sum(traj[:hh+1, var_idx])
                                                  for hh in range(h)]))
            avg_frac = avg_cum_pp / 100.0
            bucket_dollars = cost_pool * cs.weights[bucket] * avg_frac
            total += bucket_dollars
            print(f"{bucket_dollars:+>13,.0f}  ", end="")
        # Output / demand-side: cumulative level deviation at horizon h
        menu_idx = var_cols.index("mom_menu")
        traffic_idx = var_cols.index("mom_traffic")
        menu_cum = float(np.sum(traj[:h+1, menu_idx]))
        traffic_cum = float(np.sum(traj[:h+1, traffic_idx]))
        print(f"{total:+>13,.0f}  {menu_cum:+>8.2f}pp  {traffic_cum:+>10.2f}pp")

    print("\n  Note: menu Δ% is the model's exposure forecast for your")
    print("  pricing-side line. Traffic Δ% is the demand-side response.")
    print("  Customer agency: how to respond (raise menu prices, absorb")
    print("  margin compression, etc.) is the customer's decision.")

    # Persist
    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / "bvar_restaurants_6var_summary.csv", index=False)
    fevd_df.to_csv(OUT_DIR / "bvar_restaurants_6var_fevd_h12.csv")
    pd.DataFrame(
        {f"{var_cols[i]}<-{var_cols[j]}": irf[:, i, j]
            for i in range(fit.k) for j in range(fit.k)}
    ).to_csv(OUT_DIR / "bvar_restaurants_6var_irf.csv",
                 index_label="horizon_months")
    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    main()
