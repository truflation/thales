"""Phase 3.3 — 4 additional verticals BVAR comparison.

Fits a transmission VAR for each of:
  * Mid-market retail
  * Healthcare operators
  * Real estate operators
  * Manufacturing (durables)

Same architecture as 3.1/3.2: MoM-frame BVAR(1) with Minnesota prior,
Cholesky IRF, FEVD, walk-forward forecasts, customer-facing scenario
exposure map.

For brevity, each vertical reports:
  * Static fit + max|eig| stability check
  * IRF for a 1-SD shock to the most-exogenous variable
  * FEVD at h=12
  * Walk-forward 1-month forecast RMSE per variable
  * One scenario: +20pp shock to the upstream cost variable

Output: per-vertical CSVs + a single cross-vertical summary table.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

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


@dataclass
class VerticalSpec:
    name: str
    industry_key: str
    var_specs: list[tuple[str, str, str, str | None]]
    # tuple = (panel_col_name, fred_id, role_label, cost_bucket_or_None)
    upstream_var_idx: int     # which column to shock in the scenario
    upstream_label: str       # human-readable shock variable

    def load_panel(self, store: VintageStore) -> pd.DataFrame:
        """Load monthly log panel, then return MoM differences in pp."""
        cols = {}
        for panel_name, fred_id, _role, _bucket in self.var_specs:
            s = store.get_vintage(fred_id, date.today()).dropna()
            cols[panel_name] = np.log(s.resample("ME").last())
        levels = pd.concat(cols, axis=1).dropna()
        return (levels.diff() * 100.0).dropna()


VERTICALS = [
    VerticalSpec(
        name="retail_midmarket",
        industry_key="retail_midmarket",
        var_specs=[
            ("mom_cogs",       "PCU423423",       "wholesale durables", "cogs"),
            ("mom_utilities",  "CUSR0000SEHF",    "energy services",    "utilities"),
            ("mom_rent",       "PCU531120531120", "commercial rent",    "rent"),
            ("mom_labor",      "CES4200000008",   "retail wages",       "labor"),
            ("mom_sales",      "RSGMS",           "general merch sales (output/demand)", None),
        ],
        upstream_var_idx=0,
        upstream_label="cogs (wholesale goods)",
    ),

    VerticalSpec(
        name="healthcare_operators",
        industry_key="healthcare_operators",
        var_specs=[
            ("mom_pharma",      "PCU325412325412", "pharma manufacturing PPI", "pharma_supplies"),
            ("mom_utilities",   "CUSR0000SEHF",    "energy services",          "utilities"),
            ("mom_labor",       "CES6500000008",   "healthcare wages",         "labor"),
            ("mom_med_services","CUSR0000SAM2",    "medical care services CPI (output)", None),
        ],
        upstream_var_idx=0,
        upstream_label="pharma_supplies",
    ),

    VerticalSpec(
        name="real_estate_operators",
        industry_key="real_estate_operators",
        var_specs=[
            ("mom_construction", "WPUSI012011",     "construction materials PPI", "maintenance"),
            ("mom_utilities",    "CUSR0000SEHF",    "energy services",            "utilities"),
            ("mom_labor",        "CES5500000008",   "financial-activities wages", "labor"),
            ("mom_shelter",      "CUUR0000SAH1",    "shelter CPI (output)",       None),
            ("mom_construction_emp", "USCONS",      "construction employment",    None),
        ],
        upstream_var_idx=0,
        upstream_label="construction (maintenance materials)",
    ),

    VerticalSpec(
        name="manufacturing_durables",
        industry_key="manufacturing_durables",
        var_specs=[
            ("mom_raw_mat",   "PPIIDC",          "industrial commodities PPI", "raw_materials"),
            ("mom_energy",    "CUSR0000SEHF",    "energy services",            "energy"),
            ("mom_logistics", "PCU48414841",     "freight PPI",                "logistics"),
            ("mom_labor",     "CES3000000008",   "mfg wages",                  "labor"),
            ("mom_ip",        "INDPRO",          "industrial production (output)", None),
        ],
        upstream_var_idx=0,
        upstream_label="raw_materials",
    ),
]


def run_vertical(spec: VerticalSpec, store: VintageStore) -> dict:
    panel = spec.load_panel(store)
    var_cols = list(panel.columns)
    Y = panel.values
    cs = get_cost_structure(spec.industry_key)

    # Static fit
    fit = fit_bvar_minnesota(Y, p=1, overall_tightness=0.5,
                                       cross_tightness=0.5, lag_decay=1.0)
    A_list = _ar_matrices(fit.coefs, fit.k, fit.p)
    eig = np.abs(np.linalg.eigvals(_companion_matrix(A_list))).max()

    # IRF
    irf = cholesky_irf(fit, h=24)

    # FEVD
    f12 = fevd(fit, h=24)[12]    # h=12 share

    # Walk-forward forecasts per variable
    wf_results = []
    for target in var_cols:
        fc = BVARForecaster(
            var_cols=var_cols, target_col=target, horizon=1, p=1,
            overall_tightness=0.5, train_min=60,
            model_id=f"bvar_{spec.name}_{target}")
        origins = panel.index[60:-1]
        forecasts = walk_forward(fc, panel, target, origins, horizon=1)
        if not forecasts:
            continue
        df_pred = attach_actuals(forecasts, panel[target])
        block = score(df_pred)
        wf_results.append({
            "target": target,
            "n": block.n, "rmse": block.rmse,
            "rmse_naive": block.rmse_naive,
            "rmse_red_pct": block.rmse_reduction_pct,
            "cov80": block.cov80, "cov95": block.cov95,
        })

    # Scenario: +20pp shock to upstream cost variable
    traj = shock_scenario(fit, baseline=Y[-1],
                                 shock_var_idx=spec.upstream_var_idx,
                                 shock_size=20.0, h=12)

    # Cumulative level deviation at h=12 per variable (pp)
    cum_h12 = {var_cols[i]: float(np.sum(traj[:13, i]))
                    for i in range(fit.k)}

    # Cost-line dollar exposure for $10M revenue example
    revenue, op_share = 10_000_000, 0.85
    cost_pool = revenue * op_share
    bucket_dollars = {}
    for col, _fred, _role, bucket in spec.var_specs:
        if bucket is None or bucket not in cs.weights:
            continue
        idx = var_cols.index(col)
        avg_cum_pp = float(np.mean([np.sum(traj[:hh+1, idx])
                                              for hh in range(12)]))
        bucket_dollars[bucket] = (cost_pool
                                          * cs.weights[bucket]
                                          * avg_cum_pp / 100.0)
    total_d = sum(bucket_dollars.values())

    return {
        "spec": spec, "panel": panel, "fit": fit, "stability": eig,
        "irf": irf, "fevd_h12": f12, "wf_results": wf_results,
        "scenario_cum_h12": cum_h12, "bucket_dollars": bucket_dollars,
        "scenario_total_dollars": total_d,
    }


def print_vertical_report(r: dict) -> None:
    spec = r["spec"]
    var_cols = list(r["panel"].columns)
    print()
    print("=" * 78)
    print(f"  Vertical: {spec.name}")
    print("=" * 78)
    cs = get_cost_structure(spec.industry_key)
    print(f"\n  Cost structure: {dict(cs.weights)}")
    print(f"  Panel: n={len(r['panel'])} months, {r['fit'].k} variables")
    print(f"  Stability: max|eig|={r['stability']:.4f}  "
            f"({'STABLE' if r['stability'] < 1.0 else 'NON-STATIONARY'})")

    print(f"\n  ── FEVD at h=12 (variance share by shock, %) ──")
    df_fevd = pd.DataFrame((r["fevd_h12"] * 100).round(1),
                                  index=var_cols, columns=var_cols)
    df_fevd.index.name = "response"
    print("  " + df_fevd.to_string().replace("\n", "\n  "))

    print(f"\n  ── Walk-forward 1-month RMSE Δ% vs naive ──")
    for w in r["wf_results"]:
        red = (f"{w['rmse_red_pct']:+.2f}%"
                  if w["rmse_red_pct"] is not None else "—")
        cov80 = f"{w['cov80']:.1%}" if w['cov80'] is not None else "—"
        print(f"    {w['target']:<22s}  RMSE {w['rmse']:.4f}  "
                f"vs naive {w['rmse_naive']:.4f}  ({red})  cov80 {cov80}")

    print(f"\n  ── Scenario: +20pp shock to {spec.upstream_label} ──")
    print(f"  Cumulative level deviation at h=12 (pp):")
    for c, v in r["scenario_cum_h12"].items():
        print(f"    {c:<22s}  {v:+8.2f}pp")
    print(f"\n  Cost-line $ exposure ($10M shipper, 85% opex share):")
    for bucket, d in r["bucket_dollars"].items():
        print(f"    {bucket:<14s}  {d:+>14,.0f}")
    print(f"    {'TOTAL':<14s}  {r['scenario_total_dollars']:+>14,.0f}")


def main() -> None:
    print("=" * 78)
    print("Phase 3.3 — 4-vertical BVAR comparison")
    print("=" * 78)

    summary_rows = []
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        for spec in VERTICALS:
            r = run_vertical(spec, store)
            print_vertical_report(r)
            # Cross-vertical summary
            wf_df = pd.DataFrame(r["wf_results"])
            best = wf_df.loc[wf_df["rmse_red_pct"].idxmax()] if not wf_df.empty else {}
            worst = wf_df.loc[wf_df["rmse_red_pct"].idxmin()] if not wf_df.empty else {}
            summary_rows.append({
                "vertical":           spec.name,
                "n_obs":              len(r["panel"]),
                "n_vars":             r["fit"].k,
                "max_eig":            r["stability"],
                "best_target":        best.get("target", ""),
                "best_rmse_red_pct":  best.get("rmse_red_pct", float("nan")),
                "worst_target":       worst.get("target", ""),
                "worst_rmse_red_pct": worst.get("rmse_red_pct", float("nan")),
                "scenario_total_$":   r["scenario_total_dollars"],
            })

    print()
    print("=" * 78)
    print("Cross-vertical summary")
    print("=" * 78)
    summary_df = pd.DataFrame(summary_rows)
    print()
    print(summary_df.to_string(
        index=False,
        formatters={
            "max_eig":            "{:.3f}".format,
            "best_rmse_red_pct":  "{:+.2f}%".format,
            "worst_rmse_red_pct": "{:+.2f}%".format,
            "scenario_total_$":   "{:+,.0f}".format,
        }))

    out = OUT_DIR / "bvar_phase33_summary.csv"
    summary_df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
