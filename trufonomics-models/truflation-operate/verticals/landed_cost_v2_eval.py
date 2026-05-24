"""Landed-cost distribution benchmark v2 — BVAR-on-returns + regime split.

Runs the v2 benchmark across both verticals at h = 1, 3, 6 months,
sliced by regime (stable / COVID / Ukraine-post / recent).

Run::

    uv run python truflation-operate/verticals/landed_cost_v2_eval.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "truflation-operate" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


auto_mod = _load("auto_v",  ROOT / "truflation-operate" / "verticals" / "import_export_auto.py")
text_mod = _load("text_v",  ROOT / "truflation-operate" / "verticals" / "import_export_textile.py")
v2_mod   = _load("v2",      ROOT / "truflation-operate" / "scenarios" / "landed_cost_distribution_v2.py")

CostShare = v2_mod.CostShare
walk_forward_v2 = v2_mod.walk_forward_v2

AUTO_SHARES = [
    CostShare("log_truf_vehicle",   0.45),
    CostShare("log_fx_eurusd",      0.30),
    CostShare("log_freight",        0.12),
    CostShare("log_diesel",         0.06),
    CostShare("log_truf_transport", 0.07),
]
TEXTILE_SHARES = [
    CostShare("log_truf_clothing",  0.40),
    CostShare("log_fx_cnyusd",      0.30),
    CostShare("log_freight",        0.15),
    CostShare("log_diesel",         0.08),
    CostShare("log_truf_transport", 0.07),
]


def _run(label: str, panel: pd.DataFrame, shares, horizons: list[int]) -> dict:
    print()
    print("=" * 78)
    print(f"{label}")
    print("=" * 78)
    print(f"Panel: n={len(panel)} months   "
            f"{panel.index.min().date()} → {panel.index.max().date()}\n")

    summary = {"label": label, "horizons": {}}
    for h in horizons:
        print(f"── horizon h = {h} ──")
        bt = walk_forward_v2(panel, shares, h=h, train_min=60, p=1,
                                  n_samples=500, seed=42)
        for c in ("crps", "cov80", "width80", "crps_vs_naive_ar1_red_pct"):
            if c in bt.columns:
                bt[c] = bt[c].round(5)
        # Pretty-print pivoted: rows = regime, cols = method, value = crps
        crps_pivot = bt.pivot(index="regime", columns="method",
                                    values="crps")
        cov_pivot = bt.pivot(index="regime", columns="method",
                                   values="cov80")
        n_pivot = bt.pivot(index="regime", columns="method", values="n")
        red_pivot = bt.pivot(index="regime", columns="method",
                                   values="crps_vs_naive_ar1_red_pct")

        print(f"  CRPS by regime (log-deviation units, lower=better):")
        print(crps_pivot.round(5))
        print(f"  → BVAR vs naive_ar1 CRPS reduction (%) by regime:")
        for reg in crps_pivot.index:
            red = red_pivot.loc[reg, "bvar_returns"]
            if pd.notna(red):
                tag = "BVAR WINS" if red > 1 else ("tied" if abs(red) <= 1 else "BVAR loses")
                print(f"    {reg:<14s} n={int(n_pivot.loc[reg,'bvar_returns']):>3d}  "
                        f"Δ = {red:+6.2f}%   {tag}")
        print(f"  80% coverage (closer to 80% = better calibrated):")
        print((cov_pivot * 100).round(1))
        print()

        today = date.today()
        bt.to_csv(OUT_DIR / f"landed_cost_v2_{label.replace(' ','_')}_h{h}_{today}.csv",
                       index=False)
        summary["horizons"][f"h={h}"] = bt.to_dict(orient="records")
    return summary


def main() -> None:
    print("=" * 78)
    print("Landed-cost distribution v2 — BVAR-on-returns, regime split")
    print("=" * 78)
    print()
    print("Hypothesis: the BVAR's joint Σ matters most during regimes where")
    print("input correlations are highest (crisis periods). On stable")
    print("periods naive_ar1's per-input independence is approximately")
    print("right; on crisis periods BVAR should win CRPS materially.")

    horizons = [1, 3, 6]
    results = []

    panel = auto_mod.load_panel()
    results.append(_run("paris_auto_importer", panel, AUTO_SHARES, horizons))

    panel = text_mod.load_panel()
    results.append(_run("us_textile_importer", panel, TEXTILE_SHARES, horizons))

    today = date.today()
    out = OUT_DIR / f"landed_cost_v2_summary_{today}.json"
    out.write_text(json.dumps({
        "as_of_date": str(today),
        "verticals":  results,
    }, indent=2, default=str))
    print()
    print("=" * 78)
    print(f"Aggregate summary saved: {out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
