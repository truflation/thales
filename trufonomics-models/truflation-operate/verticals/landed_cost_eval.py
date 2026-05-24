"""Operator landed-cost forecast — head-to-head benchmark.

Loads both import/export verticals (auto + textile), applies their
operator-specific cost-share weights, and runs the walk-forward
benchmark in ``truflation_operate/scenarios/landed_cost_forecast.py``
that compares BVAR vs three naive baselines on the **landed-cost
aggregate**.

This is the proof point for the cross-input transmission story: the
BVAR's joint structure should help on the landed-cost aggregate even
when it's tied with naive on individual inputs.

Run::

    uv run python truflation-operate/verticals/landed_cost_eval.py
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
    # Must register in sys.modules BEFORE exec_module so dataclasses etc.
    # can resolve `cls.__module__` during decoration.
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# Load the vertical scripts as modules (for their load_panel functions)
auto_mod = _load("auto_v",    ROOT / "truflation-operate" / "verticals" / "import_export_auto.py")
text_mod = _load("text_v",    ROOT / "truflation-operate" / "verticals" / "import_export_textile.py")
lcf_mod  = _load("lcf",       ROOT / "truflation-operate" / "scenarios" / "landed_cost_forecast.py")

CostShare = lcf_mod.CostShare
walk_forward_benchmark = lcf_mod.walk_forward_benchmark
forecast_landed_cost = lcf_mod.forecast_landed_cost

# ─── Operator cost shares (from docs/cost_structures.md) ─────────────────

# Paris auto importer (per docs/cost_structures.md §"Client 1")
#   Vehicle wholesale cost      60%  → log_truf_vehicle
#   Ocean ro-ro shipping        12%  → log_freight (proxy)
#   Inland trucking             6%   → log_diesel
#   Duty (fixed, modelled separately) 10% — not part of BVAR variance
#   FX exposure                 ~60% (overlap with vehicle, but separate
#                                     covariance) → log_fx_eurusd
#   Insurance, dealer overhead  12%  — fixed, not modelled
#
# The shares applied to BVAR variables must reflect each variable's
# independent contribution to landed-cost volatility, not the static
# % of cost. Use a normalised share vector that sums to ~1.0 across
# the *modelled* variables.
AUTO_SHARES = [
    CostShare("log_truf_vehicle",   0.45),  # vehicle wholesale, dominant
    CostShare("log_fx_eurusd",      0.30),  # FX is the second biggest mover
    CostShare("log_freight",        0.12),
    CostShare("log_diesel",         0.06),
    CostShare("log_truf_transport", 0.07),
]

# US textile importer (per docs/cost_structures.md §"Client 2")
TEXTILE_SHARES = [
    CostShare("log_truf_clothing",  0.40),  # finished goods cost level
    CostShare("log_fx_cnyusd",      0.30),  # source-country FX
    CostShare("log_freight",        0.15),
    CostShare("log_diesel",         0.08),
    CostShare("log_truf_transport", 0.07),
]


def _run_one(label: str, panel: pd.DataFrame,
                  shares: list, horizons: list[int]) -> dict:
    print()
    print("=" * 78)
    print(f"{label}")
    print("=" * 78)
    print(f"Panel: n = {len(panel)} months   "
            f"{panel.index.min().date()} → {panel.index.max().date()}")
    print(f"Variables in cost basket: "
            f"{', '.join(cs.var_name for cs in shares)}")
    print(f"Weights: {[f'{cs.share:.2f}' for cs in shares]} "
            f"(sum = {sum(cs.share for cs in shares):.2f})")

    summary: dict = {"label": label, "horizons": {}}
    for h in horizons:
        print(f"\n── horizon h = {h} month{'s' if h > 1 else ''} ──")
        bt = walk_forward_benchmark(panel, shares, h=h,
                                          train_min=60, p=1)
        for c in bt.columns:
            if c not in ("method", "n"):
                bt[c] = bt[c].round(5)
        print(bt.to_string(index=False))

        # Save per-horizon CSV
        today = date.today()
        csv = OUT_DIR / f"landed_cost_eval_{label.replace(' ','_')}_h{h}_{today}.csv"
        bt.to_csv(csv, index=False)
        print(f"  Saved: {csv}")

        # Stash for summary
        summary["horizons"][f"h={h}"] = bt.to_dict(orient="records")
    return summary


def main() -> None:
    print("=" * 78)
    print("Operator Landed-Cost Forecast — BVAR vs naive baselines")
    print("=" * 78)
    print("\nMethods compared at each horizon:")
    print("  1. bvar         — joint multivariate forecast (BVAR(1) Minnesota)")
    print("  2. naive_flat   — landed cost stays at current level")
    print("  3. naive_rw     — random walk per input, aggregated")
    print("  4. naive_ar1    — per-input AR(1), no cross-effects, aggregated")
    print("\nMetric: RMSE of landed-cost log-deviation at horizon h, aggregated")
    print("via operator's cost-share weights from docs/cost_structures.md.")

    horizons = [1, 3, 6, 12]
    results: list[dict] = []

    panel = auto_mod.load_panel()
    results.append(
        _run_one("paris_auto_importer", panel, AUTO_SHARES, horizons))

    panel = text_mod.load_panel()
    results.append(
        _run_one("us_textile_importer", panel, TEXTILE_SHARES, horizons))

    today = date.today()
    out = OUT_DIR / f"landed_cost_eval_summary_{today}.json"
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
