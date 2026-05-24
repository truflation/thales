"""Operator landed-cost — joint-distribution benchmark across verticals.

Companion to ``landed_cost_eval.py``. That one tests point-forecast
RMSE. This one tests **predictive distributions**: CRPS, coverage,
sharpness. The BVAR's joint covariance should beat naive_ar1's
per-input independence here even when it doesn't on point RMSE.

Run::

    uv run python truflation-operate/verticals/landed_cost_distribution_eval.py
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
dist_mod = _load("dist",    ROOT / "truflation-operate" / "scenarios" / "landed_cost_distribution.py")

CostShare = dist_mod.CostShare
walk_forward_distribution_benchmark = dist_mod.walk_forward_distribution_benchmark

# Same operator cost shares as the point-forecast eval
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


def _run_one(label: str, panel: pd.DataFrame,
                  shares, horizons: list[int]) -> dict:
    print()
    print("=" * 78)
    print(f"{label}")
    print("=" * 78)
    print(f"Panel: n = {len(panel)} months   "
            f"{panel.index.min().date()} → {panel.index.max().date()}")
    print(f"Cost-share weights: "
            f"{', '.join(f'{cs.var_name}={cs.share:.2f}' for cs in shares)}")

    summary = {"label": label, "horizons": {}}
    for h in horizons:
        print(f"\n── horizon h = {h} month{'s' if h > 1 else ''} ──")
        bt = walk_forward_distribution_benchmark(
            panel, shares, h=h, train_min=60, p=1, n_samples=500, seed=42)
        for c in ("crps_log", "cov80", "cov95", "width80_log",
                       "crps_vs_naive_ar1_red_pct"):
            if c in bt.columns:
                bt[c] = bt[c].round(5)
        print(bt.to_string(index=False))

        # Coverage gap from nominal (80% / 95%)
        print(f"  Coverage gap from nominal:")
        for _, r in bt.iterrows():
            gap80 = (r["cov80"] - 0.80) * 100
            gap95 = (r["cov95"] - 0.95) * 100
            tag80 = "OVERCONFIDENT" if r["cov80"] < 0.75 else (
                "OK" if 0.75 <= r["cov80"] <= 0.85 else "WIDE")
            tag95 = "OVERCONFIDENT" if r["cov95"] < 0.90 else (
                "OK" if 0.90 <= r["cov95"] <= 0.98 else "WIDE")
            print(f"    {r['method']:<11s} 80%: actual {r['cov80']*100:5.1f}% "
                    f"(gap {gap80:+5.1f}pp, {tag80})    "
                    f"95%: actual {r['cov95']*100:5.1f}% "
                    f"(gap {gap95:+5.1f}pp, {tag95})")

        today = date.today()
        csv = OUT_DIR / f"landed_cost_dist_{label.replace(' ','_')}_h{h}_{today}.csv"
        bt.to_csv(csv, index=False)
        print(f"  Saved: {csv}")
        summary["horizons"][f"h={h}"] = bt.to_dict(orient="records")
    return summary


def main() -> None:
    print("=" * 78)
    print("Operator Landed-Cost — joint-distribution benchmark")
    print("=" * 78)
    print("\nThe BVAR's joint Σ should produce better-calibrated and lower-")
    print("CRPS forecasts than naive_ar1 (which assumes per-input independence")
    print("and therefore under-estimates the basket variance when inputs co-")
    print("move). This is the test where BVAR should genuinely win.")

    horizons = [1, 3, 6]
    results = []

    panel = auto_mod.load_panel()
    results.append(_run_one("paris_auto_importer", panel, AUTO_SHARES, horizons))

    panel = text_mod.load_panel()
    results.append(_run_one("us_textile_importer", panel, TEXTILE_SHARES, horizons))

    today = date.today()
    out = OUT_DIR / f"landed_cost_dist_summary_{today}.json"
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
