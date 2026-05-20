"""Print-day committee — average / median / trimmed mean across the
production print-day forecasters for the next BLS CPI and BEA PCE
releases.

Reads the JSON outputs from:
  * `forecast_next_bls_cpi.py`            — Thales standalone
  * `forecast_next_bls_cpi_blsnative.py`  — BLS-native CBDF
  * `forecast_next_bls_cpi_trufweights.py`— Truflation-weighted CPI CBDF
  * `forecast_next_bea_pce.py`            — PCE standalone
  * `forecast_next_bea_pce_native.py`     — PCE-native CBDF
  * `forecast_next_bea_pce_trufweights.py`— Truflation-weighted PCE CBDF

Emits a single committee report with multiple combination methods:
  - Simple mean across all available forecasters
  - Trimmed mean (drop highest + lowest)
  - Median
  - Pairwise 2-way averages (every pair)

If an "actual" value is available for the target month (e.g. April BLS
CPI 3.7792%), errors are computed and the closest method is flagged.

Run::

    uv run python scripts/score_committee.py
    uv run python scripts/score_committee.py --target cpi --month 2026-04-30 --actual 3.7792
    uv run python scripts/score_committee.py --target pce --month 2026-04-30
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "next_release_forecast"

CPI_FORECASTERS = {
    "thales_standalone":       "cpi_{month}_forecast_*.json",
    "bls_native_cbdf":         "cpi_blsnative_{month}_forecast_*.json",
    "truf_weighted_cpi_cbdf":  "cpi_trufweights_{month}_forecast_*.json",
}
PCE_FORECASTERS = {
    "pce_standalone":          "pce_{month}_forecast_*.json",
    "pce_native_cbdf":         "pce_native_{month}_forecast_*.json",
    "truf_weighted_pce_cbdf":  "pce_trufweights_{month}_forecast_*.json",
}


@dataclass
class ForecasterResult:
    name: str
    point: float
    source_file: str


def _latest_matching_json(pattern: str) -> Path | None:
    files = sorted(OUT_DIR.glob(pattern))
    return files[-1] if files else None


def _extract_point(payload: dict, key: str,
                      prefer_retrospective_month: str | None = None) -> float | None:
    """Each forecaster JSON has a different shape — pull the point
    forecast for the desired target month.

    If `prefer_retrospective_month` is set (e.g. '2026-04-30'), check
    `retrospective_april_forecast` first (CBDF variants stored the April
    retrospective inside their forward-targeted JSONs).
    """
    if prefer_retrospective_month == "2026-04-30":
        if "retrospective_april_forecast" in payload:
            return float(payload["retrospective_april_forecast"]["point"])
    # Thales standalone (CPI)
    if "thales_standalone" in payload:
        return float(payload["thales_standalone"]["point"])
    # PCE standalone
    if "pce_standalone" in payload:
        return float(payload["pce_standalone"]["point"])
    # CBDF variants — look at forward_next_release_forecast or forecast
    if "forward_next_release_forecast" in payload:
        return float(payload["forward_next_release_forecast"]["point"])
    if "forecast" in payload:
        return float(payload["forecast"]["point"])
    return None


def load_forecasters(target: str, month: str) -> list[ForecasterResult]:
    forecasters = CPI_FORECASTERS if target == "cpi" else PCE_FORECASTERS
    results: list[ForecasterResult] = []
    for name, pat in forecasters.items():
        # Try direct match first
        glob = pat.format(month=month)
        latest = _latest_matching_json(glob)
        if latest is None:
            # Fall back to the latest *forward* JSON for that family,
            # then look inside for the retrospective month
            family_glob = pat.format(month="*")
            latest = _latest_matching_json(family_glob)
            if latest is None:
                print(f"  [skip] {name}: no file matches {glob} or family")
                continue
        with latest.open() as f:
            payload = json.load(f)
        point = _extract_point(payload, name,
                                  prefer_retrospective_month=month)
        if point is None:
            print(f"  [skip] {name}: could not extract point from {latest.name}")
            continue
        results.append(ForecasterResult(name, point, latest.name))
    return results


def print_pairwise_averages(results: list[ForecasterResult],
                                actual: float | None) -> None:
    print()
    print("─ Pairwise averages ─")
    if actual is not None:
        print(f"  {'pair':<55s}  {'avg':>10s}  {'err (bp)':>10s}")
    else:
        print(f"  {'pair':<55s}  {'avg':>10s}")
    for a, b in combinations(results, 2):
        avg = (a.point + b.point) / 2
        label = f"avg({a.name}, {b.name})"
        if actual is not None:
            err_bp = (avg - actual) * 100
            print(f"  {label:<55s}  {avg:>9.4f}%  {err_bp:>+9.2f}")
        else:
            print(f"  {label:<55s}  {avg:>9.4f}%")


def print_committee(target: str, results: list[ForecasterResult],
                       actual: float | None) -> dict:
    if not results:
        print("  (no forecasters loaded — nothing to combine)")
        return {}

    points = np.array([r.point for r in results])
    mean = float(points.mean())
    median = float(np.median(points))

    # Trimmed mean — drop highest + lowest if ≥ 3 forecasters
    if len(points) >= 3:
        trimmed = float(np.sort(points)[1:-1].mean())
    else:
        trimmed = float("nan")

    print()
    print(f"═══ COMMITTEE — {target.upper()} ═══")
    print()
    print(f"Individual forecasters:")
    for r in results:
        if actual is not None:
            err = (r.point - actual) * 100
            print(f"  {r.name:<30s}  {r.point:>9.4f}%  "
                    f"err {err:>+8.2f} bp")
        else:
            print(f"  {r.name:<30s}  {r.point:>9.4f}%")
    if actual is not None:
        print(f"  {'ACTUAL':<30s}  {actual:>9.4f}%")

    print()
    print(f"Combination methods (n = {len(points)} forecasters):")
    for label, value in [
        ("mean",         mean),
        ("median",       median),
        ("trimmed_mean", trimmed),
    ]:
        if actual is not None and not np.isnan(value):
            err = (value - actual) * 100
            print(f"  {label:<14s}  {value:>9.4f}%  err {err:>+8.2f} bp")
        elif not np.isnan(value):
            print(f"  {label:<14s}  {value:>9.4f}%")

    # Pairwise (most useful diagnostic per the forecast-combination puzzle)
    print_pairwise_averages(results, actual)

    # Best
    if actual is not None:
        candidates = [
            ("mean",         mean),
            ("median",       median),
            ("trimmed_mean", trimmed),
        ]
        if len(results) >= 2:
            for a, b in combinations(results, 2):
                candidates.append(
                    (f"avg({a.name},{b.name})", (a.point + b.point) / 2))
        candidates = [(n, v) for n, v in candidates if not np.isnan(v)]
        candidates.sort(key=lambda x: abs(x[1] - actual))
        best_name, best_val = candidates[0]
        best_err = (best_val - actual) * 100
        print()
        print(f"  → CLOSEST METHOD:  {best_name}  =  {best_val:.4f}%  "
                f"({best_err:+.2f} bp)")

    return {
        "target": target,
        "actual": actual,
        "individuals": {r.name: r.point for r in results},
        "mean": mean,
        "median": median,
        "trimmed_mean": trimmed if not np.isnan(trimmed) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["cpi", "pce", "both"],
                        default="both")
    parser.add_argument("--month", type=str, default=None,
                        help="Target release month YYYY-MM-DD "
                              "(default: auto-detect from filenames)")
    parser.add_argument("--cpi-actual", type=float, default=None,
                        help="Actual BLS CPI YoY for the target month, "
                              "if released (e.g. 3.7792 for April 2026)")
    parser.add_argument("--pce-actual", type=float, default=None,
                        help="Actual BEA PCE YoY, if released")
    args = parser.parse_args()

    # Auto-pick month from the latest CPI or PCE JSON
    if args.month is None:
        files = sorted(OUT_DIR.glob("cpi_*_forecast_*.json")) + \
                sorted(OUT_DIR.glob("pce_*_forecast_*.json"))
        if files:
            # Filenames like cpi_2026-04-30_forecast_2026-05-07.json
            for f in reversed(files):
                parts = f.stem.split("_")
                for p in parts:
                    if p.startswith("2026-") or p.startswith("2025-") \
                            or p.startswith("2027-"):
                        args.month = p
                        break
                if args.month:
                    break
    if args.month is None:
        raise SystemExit("Could not auto-detect target month; pass --month")

    print("=" * 78)
    print(f"Print-day committee for target month {args.month}")
    print("=" * 78)

    summary = {}

    if args.target in ("cpi", "both"):
        cpi_results = load_forecasters("cpi", args.month)
        if cpi_results:
            summary["cpi"] = print_committee(
                "cpi", cpi_results, args.cpi_actual)

    if args.target in ("pce", "both"):
        pce_results = load_forecasters("pce", args.month)
        if pce_results:
            summary["pce"] = print_committee(
                "pce", pce_results, args.pce_actual)

    # Save
    out = OUT_DIR / f"committee_summary_{args.month}_{date.today()}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
