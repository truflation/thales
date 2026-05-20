"""Official BLS CPI print committee — production forecast deliverable.

Mirrors the daily_committee.py pattern for the monthly BLS Headline CPI
YoY release. Reads the latest forward forecast from each of the three
production CPI forecasters and emits a canonical committee point + bands.

The canonical committee is the **simple average of BLS-native CBDF
and Truflation-weighted CPI CBDF** — both component-based forecasters
with different weight schemes. The forecast-combination literature
(Stock-Watson 2004, Bates-Granger 1969) and our April 2026 print
empirical result (the 2-way average landed at +1.46 bp vs actual)
both motivate this choice. See `19 - Learning OS/quant-finance/
learn-forecast-combination.md` for the methodology.

Bands for the canonical committee: the **union** of the contributing
forecasters' 80% and 95% bands (conservative — admits model
uncertainty across weight schemes).

Inputs (latest matching JSON per family):
  * `cpi_<target>_forecast_<as-of>.json`             — Thales standalone
  * `cpi_blsnative_<target>_forecast_<as-of>.json`   — BLS-native CBDF
  * `cpi_trufweights_<target>_forecast_<as-of>.json` — Truflation-weighted CPI CBDF

Output:
  * `results/next_release_forecast/committee_cpi_<target>_<as-of>.json`

Run::

    uv run python scripts/cpi_committee.py
    uv run python scripts/cpi_committee.py --target 2026-05-31 --actual 3.7792
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "next_release_forecast"

FORECASTER_PATTERNS = {
    "thales_standalone":      "cpi_{target}_forecast_*.json",
    "bls_native_cbdf":        "cpi_blsnative_{target}_forecast_*.json",
    "truf_weighted_cpi_cbdf": "cpi_trufweights_{target}_forecast_*.json",
}


@dataclass
class ForecasterRead:
    name: str
    point: float
    lo80: float | None
    hi80: float | None
    lo95: float | None
    hi95: float | None
    source_file: str
    as_of: str


def _latest(pattern: str) -> Path | None:
    files = sorted(OUT_DIR.glob(pattern))
    return files[-1] if files else None


def _read_thales_standalone(payload: dict) -> ForecasterRead | None:
    if "thales_standalone" not in payload:
        return None
    fc = payload["thales_standalone"]
    return ForecasterRead(
        name="thales_standalone",
        point=float(fc["point"]),
        lo80=float(fc["band_80"][0]),
        hi80=float(fc["band_80"][1]),
        lo95=float(fc["band_95"][0]),
        hi95=float(fc["band_95"][1]),
        source_file="",
        as_of=payload.get("as_of_date", ""),
    )


def _read_cbdf(payload: dict, name: str) -> ForecasterRead | None:
    fc = payload.get("forward_next_release_forecast") or payload.get("forecast")
    if fc is None:
        return None
    return ForecasterRead(
        name=name,
        point=float(fc["point"]),
        lo80=float(fc.get("lo80")) if fc.get("lo80") is not None else None,
        hi80=float(fc.get("hi80")) if fc.get("hi80") is not None else None,
        lo95=float(fc.get("lo95")) if fc.get("lo95") is not None else None,
        hi95=float(fc.get("hi95")) if fc.get("hi95") is not None else None,
        source_file="",
        as_of=payload.get("as_of_date", ""),
    )


def load_all(target_month: str) -> list[ForecasterRead]:
    """Read latest forward forecast for each of the 3 CPI forecasters.

    Falls back to the latest family JSON if no exact target_month match.
    """
    out: list[ForecasterRead] = []
    for name, pat in FORECASTER_PATTERNS.items():
        path = _latest(pat.format(target=target_month))
        if path is None:
            path = _latest(pat.format(target="*"))
        if path is None:
            print(f"  [skip] {name}: no JSON found")
            continue
        with path.open() as f:
            payload = json.load(f)
        if name == "thales_standalone":
            r = _read_thales_standalone(payload)
        else:
            r = _read_cbdf(payload, name)
        if r is None:
            print(f"  [skip] {name}: no point in {path.name}")
            continue
        r.source_file = path.name
        out.append(r)
    return out


# ─── Committee math ──────────────────────────────────────────────────────


def canonical_committee(reads: list[ForecasterRead]) -> dict:
    """The production point: avg(BLS-native, Truflation-weighted).

    Bands: union of contributing forecasters' bands. Both 80% and 95%
    are taken as conservative envelopes — wider than either individual.
    Reasoning: each forecaster's bands reflect its own model
    uncertainty; the union spans the family of weight choices.
    """
    by_name = {r.name: r for r in reads}
    bls = by_name.get("bls_native_cbdf")
    truf = by_name.get("truf_weighted_cpi_cbdf")
    if bls is None or truf is None:
        return {"available": False,
                  "reason": "need both bls_native_cbdf and truf_weighted_cpi_cbdf"}
    point = (bls.point + truf.point) / 2.0
    los80 = [b for b in (bls.lo80, truf.lo80) if b is not None]
    his80 = [h for h in (bls.hi80, truf.hi80) if h is not None]
    los95 = [b for b in (bls.lo95, truf.lo95) if b is not None]
    his95 = [h for h in (bls.hi95, truf.hi95) if h is not None]
    return {
        "available": True,
        "point": float(point),
        "lo80_union": float(min(los80)) if los80 else None,
        "hi80_union": float(max(his80)) if his80 else None,
        "lo95_union": float(min(los95)) if los95 else None,
        "hi95_union": float(max(his95)) if his95 else None,
        "members": ["bls_native_cbdf", "truf_weighted_cpi_cbdf"],
        "bls_native_point": float(bls.point),
        "truf_weighted_point": float(truf.point),
    }


def diagnostic_combinations(reads: list[ForecasterRead]) -> dict:
    """All other committee methods for cross-check: 3-way mean, median,
    every pairwise average."""
    if len(reads) == 0:
        return {}
    points = np.array([r.point for r in reads])
    out: dict = {
        "mean":   float(points.mean()),
        "median": float(np.median(points)),
    }
    if len(points) >= 3:
        out["trimmed_mean"] = float(np.sort(points)[1:-1].mean())
    from itertools import combinations
    for a, b in combinations(reads, 2):
        out[f"avg({a.name},{b.name})"] = (a.point + b.point) / 2.0
    return out


# ─── Main ────────────────────────────────────────────────────────────────


def _auto_detect_target() -> str | None:
    """Pick the most recent target_month across all 3 forecaster JSONs."""
    candidates: set[str] = set()
    for pat in FORECASTER_PATTERNS.values():
        for p in OUT_DIR.glob(pat.format(target="*")):
            parts = p.stem.split("_")
            for q in parts:
                if q.startswith(("2026-", "2025-", "2027-")):
                    candidates.add(q)
                    break
    if not candidates:
        return None
    return sorted(candidates)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Committee forecast for the next BLS CPI YoY release")
    parser.add_argument("--target", type=str, default=None,
                        help="Target release month YYYY-MM-DD "
                              "(default: auto-detect latest)")
    parser.add_argument("--actual", type=float, default=None,
                        help="Actual BLS CPI YoY for the target month, "
                              "for retrospective scoring")
    args = parser.parse_args()

    if args.target is None:
        args.target = _auto_detect_target()
    if args.target is None:
        raise SystemExit("No forecaster JSONs found in results/next_release_forecast/")

    print("=" * 78)
    print(f"BLS CPI committee forecast — target release {args.target}")
    print("=" * 78)

    reads = load_all(args.target)
    if not reads:
        raise SystemExit("No forecasters loaded.")

    print()
    print(f"Individual forecasters (target {args.target}):")
    for r in reads:
        band = (f"  80%: [{r.lo80:.4f}, {r.hi80:.4f}]"
                 if r.lo80 is not None else "  (no band)")
        marker = (f"  err {(r.point - args.actual)*100:+.2f}bp"
                    if args.actual is not None else "")
        print(f"  {r.name:<28s}  {r.point:>9.4f}%{marker}  {band}")
        print(f"  {'':<28s}    source: {r.source_file}")
    if args.actual is not None:
        print(f"  {'ACTUAL':<28s}  {args.actual:>9.4f}%")

    # ── Canonical committee ────────────────────────────────────────
    canon = canonical_committee(reads)
    print()
    print("═══ CANONICAL COMMITTEE — avg(BLS-native, Truflation-weighted) ═══")
    print()
    if not canon["available"]:
        print(f"  Not available: {canon['reason']}")
    else:
        print(f"  Point:    {canon['point']:.4f}%")
        if canon["lo80_union"] is not None:
            print(f"  80% band (union): "
                    f"[{canon['lo80_union']:.4f}, {canon['hi80_union']:.4f}]"
                    f"   width {canon['hi80_union'] - canon['lo80_union']:.4f} pp")
        if canon["lo95_union"] is not None:
            print(f"  95% band (union): "
                    f"[{canon['lo95_union']:.4f}, {canon['hi95_union']:.4f}]"
                    f"   width {canon['hi95_union'] - canon['lo95_union']:.4f} pp")
        if args.actual is not None:
            err = (canon["point"] - args.actual) * 100
            in80 = (canon["lo80_union"] <= args.actual <= canon["hi80_union"]
                      if canon["lo80_union"] is not None else None)
            print(f"  Error vs actual {args.actual:.4f}%: {err:+.2f} bp")
            if in80 is not None:
                print(f"  Actual inside 80% band: {'✓' if in80 else '✗'}")

    # ── Diagnostic combinations ────────────────────────────────────
    diag = diagnostic_combinations(reads)
    print()
    print("─ Diagnostic combinations ─")
    if args.actual is not None:
        for name, val in diag.items():
            err = (val - args.actual) * 100
            print(f"  {name:<55s}  {val:>9.4f}%  err {err:>+8.2f} bp")
    else:
        for name, val in diag.items():
            print(f"  {name:<55s}  {val:>9.4f}%")

    # ── Closest method if actual provided ──────────────────────────
    if args.actual is not None:
        all_methods = dict(diag)
        if canon["available"]:
            all_methods["CANONICAL avg(bls_native, truf_weighted)"] = canon["point"]
        best = min(all_methods.items(), key=lambda x: abs(x[1] - args.actual))
        best_err = (best[1] - args.actual) * 100
        print()
        print(f"  → CLOSEST METHOD: {best[0]}  =  {best[1]:.4f}%  ({best_err:+.2f} bp)")

    # ── Persist ─────────────────────────────────────────────────────
    out_path = OUT_DIR / f"committee_cpi_{args.target}_{date.today()}.json"
    payload = {
        "target_release_month": args.target,
        "as_of_date":           str(date.today()),
        "individuals": [
            {
                "name":         r.name,
                "point":        r.point,
                "lo80":         r.lo80,
                "hi80":         r.hi80,
                "lo95":         r.lo95,
                "hi95":         r.hi95,
                "source_file":  r.source_file,
                "as_of":        r.as_of,
            } for r in reads
        ],
        "canonical_committee":  canon,
        "diagnostic_combinations": diag,
        "actual":               args.actual,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print()
    print(f"Saved: {out_path}")

    # ── Final ship-ready summary ──────────────────────────────────
    if canon["available"]:
        print()
        print("=" * 78)
        print("PRODUCTION POINT (CPI committee)")
        print("=" * 78)
        print(f"  Target release:  {args.target}")
        print(f"  Point:           {canon['point']:.4f}%")
        if canon["lo80_union"] is not None:
            half = (canon["hi80_union"] - canon["lo80_union"]) / 2
            mid = (canon["hi80_union"] + canon["lo80_union"]) / 2
            print(f"  80% band:        ± {half:.3f} pp around {mid:.4f}%")


if __name__ == "__main__":
    main()
