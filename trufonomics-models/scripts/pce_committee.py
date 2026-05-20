"""Official BEA PCE print committee — production forecast deliverable.

Mirror of `cpi_committee.py` for the monthly BEA Headline PCE YoY
release (~30th of the month-after). Reads the latest forward forecast
from each of the three production PCE forecasters and emits a
canonical committee point + bands.

The canonical committee is the **simple average of PCE-native CBDF
and Truflation-weighted PCE CBDF** — both component-based forecasters
targeting PCEPI, with different component+weight schemes:
  * PCE-native:        3 BEA components × OLS-fitted weights (~Fisher approx)
  * Truflation-weighted: 11 BLS components × Truflation PCE weights

Architecture-divergent pair → forecast-combination puzzle motivation.

Bands for the canonical committee: the **union** of the contributing
forecasters' 80% and 95% bands (conservative envelope).

Inputs (latest matching JSON per family):
  * `pce_<target>_forecast_<as-of>.json`              — PCE standalone
  * `pce_native_<target>_forecast_<as-of>.json`       — PCE-native CBDF
  * `pce_trufweights_<target>_forecast_<as-of>.json`  — Truflation-weighted PCE CBDF

Output:
  * `results/next_release_forecast/committee_pce_<target>_<as-of>.json`

Run::

    uv run python scripts/pce_committee.py
    uv run python scripts/pce_committee.py --target 2026-04-30 --actual 3.7800
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

FORECASTER_PATTERNS = {
    "pce_standalone":         "pce_{target}_forecast_*.json",
    "pce_native_cbdf":        "pce_native_{target}_forecast_*.json",
    "truf_weighted_pce_cbdf": "pce_trufweights_{target}_forecast_*.json",
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


def _read_pce_standalone(payload: dict) -> ForecasterRead | None:
    if "pce_standalone" not in payload:
        return None
    fc = payload["pce_standalone"]
    return ForecasterRead(
        name="pce_standalone",
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
        if name == "pce_standalone":
            r = _read_pce_standalone(payload)
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
    """The production point: avg(PCE-native, Truflation-weighted PCE).

    Bands: union of contributing forecasters' bands.
    """
    by_name = {r.name: r for r in reads}
    native = by_name.get("pce_native_cbdf")
    truf = by_name.get("truf_weighted_pce_cbdf")
    if native is None or truf is None:
        return {"available": False,
                  "reason": "need both pce_native_cbdf and truf_weighted_pce_cbdf"}
    point = (native.point + truf.point) / 2.0
    los80 = [b for b in (native.lo80, truf.lo80) if b is not None]
    his80 = [h for h in (native.hi80, truf.hi80) if h is not None]
    los95 = [b for b in (native.lo95, truf.lo95) if b is not None]
    his95 = [h for h in (native.hi95, truf.hi95) if h is not None]
    return {
        "available": True,
        "point": float(point),
        "lo80_union": float(min(los80)) if los80 else None,
        "hi80_union": float(max(his80)) if his80 else None,
        "lo95_union": float(min(los95)) if los95 else None,
        "hi95_union": float(max(his95)) if his95 else None,
        "members": ["pce_native_cbdf", "truf_weighted_pce_cbdf"],
        "pce_native_point": float(native.point),
        "truf_weighted_point": float(truf.point),
    }


def diagnostic_combinations(reads: list[ForecasterRead]) -> dict:
    if not reads:
        return {}
    points = np.array([r.point for r in reads])
    out: dict = {
        "mean":   float(points.mean()),
        "median": float(np.median(points)),
    }
    if len(points) >= 3:
        out["trimmed_mean"] = float(np.sort(points)[1:-1].mean())
    for a, b in combinations(reads, 2):
        out[f"avg({a.name},{b.name})"] = (a.point + b.point) / 2.0
    return out


# ─── Main ────────────────────────────────────────────────────────────────


def _auto_detect_target() -> str | None:
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
        description="Committee forecast for the next BEA PCE YoY release")
    parser.add_argument("--target", type=str, default=None,
                        help="Target release month YYYY-MM-DD "
                              "(default: auto-detect latest)")
    parser.add_argument("--actual", type=float, default=None,
                        help="Actual BEA PCE YoY for the target month, "
                              "for retrospective scoring")
    args = parser.parse_args()

    if args.target is None:
        args.target = _auto_detect_target()
    if args.target is None:
        raise SystemExit("No forecaster JSONs found in results/next_release_forecast/")

    print("=" * 78)
    print(f"BEA PCE committee forecast — target release {args.target}")
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
    print("═══ CANONICAL COMMITTEE — avg(PCE-native, Truflation-weighted PCE) ═══")
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

    if args.actual is not None:
        all_methods = dict(diag)
        if canon["available"]:
            all_methods["CANONICAL avg(pce_native, truf_weighted_pce)"] = canon["point"]
        best = min(all_methods.items(), key=lambda x: abs(x[1] - args.actual))
        best_err = (best[1] - args.actual) * 100
        print()
        print(f"  → CLOSEST METHOD: {best[0]}  =  {best[1]:.4f}%  ({best_err:+.2f} bp)")

    # ── Persist ─────────────────────────────────────────────────────
    out_path = OUT_DIR / f"committee_pce_{args.target}_{date.today()}.json"
    payload = {
        "target_release_month": args.target,
        "as_of_date":           str(date.today()),
        "individuals": [
            {
                "name":        r.name,
                "point":       r.point,
                "lo80":        r.lo80,
                "hi80":        r.hi80,
                "lo95":        r.lo95,
                "hi95":        r.hi95,
                "source_file": r.source_file,
                "as_of":       r.as_of,
            } for r in reads
        ],
        "canonical_committee":     canon,
        "diagnostic_combinations": diag,
        "actual":                  args.actual,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print()
    print(f"Saved: {out_path}")

    # ── Final ship-ready summary ──────────────────────────────────
    if canon["available"]:
        print()
        print("=" * 78)
        print("PRODUCTION POINT (PCE committee)")
        print("=" * 78)
        print(f"  Target release:  {args.target}")
        print(f"  Point:           {canon['point']:.4f}%")
        if canon["lo80_union"] is not None:
            half = (canon["hi80_union"] - canon["lo80_union"]) / 2
            mid = (canon["hi80_union"] + canon["lo80_union"]) / 2
            print(f"  80% band:        ± {half:.3f} pp around {mid:.4f}%")


if __name__ == "__main__":
    main()
