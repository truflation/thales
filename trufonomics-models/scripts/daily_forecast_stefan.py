"""Stefan day-ahead forecaster — produces tomorrow's Truflation US CPI YoY
and a social-media-ready post draft.

Method: per-component OLS (12 top-level categories) + weighted composition
via 2026 v2 Truflation category weights. Bootstrap residual bands.
Attribution from per-component index moves. Spec: docs/planning/01-architecture.md
§Methodology review 2026-04-24 and the design review discussion.

Usage:
    uv run python scripts/daily_forecast_stefan.py

Writes:
    prints post text + debug block to stdout
    results/daily_forecast/forecast_<YYYY-MM-DD>.json — machine-readable dump
    vintage_store gets one row logging the point forecast under
    series='thales_daily_forecast_v2', as_of=origin, reference_date=target
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.direct_forecaster import direct_target_forecast  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "daily_forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_SERIES = "thales_daily_forecast_v3_direct"


def _arrow(delta: float, threshold: float = 0.005) -> str:
    if delta > threshold:
        return "↑"
    if delta < -threshold:
        return "↓"
    return "→"


def format_post(fc) -> str:
    delta = fc.point_yoy_pct - fc.today_published_yoy
    arrow = _arrow(delta)
    top3 = fc.contributions[:3]
    lines = [
        "Thales — Day-ahead Truflation US CPI YoY forecast",
        "",
        f"{fc.target_date:%b %d, %Y}:  {fc.point_yoy_pct:.2f}%  {arrow}",
        f"  80% band:  [{fc.band_80[0]:.2f}%, {fc.band_80[1]:.2f}%]",
        f"  Today:     {fc.today_published_yoy:.2f}%",
        "",
        "Top drivers (7-day component moves):",
    ]
    for c in top3:
        ar = _arrow(c.contribution_pp)
        lines.append(
            f"  {ar} {c.category_name:<32s}"
            f"  {c.recent_move_pct:+.2f}% → {c.contribution_pp:+.3f} pp"
        )
    lines.append("")
    lines.append(
        "Method: RidgeCV on 12 component index values + YoY lag → split-conformal "
        "bands calibrated on last 30 days."
    )
    return "\n".join(lines)


def format_debug(fc) -> str:
    lines = [
        "=" * 72,
        "DEBUG (not for public post)",
        "=" * 72,
        f"Origin:          {fc.origin_date}",
        f"Target:          {fc.target_date}",
        f"Today published: {fc.today_published_yoy:+.4f}%",
        f"Point T+1:       {fc.point_yoy_pct:+.4f}%",
        f"80% band:        [{fc.band_80[0]:+.4f}, {fc.band_80[1]:+.4f}]  "
        f"(width {fc.band_80[1]-fc.band_80[0]:.4f})",
        f"95% band:        [{fc.band_95[0]:+.4f}, {fc.band_95[1]:+.4f}]",
        f"Ridge α:         {fc.ridge_alpha}",
        f"Residual SD:     {fc.residual_sd_pct:.4f} pp  (n_train={fc.n_train})",
        f"Intercept:       {fc.intercept:.4f}",
        f"Lag coef (φ):    {fc.lag_coef:.4f}",
        "",
        "Per-component coefficients + recent moves:",
    ]
    for c in fc.contributions:
        lines.append(
            f"  [{c.category_id:>3d}] {c.category_name:<38s}  "
            f"β={c.coef:+.6f}  today={c.today_index:>8.3f}  "
            f"7d move={c.recent_move_pct:+.2f}%  contrib={c.contribution_pp:+.4f}pp"
        )
    return "\n".join(lines)


def log_to_vintage_store(fc) -> None:
    with VintageStore(VINTAGE_DB) as store:
        store.ingest(
            series_id=FORECAST_SERIES,
            observations=[(fc.target_date, fc.point_yoy_pct)],
            as_of_date=fc.origin_date,
            source=FORECAST_SERIES,
        )


def main() -> None:
    print(f"Loading vintage store: {VINTAGE_DB}")
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        fc = direct_target_forecast(store)

    print()
    print(format_post(fc))
    print()
    print(format_debug(fc))

    # Save JSON dump
    dump = {
        "origin_date": str(fc.origin_date),
        "target_date": str(fc.target_date),
        "point_yoy_pct": fc.point_yoy_pct,
        "band_80": list(fc.band_80),
        "band_95": list(fc.band_95),
        "today_published_yoy": fc.today_published_yoy,
        "residual_sd_pct": fc.residual_sd_pct,
        "ridge_alpha": fc.ridge_alpha,
        "intercept": fc.intercept,
        "lag_coef": fc.lag_coef,
        "n_train": fc.n_train,
        "contributions": [
            {
                "category_id": c.category_id,
                "category_name": c.category_name,
                "raw_name": c.raw_name,
                "today_index": c.today_index,
                "coef": c.coef,
                "recent_move_pct": c.recent_move_pct,
                "contribution_pp": c.contribution_pp,
            }
            for c in fc.contributions
        ],
    }
    out = OUT_DIR / f"forecast_direct_{fc.origin_date}.json"
    out.write_text(json.dumps(dump, indent=2, default=str))
    print(f"\nSaved: {out}")

    # Log to vintage store
    log_to_vintage_store(fc)
    print(f"Logged to store: series={FORECAST_SERIES} "
          f"as_of={fc.origin_date} target={fc.target_date}")


if __name__ == "__main__":
    main()
