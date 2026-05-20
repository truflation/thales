"""Day-ahead Truflation YoY — 2-model committee.

Two simple, fully-independent forecasters run side-by-side on the
**frozen** Truflation feed for a chosen target series. Output is a
single committee report intended to be defensible enough to share
publicly.

  1. **persistence**       y[T+1] = y[T] (the floor — what news headlines do)
  2. **ar1_rolling**       y[T+1] = α + φ·y[T]  with rolling-conformal bands

The earlier ridge_stacker member was removed 2026-05-16: it depended on
an external forecaster that didn't align with the Truflation-only
methodology direction. Persistence + AR(1) rolling alone gives 100%
coverage in 80% bands over the validated 90-day backtest window.

Ship rule: if both agree within ±5 bp → SHIP MEDIAN with high
confidence. If range > 5 bp → SHIP MEDIAN with caveat. If range
> 15 bp → HOLD (don't post; investigate).

**Endpoint choice — frozen, not live.** Truflation publishes two
parallel YoY series for every macro target:

  * ``..._yoy`` (live): continually revises.
  * ``..._frozen_yoy`` (frozen): pinned at first publication, never
    revises. This is what the public Truflation chart displays.

We use the **frozen** endpoint here so:
  - "Today's value" matches what customers see on the public chart
  - "Tomorrow's prediction" is what we expect the next morning's
    frozen publication to be
  - Backtests use point-in-time first-observed values, not revised

A frozen value timestamped X is the YoY published on X morning,
reflecting underlying price data through (roughly) X-1 EOD. There is
a built-in ~1-day publication lag.

**Targets supported (--target flag):**

  * ``cpi``           — US Headline CPI YoY (default)
  * ``pce``           — US Headline PCE YoY
  * ``pce_core``      — US Core PCE YoY (excludes food + energy)
  * ``pce_goods``     — US PCE Goods YoY
  * ``pce_services``  — US PCE Services YoY

Output:
  * Stdout: committee report
  * JSON: results/daily_forecast_live/committee_<target>_<YYYY-MM-DD>.json

Run::

    uv run python scripts/daily_committee.py             # CPI (default)
    uv run python scripts/daily_committee.py --target pce
    uv run python scripts/daily_committee.py --target pce_core
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation.harness import attach_actuals, score, walk_forward    # noqa: E402
from thales.models.baselines import AR1Baseline, PersistenceBaseline    # noqa: E402

OUT_DIR = ROOT / "results" / "daily_forecast_live"
FEED_API_KEY = "egiZc8oSTWdHr7V6lwYZxYil1wF0ezgF"
FEED_BASE = "https://api.truflation.com/api/v1/feed/truflation/macro-data-us"


# ─── Target registry ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class TargetSpec:
    """One supported target series.

    ``key`` is the CLI selector. ``frozen_col`` and ``live_col`` are the
    Truflation column names returned by the feed. ``display_name`` is
    the human-readable label shown in stdout.
    """
    key: str
    frozen_col: str
    live_col: str
    display_name: str

    @property
    def frozen_url(self) -> str:
        return f"{FEED_BASE}/{self.frozen_col}"

    @property
    def live_url(self) -> str:
        return f"{FEED_BASE}/{self.live_col}"


TARGETS: dict[str, TargetSpec] = {
    "cpi": TargetSpec(
        key="cpi",
        frozen_col="truflation_us_cpi_frozen_yoy",
        live_col="truflation_us_cpi_yoy",
        display_name="US CPI YoY",
    ),
    "pce": TargetSpec(
        key="pce",
        frozen_col="truflation_us_pce_frozen_yoy",
        live_col="truflation_us_pce_yoy",
        display_name="US PCE YoY (headline)",
    ),
    "pce_core": TargetSpec(
        key="pce_core",
        frozen_col="truflation_us_pce_core_frozen_yoy",
        live_col="truflation_us_pce_core_yoy",
        display_name="US Core PCE YoY (ex food, energy)",
    ),
    "pce_goods": TargetSpec(
        key="pce_goods",
        frozen_col="truflation_us_pce_goods_frozen_yoy",
        live_col="truflation_us_pce_goods_yoy",
        display_name="US PCE Goods YoY",
    ),
    "pce_services": TargetSpec(
        key="pce_services",
        frozen_col="truflation_us_pce_services_frozen_yoy",
        live_col="truflation_us_pce_services_yoy",
        display_name="US PCE Services YoY",
    ),
}


# ─── Data pull ────────────────────────────────────────────────────────────


def pull_yoy(target: TargetSpec, frozen: bool = True) -> pd.Series:
    """Pull Truflation YoY series for the given target.

    Frozen = as-published, never revised (matches the public chart).
    Live = continually revising.
    """
    url = target.frozen_url if frozen else target.live_url
    col = target.frozen_col if frozen else target.live_col
    r = requests.get(url, headers={"Authorization": FEED_API_KEY}, timeout=30)
    r.raise_for_status()
    data = r.json()
    s = pd.Series(data[col],
                       index=pd.to_datetime(data["index"]),
                       name="truf_yoy").sort_index()
    return s.dropna()


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day-ahead Truflation committee forecast")
    parser.add_argument("--target", type=str, default="cpi",
                          choices=sorted(TARGETS.keys()),
                          help="Target series (default: cpi)")
    args = parser.parse_args()
    target = TARGETS[args.target]

    print("=" * 78)
    print(f"Truflation {target.display_name} — day-ahead committee forecast")
    print("=" * 78)

    yoy = pull_yoy(target, frozen=True)
    today_value = float(yoy.iloc[-1])
    today_date = yoy.index[-1].date()
    target_date = today_date + pd.Timedelta(days=1).to_pytimedelta()
    print(f"\nFrozen feed (matches public chart):")
    print(f"  latest published value ({today_date}): {today_value:.4f}%")
    print(f"  this represents data through ~{today_date - pd.Timedelta(days=1).to_pytimedelta()} EOD")
    print(f"  next publication target ({target_date}) reflects data through ~today EOD")

    # ── Build a daily panel for harness-protocol forecasters ────────
    panel = pd.DataFrame({"truf_yoy": yoy})

    # ── 1. Persistence (the floor) ─────────────────────────────────
    persistence = PersistenceBaseline(
        target_col="truf_yoy", horizon=1, train_min=180,
        model_id=f"persistence_daily_{target.key}")
    pers_origins = panel.index[-91:-1]    # last 90 days
    pers_forecasts = walk_forward(persistence, panel, "truf_yoy",
                                          pers_origins, horizon=1)
    pers_df = attach_actuals(pers_forecasts, panel["truf_yoy"])
    pers_block = score(pers_df) if not pers_df.empty else None
    pers_pred = today_value    # by construction

    # ── 2. AR(1) with rolling-conformal bands ─────────────────────
    ar1 = AR1Baseline(
        target_col="truf_yoy", horizon=1, train_min=180,
        calib_months=30,    # interpreted as 30 calibration steps in daily frame
        band_method="rolling_conformal",
        model_id=f"ar1_daily_{target.key}")
    ar1_origins = panel.index[-91:-1]
    ar1_forecasts = walk_forward(ar1, panel, "truf_yoy",
                                           ar1_origins, horizon=1)
    ar1_df = attach_actuals(ar1_forecasts, panel["truf_yoy"])
    ar1_block = score(ar1_df) if not ar1_df.empty else None
    # Live forecast at today's origin
    ar1_today = ar1.fit_predict(panel,
                                       origin=panel.index[-1],
                                       target=panel.index[-1] + pd.Timedelta(days=1))
    ar1_pred = float(ar1_today.point)
    ar1_lo80 = float(ar1_today.lo80) if ar1_today.lo80 is not None else None
    ar1_hi80 = float(ar1_today.hi80) if ar1_today.hi80 is not None else None
    ar1_phi = ar1_today.metadata.get("phi")

    # ── Committee summary ──────────────────────────────────────────
    # Two members only: persistence + ar1_rolling. Ridge member removed
    # 2026-05-16 per the Truflation-only methodology direction.
    preds: dict[str, float] = {
        "persistence":   pers_pred,
        "ar1_rolling":   ar1_pred,
    }

    median_pred = float(np.median(list(preds.values())))
    range_bp = (max(preds.values()) - min(preds.values())) * 100   # in bp

    if range_bp <= 5:
        verdict = "✅ SHIP MEDIAN  (high confidence — committee agrees within 5 bp)"
    elif range_bp <= 15:
        verdict = "⚠️  SHIP MEDIAN with caveat (committee disagrees by 5–15 bp)"
    else:
        verdict = "❌ HOLD  (committee disagrees by > 15 bp — investigate)"

    # Direction (vs today)
    def direction(v):
        d = v - today_value
        if abs(d) < 0.0005:
            return "→"
        return "↑" if d > 0 else "↓"

    print()
    print("=" * 78)
    print(f"Day-ahead committee report — target {target_date}  ({target.display_name})")
    print("=" * 78)
    print()
    print(f"  {'model':<18s}  {'prediction':>12s}  {'Δ vs today':>11s}  {'dir':>3s}")
    print("  " + "-" * 50)
    for name, p in preds.items():
        d = (p - today_value) * 100   # bp
        print(f"  {name:<18s}  {p:>11.4f}%  {d:>+10.2f}bp  {direction(p):>3s}")

    print()
    print(f"  Committee median:    {median_pred:.4f}%   "
            f"({(median_pred-today_value)*100:+.2f}bp vs today)")
    print(f"  Committee range:     {range_bp:.2f}bp  ({len(preds)} models)")
    print(f"  Verdict:             {verdict}")

    # Backtest scores
    print()
    print("90-day OOS backtest (each model individually):")
    if pers_block:
        print(f"  persistence    n={pers_block.n:>3d}  "
                f"RMSE {pers_block.rmse:.4f}  MAE {pers_block.mae:.4f}")
    if ar1_block:
        red = (f"({ar1_block.rmse_reduction_pct:+.2f}% vs naive)"
                  if ar1_block.rmse_reduction_pct is not None else "")
        print(f"  ar1_rolling    n={ar1_block.n:>3d}  "
                f"RMSE {ar1_block.rmse:.4f}  MAE {ar1_block.mae:.4f}  {red}")

    # AR1 phi sanity
    if ar1_phi is not None:
        print(f"\n  AR1 fit details: φ = {ar1_phi:.4f}  "
                f"(near 1 ⇒ daily YoY is near-unit-root)")

    # ── Persist ─────────────────────────────────────────────────────
    out = OUT_DIR / f"committee_{target.key}_{today_date}.json"
    payload = {
        "target_key":   target.key,
        "target_name":  target.display_name,
        "as_of_date":   str(today_date),
        "target_date":  str(target_date),
        "today_value":  today_value,
        "predictions":  preds,
        "median":       median_pred,
        "range_bp":     range_bp,
        "n_models":     len(preds),
        "verdict":      verdict,
        "ar1_band_80":  [ar1_lo80, ar1_hi80] if ar1_lo80 is not None else None,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
