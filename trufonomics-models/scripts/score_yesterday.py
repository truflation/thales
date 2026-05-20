"""Score yesterday's day-ahead forecast against today's published LIVE YoY.

Reads the JSON dump produced by `forecast_live_tomorrow.py` whose
`target_date` == today, looks up the realized value from the Truflation
Feed API, and appends one row to `results/daily_forecast_live/scoring.csv`
with hit/miss flags on the 80% and 95% bands and on direction.

Idempotent: re-running on the same day overwrites the existing row for
that target_date instead of duplicating.

Usage:
    uv run python scripts/score_yesterday.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "results" / "daily_forecast_live"
SCORING_CSV = OUT_DIR / "scoring.csv"
FEED_API_KEY = "egiZc8oSTWdHr7V6lwYZxYil1wF0ezgF"
FEED_URL = "https://api.truflation.com/api/v1/feed/truflation/macro-data-us/truflation_us_cpi_yoy"


def pull_live_yoy() -> pd.Series:
    r = requests.get(FEED_URL, headers={"Authorization": FEED_API_KEY}, timeout=30)
    r.raise_for_status()
    body = r.json()
    s = pd.Series(body["truflation_us_cpi_yoy"],
                   index=pd.to_datetime(body["index"])).dropna()
    s.name = "live_yoy"
    return s


def find_prediction_for_target(target_date: pd.Timestamp) -> dict | None:
    """Find the JSON whose target_date matches; origin_date will be target-1."""
    expected_origin = (target_date - pd.Timedelta(days=1)).date()
    path = OUT_DIR / f"forecast_live_{expected_origin}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def first_seen_actual_for_target(target_date: pd.Timestamp,
                                 live: pd.Series) -> tuple[float, str] | None:
    """Return the target day's first-seen live print when available.

    The Truflation live feed can revise historical reference dates. For
    forecast scoring, the clean target is the value visible when that
    reference date first entered our daily run. `forecast_live_{target}.json`
    stores that day's `today_published_yoy`, so prefer it over a later API
    pull. Fall back to the current API only when we have no local snapshot.
    """
    snapshot = OUT_DIR / f"forecast_live_{target_date.date()}.json"
    if snapshot.exists():
        body = json.loads(snapshot.read_text())
        if body.get("origin_date") == str(target_date.date()):
            return float(body["today_published_yoy"]), "stored_first_seen_snapshot"
    if target_date in live.index:
        return float(live.loc[target_date]), "current_live_api_may_be_revised"
    return None


def score_one(target_date: pd.Timestamp, live: pd.Series) -> dict | None:
    pred = find_prediction_for_target(target_date)
    if pred is None:
        return None
    actual_pair = first_seen_actual_for_target(target_date, live)
    if actual_pair is None:
        return None
    actual, actual_source = actual_pair

    point = pred["point_yoy_pct"]
    lo80, hi80 = pred["band_80"]
    lo95, hi95 = pred["band_95"]
    today_pub = pred["today_published_yoy"]

    pred_up = point > today_pub
    actual_up = actual > today_pub

    return {
        "origin_date": pred["origin_date"],
        "target_date": str(target_date.date()),
        "point_pred": point,
        "lo80": lo80, "hi80": hi80,
        "lo95": lo95, "hi95": hi95,
        "today_published": today_pub,
        "actual": actual,
        "actual_source": actual_source,
        "error": point - actual,
        "abs_error": abs(point - actual),
        "naive_error": today_pub - actual,
        "hit_80": bool(lo80 <= actual <= hi80),
        "hit_95": bool(lo95 <= actual <= hi95),
        "pred_up": bool(pred_up),
        "actual_up": bool(actual_up),
        "direction_hit": bool(pred_up == actual_up),
    }


def append_or_replace(row: dict) -> None:
    if SCORING_CSV.exists():
        df = pd.read_csv(SCORING_CSV)
        df = df[df["target_date"] != row["target_date"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values("target_date").reset_index(drop=True)
    df.to_csv(SCORING_CSV, index=False)


def main() -> None:
    print("Pulling LIVE YoY from Feed API...")
    live = pull_live_yoy()
    today = pd.Timestamp(live.index.max())
    print(f"  latest live obs: {today:%Y-%m-%d}  value={live.iloc[-1]:.4f}%")

    # Score the most recent day where we have both a prediction AND realized
    target = today
    row = score_one(target, live)
    if row is None:
        print(f"\nNo prediction found for target={target:%Y-%m-%d} "
              f"(expected JSON at origin={target - pd.Timedelta(days=1):%Y-%m-%d}).")
        print("Either no forecast was made for that day, or the live series doesn't yet have it.")
        sys.exit(0)

    append_or_replace(row)
    print()
    print("=" * 64)
    print(f"Scored target_date = {row['target_date']}")
    print("=" * 64)
    print(f"  Point pred:    {row['point_pred']:.4f}%")
    print(f"  Today (T):     {row['today_published']:.4f}%")
    print(f"  Actual (T+1):  {row['actual']:.4f}%")
    print(f"  Error:         {row['error']:+.4f} pp  (|err| {row['abs_error']:.4f})")
    print(f"  Naive error:   {row['naive_error']:+.4f} pp")
    print(f"  80% band:      [{row['lo80']:.4f}, {row['hi80']:.4f}]   "
          f"hit={row['hit_80']}")
    print(f"  95% band:      [{row['lo95']:.4f}, {row['hi95']:.4f}]   "
          f"hit={row['hit_95']}")
    print(f"  Direction:     pred_up={row['pred_up']}  actual_up={row['actual_up']}  "
          f"hit={row['direction_hit']}")
    print()
    print(f"Wrote: {SCORING_CSV}")


if __name__ == "__main__":
    main()
