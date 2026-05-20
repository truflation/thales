"""Weekly rollup of the Stefan day-ahead LIVE forecaster.

Reads `results/daily_forecast_live/scoring.csv`, takes the last 7 days
(default; configurable via --days), and prints the headline metric block:

    n predictions  ·  MAE  ·  80%/95% coverage  ·  direction vs base-rate
    SHIP/HOLD verdict using same gates as the historical backtest

Use this once a week before deciding whether to extend the pilot or
publish externally.

Usage:
    uv run python scripts/weekly_rollup.py             # last 7 days
    uv run python scripts/weekly_rollup.py --days 14   # last 2 weeks
    uv run python scripts/weekly_rollup.py --all       # full history
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCORING_CSV = ROOT / "results" / "daily_forecast_live" / "scoring.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7,
                     help="Number of most-recent days to include (default 7)")
    ap.add_argument("--all", action="store_true",
                     help="Include full scoring history (overrides --days)")
    args = ap.parse_args()

    if not SCORING_CSV.exists():
        print(f"No scoring history yet at {SCORING_CSV}.")
        print("Run scripts/score_yesterday.py for at least one day first.")
        return

    df = pd.read_csv(SCORING_CSV)
    df["target_date"] = pd.to_datetime(df["target_date"])
    df = df.sort_values("target_date").reset_index(drop=True)

    if not args.all:
        df = df.tail(args.days).reset_index(drop=True)

    if df.empty:
        print("No rows in window.")
        return

    n = len(df)
    mae = df["abs_error"].mean()
    rmse = (df["error"] ** 2).mean() ** 0.5
    naive_rmse = (df["naive_error"] ** 2).mean() ** 0.5
    rmse_red = (1 - rmse / naive_rmse) * 100 if naive_rmse > 0 else float("nan")
    cov80 = df["hit_80"].mean()
    cov95 = df["hit_95"].mean()
    dir_hit = df["direction_hit"].mean()
    base_up = df["actual_up"].mean()

    print("=" * 72)
    print("Thales Day-Ahead LIVE Forecaster — Rollup")
    print("=" * 72)
    print(f"Window:  {df['target_date'].min():%Y-%m-%d} → "
          f"{df['target_date'].max():%Y-%m-%d}  (n={n})")
    print()
    print(f"  RMSE model / naive:   {rmse:.4f} / {naive_rmse:.4f} pp  "
          f"({rmse_red:+.2f}%)")
    print(f"  MAE:                  {mae:.4f} pp")
    print(f"  80% coverage:         {cov80:.1%}  (nominal 80%)   "
          f"{'✓' if abs(cov80 - 0.80) < 0.07 else '✗'}")
    print(f"  95% coverage:         {cov95:.1%}  (nominal 95%)   "
          f"{'✓' if abs(cov95 - 0.95) < 0.04 else '✗'}")
    print(f"  Directional acc:     {dir_hit:.1%}  "
          f"(base-rate up: {base_up:.1%})")
    print()

    if n < 7:
        print(f"  Verdict: insufficient sample (n={n} < 7), "
              f"hold ship-decision.")
    else:
        calibrated = (abs(cov80 - 0.80) < 0.07 and abs(cov95 - 0.95) < 0.04)
        better_than_naive = rmse_red > -10
        ship = calibrated and better_than_naive
        print(f"  Verdict: {'SHIP' if ship else 'HOLD'} — "
              f"calibration={'OK' if calibrated else 'OFF'}, "
              f"RMSE-vs-naive={'OK' if better_than_naive else 'WORSE'}")
    print()
    print("Per-day:")
    for _, r in df.iterrows():
        m80 = "✓" if r["hit_80"] else "✗"
        m95 = "✓" if r["hit_95"] else "✗"
        md = "✓" if r["direction_hit"] else "✗"
        print(f"  {r['target_date']:%Y-%m-%d}  pred={r['point_pred']:.4f} "
              f"actual={r['actual']:.4f}  err={r['error']:+.4f}  "
              f"80{m80} 95{m95} dir{md}")


if __name__ == "__main__":
    main()
