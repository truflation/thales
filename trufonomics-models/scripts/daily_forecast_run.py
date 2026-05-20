"""Daily forecast driver — produces tomorrow's Truflation prediction.

Usage:
    uv run python scripts/daily_forecast_run.py               # latest available
    uv run python scripts/daily_forecast_run.py --backtest    # also print backtest metrics

Workflow:
  1. Load Truflation daily history from kairos parquet + FRED covariates
     from our vintage store.
  2. Fit walk-forward OLS on all data up to the latest available origin.
  3. Predict T+1 with bootstrap density for 80/95 bands and P(up).
  4. Log the prediction to the vintage store (series='thales_daily_forecast').
  5. Print three posting-format drafts.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.daily_forecaster import (  # noqa: E402
    FEATURES,
    Forecast,
    load_panel_from_existing_sources,
    predict_next_day,
    walk_forward_backtest,
)
from thales.evaluation import metrics as M  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

KAIROS_PARQUET = Path("/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
FORECAST_SERIES = "thales_daily_forecast"


def log_prediction(store: VintageStore, fc: Forecast) -> None:
    """Write the point forecast into the vintage store for track-record logging.

    reference_date = the target date (T+1); as_of_date = origin date (when we
    made the call). So a future scoring query can ask "what did we predict
    for 2026-04-25 as of 2026-04-24".
    """
    store.ingest(
        series_id=FORECAST_SERIES,
        observations=[(fc.target_date.date(), fc.point)],
        as_of_date=fc.origin_date.date(),
        source=FORECAST_SERIES,
    )


def format_post_concise(fc: Forecast) -> str:
    direction = "^" if fc.point > fc.today_value else (
        "v" if fc.point < fc.today_value else "=")
    return (
        f"Thales forecast - Truflation US CPI YoY for {fc.target_date:%b %d}\n"
        f"{fc.point:.2f}% {direction}  (today: {fc.today_value:.2f}%)"
    )


def format_post_with_band(fc: Forecast) -> str:
    lo, hi = fc.band_80
    return (
        f"Thales day-ahead Truflation US CPI YoY forecast\n"
        f"{fc.target_date:%B %d, %Y}:  {fc.point:.2f}%  "
        f"[80% band: {lo:.2f}%-{hi:.2f}%]\n"
        f"Today: {fc.today_value:.2f}%   P(up): {fc.p_up:.0%}"
    )


def format_post_with_drivers(fc: Forecast) -> str:
    coefs = {k: v for k, v in fc.coefficients.items() if k != "intercept"}
    top = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    lines = [
        "Thales day-ahead Truflation US CPI YoY forecast",
        f"{fc.target_date:%b %d}: {fc.point:.2f}% (today {fc.today_value:.2f}%)",
        f"80% band: [{fc.band_80[0]:.2f}%, {fc.band_80[1]:.2f}%]",
        f"P(up): {fc.p_up:.0%}",
        "Top weights:",
    ] + [f"  {k:<12s} {v:+.4f}" for k, v in top]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--backtest-start", default="2023-01-01")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    print("Loading Truflation daily + FRED covariates...")
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        panel = load_panel_from_existing_sources(KAIROS_PARQUET, store,
                                                   as_of_date=date.today())
    latest = panel[FEATURES].dropna().index.max()
    print(f"  latest usable origin: {latest:%Y-%m-%d}")
    print(f"  Truflation latest value: {panel.loc[latest, 'y_t']:.4f}")

    fc = predict_next_day(panel, origin=latest)
    print(f"  Training rows: {fc.n_train}  (from 2021-01-01 through {latest:%Y-%m-%d})")

    if not args.no_log:
        with VintageStore(VINTAGE_DB) as store:
            log_prediction(store, fc)
        print(f"  Logged: {FORECAST_SERIES} "
              f"target={fc.target_date:%Y-%m-%d} as_of={fc.origin_date:%Y-%m-%d}")

    print("\n" + "=" * 70)
    print("POST FORMAT 1 - Concise")
    print("=" * 70)
    print(format_post_concise(fc))

    print("\n" + "=" * 70)
    print("POST FORMAT 2 - Point + band + P(up)")
    print("=" * 70)
    print(format_post_with_band(fc))

    print("\n" + "=" * 70)
    print("POST FORMAT 3 - With drivers")
    print("=" * 70)
    print(format_post_with_drivers(fc))

    print("\n" + "=" * 70)
    print("Model internals (for you, not the post)")
    print("=" * 70)
    for feat, coef in fc.coefficients.items():
        print(f"  {feat:<12s} {coef:+.6f}")
    residuals_sd = float(np.std(fc.samples - fc.point))
    print(f"  Bootstrap residual SD: {residuals_sd:.5f}")
    print(f"  80% band half-width:   {(fc.band_80[1] - fc.band_80[0]) / 2:.4f}")
    print(f"  95% band half-width:   {(fc.band_95[1] - fc.band_95[0]) / 2:.4f}")

    if args.backtest:
        print("\n" + "=" * 70)
        print("ROLLING-ORIGIN BACKTEST")
        print("=" * 70)
        print(f"Running walk-forward from {args.backtest_start}...")
        bt = walk_forward_backtest(panel, start=args.backtest_start, end=latest)
        if bt.predictions.empty:
            print("  no backtest predictions produced")
            return
        df = bt.predictions
        df["naive"] = df["today_value"]
        rmse_model = M.rmse(df["pred"].values, df["actual"].values)
        rmse_naive = M.rmse(df["naive"].values, df["actual"].values)
        mae_model = M.mae(df["pred"].values, df["actual"].values)
        mae_naive = M.mae(df["naive"].values, df["actual"].values)
        dir_model = M.directional_accuracy(
            df["pred"].values, df["actual"].values,
            reference=df["today_value"].values)
        cov80 = ((df["actual"] >= df["lo80"]) & (df["actual"] <= df["hi80"])).mean()
        cov95 = ((df["actual"] >= df["lo95"]) & (df["actual"] <= df["hi95"])).mean()
        rmse_reduction = 100 * (1 - rmse_model / rmse_naive) if rmse_naive > 0 else float("nan")

        print(f"  n_origins:         {len(df)}")
        print(f"  RMSE model/naive:  {rmse_model:.5f} / {rmse_naive:.5f}  "
              f"({rmse_reduction:+.2f}%)")
        print(f"  MAE  model/naive:  {mae_model:.5f} / {mae_naive:.5f}")
        print(f"  Directional acc:   {dir_model:.3f}  (vs today's value)")
        print(f"  80% coverage:      {cov80:.3f}  (nominal 0.80)")
        print(f"  95% coverage:      {cov95:.3f}  (nominal 0.95)")

        out = ROOT / "results" / "daily_forecast"
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "backtest_predictions.csv", index=False)
        print(f"  saved: {out / 'backtest_predictions.csv'}")


if __name__ == "__main__":
    main()
