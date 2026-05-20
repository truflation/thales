"""Truflation US CPI YoY — Phase 2 long-horizon forecaster (UC+SV+MS).

Phase 2 closes the h=90 gap that Phase 1's bottom-up AR(1) drift can't
handle (Phase 1 lost to persistence by 7.8% RMSE at h=90; Phase 1.5 by
10.1%). Method:

  * Resample daily Truflation YoY to monthly (last day of month).
  * Walk-forward at quarterly origins (every 3 months from 2018-01).
  * At each origin, fit UC + SV + MS (Stock-Watson 2007 / Phase 2.2c
    machinery) on the trailing monthly history.
  * Monte-Carlo forecast at h ∈ {1, 3} months (≈ 30, 90 days).
  * Score per-horizon RMSE/MAE/coverage; compare to Phase 1.

This produces the long-horizon side of the eventual Phase 1 + Phase 2
ensemble: bottom-up for h ≤ 14 days, UC+SV+MS for h ≥ 30 days, blend
around the cross-over.

Run::

    uv run python scripts/forecast_truflation_cpi_phase2.py
    uv run python scripts/forecast_truflation_cpi_phase2.py --start 2018-01-01 \\
        --step-months 3 --num-samples 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.uc_sv_ms import (    # noqa: E402
    fit_uc_sv_ms,
    forecast_uc_sv_ms,
)

OUT_DIR = ROOT / "results" / "truflation_cpi_forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEED_API_KEY = "egiZc8oSTWdHr7V6lwYZxYil1wF0ezgF"
FEED_BASE = "https://api.truflation.com/api/v1/feed/truflation/macro-data-us"
HEADLINE_FROZEN_COL = "truflation_us_cpi_frozen_yoy"

HORIZONS_MONTHS = [1, 3]    # h=30d ≈ 1m, h=90d ≈ 3m
HORIZONS_DAYS_LABEL = {1: 30, 3: 90}    # for output column naming


# ─── Data loading ────────────────────────────────────────────────────────


def load_truflation_yoy_monthly() -> pd.Series:
    """Fetch daily Truflation US CPI YoY, resample to monthly (last value
    per month-end). The series is so slow-moving day-to-day that monthly
    sampling captures essentially all the trend dynamics needed for
    UC+SV+MS estimation, while keeping the fit tractable.
    """
    r = requests.get(f"{FEED_BASE}/{HEADLINE_FROZEN_COL}",
                       headers={"Authorization": FEED_API_KEY},
                       timeout=30)
    r.raise_for_status()
    data = r.json()
    daily = pd.Series(data[HEADLINE_FROZEN_COL],
                       index=pd.to_datetime(data["index"])
                       ).sort_index().dropna()
    monthly = daily.resample("ME").last().dropna()
    monthly.name = "truflation_us_cpi_yoy_monthly"
    return monthly


# ─── Walk-forward driver ─────────────────────────────────────────────────


def walk_forward(monthly_yoy: pd.Series,
                    start_date: str = "2018-01-01",
                    step_months: int = 3,
                    horizons_m: list[int] = HORIZONS_MONTHS,
                    num_warmup: int = 400,
                    num_samples: int = 400,
                    n_forecast_paths: int = 500,
                    sigma_eta_prior_scale: float = 0.05,
                    ) -> pd.DataFrame:
    """Walk forward through monthly history; at each origin, fit
    UC+SV+MS on history up to origin and Monte-Carlo forecast at
    horizons_m months ahead.
    """
    start = pd.Timestamp(start_date) + pd.offsets.MonthEnd(0)
    last_origin = monthly_yoy.index.max() - pd.offsets.MonthEnd(max(horizons_m))
    origins: list[pd.Timestamp] = []
    cursor = start
    while cursor <= last_origin:
        origins.append(cursor)
        cursor = cursor + pd.offsets.MonthEnd(step_months)

    rows = []
    for i, origin in enumerate(origins, 1):
        if origin not in monthly_yoy.index:
            continue
        history = monthly_yoy.loc[monthly_yoy.index <= origin]
        if len(history) < 60:
            continue

        print(f"  [{i:>3d}/{len(origins)}] origin={origin.date()}  "
                f"|y|={len(history)}  fitting…", flush=True)
        fit = fit_uc_sv_ms(history.values,
                              num_warmup=num_warmup,
                              num_samples=num_samples,
                              seed=int(origin.value % 10_000),
                              sigma_eta_prior_scale=sigma_eta_prior_scale,
                              return_samples=True)
        snapshots = forecast_uc_sv_ms(fit, horizons_m,
                                          n_paths=n_forecast_paths,
                                          seed=int(origin.value % 10_000))
        for h in horizons_m:
            target = origin + pd.offsets.MonthEnd(h)
            samples = snapshots[h]
            point = float(np.median(samples))
            lo80 = float(np.quantile(samples, 0.10))
            hi80 = float(np.quantile(samples, 0.90))
            lo95 = float(np.quantile(samples, 0.025))
            hi95 = float(np.quantile(samples, 0.975))
            actual = (float(monthly_yoy.loc[target])
                      if target in monthly_yoy.index else None)
            err = (point - actual) if actual is not None else None
            in_80 = (lo80 <= actual <= hi80) if actual is not None else None
            in_95 = (lo95 <= actual <= hi95) if actual is not None else None
            rows.append({
                "origin": origin,
                "horizon_months": h,
                "horizon_days_label": HORIZONS_DAYS_LABEL[h],
                "target_date": target,
                "point": point,
                "lo80": lo80, "hi80": hi80,
                "lo95": lo95, "hi95": hi95,
                "actual": actual,
                "error_pp": err,
                "in_80": in_80,
                "in_95": in_95,
                "width80_pp": hi80 - lo80,
                "width95_pp": hi95 - lo95,
                "sigma_low": fit.sigma_low,
                "sigma_high": fit.sigma_high,
                "p_high_at_origin": float(fit.smoothed_prob_high[-1]),
            })
    return pd.DataFrame(rows)


# ─── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--num-warmup", type=int, default=400)
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--n-forecast-paths", type=int, default=500)
    parser.add_argument("--sigma-eta-prior-scale", type=float, default=0.05)
    args = parser.parse_args()

    print("=" * 78)
    print("Truflation US CPI YoY — Phase 2 long-horizon (UC+SV+MS) forecaster")
    print("=" * 78)

    print("\nLoading actual Truflation US CPI YoY (Feed API)…")
    monthly = load_truflation_yoy_monthly()
    print(f"  Monthly: n={len(monthly)}, "
            f"{monthly.index.min().date()} → {monthly.index.max().date()}")
    print(f"  range = [{monthly.min():.2f}, {monthly.max():.2f}]  "
            f"mean = {monthly.mean():.2f}  std = {monthly.std():.2f}")

    print(f"\nWalk-forward from {args.start}, step {args.step_months}m, "
            f"horizons {HORIZONS_MONTHS}m (≈ "
            f"{[HORIZONS_DAYS_LABEL[h] for h in HORIZONS_MONTHS]}d)…")
    df = walk_forward(monthly,
                        start_date=args.start,
                        step_months=args.step_months,
                        horizons_m=HORIZONS_MONTHS,
                        num_warmup=args.num_warmup,
                        num_samples=args.num_samples,
                        n_forecast_paths=args.n_forecast_paths,
                        sigma_eta_prior_scale=args.sigma_eta_prior_scale)
    print(f"\n  Generated {len(df)} forecast points across "
            f"{df['origin'].nunique()} origins")

    if len(df):
        print("\nWalk-forward summary by horizon:")
        scored = df.dropna(subset=["actual"])
        agg = scored.groupby("horizon_months").agg(
            n=("actual", "count"),
            rmse=("error_pp", lambda x: float(np.sqrt(np.mean(x ** 2)))),
            mae=("error_pp", lambda x: float(np.mean(np.abs(x)))),
            mean_err=("error_pp", "mean"),
            cov_80=("in_80", "mean"),
            cov_95=("in_95", "mean"),
            width80=("width80_pp", "mean"),
            width95=("width95_pp", "mean"),
        ).reset_index()
        print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        out_csv = OUT_DIR / "walk_forward_summary_phase2.csv"
        df.to_csv(out_csv, index=False)
        agg_csv = OUT_DIR / "walk_forward_aggregate_phase2.csv"
        agg.to_csv(agg_csv, index=False)
        print(f"\nSaved walk-forward results: {out_csv}")
        print(f"Saved aggregate metrics:    {agg_csv}")


if __name__ == "__main__":
    main()
