"""Conservative day-ahead Truflation CPI YoY forecast.

This script is intentionally boring. Recent backtests show that daily
Truflation YoY is close to a random walk and most feature-rich variants do
not beat `y[T+1] = y[T]` reliably. A production forecast should therefore:

  1. Benchmark every candidate against persistence.
  2. Use a candidate only if it clears a small recent-validation hurdle.
  3. Calibrate bands from recent one-step-ahead errors.
  4. Warn loudly when the input CSV is stale.

Input CSV schema expected:
    date, inflation, cpiIndex, cpiIndexYearAgo, created_at

Usage:
    uv run python scripts/forecast_truflation_tomorrow_real.py
    uv run python scripts/forecast_truflation_tomorrow_real.py --csv /path/to/Truflation_US_CPI_Data.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation.conformal import conformal_band_offsets  # noqa: E402
from thales.evaluation import metrics as M  # noqa: E402


OUT_DIR = ROOT / "results" / "daily_forecast_real"
FEED_API_KEY = "egiZc8oSTWdHr7V6lwYZxYil1wF0ezgF"
FEED_URL = (
    "https://api.truflation.com/api/v1/feed/truflation/"
    "macro-data-us/truflation_us_cpi_yoy"
)
DEFAULT_CSV_CANDIDATES = (
    Path("/Users/kluless/Downloads/Truflation_US_CPI_Data_(Frozen) (1).csv"),
    Path("/Users/kluless/Downloads/Truflation_US_CPI_Data_(Frozen).csv"),
    Path("/Users/kluless/Downloads/Truflation_US_CPI_Data.csv"),
    ROOT.parent / "data" / "truflation" / "frozen" / "us_cpi.csv",
    ROOT.parent / "data" / "truflation" / "unfrozen" / "us_cpi.csv",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    forecast: Callable[[pd.Series, pd.Timestamp], float]


def _resolve_csv(path: str | None) -> Path:
    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for p in DEFAULT_CSV_CANDIDATES:
        if p.exists():
            return p
    searched = "\n  ".join(str(p) for p in DEFAULT_CSV_CANDIDATES)
    raise FileNotFoundError(f"No Truflation CPI CSV found. Searched:\n  {searched}")


def load_cpi_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "inflation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[["inflation"]].dropna()
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_live_feed() -> pd.DataFrame:
    r = requests.get(FEED_URL, headers={"Authorization": FEED_API_KEY}, timeout=30)
    r.raise_for_status()
    body = r.json()
    df = pd.DataFrame({
        "date": pd.to_datetime(body["index"]),
        "inflation": body["truflation_us_cpi_yoy"],
    })
    df = df.set_index("date").sort_index()
    df = df.dropna(subset=["inflation"])
    df = df[~df.index.duplicated(keep="last")]
    return df


def _persistence(y: pd.Series, origin: pd.Timestamp) -> float:
    return float(y.loc[origin])


def _median_drift(days: int) -> Callable[[pd.Series, pd.Timestamp], float]:
    def inner(y: pd.Series, origin: pd.Timestamp) -> float:
        hist = y.loc[:origin].diff().dropna().tail(days)
        if hist.empty:
            return float(y.loc[origin])
        return float(y.loc[origin] + hist.median())

    return inner


def _mean_drift(days: int) -> Callable[[pd.Series, pd.Timestamp], float]:
    def inner(y: pd.Series, origin: pd.Timestamp) -> float:
        hist = y.loc[:origin].diff().dropna().tail(days)
        if hist.empty:
            return float(y.loc[origin])
        return float(y.loc[origin] + hist.mean())

    return inner


def _ewma_drift(span: int) -> Callable[[pd.Series, pd.Timestamp], float]:
    def inner(y: pd.Series, origin: pd.Timestamp) -> float:
        hist = y.loc[:origin].diff().dropna()
        if hist.empty:
            return float(y.loc[origin])
        drift = hist.ewm(span=span, adjust=False).mean().iloc[-1]
        return float(y.loc[origin] + drift)

    return inner


CANDIDATES = (
    Candidate("persistence", _persistence),
    Candidate("median_drift_7d", _median_drift(7)),
    Candidate("median_drift_30d", _median_drift(30)),
    Candidate("mean_drift_30d", _mean_drift(30)),
    Candidate("ewma_drift_14d", _ewma_drift(14)),
)


def candidate_frame(y: pd.Series,
                    start: pd.Timestamp,
                    end: pd.Timestamp | None = None) -> pd.DataFrame:
    end = end or y.index.max()
    rows: list[dict] = []
    for origin in y.loc[start:end].index:
        target = origin + pd.Timedelta(days=1)
        if target not in y.index or pd.isna(y.loc[target]):
            continue
        row = {
            "origin": origin,
            "target": target,
            "today": float(y.loc[origin]),
            "actual": float(y.loc[target]),
        }
        for cand in CANDIDATES:
            row[cand.name] = cand.forecast(y, origin)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_candidates(df: pd.DataFrame,
                         candidate_names: list[str]) -> pd.DataFrame:
    rows = []
    naive_rmse = M.rmse(df["today"].values, df["actual"].values)
    for name in candidate_names:
        pred = df[name].values
        actual = df["actual"].values
        rmse = M.rmse(pred, actual)
        rows.append({
            "model": name,
            "n": len(df),
            "rmse": rmse,
            "mae": M.mae(pred, actual),
            "rmse_reduction_vs_persistence_pct":
                100 * (1.0 - rmse / naive_rmse) if naive_rmse > 0 else np.nan,
            "directional_accuracy":
                M.directional_accuracy(pred, actual, reference=df["today"].values),
        })
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def select_model(summary: pd.DataFrame, min_edge_pct: float) -> str:
    best = summary.iloc[0]
    edge = float(best["rmse_reduction_vs_persistence_pct"])
    if best["model"] != "persistence" and edge >= min_edge_pct:
        return str(best["model"])
    return "persistence"


def calibration_errors(y: pd.Series,
                       model_name: str,
                       origin: pd.Timestamp,
                       days: int) -> np.ndarray:
    cand = {c.name: c for c in CANDIDATES}[model_name]
    rows = []
    # Last `days` scored origins before the live origin. The target for
    # origin - 1 is known at the live origin, so this is usable in real time.
    scored_origins = []
    for o in y.index[y.index < origin]:
        target = o + pd.Timedelta(days=1)
        if target in y.index and target <= origin:
            scored_origins.append(o)
    for o in scored_origins[-days:]:
        target = o + pd.Timedelta(days=1)
        rows.append(float(y.loc[target] - cand.forecast(y, o)))
    if len(rows) < 20:
        raise ValueError(f"Need at least 20 calibration errors; got {len(rows)}")
    return np.asarray(rows, dtype=float)


def forecast_current(y: pd.Series,
                     model_name: str,
                     calib_days: int) -> dict:
    origin = y.index.max()
    target = origin + pd.Timedelta(days=1)
    cand = {c.name: c for c in CANDIDATES}[model_name]
    raw_point = cand.forecast(y, origin)

    errs = calibration_errors(y, model_name, origin, calib_days)
    # Keep the point forecast exactly equal to the selected candidate. Any
    # recent skew/bias is handled by asymmetric signed conformal offsets.
    median_error = float(np.median(errs))
    point = float(raw_point)
    lo80, hi80 = conformal_band_offsets(errs, alpha=0.20)
    lo95, hi95 = conformal_band_offsets(errs, alpha=0.05)
    samples = point + errs

    today = float(y.loc[origin])
    return {
        "origin_date": str(origin.date()),
        "target_date": str(target.date()),
        "selected_model": model_name,
        "today_yoy_pct": today,
        "raw_point_yoy_pct": raw_point,
        "bias_correction_pp": 0.0,
        "median_calibration_error_pp": median_error,
        "point_yoy_pct": point,
        "delta_vs_today_pp": point - today,
        "band_80": [point + lo80, point + hi80],
        "band_95": [point + lo95, point + hi95],
        "p_up": float((samples > today).mean()),
        "n_calibration": int(len(errs)),
        "calibration_error_mae_pp": float(np.mean(np.abs(errs))),
    }


def rolling_production_backtest(y: pd.Series,
                                model_name: str,
                                start: pd.Timestamp,
                                calib_days: int) -> pd.DataFrame:
    """Backtest the exact production rule used by `forecast_current`.

    For every scored origin:
      * raw forecast from the selected candidate
      * finite-sample conformal offsets from trailing signed errors
    """
    cand = {c.name: c for c in CANDIDATES}[model_name]
    rows: list[dict] = []
    for origin in y.loc[start:].index:
        target = origin + pd.Timedelta(days=1)
        if target not in y.index or pd.isna(y.loc[target]):
            continue
        try:
            errs = calibration_errors(y, model_name, origin, calib_days)
        except ValueError:
            continue
        raw_point = cand.forecast(y, origin)
        median_error = float(np.median(errs))
        point = float(raw_point)
        lo80, hi80 = conformal_band_offsets(errs, alpha=0.20)
        lo95, hi95 = conformal_band_offsets(errs, alpha=0.05)
        today = float(y.loc[origin])
        actual = float(y.loc[target])
        rows.append({
            "origin": origin,
            "target": target,
            "today": today,
            "actual": actual,
            "raw_point": raw_point,
            "bias": 0.0,
            "median_calibration_error": median_error,
            "pred": point,
            "lo80": point + lo80,
            "hi80": point + hi80,
            "lo95": point + lo95,
            "hi95": point + hi95,
            "error": point - actual,
            "naive_error": today - actual,
            "pred_up": point > today,
            "actual_up": actual > today,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("live", "csv"), default="live",
                    help="Use Truflation live feed by default; csv is offline diagnostics only")
    ap.add_argument("--csv", help="Path to Truflation CPI CSV")
    ap.add_argument("--backtest-start", default="2025-01-01")
    ap.add_argument("--selection-days", type=int, default=180)
    ap.add_argument("--calib-days", type=int, default=90)
    ap.add_argument("--min-edge-pct", type=float, default=2.0,
                    help="Required RMSE reduction vs persistence before using a non-persistence model")
    args = ap.parse_args()

    if args.source == "live":
        csv_path = None
        df = load_live_feed()
        source_label = "Truflation live feed"
    else:
        csv_path = _resolve_csv(args.csv)
        df = load_cpi_csv(csv_path)
        source_label = str(csv_path)
    y = df["inflation"]
    candidate_names = [c.name for c in CANDIDATES]

    latest = y.index.max()
    stale_days = (pd.Timestamp(date.today()) - latest.normalize()).days

    print("Loading Truflation CPI YoY")
    print(f"  source: {source_label}")
    print(f"  range:  {y.index.min():%Y-%m-%d} -> {latest:%Y-%m-%d}  n={len(y)}")
    print(f"  latest: {y.iloc[-1]:.6f}%")
    print(f"  next reference date to forecast: {(latest + pd.Timedelta(days=1)):%Y-%m-%d}")
    if args.source == "csv" and stale_days > 1:
        print(f"  WARNING: CSV latest observation is {stale_days} days before today; "
              "do not use this for posting if the live site has newer values.")

    full_bt = candidate_frame(y, pd.Timestamp(args.backtest_start))
    if full_bt.empty:
        raise RuntimeError("No scored backtest rows produced")

    select_bt = full_bt.tail(args.selection_days)
    summary = summarize_candidates(select_bt, candidate_names)
    selected = select_model(summary, args.min_edge_pct)

    forecast = forecast_current(y, selected, args.calib_days)

    prod_bt = rolling_production_backtest(
        y, selected, pd.Timestamp(args.backtest_start), args.calib_days)
    if prod_bt.empty:
        raise RuntimeError("No production-rule backtest rows produced")

    selected_pred = prod_bt["pred"].values
    actual = prod_bt["actual"].values
    today = prod_bt["today"].values
    rmse_model = M.rmse(selected_pred, actual)
    rmse_naive = M.rmse(today, actual)
    mae_model = M.mae(selected_pred, actual)
    mae_naive = M.mae(today, actual)
    dir_hit = (prod_bt["pred_up"] == prod_bt["actual_up"]).mean()
    base_up = prod_bt["actual_up"].mean()
    cov80 = ((prod_bt["actual"] >= prod_bt["lo80"]) &
             (prod_bt["actual"] <= prod_bt["hi80"])).mean()
    cov95 = ((prod_bt["actual"] >= prod_bt["lo95"]) &
             (prod_bt["actual"] <= prod_bt["hi95"])).mean()
    width80 = (prod_bt["hi80"] - prod_bt["lo80"]).mean()
    width95 = (prod_bt["hi95"] - prod_bt["lo95"]).mean()

    print()
    print("=" * 76)
    print("Candidate validation")
    print("=" * 76)
    print(f"Window: last {len(select_bt)} scored origins "
          f"({select_bt['origin'].min():%Y-%m-%d} -> {select_bt['origin'].max():%Y-%m-%d})")
    for _, r in summary.iterrows():
        print(f"  {r['model']:<18s} RMSE={r['rmse']:.5f}  "
              f"MAE={r['mae']:.5f}  "
              f"vs_persist={r['rmse_reduction_vs_persistence_pct']:+.2f}%  "
              f"dir={r['directional_accuracy']:.1%}")
    print(f"Selected point model: {selected}")

    print()
    print("=" * 76)
    print("Forecast")
    print("=" * 76)
    print(f"{forecast['target_date']}: {forecast['point_yoy_pct']:.6f}%")
    print(f"  today:       {forecast['today_yoy_pct']:.6f}%")
    print(f"  delta:       {forecast['delta_vs_today_pp']:+.6f} pp")
    print(f"  80% band:    [{forecast['band_80'][0]:.6f}%, {forecast['band_80'][1]:.6f}%]")
    print(f"  95% band:    [{forecast['band_95'][0]:.6f}%, {forecast['band_95'][1]:.6f}%]")
    print(f"  P(up):       {forecast['p_up']:.1%}")
    print(f"  calib n:     {forecast['n_calibration']}")

    print()
    print("=" * 76)
    print("Backtest context")
    print("=" * 76)
    print(f"Window: {prod_bt['origin'].min():%Y-%m-%d} -> {prod_bt['origin'].max():%Y-%m-%d}  "
          f"n={len(prod_bt)}")
    print(f"  selected RMSE / persistence RMSE: {rmse_model:.5f} / {rmse_naive:.5f} "
          f"({100 * (1 - rmse_model / rmse_naive):+.2f}%)")
    print(f"  selected MAE / persistence MAE:  {mae_model:.5f} / {mae_naive:.5f} pp")
    print(f"  80% coverage:                    {cov80:.1%}  width {width80:.5f} pp")
    print(f"  95% coverage:                    {cov95:.1%}  width {width95:.5f} pp")
    print(f"  selected directional accuracy:   {dir_hit:.1%}  "
          f"(base-rate up: {base_up:.1%})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_bt = OUT_DIR / "candidate_backtest.csv"
    full_bt.to_csv(out_bt, index=False)
    out_prod = OUT_DIR / "production_backtest.csv"
    prod_bt.to_csv(out_prod, index=False)
    out_json = OUT_DIR / f"forecast_real_{forecast['origin_date']}.json"
    payload = {
        "source": args.source,
        "csv_path": str(csv_path) if csv_path else None,
        "data_range": [str(y.index.min().date()), str(latest.date())],
        "stale_days": int(stale_days),
        "selection_window_days": args.selection_days,
        "min_edge_pct": args.min_edge_pct,
        "candidate_summary": summary.to_dict(orient="records"),
        "forecast": forecast,
        "backtest": {
            "window_start": str(prod_bt["origin"].min().date()),
            "window_end": str(prod_bt["origin"].max().date()),
            "n": int(len(prod_bt)),
            "selected_model": selected,
            "rmse_selected": float(rmse_model),
            "rmse_persistence": float(rmse_naive),
            "rmse_reduction_vs_persistence_pct":
                float(100 * (1 - rmse_model / rmse_naive)),
            "mae_selected": float(mae_model),
            "mae_persistence": float(mae_naive),
            "coverage_80": float(cov80),
            "coverage_95": float(cov95),
            "width_80": float(width80),
            "width_95": float(width95),
            "directional_accuracy": float(dir_hit),
            "base_rate_up": float(base_up),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print()
    print(f"Saved: {out_bt}")
    print(f"Saved: {out_prod}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
