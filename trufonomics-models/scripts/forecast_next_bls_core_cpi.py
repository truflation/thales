"""Forecast the next BLS Core CPI YoY release — **standalone (headline-only)**.

Core CPI = CPILFESL = CPI All Items less Food and Energy. The
Fed-preferred CPI cut for monetary policy; traders care about it more
than headline because it excludes the volatile components.

Method: MoM-composed AR(1) on log-MoM of CPILFESL — direct analog of
Thales standalone for headline. Density via rolling-conformal residuals.

Run::

    uv run python scripts/forecast_next_bls_core_cpi.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "next_release_forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALIB_WINDOW_AR = 24
CALIB_WINDOW_RESIDUAL = 24
N_SAMPLES = 500


def _yoy_from_levels(levels: pd.Series) -> pd.Series:
    out = {}
    for t in levels.index:
        denom = pd.Timestamp(year=t.year - 1, month=t.month, day=1) \
                    + pd.offsets.MonthEnd(0)
        if denom in levels.index:
            out[t] = (levels.loc[t] / levels.loc[denom] - 1.0) * 100.0
    return pd.Series(out).sort_index()


def load_cpilfesl_levels() -> pd.Series:
    with duckdb.connect(str(VINTAGE_DB), read_only=True) as con:
        rows = con.execute(
            "SELECT reference_date, value FROM vintage "
            "WHERE series_id = 'CPILFESL' AND source = 'fred_alfred_target' "
            "AND (series_id, reference_date, as_of_date) IN ("
            "  SELECT series_id, reference_date, MAX(as_of_date) "
            "  FROM vintage WHERE series_id = 'CPILFESL' "
            "    AND source = 'fred_alfred_target' "
            "  GROUP BY series_id, reference_date) "
            "ORDER BY reference_date",
        ).fetchall()
    idx = [pd.Timestamp(r[0]) + pd.offsets.MonthEnd(0) for r in rows]
    return pd.Series([r[1] for r in rows], index=idx, name="cpilfesl")


def fit_ar1(x: np.ndarray) -> tuple[float, float, np.ndarray]:
    if len(x) < 4:
        return 0.0, 0.0, np.array([])
    a, b = x[:-1], x[1:]
    X = np.column_stack([np.ones_like(a), a])
    coef, *_ = np.linalg.lstsq(X, b, rcond=None)
    alpha, phi = float(coef[0]), float(coef[1])
    resid = b - (alpha + phi * a)
    return alpha, phi, resid


def forecast_core_standalone(levels: pd.Series,
                                  yoy: pd.Series,
                                  origin: pd.Timestamp,
                                  n_samples: int = N_SAMPLES,
                                  calib_ar: int = CALIB_WINDOW_AR,
                                  seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    history = levels.loc[levels.index <= origin]
    if len(history) < calib_ar + 12 + 1:
        raise ValueError(f"insufficient history: {len(history)}")
    log_levels = np.log(history.values)
    log_mom = np.diff(log_levels) * 100.0
    calib = log_mom[-calib_ar:]
    alpha, phi, resid = fit_ar1(calib)
    point_log_mom_next = alpha + phi * float(log_mom[-1])

    next_month = origin + pd.offsets.MonthEnd(1)
    drop_off = origin - pd.offsets.MonthEnd(11)
    yoy_at_origin = float(yoy.loc[origin]) if origin in yoy.index else float("nan")

    if drop_off in history.index:
        idx_drop = history.index.get_loc(drop_off)
        log_mom_drop = (np.log(history.iloc[idx_drop])
                          - np.log(history.iloc[idx_drop - 1])) * 100.0
    else:
        log_mom_drop = float("nan")

    point_yoy = yoy_at_origin + point_log_mom_next - log_mom_drop

    eps = rng.choice(resid[-CALIB_WINDOW_RESIDUAL:], size=n_samples)
    log_mom_samples = point_log_mom_next + eps
    yoy_samples = yoy_at_origin + log_mom_samples - log_mom_drop

    return {
        "origin":               str(origin.date()),
        "next_month":           str(next_month.date()),
        "yoy_at_origin":        yoy_at_origin,
        "log_mom_drop_off":     float(log_mom_drop),
        "ar1_alpha":            alpha,
        "ar1_phi":              phi,
        "point_log_mom_next":   float(point_log_mom_next),
        "point":                float(point_yoy),
        "lo80":                 float(np.quantile(yoy_samples, 0.10)),
        "hi80":                 float(np.quantile(yoy_samples, 0.90)),
        "lo95":                 float(np.quantile(yoy_samples, 0.025)),
        "hi95":                 float(np.quantile(yoy_samples, 0.975)),
        "n_samples":            n_samples,
    }


def main() -> None:
    print("=" * 78)
    print("Next BLS Core CPI YoY release — standalone forecast")
    print("=" * 78)

    levels = load_cpilfesl_levels()
    yoy = _yoy_from_levels(levels)
    latest = levels.index.max()
    latest_yoy = float(yoy.loc[latest])
    next_release = latest + pd.offsets.MonthEnd(1)

    print(f"\nLatest CPILFESL: {latest.date()}  level = {float(levels.iloc[-1]):.4f}  "
            f"YoY = {latest_yoy:.4f}%")
    print(f"Next release target: {next_release.date()}")

    fc = forecast_core_standalone(levels, yoy, latest)
    width80 = fc["hi80"] - fc["lo80"]
    width95 = fc["hi95"] - fc["lo95"]
    print()
    print("── Core CPI standalone (MoM-composed AR(1) on CPILFESL) ──")
    print(f"  AR(1):    α = {fc['ar1_alpha']:+.4f}  φ = {fc['ar1_phi']:+.4f}")
    print(f"  log-MoM[T+1] predicted: {fc['point_log_mom_next']:+.4f} pp")
    print(f"  log-MoM 12m ago (drops off): {fc['log_mom_drop_off']:+.4f} pp")
    print(f"  Point:    {fc['point']:.4f}%")
    print(f"  80% band: [{fc['lo80']:.4f}, {fc['hi80']:.4f}]   width {width80:.4f} pp")
    print(f"  95% band: [{fc['lo95']:.4f}, {fc['hi95']:.4f}]   width {width95:.4f} pp")

    out_path = OUT_DIR / f"core_cpi_{next_release.date()}_forecast_{date.today()}.json"
    out_path.write_text(json.dumps({
        "target":               "bls_headline_core_cpi_yoy",
        "as_of_date":           str(date.today()),
        "latest_cpilfesl_month": str(latest.date()),
        "latest_cpilfesl_yoy":  latest_yoy,
        "next_release_month":   str(next_release.date()),
        "core_cpi_standalone": {
            "point":            fc["point"],
            "band_80":          [fc["lo80"], fc["hi80"]],
            "band_95":          [fc["lo95"], fc["hi95"]],
            "ar1_alpha":        fc["ar1_alpha"],
            "ar1_phi":          fc["ar1_phi"],
            "method":           "mom_composed_ar1_on_cpilfesl",
        },
    }, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
