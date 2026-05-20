"""Forecast the next BEA PCE Headline YoY release.

Two flavours, mirroring the CPI side:

  1. **PCE standalone** — MoM-composed AR(1) on PCEPI log-MoM. Direct
     PCE analog of Thales standalone for CPI.

  2. **Cleveland Fed comparator** — Cleveland publishes daily nowcasts
     of headline PCE YoY (clevfed_pce_yoy); we surface the latest
     value for the target month and (eventually) blend it with the
     standalone.

Run::

    uv run python scripts/forecast_next_bea_pce.py
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

# Rolling calibration windows — matches the CPI script
CALIB_WINDOW_AR = 24
CALIB_WINDOW_RESIDUAL = 24
N_SAMPLES = 500


# ─── Data loading ────────────────────────────────────────────────────────


def load_pcepi_levels() -> pd.Series:
    """Latest-as-of PCEPI level series (BEA headline PCE price index)."""
    with duckdb.connect(str(VINTAGE_DB), read_only=True) as con:
        rows = con.execute(
            "SELECT reference_date, value FROM vintage "
            "WHERE series_id = 'PCEPI' AND source = 'fred_alfred_target' "
            "AND (series_id, reference_date, as_of_date) IN ("
            "  SELECT series_id, reference_date, MAX(as_of_date) "
            "  FROM vintage WHERE series_id = 'PCEPI' "
            "    AND source = 'fred_alfred_target' "
            "  GROUP BY series_id, reference_date) "
            "ORDER BY reference_date",
        ).fetchall()
    # Convert each reference_date (always month-start in FRED) to month-end
    idx = [pd.Timestamp(r[0]) + pd.offsets.MonthEnd(0) for r in rows]
    return pd.Series([r[1] for r in rows], index=idx, name="pcepi")


def _yoy_from_levels(levels: pd.Series) -> pd.Series:
    """Date-based YoY: level[t] / level[t-1y] - 1.  Resilient to gaps."""
    out = {}
    for t in levels.index:
        denom = pd.Timestamp(year=t.year - 1, month=t.month, day=1) \
                    + pd.offsets.MonthEnd(0)
        if denom in levels.index:
            out[t] = (levels.loc[t] / levels.loc[denom] - 1.0) * 100.0
    return pd.Series(out).sort_index()


def load_cleveland_pce_yoy() -> pd.Series:
    """Latest Cleveland Fed PCE YoY nowcast per month-end."""
    with duckdb.connect(str(VINTAGE_DB), read_only=True) as con:
        rows = con.execute(
            "SELECT reference_date, value FROM vintage "
            "WHERE series_id = 'clevfed_pce_yoy' "
            "AND (series_id, reference_date, as_of_date) IN ("
            "  SELECT series_id, reference_date, MAX(as_of_date) "
            "  FROM vintage WHERE series_id = 'clevfed_pce_yoy' "
            "  GROUP BY series_id, reference_date) "
            "ORDER BY reference_date",
        ).fetchall()
    return pd.Series([r[1] for r in rows],
                       index=pd.to_datetime([r[0] for r in rows]),
                       name="clevfed_pce_yoy")


# ─── PCE standalone — MoM-composed AR(1) ─────────────────────────────────


def fit_ar1(x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """OLS AR(1): x[t] = α + φ · x[t-1] + ε.  Returns (α, φ, residuals)."""
    if len(x) < 4:
        return 0.0, 0.0, np.array([])
    a = x[:-1]
    b = x[1:]
    X = np.column_stack([np.ones_like(a), a])
    coef, *_ = np.linalg.lstsq(X, b, rcond=None)
    alpha, phi = float(coef[0]), float(coef[1])
    resid = b - (alpha + phi * a)
    return alpha, phi, resid


def forecast_pce_standalone(levels: pd.Series,
                                yoy: pd.Series,
                                origin: pd.Timestamp,
                                n_samples: int = N_SAMPLES,
                                calib_ar: int = CALIB_WINDOW_AR,
                                calib_resid: int = CALIB_WINDOW_RESIDUAL,
                                seed: int = 0) -> dict:
    """Forecast the next-month YoY by MoM-composing an AR(1) forecast of
    next month's log-MoM, then resolving the YoY identity::

        YoY[T+1] = YoY[T] + MoM_log[T+1] − MoM_log[T+1-12]

    Density via bootstrap of AR(1) calibration residuals."""
    rng = np.random.default_rng(seed)
    history = levels.loc[levels.index <= origin]
    if len(history) < calib_ar + 12 + 1:
        raise ValueError(f"insufficient history ({len(history)}) for "
                          f"calib_ar={calib_ar} + 12 + 1")

    log_levels = np.log(history.values)
    log_mom = np.diff(log_levels) * 100.0    # percent
    calib = log_mom[-calib_ar:]
    alpha, phi, resid = fit_ar1(calib)
    last_log_mom = float(log_mom[-1])
    point_log_mom_next = alpha + phi * last_log_mom

    next_month = origin + pd.offsets.MonthEnd(1)
    drop_off = origin - pd.offsets.MonthEnd(11)     # MoM 12 months ago
    yoy_at_origin = float(yoy.loc[origin]) if origin in yoy.index else float("nan")

    # The MoM 12 months ago (in log percent)
    if drop_off in history.index:
        idx_drop = history.index.get_loc(drop_off)
        log_mom_drop = (np.log(history.iloc[idx_drop])
                          - np.log(history.iloc[idx_drop - 1])) * 100.0
    else:
        log_mom_drop = float("nan")

    point_yoy_next = yoy_at_origin + point_log_mom_next - log_mom_drop

    # Density: bootstrap residuals around point_log_mom_next, propagate to YoY
    resid_calib = resid[-calib_resid:]
    eps_samples = rng.choice(resid_calib, size=n_samples)
    log_mom_samples = point_log_mom_next + eps_samples
    yoy_samples = yoy_at_origin + log_mom_samples - log_mom_drop

    return {
        "origin": str(origin.date()),
        "next_month": str(next_month.date()),
        "yoy_at_origin": yoy_at_origin,
        "log_mom_drop_off": float(log_mom_drop),
        "ar1_alpha": alpha,
        "ar1_phi": phi,
        "point_log_mom_next": float(point_log_mom_next),
        "point": float(point_yoy_next),
        "lo80": float(np.quantile(yoy_samples, 0.10)),
        "hi80": float(np.quantile(yoy_samples, 0.90)),
        "lo95": float(np.quantile(yoy_samples, 0.025)),
        "hi95": float(np.quantile(yoy_samples, 0.975)),
        "n_samples": n_samples,
    }


# ─── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 78)
    print("Next BEA Headline PCE YoY release — forecast")
    print("=" * 78)

    levels = load_pcepi_levels()
    yoy = _yoy_from_levels(levels)
    clev = load_cleveland_pce_yoy()

    latest_month = levels.index.max()
    next_release_month = latest_month + pd.offsets.MonthEnd(1)
    latest_yoy = float(yoy.loc[latest_month]) if latest_month in yoy.index else float("nan")
    latest_level = float(levels.loc[latest_month])

    print(f"\nLatest PCEPI: {latest_month.date()}  level = {latest_level:.4f}  "
            f"YoY = {latest_yoy:.4f}%")
    print(f"Next release target: {next_release_month.date()}")

    # ── PCE standalone ────────────────────────────────────────────
    fc = forecast_pce_standalone(levels, yoy, latest_month)
    width80 = fc["hi80"] - fc["lo80"]
    width95 = fc["hi95"] - fc["lo95"]
    print()
    print("── PCE standalone (MoM-composed AR(1) on PCEPI) ──")
    print(f"  Method:   mirror of Thales standalone for CPI, applied to PCEPI")
    print(f"  AR(1):    α = {fc['ar1_alpha']:+.4f}  φ = {fc['ar1_phi']:+.4f}")
    print(f"  log-MoM[T+1] predicted: {fc['point_log_mom_next']:+.4f} pp")
    print(f"  log-MoM 12m ago (drops off): {fc['log_mom_drop_off']:+.4f} pp")
    print(f"  Point:    {fc['point']:.4f}%")
    print(f"  80% band: [{fc['lo80']:.4f}, {fc['hi80']:.4f}]   width {width80:.4f} pp")
    print(f"  95% band: [{fc['lo95']:.4f}, {fc['hi95']:.4f}]   width {width95:.4f} pp")

    # ── Cleveland Fed PCE comparator ────────────────────────────────
    print()
    if next_release_month in clev.index:
        clev_at_target = float(clev.loc[next_release_month])
        clev_label = f"for {next_release_month.date()} (latest as_of {date.today()})"
    elif latest_month in clev.index:
        clev_at_target = float(clev.loc[latest_month])
        clev_label = (f"latest available ({latest_month.date()}) — "
                          f"may not yet cover {next_release_month.date()}")
    else:
        clev_at_target = float("nan")
        clev_label = "unavailable"
    print(f"── Cleveland Fed PCE nowcast ({clev_label}) ──")
    print(f"  Point: {clev_at_target:.4f}%")

    # Save
    out_path = OUT_DIR / (
        f"pce_{next_release_month.date()}_forecast_{date.today()}.json")
    out_path.write_text(json.dumps({
        "target": "bea_headline_pce_yoy",
        "as_of_date": str(date.today()),
        "latest_pcepi_month": str(latest_month.date()),
        "latest_pcepi_level": latest_level,
        "latest_pcepi_yoy": latest_yoy,
        "next_release_month": str(next_release_month.date()),
        "pce_standalone": {
            "point": fc["point"],
            "band_80": [fc["lo80"], fc["hi80"]],
            "band_95": [fc["lo95"], fc["hi95"]],
            "ar1_alpha": fc["ar1_alpha"],
            "ar1_phi": fc["ar1_phi"],
            "method": "mom_composed_ar1_on_pcepi",
        },
        "cleveland_fed": {
            "point": clev_at_target,
            "label": clev_label,
        },
    }, indent=2, default=str))
    print(f"\nSaved: {out_path}")

    # ── Summary ─────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Latest PCEPI print  ({latest_month.date()}):  {latest_yoy:.4f}%")
    print(f"  Cleveland Fed PCE nowcast:                  {clev_at_target:.4f}%")
    print(f"  PCE standalone (MoM-composed AR(1)):        {fc['point']:.4f}%  "
            f"± {(fc['hi80'] - fc['point']):.4f} pp (80%)")


if __name__ == "__main__":
    main()
