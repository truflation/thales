"""Forecast the next BLS Headline CPI YoY release.

Two numbers with validated track records on CPI:

  1. **Thales standalone** — MoM-composed AR(1).
     +37.6 % RMSE reduction over Stock-Watson DFM (p = 0.0003, n = 25 OOS
     months). Methodology winner from the O'Keeffe head-to-head.

  2. **Cleveland-plus-Thales blend** — rolling-OLS combination
     `actual ~ α + β · Cleveland + γ · Thales`.
     +67.8 % RMSE reduction over Cleveland Fed alone (p = 0.04,
     n = 36 OOS months). The operational deployment claim.

Both formulas, windows, and significance tests are documented in
``results/baseline_eval/OKEEFE_HEADTOHEAD_FINDINGS.md``. This script
applies them to *today's* origin to produce a forecast for the next
BLS Headline CPI release (which lands roughly mid-month-following).

The script is generic across release cycles — at any point in time it
takes the latest released BLS CPI month, projects one month ahead, and
emits the prediction. Re-run on any date to refresh.

Run::

    uv run python scripts/forecast_next_bls_cpi.py
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

from thales import targets as T    # noqa: E402
from thales.models.baselines import AR1Baseline    # noqa: E402
from thales.models.mom_composed import MoMComposedForecaster    # noqa: E402
from thales.vintage import VintageStore    # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "next_release_forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Calibration window for the Cleveland+Thales blend — matches the
# head-to-head spec (36 trailing months of completed predictions).
CALIB_WINDOW = 36

# History depth for assembling the blend training data — needs to span
# at least CALIB_WINDOW + Truflation overlap. 60 covers comfortably.
HISTORY_DEPTH = 60


# ─── Panel + forecaster ──────────────────────────────────────────────────


def build_panel() -> pd.DataFrame:
    """Load BLS Headline CPI panel from the vintage store.

    Columns: ``y`` (YoY %), ``level`` (CPI index), ``clevfed`` (Cleveland
    Fed CPI nowcast YoY at each month), ``bls_mom`` (log MoM, in pp).
    Indexed at month-end, dropna on y/level.
    """
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        panel = T.load_panel(store, "cpi", as_of=date.today())
    panel = panel.dropna(subset=["y", "level"]).copy()
    panel["bls_mom"] = (
        np.log(panel["level"]) - np.log(panel["level"].shift(1))) * 100.0
    return panel


def build_thales_forecaster() -> MoMComposedForecaster:
    """The validated MoM-composed AR(1) forecaster — production spec."""
    inner = AR1Baseline(
        target_col="bls_mom",
        horizon=1,
        train_min=24,
        calib_months=24,
        band_method="rolling_conformal",
        n_samples=500,
        model_id="ar1_mom_inner",
    )
    return MoMComposedForecaster(
        inner=inner,
        bls_level_col="level",
        bls_yoy_col="y",
        mom_col="bls_mom",
        log_mom=True,
        horizon=1,
        n_samples=500,
        model_id="mom_composed_ar1",
    )


# ─── Historical Thales predictions for blend fitting ─────────────────────


def historical_thales_predictions(
    panel: pd.DataFrame,
    forecaster: MoMComposedForecaster,
    n_history: int = HISTORY_DEPTH,
) -> pd.DataFrame:
    """Walk the Thales forecaster across the last ``n_history`` origins.

    For each (origin, target) pair where target has a realised BLS
    value, compute the Thales prediction at that origin. Used to fit
    the rolling-OLS Cleveland-plus-Thales blend.

    Returns a frame indexed by ``target`` with columns
    ``thales_pred``, ``actual``, ``clevfed``.
    """
    origins = panel.index[-n_history - 1: -1]
    targets = panel.index[-n_history:]
    rows = []
    for origin, target in zip(origins, targets):
        try:
            f = forecaster.fit_predict(panel, origin, target)
        except (ValueError, KeyError):
            continue
        actual = float(panel.loc[target, "y"])
        clev = (float(panel.loc[target, "clevfed"])
                if "clevfed" in panel.columns
                and pd.notna(panel.loc[target, "clevfed"])
                else float("nan"))
        rows.append({
            "origin": origin,
            "target": target,
            "thales_pred": float(f.point),
            "actual": actual,
            "clevfed": clev,
        })
    return pd.DataFrame(rows).set_index("target")


def fit_blend(history: pd.DataFrame,
                calib_window: int = CALIB_WINDOW
                ) -> tuple[float, float, float, int]:
    """Fit the rolling-OLS blend on the trailing ``calib_window`` months.

    Returns (α, β_cleveland, γ_thales, n_used).
    """
    train = history.dropna(
        subset=["actual", "clevfed", "thales_pred"]).iloc[-calib_window:]
    if len(train) < 12:
        raise ValueError(
            f"need ≥12 valid blend rows, got {len(train)}")
    X = np.column_stack([
        np.ones(len(train)),
        train["clevfed"].values,
        train["thales_pred"].values,
    ])
    y = train["actual"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2]), len(train)


# ─── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 78)
    print("Next BLS Headline CPI YoY release — forecast")
    print("=" * 78)

    panel = build_panel()
    latest_bls_month = panel.index.max()
    next_release_month = (latest_bls_month + pd.offsets.MonthEnd(1)
                                ).normalize()

    print(f"\nLatest released BLS month: {latest_bls_month.date()}  "
            f"YoY = {float(panel.loc[latest_bls_month, 'y']):.4f}%")
    print(f"Next release target:       {next_release_month.date()}  "
            f"(forecasting this)")

    # ── Thales standalone ─────────────────────────────────────────
    fc = build_thales_forecaster()
    f = fc.fit_predict(panel, latest_bls_month, next_release_month)
    width80 = f.hi80 - f.lo80
    width95 = f.hi95 - f.lo95
    print()
    print("── Thales standalone (MoM-composed AR(1)) ──")
    print(f"  Track record: +37.6 % RMSE vs Stock-Watson DFM (p=0.0003, n=25)")
    print(f"  Point:    {f.point:.4f}%")
    print(f"  80% band: [{f.lo80:.4f}, {f.hi80:.4f}]   width {width80:.4f} pp")
    print(f"  95% band: [{f.lo95:.4f}, {f.hi95:.4f}]   width {width95:.4f} pp")

    # ── Cleveland Fed nowcast ─────────────────────────────────────
    # The shared panel is anchored to BLS-released months, so it won't
    # contain a Cleveland row for `next_release_month` until BLS prints.
    # Query the vintage store directly for the freshest Cleveland nowcast
    # of next_release_month so we don't fall back to the prior month.
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        clev_fresh = store.get_vintage(
            "clevfed_cpi_yoy", as_of_date=date.today())
    clev_fresh = clev_fresh.dropna()
    if next_release_month in clev_fresh.index:
        clev_at_target = float(clev_fresh.loc[next_release_month])
        clev_label = (f"for {next_release_month.date()} "
                        f"(latest as_of {date.today()})")
    else:
        clev_series = panel["clevfed"].dropna()
        clev_at_target = float(clev_series.iloc[-1])
        clev_label = (f"latest available ({clev_series.index[-1].date()}) — "
                          f"may not yet cover {next_release_month.date()}")
    print()
    print(f"── Cleveland Fed nowcast ({clev_label}) ──")
    print(f"  Point: {clev_at_target:.4f}%")

    # ── Cleveland+Thales blend ─────────────────────────────────────
    print()
    print(f"Building historical Thales predictions for blend fit "
            f"(walk-forward over last {HISTORY_DEPTH} months)…")
    history = historical_thales_predictions(panel, fc, n_history=HISTORY_DEPTH)
    history_with_clev = history.dropna(subset=["clevfed"])
    print(f"  History assembled: {len(history)} thales-prediction rows; "
            f"{len(history_with_clev)} with Cleveland coverage")

    blend_point: float | None = None
    alpha = beta = gamma = None
    n_blend = 0
    if len(history_with_clev) >= 12:
        alpha, beta, gamma, n_blend = fit_blend(
            history_with_clev, calib_window=CALIB_WINDOW)
        blend_point = alpha + beta * clev_at_target + gamma * f.point
        print()
        print("── Cleveland + Thales blend ──")
        print(f"  Track record: +67.8 % RMSE vs Cleveland alone (p=0.04, n=36)")
        print(f"  Formula:    α + β · Cleveland + γ · Thales")
        print(f"  Fitted on:  trailing {n_blend} months "
                f"(target {CALIB_WINDOW})")
        print(f"  Coefficients: α={alpha:+.4f}  "
                f"β={beta:+.4f}  γ={gamma:+.4f}")
        print(f"  Inputs:")
        print(f"    Cleveland:  {clev_at_target:.4f}%  × β  = "
                f"{beta * clev_at_target:+.4f}")
        print(f"    Thales:     {f.point:.4f}%  × γ  = "
                f"{gamma * f.point:+.4f}")
        print(f"    Intercept:  α                       = {alpha:+.4f}")
        print(f"  Blend point: {blend_point:.4f}%")
    else:
        print()
        print(f"  ⚠ insufficient history ({len(history_with_clev)} rows) "
                f"to fit blend; skipping")

    # ── Persist ─────────────────────────────────────────────────────
    out_path = (OUT_DIR /
                  f"cpi_{next_release_month.date()}_forecast_"
                  f"{date.today()}.json")
    payload = {
        "target": "bls_headline_cpi_yoy",
        "as_of_date": str(date.today()),
        "latest_bls_month": str(latest_bls_month.date()),
        "latest_bls_yoy": float(panel.loc[latest_bls_month, "y"]),
        "next_release_month": str(next_release_month.date()),
        "thales_standalone": {
            "point": float(f.point),
            "band_80": [float(f.lo80), float(f.hi80)],
            "band_95": [float(f.lo95), float(f.hi95)],
            "model": "mom_composed_ar1",
            "track_record": (
                "+37.6% RMSE over Stock-Watson DFM (p=0.0003, n=25)"),
        },
        "cleveland_fed": {
            "point": clev_at_target,
            "label": clev_label,
        },
        "blend": ({
            "point": blend_point,
            "alpha": alpha,
            "beta_cleveland": beta,
            "gamma_thales": gamma,
            "n_calibration": n_blend,
            "track_record": (
                "+67.8% RMSE over Cleveland alone (p=0.04, n=36)"),
        } if blend_point is not None else None),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved: {out_path}")

    # ── One-line summary ─────────────────────────────────────────────
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Latest BLS print  ({latest_bls_month.date()}):  "
            f"{float(panel.loc[latest_bls_month, 'y']):.2f}%")
    print(f"  Cleveland Fed nowcast:                  "
            f"{clev_at_target:.2f}%")
    print(f"  Thales standalone (validated):          "
            f"{f.point:.2f}%  ± {(f.hi80 - f.point):.2f} pp (80%)")
    if blend_point is not None:
        print(f"  Cleveland + Thales blend (validated):   "
                f"{blend_point:.2f}%   ← +67.8% RMSE vs Cleveland alone")


if __name__ == "__main__":
    main()
