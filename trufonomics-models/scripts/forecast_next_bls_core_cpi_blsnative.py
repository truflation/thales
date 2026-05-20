"""Forecast the next BLS Core CPI YoY release — **BLS-native core CBDF v2**.

Clean 2-component decomposition using BLS's own published aggregates
that exclude food and energy by construction:

  * **CUSR0000SACL1E** — Commodities less food and energy commodities ("Core goods")
  * **CUSR0000SASLE**  — Services less energy services ("Core services")

These two series, by BLS Laspeyres construction, sum (with appropriate
weights) to CPILFESL. We fit the weights empirically via OLS — same
approach as PCE-native CBDF — which yields per-component weights that
match BLS's published expenditure shares.

This v2 replaces the 9-component version that included SAT
(Transportation, which contained gasoline within and contaminated the
core composition with energy volatility). The 2-component approach
gives a much smaller composition residual (target < 0.15 pp SD; PCE-
native CBDF achieves 0.09 pp with the analogous 3-BEA-component setup).

Run::

    uv run python scripts/forecast_next_bls_core_cpi_blsnative.py
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

CORE_COMPONENTS = [
    "CUSR0000SACL1E",   # Core goods
    "CUSR0000SASLE",    # Core services
]
CALIB_WINDOW = 24
N_SAMPLES = 500


# ─── Data loading ────────────────────────────────────────────────────────


def _latest_per_ref(con, series_id, source):
    rows = con.execute(
        "SELECT reference_date, value FROM vintage "
        "WHERE series_id = ? AND source = ? "
        "AND (series_id, reference_date, as_of_date) IN ("
        "  SELECT series_id, reference_date, MAX(as_of_date) "
        "  FROM vintage WHERE series_id = ? AND source = ? "
        "  GROUP BY series_id, reference_date) "
        "ORDER BY reference_date",
        [series_id, source, series_id, source],
    ).fetchall()
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.Series([r[1] for r in rows], index=idx, name=series_id)


def load_core_component_levels() -> pd.DataFrame:
    with duckdb.connect(str(VINTAGE_DB), read_only=True) as con:
        frames = [_latest_per_ref(con, sid, "bls_direct")
                    for sid in CORE_COMPONENTS]
    return pd.concat(frames, axis=1).dropna()


def _yoy_from_levels(levels: pd.Series) -> pd.Series:
    out = {}
    for t in levels.index:
        denom = pd.Timestamp(year=t.year - 1, month=t.month, day=1) \
                    + pd.offsets.MonthEnd(0)
        if denom in levels.index:
            out[t] = (levels.loc[t] / levels.loc[denom] - 1.0) * 100.0
    return pd.Series(out).sort_index()


def load_core_cpi_yoy_and_levels() -> tuple[pd.Series, pd.Series]:
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
    levels = pd.Series([r[1] for r in rows], index=idx, name="cpilfesl")
    return levels, _yoy_from_levels(levels)


# ─── Empirical weight calibration (OLS) ──────────────────────────────────


def fit_weights_ols(component_levels: pd.DataFrame,
                       core_levels: pd.Series) -> dict:
    """OLS: CPILFESL[t] = α + w_g · SACL1E[t] + w_s · SASLE[t] + ε."""
    common = component_levels.index.intersection(core_levels.index)
    X = component_levels.loc[common].values
    y = core_levels.loc[common].values
    Xb = np.column_stack([np.ones_like(y), X])
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    alpha = float(coef[0])
    w = coef[1:].astype(float)
    pred = Xb @ coef
    resid = y - pred
    return {
        "alpha": alpha,
        "weights": dict(zip(component_levels.columns, w.tolist())),
        "sum_weights": float(w.sum()),
        "fit_rmse_level": float(np.sqrt((resid ** 2).mean())),
        "n_obs": len(common),
    }


def compose_level(component_levels, weights, alpha=0.0):
    out = pd.Series(alpha, index=component_levels.index)
    for col, w in weights.items():
        out = out + w * component_levels[col]
    return out


def validate_composition(component_levels, weights, alpha, core_yoy):
    composed = compose_level(component_levels, weights, alpha)
    composed_yoy = _yoy_from_levels(composed)
    common = composed_yoy.index.intersection(core_yoy.index)
    resid = (composed_yoy.loc[common] - core_yoy.loc[common]).dropna()
    return {
        "n":            len(resid),
        "mean_resid":   float(resid.mean()),
        "median_resid": float(resid.median()),
        "sd_resid":     float(resid.std()),
        "abs_max":      float(resid.abs().max()),
        "within_0.1pp": float((resid.abs() < 0.1).mean()),
        "within_0.3pp": float((resid.abs() < 0.3).mean()),
    }


# ─── Per-component AR(1) on log-MoM ──────────────────────────────────────


def forecast_component_level_one_step(level_history, calib_window=CALIB_WINDOW,
                                          n_samples=0, rng=None):
    log_levels = np.log(level_history.values)
    log_mom = np.diff(log_levels)
    last_level = float(level_history.iloc[-1])
    if len(log_mom) < calib_window + 1:
        empty = np.full(n_samples, last_level) if n_samples > 0 else np.array([])
        return last_level, empty
    calib = log_mom[-calib_window:]
    x, y = calib[:-1], calib[1:]
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, phi = float(coef[0]), float(coef[1])
    resid = y - (alpha + phi * x)
    point_log_mom = alpha + phi * float(log_mom[-1])
    next_level = last_level * np.exp(point_log_mom)
    if n_samples > 0 and rng is not None:
        eps = rng.choice(resid, size=n_samples)
        sample_levels = last_level * np.exp(point_log_mom + eps)
    else:
        sample_levels = np.array([])
    return next_level, sample_levels


# ─── Forecast next month's Core CPI YoY ──────────────────────────────────


def forecast_next_yoy(component_levels, weights, alpha_intercept,
                          core_yoy, origin,
                          n_samples=N_SAMPLES, seed=0):
    rng = np.random.default_rng(seed)
    history = component_levels.loc[component_levels.index <= origin]
    next_month = origin + pd.offsets.MonthEnd(1)

    next_levels: dict[str, float] = {}
    next_level_samples: dict[str, np.ndarray] = {}
    for col in history.columns:
        pt, samples = forecast_component_level_one_step(
            history[col], calib_window=CALIB_WINDOW,
            n_samples=n_samples, rng=rng)
        next_levels[col] = pt
        next_level_samples[col] = samples

    composed_next = alpha_intercept + sum(
        weights[col] * next_levels[col] for col in history.columns)

    denom_date = next_month - pd.DateOffset(years=1)
    denom_target = pd.Timestamp(year=denom_date.year, month=denom_date.month,
                                       day=1) + pd.offsets.MonthEnd(0)
    if denom_target not in history.index:
        avail = history.index[history.index <= denom_target]
        denom_target = avail[-1]
    denom_levels = history.loc[denom_target]
    composed_denom = alpha_intercept + sum(
        weights[col] * denom_levels[col] for col in history.columns)
    composed_yoy_next = (composed_next / composed_denom - 1.0) * 100.0

    origin_levels = history.loc[origin]
    denom_origin_date = origin - pd.DateOffset(years=1)
    denom_origin_target = pd.Timestamp(year=denom_origin_date.year,
                                            month=denom_origin_date.month,
                                            day=1) + pd.offsets.MonthEnd(0)
    if denom_origin_target not in history.index:
        avail = history.index[history.index <= denom_origin_target]
        denom_origin_target = avail[-1]
    denom_origin_levels = history.loc[denom_origin_target]
    composed_origin = alpha_intercept + sum(
        weights[col] * origin_levels[col] for col in history.columns)
    composed_origin_denom = alpha_intercept + sum(
        weights[col] * denom_origin_levels[col] for col in history.columns)
    composed_yoy_at_origin = (composed_origin / composed_origin_denom - 1.0) * 100.0

    if origin in core_yoy.index:
        actual_at_origin = float(core_yoy.loc[origin])
        anchor_offset = actual_at_origin - composed_yoy_at_origin
    else:
        actual_at_origin = float("nan")
        anchor_offset = 0.0
    yoy_point = composed_yoy_next + anchor_offset

    bands = {}
    if n_samples > 0:
        composed_samples = np.full(n_samples, alpha_intercept)
        for col in history.columns:
            composed_samples = composed_samples + (
                weights[col] * next_level_samples[col])
        yoy_samples = (composed_samples / composed_denom - 1.0) * 100.0
        yoy_samples = yoy_samples + anchor_offset
        bands = {
            "lo80": float(np.quantile(yoy_samples, 0.10)),
            "hi80": float(np.quantile(yoy_samples, 0.90)),
            "lo95": float(np.quantile(yoy_samples, 0.025)),
            "hi95": float(np.quantile(yoy_samples, 0.975)),
            "n_samples": int(n_samples),
        }

    return {
        "origin":                    str(origin.date()),
        "next_month":                str(next_month.date()),
        "composed_yoy_at_origin":    composed_yoy_at_origin,
        "actual_core_yoy_at_origin": actual_at_origin,
        "anchor_offset_pp":          anchor_offset,
        "composed_yoy_next_raw":     composed_yoy_next,
        "point":                     float(yoy_point),
        **bands,
    }


def main() -> None:
    print("=" * 78)
    print("BLS-native Core CPI CBDF v2 — clean 2-component (Goods + Services)")
    print("=" * 78)

    component_levels = load_core_component_levels()
    core_levels, core_yoy = load_core_cpi_yoy_and_levels()

    print(f"\n2-component panel: {component_levels.shape}, "
            f"{component_levels.index.min().date()} → {component_levels.index.max().date()}")
    print(f"CPILFESL panel:    {len(core_levels)} months")

    fit = fit_weights_ols(component_levels, core_levels)
    print(f"\nOLS weight calibration (n={fit['n_obs']}):")
    print(f"  α (intercept) = {fit['alpha']:+.4f}")
    for c, w in fit["weights"].items():
        print(f"  w[{c:<18s}] = {w:+.5f}")
    print(f"  Σ weights = {fit['sum_weights']:.5f}  (should be ≈ 1.0)")
    print(f"  Level RMSE = {fit['fit_rmse_level']:.4f}")

    val = validate_composition(component_levels, fit["weights"], fit["alpha"],
                                    core_yoy)
    print(f"\nComposition validation (composed YoY vs actual CPILFESL YoY):")
    print(f"  n = {val['n']}, mean residual = {val['mean_resid']:+.4f} pp, "
            f"median = {val['median_resid']:+.4f} pp, sd = {val['sd_resid']:.4f}")
    print(f"  |residual| max = {val['abs_max']:.4f} pp, "
            f"within 0.1 pp = {val['within_0.1pp']*100:.1f}%, "
            f"within 0.3 pp = {val['within_0.3pp']*100:.1f}%")

    # Forecast for next release
    latest = core_yoy.index.max()
    latest_yoy = float(core_yoy.loc[latest])
    next_release = latest + pd.offsets.MonthEnd(1)
    print(f"\nLatest Core CPI print: {latest.date()}  YoY = {latest_yoy:.4f}%")
    print(f"Next release target:   {next_release.date()}")

    fc = forecast_next_yoy(component_levels, fit["weights"], fit["alpha"],
                                core_yoy, latest,
                                n_samples=N_SAMPLES,
                                seed=int(latest.value % 1_000_000))
    print(f"\n── Core CPI BLS-native CBDF v2 forecast at origin {latest.date()} ──")
    print(f"  Composed YoY at origin (raw):     {fc['composed_yoy_at_origin']:.4f}%")
    print(f"  Actual Core CPI YoY at origin:    {fc['actual_core_yoy_at_origin']:.4f}%")
    print(f"  Anchor offset:                    {fc['anchor_offset_pp']:+.4f} pp")
    print(f"  Composed forecast YoY (raw):      {fc['composed_yoy_next_raw']:.4f}%")
    print(f"  Anchored point forecast:          {fc['point']:.4f}%")
    if "lo80" in fc:
        print(f"  80% band: [{fc['lo80']:.4f}, {fc['hi80']:.4f}]   "
                f"width {fc['hi80'] - fc['lo80']:.4f} pp")
        print(f"  95% band: [{fc['lo95']:.4f}, {fc['hi95']:.4f}]   "
                f"width {fc['hi95'] - fc['lo95']:.4f} pp")

    out_path = OUT_DIR / (f"core_cpi_blsnative_{next_release.date()}"
                              f"_forecast_{date.today()}.json")
    out_path.write_text(json.dumps({
        "target":                "bls_headline_core_cpi_yoy",
        "method":                "bls_native_core_cbdf_v2",
        "as_of_date":            str(date.today()),
        "latest_core_cpi_month": str(latest.date()),
        "latest_core_cpi_yoy":   latest_yoy,
        "next_release_month":    str(next_release.date()),
        "forward_next_release_forecast": fc,
        "composition_validation": val,
        "weight_calibration":    fit,
    }, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
