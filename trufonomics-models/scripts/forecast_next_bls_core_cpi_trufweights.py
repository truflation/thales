"""Forecast the next BLS Core CPI YoY release — **Truflation-weighted core CBDF**.

Core CPI = CPILFESL = CPI All Items less Food and Energy.

Components: 9 BLS subindexes (the 11 used for headline CPI MINUS
  CUSR0000SAF1 (Food) and CUSR0000SAH2 (Fuels and utilities — the
  energy block for residential).

Weights: BLS official weights, renormalized to sum to 100% across the
9 retained components.

Caveat: CUSR0000SAT (Transportation) still includes gasoline within.
True BLS Core CPI excludes ALL energy including motor fuel. This
inclusion is a small leak (~3-4% of headline weight, less than ~5% of
core). The anchor offset absorbs the constant bias from this leak,
and the composition residual measures the impact on dynamics.

Run::

    uv run python scripts/forecast_next_bls_core_cpi_trufweights.py
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
WEIGHTS_CSV = ROOT / "data" / "truflation" / "weights" / "categories-tables-v2.csv"
OUT_DIR = ROOT / "results" / "next_release_forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 9 BLS subindexes constituting Core CPI (exclude Food cat 78 and Utilities cat 81)
BLS_CORE_COMPONENTS = {
    "CUSR0000SAH1":  [79],          # Shelter
    "CUSR0000SAT":   [80],          # Transportation (includes gasoline — small leak)
    "CUSR0000SAM":   [82],          # Medical care
    "CUSR0000SAH3":  [83],          # Household furnishings
    "CUSR0000SEFW":  [84],          # Alcoholic beverages
    "CUSR0000SAA":   [85],          # Apparel
    "CUSR0000SAE":   [86, 87],      # Education + Communication
    "CUSR0000SAR":   [88],          # Recreation
    "CUSR0000SAG":   [89],          # Other goods and services
}

CALIB_WINDOW = 24
N_SAMPLES = 500


def _yoy_from_levels(levels: pd.Series) -> pd.Series:
    out = {}
    for t in levels.index:
        denom = pd.Timestamp(year=t.year - 1, month=t.month, day=1) \
                    + pd.offsets.MonthEnd(0)
        if denom in levels.index:
            out[t] = (levels.loc[t] / levels.loc[denom] - 1.0) * 100.0
    return pd.Series(out).sort_index()


def load_bls_component_levels(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    frames = []
    for series_id in BLS_CORE_COMPONENTS:
        rows = con.execute(
            "SELECT reference_date, value FROM vintage "
            "WHERE series_id = ? AND source = 'bls_direct' "
            "AND (series_id, reference_date, as_of_date) IN ("
            "  SELECT series_id, reference_date, MAX(as_of_date) "
            "  FROM vintage WHERE series_id = ? AND source = 'bls_direct' "
            "  GROUP BY series_id, reference_date) "
            "ORDER BY reference_date",
            [series_id, series_id],
        ).fetchall()
        s = pd.Series([r[1] for r in rows],
                       index=pd.to_datetime([r[0] for r in rows]),
                       name=series_id)
        frames.append(s)
    return pd.concat(frames, axis=1).dropna()


def load_core_cpi_yoy() -> pd.Series:
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
    levels = pd.Series([r[1] for r in rows], index=idx)
    return _yoy_from_levels(levels)


def load_core_truflation_weights() -> dict[str, float]:
    """BLS weights renormalized to 100% across the 9 core components."""
    wt = pd.read_csv(WEIGHTS_CSV)
    top = wt[(wt["subcategory_id"] == 0) & (wt["source_id"] == 0)]
    cat_w = dict(zip(top["category_id"].astype(int),
                       top["relative_importance"].astype(float)))
    raw = {sid: sum(cat_w[c] for c in cat_ids)
              for sid, cat_ids in BLS_CORE_COMPONENTS.items()}
    total = sum(raw.values())
    return {k: v * 100.0 / total for k, v in raw.items()}


def compose_level(component_levels, weights_pct, base_date):
    base_levels = component_levels.loc[base_date]
    composite = pd.Series(0.0, index=component_levels.index)
    for col in component_levels.columns:
        composite += weights_pct[col] * (component_levels[col] / base_levels[col]) * 100.0
    composite /= 100.0
    return composite


def validate_composition(component_levels, weights_pct, target_yoy):
    base = component_levels.index.min()
    composed = compose_level(component_levels, weights_pct, base)
    composed_yoy = _yoy_from_levels(composed)
    common = composed_yoy.index.intersection(target_yoy.index)
    resid = (composed_yoy.loc[common] - target_yoy.loc[common]).dropna()
    return {
        "n":            len(resid),
        "mean_resid":   float(resid.mean()),
        "median_resid": float(resid.median()),
        "sd_resid":     float(resid.std()),
        "abs_max":      float(resid.abs().max()),
        "within_0.1pp": float((resid.abs() < 0.1).mean()),
        "within_0.3pp": float((resid.abs() < 0.3).mean()),
    }


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


def forecast_next_yoy(component_levels, weights_pct, target_yoy, origin,
                          n_samples=N_SAMPLES, seed=0):
    rng = np.random.default_rng(seed)
    base_date = component_levels.index.min()
    history = component_levels.loc[component_levels.index <= origin]
    base_levels = history.loc[base_date]

    next_levels: dict[str, float] = {}
    next_level_samples: dict[str, np.ndarray] = {}
    for col in history.columns:
        pt, samples = forecast_component_level_one_step(
            history[col], calib_window=CALIB_WINDOW,
            n_samples=n_samples, rng=rng)
        next_levels[col] = pt
        next_level_samples[col] = samples
    next_month = origin + pd.offsets.MonthEnd(1)

    composed_next = sum(
        weights_pct[col] * (next_levels[col] / base_levels[col]) * 100.0
        for col in history.columns
    ) / 100.0

    denom_date = next_month - pd.DateOffset(years=1)
    denom_target = pd.Timestamp(year=denom_date.year, month=denom_date.month,
                                       day=1) + pd.offsets.MonthEnd(0)
    if denom_target not in history.index:
        avail = history.index[history.index <= denom_target]
        denom_target = avail[-1]
    denom_levels = history.loc[denom_target]
    composed_denom = sum(
        weights_pct[col] * (denom_levels[col] / base_levels[col]) * 100.0
        for col in history.columns
    ) / 100.0
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
    composed_origin = sum(
        weights_pct[col] * (origin_levels[col] / base_levels[col]) * 100.0
        for col in history.columns
    ) / 100.0
    composed_origin_denom = sum(
        weights_pct[col] * (denom_origin_levels[col] / base_levels[col]) * 100.0
        for col in history.columns
    ) / 100.0
    composed_yoy_at_origin = (composed_origin / composed_origin_denom - 1.0) * 100.0

    if origin in target_yoy.index:
        actual_at_origin = float(target_yoy.loc[origin])
        anchor_offset = actual_at_origin - composed_yoy_at_origin
    else:
        actual_at_origin = float("nan")
        anchor_offset = 0.0
    yoy_point = composed_yoy_next + anchor_offset

    bands = {}
    if n_samples > 0:
        composed_samples = np.zeros(n_samples)
        for col in history.columns:
            composed_samples += (weights_pct[col]
                                  * (next_level_samples[col] / base_levels[col])
                                  * 100.0)
        composed_samples /= 100.0
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
        "origin":                 str(origin.date()),
        "next_month":             str(next_month.date()),
        "composed_yoy_at_origin": composed_yoy_at_origin,
        "actual_core_yoy_at_origin": actual_at_origin,
        "anchor_offset_pp":       anchor_offset,
        "composed_yoy_next_raw":  composed_yoy_next,
        "point":                  float(yoy_point),
        **bands,
    }


def main() -> None:
    print("=" * 78)
    print("Truflation-weighted Core CPI CBDF — Next Core CPI YoY release forecast")
    print("=" * 78)

    con = duckdb.connect(str(VINTAGE_DB), read_only=True)
    component_levels = load_bls_component_levels(con)
    con.close()
    core_yoy = load_core_cpi_yoy()
    weights_pct = load_core_truflation_weights()

    print(f"\n9-component BLS core panel: {component_levels.shape}, "
            f"{component_levels.index.min().date()} → {component_levels.index.max().date()}")
    print(f"\nTruflation weights, renormalized to {sum(weights_pct.values()):.3f}% across core:")
    for sid, w in weights_pct.items():
        print(f"  {sid:<16s} {w:>6.3f}%")

    val = validate_composition(component_levels, weights_pct, core_yoy)
    print(f"\nComposition validation (9-component Truflation-weighted vs CPILFESL YoY):")
    print(f"  n = {val['n']}, mean residual = {val['mean_resid']:+.4f} pp, "
            f"median = {val['median_resid']:+.4f} pp, sd = {val['sd_resid']:.4f}")
    print(f"  |residual| max = {val['abs_max']:.4f} pp, "
            f"within 0.1 pp = {val['within_0.1pp']*100:.1f}%, "
            f"within 0.3 pp = {val['within_0.3pp']*100:.1f}%")

    latest = core_yoy.index.max()
    latest_yoy = float(core_yoy.loc[latest])
    next_release = latest + pd.offsets.MonthEnd(1)
    print(f"\nLatest Core CPI print: {latest.date()}  YoY = {latest_yoy:.4f}%")
    print(f"Next release target:   {next_release.date()}")

    fc = forecast_next_yoy(component_levels, weights_pct, core_yoy, latest,
                                n_samples=N_SAMPLES,
                                seed=int(latest.value % 1_000_000))
    print(f"\n── Truflation-weighted Core CBDF forecast at origin {latest.date()} ──")
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

    out_path = OUT_DIR / (f"core_cpi_trufweights_{next_release.date()}"
                              f"_forecast_{date.today()}.json")
    out_path.write_text(json.dumps({
        "target":                "bls_headline_core_cpi_yoy",
        "method":                "truf_weighted_core_cbdf",
        "as_of_date":            str(date.today()),
        "latest_core_cpi_month": str(latest.date()),
        "latest_core_cpi_yoy":   latest_yoy,
        "next_release_month":    str(next_release.date()),
        "forward_next_release_forecast": fc,
        "composition_validation": val,
        "weights_used":          weights_pct,
    }, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
