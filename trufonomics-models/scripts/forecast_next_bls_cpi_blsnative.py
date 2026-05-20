"""Forecast the next BLS Headline CPI YoY release — **BLS-native CBDF**.

Architecture:
  * 11 BLS subindex levels (Food, Shelter, Transport, Fuels & Utilities,
    Medical, Household Furnishings, Alcohol, Apparel, Edu+Comm, Recreation,
    Other) — all from CUSR0000 series, ingested via BLS direct API.
  * BLS relative-importance weights from the v2 weights CSV
    (`bls_relative_importance`); cats 86 + 87 are summed to match the BLS
    SAE (Education + Communication) series.
  * Per-component MoM-AR(1) on log-MoM of each subindex level.
  * M2 composition: composite_level[T+1] = Σ_c w_c × (level_c[T+1] /
    level_c[base]) × 100.
  * YoY computed on the composite, anchored to actual BLS headline YoY at
    origin so the prediction lines up cleanly with the latest known print.

The hypothesis to test: does disaggregating BLS into 11 of its OWN
sub-components + applying BLS's OWN weights beat the headline-only
MoM-AR(1) baseline? The Bridged-CBDF earlier (Truflation components →
bridge → BLS) lost to MoM-AR(1) by ~7-12 percentage points of RMSE
reduction. Removing the cross-source noise + the composition residual
might close that gap.

Compares directly with Thales standalone (`forecast_next_bls_cpi.py`)
on the same target month — same architecture spirit, different
information set (components vs aggregate).

Run::

    uv run python scripts/forecast_next_bls_cpi_blsnative.py
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

# 11 BLS subindex series covering all 12 Truflation top-level categories.
# Cat 86 (Communications) + cat 87 (Education) share the BLS SAE series.
BLS_COMPONENTS = {
    "CUSR0000SAF1":  [78],          # Food
    "CUSR0000SAH1":  [79],          # Shelter (Housing)
    "CUSR0000SAT":   [80],          # Transportation
    "CUSR0000SAH2":  [81],          # Fuels and utilities
    "CUSR0000SAM":   [82],          # Medical care
    "CUSR0000SAH3":  [83],          # Household furnishings
    "CUSR0000SEFW":  [84],          # Alcoholic beverages
    "CUSR0000SAA":   [85],          # Apparel
    "CUSR0000SAE":   [86, 87],      # Education + Communication (BLS combines)
    "CUSR0000SAR":   [88],          # Recreation
    "CUSR0000SAG":   [89],          # Other goods and services
}

# 24-month rolling calibration window for the per-component AR(1)
CALIB_WINDOW = 24
N_SAMPLES = 500


# ─── Data loading ────────────────────────────────────────────────────────


def load_bls_component_levels(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load 11 BLS subindex level series. Wide DataFrame, columns =
    BLS series IDs, indexed at month-end."""
    frames = []
    for series_id in BLS_COMPONENTS:
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


def _yoy_from_levels(levels: pd.Series) -> pd.Series:
    """Compute YoY by date-based lookup (level[t] / level[t-1y]) - 1.
    Resilient to missing months in the series (some BLS revisions can
    leave gaps; positional shift(12) breaks in that case).
    """
    out = {}
    for t in levels.index:
        denom_target = pd.Timestamp(year=t.year - 1, month=t.month, day=1) \
                            + pd.offsets.MonthEnd(0)
        if denom_target in levels.index:
            out[t] = (levels.loc[t] / levels.loc[denom_target] - 1.0) * 100.0
    return pd.Series(out).sort_index()


def load_bls_headline_yoy(con: duckdb.DuckDBPyConnection) -> pd.Series:
    """BLS Headline CPI YoY series (computed from CUSR0000SA0 levels)."""
    rows = con.execute(
        "SELECT reference_date, value FROM vintage "
        "WHERE series_id = 'CUSR0000SA0' AND source = 'bls_direct' "
        "AND (series_id, reference_date, as_of_date) IN ("
        "  SELECT series_id, reference_date, MAX(as_of_date) "
        "  FROM vintage WHERE series_id = 'CUSR0000SA0' AND source = 'bls_direct' "
        "  GROUP BY series_id, reference_date) "
        "ORDER BY reference_date",
    ).fetchall()
    levels = pd.Series([r[1] for r in rows],
                          index=pd.to_datetime([r[0] for r in rows]))
    return _yoy_from_levels(levels)


def load_bls_weights() -> dict[str, float]:
    """Build {bls_series_id: weight_pct} dict. Sums cats 86+87 for SAE."""
    wt = pd.read_csv(WEIGHTS_CSV)
    top = wt[(wt["subcategory_id"] == 0) & (wt["source_id"] == 0)]
    cat_weight = dict(zip(top["category_id"].astype(int),
                            top["bls_relative_importance"].astype(float)))
    out: dict[str, float] = {}
    for series_id, cat_ids in BLS_COMPONENTS.items():
        out[series_id] = sum(cat_weight[c] for c in cat_ids)
    return out


# ─── Composition (M2) + diagnostics ──────────────────────────────────────


def compose_level(component_levels: pd.DataFrame,
                    weights_pct: dict[str, float],
                    base_date: pd.Timestamp) -> pd.Series:
    """M2 composition. composite[t] = (Σ_c w_c × (level_c[t]/level_c[base]) × 100) / 100."""
    base_levels = component_levels.loc[base_date]
    composite = pd.Series(0.0, index=component_levels.index)
    for col in component_levels.columns:
        composite += weights_pct[col] * (component_levels[col] / base_levels[col]) * 100.0
    composite /= 100.0
    return composite


def validate_composition(component_levels: pd.DataFrame,
                            weights_pct: dict[str, float],
                            bls_headline_yoy: pd.Series) -> dict:
    """Compose 11 BLS subindexes with BLS weights, compute YoY, compare to
    published BLS Headline CPI YoY. The 'accounting identity' check —
    residuals should be small (<0.5 pp) because BLS components × BLS
    weights = BLS aggregate by construction (up to rounding + the 12th
    aggregation gap).
    """
    base = component_levels.index.min()
    composed = compose_level(component_levels, weights_pct, base)
    composed_yoy = _yoy_from_levels(composed)
    common = composed_yoy.index.intersection(bls_headline_yoy.index)
    resid = composed_yoy.loc[common] - bls_headline_yoy.loc[common]
    resid = resid.dropna()
    return {
        "n": len(resid),
        "mean_resid": float(resid.mean()),
        "median_resid": float(resid.median()),
        "sd_resid": float(resid.std()),
        "abs_max": float(resid.abs().max()),
        "within_0.1pp": float((resid.abs() < 0.1).mean()),
        "within_0.3pp": float((resid.abs() < 0.3).mean()),
    }


# ─── Per-component AR(1) on log-MoM ──────────────────────────────────────


def forecast_component_level_one_step(level_history: pd.Series,
                                          calib_window: int = CALIB_WINDOW,
                                          n_samples: int = 0,
                                          rng: np.random.Generator | None = None,
                                          ) -> tuple[float, np.ndarray]:
    """One-step-ahead forecast of next month's component level via AR(1)
    on log-MoM. Returns (point_level, samples_array).

    samples_array has length n_samples (empty array if n_samples=0).
    Density via bootstrap of the AR(1) calibration residuals.
    """
    log_levels = np.log(level_history.values)
    log_mom = np.diff(log_levels)
    last_level = float(level_history.iloc[-1])
    if len(log_mom) < calib_window + 1:
        empty = np.full(n_samples, last_level) if n_samples > 0 else np.array([])
        return last_level, empty
    calib = log_mom[-calib_window:]
    x = calib[:-1]
    y = calib[1:]
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, phi = float(coef[0]), float(coef[1])
    resid = y - (alpha + phi * x)
    point_log_mom = alpha + phi * float(log_mom[-1])
    next_level = last_level * np.exp(point_log_mom)
    if n_samples > 0 and rng is not None:
        eps = rng.choice(resid, size=n_samples)
        sample_log_moms = point_log_mom + eps
        sample_levels = last_level * np.exp(sample_log_moms)
    else:
        sample_levels = np.array([])
    return next_level, sample_levels


# ─── Forecast one step ahead ─────────────────────────────────────────────


def forecast_next_yoy(component_levels: pd.DataFrame,
                          weights_pct: dict[str, float],
                          bls_headline_yoy: pd.Series,
                          origin: pd.Timestamp,
                          n_samples: int = N_SAMPLES,
                          seed: int = 0,
                          ) -> dict:
    """Forecast the YoY level for the month after `origin`, with density."""
    rng = np.random.default_rng(seed)
    base_date = component_levels.index.min()
    history = component_levels.loc[component_levels.index <= origin]
    base_levels = history.loc[base_date]
    base_weights = weights_pct

    # Per-component one-step level forecast + sample paths
    next_levels: dict[str, float] = {}
    next_level_samples: dict[str, np.ndarray] = {}
    for col in history.columns:
        pt, samples = forecast_component_level_one_step(
            history[col], calib_window=CALIB_WINDOW,
            n_samples=n_samples, rng=rng)
        next_levels[col] = pt
        next_level_samples[col] = samples
    next_month = origin + pd.offsets.MonthEnd(1)

    # Compose forecasted level
    composed_next = sum(
        base_weights[col] * (next_levels[col] / base_levels[col]) * 100.0
        for col in history.columns
    ) / 100.0

    # Composed level a year before next month (use actual history)
    denom_date = next_month - pd.DateOffset(years=1)
    denom_target_date = pd.Timestamp(year=denom_date.year,
                                          month=denom_date.month,
                                          day=1) + pd.offsets.MonthEnd(0)
    if denom_target_date not in history.index:
        # Find closest
        avail = history.index[history.index <= denom_target_date]
        denom_target_date = avail[-1]
    denom_levels = history.loc[denom_target_date]
    composed_denom = sum(
        base_weights[col] * (denom_levels[col] / base_levels[col]) * 100.0
        for col in history.columns
    ) / 100.0
    composed_yoy_next = (composed_next / composed_denom - 1.0) * 100.0

    # Composed YoY at origin (for anchor offset)
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
        base_weights[col] * (origin_levels[col] / base_levels[col]) * 100.0
        for col in history.columns
    ) / 100.0
    composed_origin_denom = sum(
        base_weights[col] * (denom_origin_levels[col] / base_levels[col]) * 100.0
        for col in history.columns
    ) / 100.0
    composed_yoy_at_origin = (composed_origin / composed_origin_denom - 1.0) * 100.0

    # Anchor offset
    if origin in bls_headline_yoy.index:
        actual_at_origin = float(bls_headline_yoy.loc[origin])
        anchor_offset = actual_at_origin - composed_yoy_at_origin
    else:
        actual_at_origin = float("nan")
        anchor_offset = 0.0

    yoy_point = composed_yoy_next + anchor_offset

    # Density: compose sample paths through M2 → YoY → bands
    bands: dict = {}
    if n_samples > 0:
        # composed_target_samples[s] = (Σ_c w_c × (level_c[s] / base_c) × 100) / 100
        composed_samples = np.zeros(n_samples)
        for col in history.columns:
            composed_samples += (base_weights[col]
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
        "origin": str(origin.date()),
        "next_month": str(next_month.date()),
        "composed_yoy_at_origin": composed_yoy_at_origin,
        "actual_bls_yoy_at_origin": actual_at_origin,
        "anchor_offset_pp": anchor_offset,
        "composed_yoy_next_raw": composed_yoy_next,
        "point": float(yoy_point),
        **bands,
    }


# ─── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 78)
    print("BLS-native CBDF — Next BLS Headline CPI YoY release forecast")
    print("=" * 78)

    con = duckdb.connect(str(VINTAGE_DB), read_only=True)
    component_levels = load_bls_component_levels(con)
    bls_yoy = load_bls_headline_yoy(con)
    con.close()
    weights_pct = load_bls_weights()

    print(f"\nBLS component panel: {component_levels.shape}, "
            f"{component_levels.index.min().date()} → {component_levels.index.max().date()}")
    print(f"BLS weights (sum {sum(weights_pct.values()):.3f}%):")
    for sid, w in weights_pct.items():
        print(f"  {sid:<16s} {w:>6.3f}%")

    # Composition validation
    print("\nComposition validation (BLS-components × BLS-weights vs actual BLS Headline YoY):")
    val = validate_composition(component_levels, weights_pct, bls_yoy)
    print(f"  n = {val['n']}, mean residual = {val['mean_resid']:+.4f} pp, "
            f"median = {val['median_resid']:+.4f} pp, sd = {val['sd_resid']:.4f}")
    print(f"  |residual| max = {val['abs_max']:.4f} pp, "
            f"within 0.1 pp = {val['within_0.1pp']*100:.1f}%, "
            f"within 0.3 pp = {val['within_0.3pp']*100:.1f}%")

    # Forecast for next month after the latest BLS print
    latest_bls_month = bls_yoy.index.max()
    latest_bls_yoy = float(bls_yoy.loc[latest_bls_month])
    print(f"\nLatest BLS print:  {latest_bls_month.date()}  YoY = {latest_bls_yoy:.4f}%")

    next_release_month = latest_bls_month + pd.offsets.MonthEnd(1)
    print(f"Next release target: {next_release_month.date()}")

    # March 2026 was the origin we'd have used to forecast the April 2026 print
    # We want the prediction we WOULD HAVE made before April released.
    # Origin for that prediction: 2026-03-31 (when only March data was available).
    march_origin = pd.Timestamp("2026-03-31")
    print(f"\n── Retrospective: BLS-native CBDF forecast at origin {march_origin.date()} ──")
    print(f"  (forecasting the April 2026 print that released 2026-05-12)")
    march_history = component_levels.loc[component_levels.index <= march_origin]
    march_bls_yoy = bls_yoy.loc[bls_yoy.index <= march_origin]
    fc_apr = forecast_next_yoy(march_history, weights_pct, march_bls_yoy,
                                    march_origin, n_samples=N_SAMPLES,
                                    seed=int(march_origin.value % 1_000_000))
    print(f"  Composed YoY at origin (raw):     {fc_apr['composed_yoy_at_origin']:.4f}%")
    print(f"  Actual BLS YoY at origin (March): {fc_apr['actual_bls_yoy_at_origin']:.4f}%")
    print(f"  Anchor offset:                    {fc_apr['anchor_offset_pp']:+.4f} pp")
    print(f"  Composed forecast YoY (raw):      {fc_apr['composed_yoy_next_raw']:.4f}%")
    print(f"  Anchored point forecast:          {fc_apr['point']:.4f}%")
    if "lo80" in fc_apr:
        print(f"  80% band: [{fc_apr['lo80']:.4f}, {fc_apr['hi80']:.4f}]   "
                f"width {fc_apr['hi80'] - fc_apr['lo80']:.4f} pp")
        print(f"  95% band: [{fc_apr['lo95']:.4f}, {fc_apr['hi95']:.4f}]   "
                f"width {fc_apr['hi95'] - fc_apr['lo95']:.4f} pp")

    # Compare to actual (now known)
    apr_target = pd.Timestamp("2026-04-30")
    if apr_target in bls_yoy.index:
        actual_apr = float(bls_yoy.loc[apr_target])
        err = fc_apr["point"] - actual_apr
        print(f"\n  Actual April BLS print: {actual_apr:.4f}%")
        print(f"  BLS-native CBDF error:  {err:+.4f} pp ({err*100:+.2f} bp)")
        # vs Thales standalone
        thales_standalone_apr = 3.855936274456584   # from May 7 forecast JSON
        thales_err = thales_standalone_apr - actual_apr
        print(f"\n  Thales standalone (headline-only AR(1)): {thales_standalone_apr:.4f}%  "
                f"err {thales_err:+.4f} pp ({thales_err*100:+.2f} bp)")

    # Now-forecast: from the latest available origin
    print(f"\n── Forward: BLS-native CBDF forecast at origin {latest_bls_month.date()} "
            f"(next month = {next_release_month.date()}) ──")
    fc_next = forecast_next_yoy(component_levels, weights_pct, bls_yoy,
                                     latest_bls_month, n_samples=N_SAMPLES,
                                     seed=int(latest_bls_month.value % 1_000_000))
    print(f"  Composed YoY at origin (raw):     {fc_next['composed_yoy_at_origin']:.4f}%")
    print(f"  Actual BLS YoY at origin:         {fc_next['actual_bls_yoy_at_origin']:.4f}%")
    print(f"  Anchor offset:                    {fc_next['anchor_offset_pp']:+.4f} pp")
    print(f"  Composed forecast YoY (raw):      {fc_next['composed_yoy_next_raw']:.4f}%")
    print(f"  Anchored point forecast:          {fc_next['point']:.4f}%")
    if "lo80" in fc_next:
        print(f"  80% band: [{fc_next['lo80']:.4f}, {fc_next['hi80']:.4f}]   "
                f"width {fc_next['hi80'] - fc_next['lo80']:.4f} pp")
        print(f"  95% band: [{fc_next['lo95']:.4f}, {fc_next['hi95']:.4f}]   "
                f"width {fc_next['hi95'] - fc_next['lo95']:.4f} pp")

    # Persist
    out_path = OUT_DIR / f"cpi_blsnative_{next_release_month.date()}_forecast_{date.today()}.json"
    out_path.write_text(json.dumps({
        "target": "bls_headline_cpi_yoy",
        "method": "bls_native_cbdf",
        "as_of_date": str(date.today()),
        "latest_bls_month": str(latest_bls_month.date()),
        "latest_bls_yoy": latest_bls_yoy,
        "next_release_month": str(next_release_month.date()),
        "retrospective_april_forecast": fc_apr,
        "forward_next_release_forecast": fc_next,
        "composition_validation": val,
        "weights_used": weights_pct,
    }, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
