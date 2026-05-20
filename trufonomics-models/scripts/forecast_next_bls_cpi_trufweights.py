"""Forecast the next BLS Headline CPI YoY release — **Truflation-weighted CPI CBDF**.

Architecture mirrors the BLS-native CBDF (`forecast_next_bls_cpi_blsnative.py`)
but swaps the weights:

  * 11 BLS subindex levels (same as BLS-native)
  * **Truflation's `relative_importance` weights** (the v2 CSV column —
    Truflation's own CPI weight curation, distinct from BLS's official
    weights). Cats 86 (Communications) + 87 (Education) summed for the
    BLS SAE series.
  * Per-component MoM-AR(1) + M2 Laspeyres composition + anchor offset.
  * Density via per-component AR(1) residual bootstrap.

The hypothesis: does Truflation's CPI weight curation carry useful
information beyond BLS's published weights on the same components? If
the composition residual is similar to BLS-native (~0.1 pp SD) the
weights are doing similar work; if it's much worse, Truflation's
weighting structure diverges meaningfully from BLS's.

Symmetric experiment to `forecast_next_bea_pce_trufweights.py` on the
PCE side (which tests BLS components + Truflation's PCE weights → BEA
PCE target).

Run::

    uv run python scripts/forecast_next_bls_cpi_trufweights.py
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

# Same 11 BLS subindexes as BLS-native CBDF
BLS_COMPONENTS = {
    "CUSR0000SAF1":  [78],          # Food
    "CUSR0000SAH1":  [79],          # Shelter
    "CUSR0000SAT":   [80],          # Transportation
    "CUSR0000SAH2":  [81],          # Fuels and utilities
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


# ─── Data loading ────────────────────────────────────────────────────────


def load_bls_component_levels(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
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
    out = {}
    for t in levels.index:
        denom = pd.Timestamp(year=t.year - 1, month=t.month, day=1) \
                    + pd.offsets.MonthEnd(0)
        if denom in levels.index:
            out[t] = (levels.loc[t] / levels.loc[denom] - 1.0) * 100.0
    return pd.Series(out).sort_index()


def load_bls_headline_yoy(con: duckdb.DuckDBPyConnection) -> pd.Series:
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


def load_truflation_cpi_weights() -> dict[str, float]:
    """Truflation's own CPI weights (the `relative_importance` column).
    Maps cat 86+87 → SAE; should sum to 100% already (Phase 1 validated
    this on the Truflation side)."""
    wt = pd.read_csv(WEIGHTS_CSV)
    top = wt[(wt["subcategory_id"] == 0) & (wt["source_id"] == 0)]
    cat_weight = dict(zip(top["category_id"].astype(int),
                            top["relative_importance"].astype(float)))
    out: dict[str, float] = {}
    for series_id, cat_ids in BLS_COMPONENTS.items():
        out[series_id] = sum(cat_weight[c] for c in cat_ids)
    total = sum(out.values())
    # Renormalize if not exactly 100% (typically 99.999-100.001 due to rounding)
    return {k: v * 100.0 / total for k, v in out.items()}


# ─── Composition (M2 Laspeyres) ──────────────────────────────────────────


def compose_level(component_levels: pd.DataFrame,
                    weights_pct: dict[str, float],
                    base_date: pd.Timestamp) -> pd.Series:
    base_levels = component_levels.loc[base_date]
    composite = pd.Series(0.0, index=component_levels.index)
    for col in component_levels.columns:
        composite += weights_pct[col] * (component_levels[col] / base_levels[col]) * 100.0
    composite /= 100.0
    return composite


def validate_composition(component_levels: pd.DataFrame,
                            weights_pct: dict[str, float],
                            bls_headline_yoy: pd.Series) -> dict:
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


# ─── Per-component AR(1) with density ────────────────────────────────────


def forecast_component_level_one_step(level_history: pd.Series,
                                          calib_window: int = CALIB_WINDOW,
                                          n_samples: int = 0,
                                          rng: np.random.Generator | None = None,
                                          ) -> tuple[float, np.ndarray]:
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
        sample_levels = last_level * np.exp(point_log_mom + eps)
    else:
        sample_levels = np.array([])
    return next_level, sample_levels


# ─── Forecast next month's BLS CPI YoY ───────────────────────────────────


def forecast_next_yoy(component_levels: pd.DataFrame,
                          weights_pct: dict[str, float],
                          bls_headline_yoy: pd.Series,
                          origin: pd.Timestamp,
                          n_samples: int = N_SAMPLES,
                          seed: int = 0) -> dict:
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

    if origin in bls_headline_yoy.index:
        actual_at_origin = float(bls_headline_yoy.loc[origin])
        anchor_offset = actual_at_origin - composed_yoy_at_origin
    else:
        actual_at_origin = float("nan")
        anchor_offset = 0.0

    yoy_point = composed_yoy_next + anchor_offset

    bands: dict = {}
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
    print("Truflation-weighted CPI CBDF — Next BLS Headline CPI YoY forecast")
    print("=" * 78)

    con = duckdb.connect(str(VINTAGE_DB), read_only=True)
    component_levels = load_bls_component_levels(con)
    bls_yoy = load_bls_headline_yoy(con)
    con.close()
    weights_pct = load_truflation_cpi_weights()

    print(f"\nBLS component panel: {component_levels.shape}, "
            f"{component_levels.index.min().date()} → "
            f"{component_levels.index.max().date()}")
    print(f"\nTruflation CPI weights (renormalized to {sum(weights_pct.values()):.3f}%):")
    for sid, w in weights_pct.items():
        print(f"  {sid:<16s} {w:>6.3f}%")

    val = validate_composition(component_levels, weights_pct, bls_yoy)
    print(f"\nComposition validation (BLS-components × Truflation-CPI-weights vs BLS Headline YoY):")
    print(f"  n = {val['n']}, mean residual = {val['mean_resid']:+.4f} pp, "
            f"median = {val['median_resid']:+.4f} pp, sd = {val['sd_resid']:.4f}")
    print(f"  |residual| max = {val['abs_max']:.4f} pp, "
            f"within 0.1 pp = {val['within_0.1pp']*100:.1f}%, "
            f"within 0.3 pp = {val['within_0.3pp']*100:.1f}%")

    # Retrospective: April 2026 forecast at March 2026 origin
    march_origin = pd.Timestamp("2026-03-31")
    print(f"\n── Retrospective at origin {march_origin.date()} "
            f"(forecasting April 2026 print that released 2026-05-12) ──")
    march_history = component_levels.loc[component_levels.index <= march_origin]
    march_bls = bls_yoy.loc[bls_yoy.index <= march_origin]
    fc_apr = forecast_next_yoy(march_history, weights_pct, march_bls,
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

    apr_target = pd.Timestamp("2026-04-30")
    if apr_target in bls_yoy.index:
        actual_apr = float(bls_yoy.loc[apr_target])
        err = fc_apr["point"] - actual_apr
        print(f"\n  Actual April BLS print:                  {actual_apr:.4f}%")
        print(f"  Truflation-weighted CPI CBDF error:      {err:+.4f} pp ({err*100:+.2f} bp)")
        # Comparators
        print(f"\n  Comparison to other CPI forecasters on this print:")
        print(f"    BLS-native CBDF:                       3.7359%  (-4.33 bp)")
        print(f"    Thales standalone:                     3.8559%  (+7.67 bp)")
        print(f"    Cleveland Fed:                         3.5600%  (-21.92 bp)")
        print(f"    Cleveland + Thales blend:              3.6300%  (-14.92 bp)")
        print(f"    Truflation-weighted CPI CBDF (this):   {fc_apr['point']:.4f}%  "
                f"({err*100:+.2f} bp)")
        avg = (3.8559 + 3.7359) / 2.0
        print(f"    Simple avg(standalone, BLS-native):    {avg:.4f}%  "
                f"({(avg - actual_apr)*100:+.2f} bp)")

    # Forward: from latest available origin
    latest_bls_month = bls_yoy.index.max()
    latest_yoy = float(bls_yoy.loc[latest_bls_month])
    next_release_month = latest_bls_month + pd.offsets.MonthEnd(1)
    print(f"\n── Forward forecast at origin {latest_bls_month.date()} "
            f"(next release = {next_release_month.date()}) ──")
    fc_next = forecast_next_yoy(component_levels, weights_pct, bls_yoy,
                                     latest_bls_month, n_samples=N_SAMPLES,
                                     seed=int(latest_bls_month.value % 1_000_000))
    print(f"  Composed YoY at origin (raw):     {fc_next['composed_yoy_at_origin']:.4f}%")
    print(f"  Actual BLS YoY at origin:         {fc_next['actual_bls_yoy_at_origin']:.4f}%")
    print(f"  Anchor offset:                    {fc_next['anchor_offset_pp']:+.4f} pp")
    print(f"  Anchored point forecast:          {fc_next['point']:.4f}%")
    if "lo80" in fc_next:
        print(f"  80% band: [{fc_next['lo80']:.4f}, {fc_next['hi80']:.4f}]")
        print(f"  95% band: [{fc_next['lo95']:.4f}, {fc_next['hi95']:.4f}]")

    out_path = OUT_DIR / (
        f"cpi_trufweights_{next_release_month.date()}_forecast_{date.today()}.json")
    out_path.write_text(json.dumps({
        "target": "bls_headline_cpi_yoy",
        "method": "truflation_weighted_cbdf_on_bls_components",
        "as_of_date": str(date.today()),
        "latest_bls_month": str(latest_bls_month.date()),
        "latest_bls_yoy": latest_yoy,
        "next_release_month": str(next_release_month.date()),
        "retrospective_april_forecast": fc_apr,
        "forward_next_release_forecast": fc_next,
        "composition_validation": val,
        "weights_used": weights_pct,
    }, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
