"""Forecast the next BEA PCE Headline YoY release — **Truflation-weighted PCE CBDF**.

Architecture:

  * 11 BLS subindex levels (the same components used by the BLS-native
    CBDF: Food, Shelter, Transport, Fuels & Utilities, Medical,
    Household Furnishings, Alcohol, Apparel, Edu+Comm, Recreation, Other).
  * **Truflation's curated PCE-equivalent weights** (the
    `pce_relative_importance` column in the v2 weights CSV).
    The 12 BLS-categorized PCE weights sum to 96.74% (the 3.26% gap is
    NPISH which doesn't fit the BLS CPI hierarchy); we renormalize to
    100% so the weighted composition is well-defined.
  * Cats 86 (Communications) + 87 (Education) are summed to match the
    BLS SAE series (same mapping as BLS-native CBDF).
  * Per-component MoM-AR(1) on log-MoM + M2 composition.
  * Anchor-correct to actual PCEPI YoY at origin.
  * Density via per-component AR(1) residual bootstrap (500 sample
    paths), composed through M2 → quantile bands.

Mirrors the BLS-native CBDF architecture but swaps the target (BLS
→ BEA PCEPI) and the weights (BLS → Truflation's PCE-equivalent).
Useful as a third PCE forecaster alongside PCE standalone and
PCE-native CBDF (3 BEA components, empirical OLS weights).

Run::

    uv run python scripts/forecast_next_bea_pce_trufweights.py
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

# Same 11 BLS subindex series as BLS-native CBDF
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


def load_pcepi_yoy(con: duckdb.DuckDBPyConnection) -> pd.Series:
    rows = con.execute(
        "SELECT reference_date, value FROM vintage "
        "WHERE series_id = 'PCEPI' AND source = 'fred_alfred_target' "
        "AND (series_id, reference_date, as_of_date) IN ("
        "  SELECT series_id, reference_date, MAX(as_of_date) "
        "  FROM vintage WHERE series_id = 'PCEPI' AND source = 'fred_alfred_target' "
        "  GROUP BY series_id, reference_date) "
        "ORDER BY reference_date",
    ).fetchall()
    idx = [pd.Timestamp(r[0]) + pd.offsets.MonthEnd(0) for r in rows]
    levels = pd.Series([r[1] for r in rows], index=idx)
    return _yoy_from_levels(levels)


def load_truflation_pce_weights() -> dict[str, float]:
    """Truflation's curated PCE-equivalent weights at the 12 BLS-CPI
    category level, mapped to the 11 BLS series and renormalized to 100%."""
    wt = pd.read_csv(WEIGHTS_CSV)
    top = wt[(wt["subcategory_id"] == 0) & (wt["source_id"] == 0)]
    cat_weight = dict(zip(top["category_id"].astype(int),
                            top["pce_relative_importance"].astype(float)))
    raw: dict[str, float] = {}
    for series_id, cat_ids in BLS_COMPONENTS.items():
        raw[series_id] = sum(cat_weight[c] for c in cat_ids)
    raw_total = sum(raw.values())
    # Renormalize to 100% — absorbs the NPISH gap proportionally
    return {k: v * 100.0 / raw_total for k, v in raw.items()}


# ─── Composition (M2) ────────────────────────────────────────────────────


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
                            pcepi_yoy: pd.Series) -> dict:
    base = component_levels.index.min()
    composed = compose_level(component_levels, weights_pct, base)
    composed_yoy = _yoy_from_levels(composed)
    common = composed_yoy.index.intersection(pcepi_yoy.index)
    resid = composed_yoy.loc[common] - pcepi_yoy.loc[common]
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


# ─── Per-component AR(1) on log-MoM, with density ────────────────────────


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


# ─── Forecast next month's PCE YoY ───────────────────────────────────────


def forecast_next_yoy(component_levels: pd.DataFrame,
                          weights_pct: dict[str, float],
                          pcepi_yoy: pd.Series,
                          origin: pd.Timestamp,
                          n_samples: int = N_SAMPLES,
                          seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    base_date = component_levels.index.min()
    history = component_levels.loc[component_levels.index <= origin]
    base_levels = history.loc[base_date]
    next_month = origin + pd.offsets.MonthEnd(1)

    # Fix 1 (2026-05-19): when observed BLS subindex levels exist for
    # next_month (typical: BLS releases ~mid-month, PCEPI releases
    # ~end-of-month — so the BLS panel runs one month ahead of PCEPI),
    # use them directly instead of projecting from origin via AR(1).
    # This converts the model into a Cleveland-Fed-style BLS→PCE bridge
    # nowcast whenever the BLS panel has fresher data.
    use_observed_next = (
        next_month in component_levels.index
        and not component_levels.loc[next_month].isna().any()
    )

    next_levels: dict[str, float] = {}
    next_level_samples: dict[str, np.ndarray] = {}
    for col in history.columns:
        if use_observed_next:
            obs_level = float(component_levels.loc[next_month, col])
            next_levels[col] = obs_level
            # Observed BLS level is known exactly — zero AR(1) sampling
            # noise from the component layer. Bridge uncertainty is added
            # below via the historical composition-residual SD.
            next_level_samples[col] = (np.full(n_samples, obs_level)
                                          if n_samples > 0 else np.array([]))
        else:
            pt, samples = forecast_component_level_one_step(
                history[col], calib_window=CALIB_WINDOW,
                n_samples=n_samples, rng=rng)
            next_levels[col] = pt
            next_level_samples[col] = samples

    # Composed forecast level (point)
    composed_next = sum(
        weights_pct[col] * (next_levels[col] / base_levels[col]) * 100.0
        for col in history.columns
    ) / 100.0

    # Composed denom level at (next_month - 1y)
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
        weights_pct[col] * (origin_levels[col] / base_levels[col]) * 100.0
        for col in history.columns
    ) / 100.0
    composed_origin_denom = sum(
        weights_pct[col] * (denom_origin_levels[col] / base_levels[col]) * 100.0
        for col in history.columns
    ) / 100.0
    composed_yoy_at_origin = (composed_origin / composed_origin_denom - 1.0) * 100.0

    if origin in pcepi_yoy.index:
        actual_at_origin = float(pcepi_yoy.loc[origin])
        anchor_offset = actual_at_origin - composed_yoy_at_origin
    else:
        actual_at_origin = float("nan")
        anchor_offset = 0.0

    yoy_point = composed_yoy_next + anchor_offset

    # Density:
    #   • Forecast mode (no observed BLS for next_month): per-component
    #     AR(1) residual bootstrap composed through M2 — original logic.
    #   • Nowcast mode (observed BLS available): per-component samples are
    #     constant (BLS known exactly), so the bootstrap would collapse to
    #     zero width. Use the historical BLS→PCE composition-residual SD
    #     (trailing 24 months ending at origin) as the bridge uncertainty,
    #     applied as Gaussian noise around the anchored point.
    bands: dict = {}
    if n_samples > 0:
        if use_observed_next:
            base = component_levels.index.min()
            composed_all = compose_level(component_levels, weights_pct, base)
            composed_yoy_all = _yoy_from_levels(composed_all)
            common = composed_yoy_all.index.intersection(pcepi_yoy.index)
            common = common[common <= origin]
            resid = (composed_yoy_all.loc[common]
                       - pcepi_yoy.loc[common]).dropna()
            trailing = resid.iloc[-24:] if len(resid) >= 24 else resid
            sigma_bridge = float(trailing.std()) if len(trailing) >= 4 else 0.0
            yoy_samples = yoy_point + rng.normal(0.0, sigma_bridge,
                                                      size=n_samples)
            bands = {
                "lo80": float(np.quantile(yoy_samples, 0.10)),
                "hi80": float(np.quantile(yoy_samples, 0.90)),
                "lo95": float(np.quantile(yoy_samples, 0.025)),
                "hi95": float(np.quantile(yoy_samples, 0.975)),
                "n_samples": int(n_samples),
                "band_method": "bridge_residual_sd",
                "sigma_bridge_pp": sigma_bridge,
            }
        else:
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
                "band_method": "ar1_residual_bootstrap",
            }

    return {
        "origin": str(origin.date()),
        "next_month": str(next_month.date()),
        "composed_yoy_at_origin": composed_yoy_at_origin,
        "actual_pcepi_yoy_at_origin": actual_at_origin,
        "anchor_offset_pp": anchor_offset,
        "composed_yoy_next_raw": composed_yoy_next,
        "point": float(yoy_point),
        **bands,
    }


# ─── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 78)
    print("Truflation-weighted PCE CBDF — Next BEA PCE Headline YoY forecast")
    print("=" * 78)

    con = duckdb.connect(str(VINTAGE_DB), read_only=True)
    component_levels = load_bls_component_levels(con)
    pcepi_yoy = load_pcepi_yoy(con)
    con.close()
    weights_pct = load_truflation_pce_weights()

    print(f"\nBLS component panel: {component_levels.shape}, "
            f"{component_levels.index.min().date()} → {component_levels.index.max().date()}")
    print(f"\nTruflation PCE-equivalent weights (renormalized to {sum(weights_pct.values()):.3f}%):")
    for sid, w in weights_pct.items():
        print(f"  {sid:<16s} {w:>6.3f}%")

    val = validate_composition(component_levels, weights_pct, pcepi_yoy)
    print(f"\nComposition validation (BLS-components × Truflation-PCE-weights vs PCEPI YoY):")
    print(f"  n = {val['n']}, mean residual = {val['mean_resid']:+.4f} pp, "
            f"median = {val['median_resid']:+.4f} pp, sd = {val['sd_resid']:.4f}")
    print(f"  |residual| max = {val['abs_max']:.4f} pp, "
            f"within 0.1 pp = {val['within_0.1pp']*100:.1f}%, "
            f"within 0.3 pp = {val['within_0.3pp']*100:.1f}%")

    # Latest available PCEPI month (target anchor)
    latest_pce = pcepi_yoy.index.max()
    latest_pce_yoy = float(pcepi_yoy.loc[latest_pce])
    next_release_month = latest_pce + pd.offsets.MonthEnd(1)
    print(f"\nLatest PCEPI print:   {latest_pce.date()}  YoY = {latest_pce_yoy:.4f}%")
    print(f"Next PCE release:     {next_release_month.date()}")

    # The forecaster uses BLS component history (which can extend past PCEPI's
    # release lag), but anchors to the actual PCEPI YoY at the chosen origin.
    # We forecast for `next_release_month` from origin `latest_pce`.
    origin = latest_pce
    fc = forecast_next_yoy(component_levels, weights_pct, pcepi_yoy, origin,
                                n_samples=N_SAMPLES,
                                seed=int(origin.value % 1_000_000))
    print(f"\n── Truflation-weighted PCE CBDF forecast at origin {origin.date()} ──")
    print(f"  Composed YoY at origin (raw):     {fc['composed_yoy_at_origin']:.4f}%")
    print(f"  Actual PCEPI YoY at origin:       {fc['actual_pcepi_yoy_at_origin']:.4f}%")
    print(f"  Anchor offset:                    {fc['anchor_offset_pp']:+.4f} pp")
    print(f"  Composed forecast YoY (raw):      {fc['composed_yoy_next_raw']:.4f}%")
    print(f"  Anchored point forecast:          {fc['point']:.4f}%")
    if "lo80" in fc:
        print(f"  80% band: [{fc['lo80']:.4f}, {fc['hi80']:.4f}]   "
                f"width {fc['hi80'] - fc['lo80']:.4f} pp")
        print(f"  95% band: [{fc['lo95']:.4f}, {fc['hi95']:.4f}]   "
                f"width {fc['hi95'] - fc['lo95']:.4f} pp")

    out_path = OUT_DIR / (
        f"pce_trufweights_{next_release_month.date()}_forecast_{date.today()}.json")
    out_path.write_text(json.dumps({
        "target": "bea_headline_pce_yoy",
        "method": "truflation_weighted_cbdf_on_bls_components",
        "as_of_date": str(date.today()),
        "latest_pcepi_month": str(latest_pce.date()),
        "latest_pcepi_yoy": latest_pce_yoy,
        "next_release_month": str(next_release_month.date()),
        "forecast": fc,
        "composition_validation": val,
        "weights_used": weights_pct,
    }, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
