"""Forecast the next BEA PCE Headline YoY release — **PCE-native CBDF**.

Architecture mirror of `forecast_next_bls_cpi_blsnative.py`:

  * 3 BEA PCE chain-type sub-component price indexes:
      - DDURRG3M086SBEA   Durable Goods
      - DNDGRG3M086SBEA   Nondurable Goods
      - DSERRG3M086SBEA   Services
  * Weights estimated empirically via OLS over the full sample
    (`PCEPI[t] ≈ w_D · D[t] + w_N · N[t] + w_S · S[t]`).
    PCE is a Fisher chain-type aggregate so a linear weighted sum is
    not the exact accounting identity (unlike BLS CPI Laspeyres) — but
    the OLS-fitted weights minimize the residual and give a usable
    composition. Composition residual ~0.1-0.3 pp expected.
  * Per-component MoM-AR(1) on log-MoM of each sub-component level.
  * M2 composition + anchor-offset to actual PCEPI YoY at origin.

Run::

    uv run python scripts/forecast_next_bea_pce_native.py
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

PCE_COMPONENTS = [
    "DDURRG3M086SBEA",   # Durable Goods
    "DNDGRG3M086SBEA",   # Nondurable Goods
    "DSERRG3M086SBEA",   # Services
]

CALIB_WINDOW = 24
N_SAMPLES = 500


# ─── Data loading ────────────────────────────────────────────────────────


def _latest_per_ref_date(con, series_id, source):
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
    # FRED PCE series are dated by month-start; bump to month-end for consistency
    idx = [pd.Timestamp(r[0]) + pd.offsets.MonthEnd(0) for r in rows]
    return pd.Series([r[1] for r in rows], index=idx, name=series_id)


def load_pce_component_levels() -> pd.DataFrame:
    with duckdb.connect(str(VINTAGE_DB), read_only=True) as con:
        frames = [_latest_per_ref_date(con, sid, "fred_alfred")
                    for sid in PCE_COMPONENTS]
    return pd.concat(frames, axis=1).dropna()


def load_pcepi_levels() -> pd.Series:
    with duckdb.connect(str(VINTAGE_DB), read_only=True) as con:
        s = _latest_per_ref_date(con, "PCEPI", "fred_alfred_target")
    s.name = "pcepi"
    return s


def _yoy_from_levels(levels: pd.Series) -> pd.Series:
    out = {}
    for t in levels.index:
        denom = pd.Timestamp(year=t.year - 1, month=t.month, day=1) \
                    + pd.offsets.MonthEnd(0)
        if denom in levels.index:
            out[t] = (levels.loc[t] / levels.loc[denom] - 1.0) * 100.0
    return pd.Series(out).sort_index()


# ─── Empirical weight calibration ────────────────────────────────────────


def fit_weights_ols(component_levels: pd.DataFrame,
                       pcepi_levels: pd.Series) -> dict:
    """OLS: PCEPI[t] = α + Σ w_c · level_c[t] + ε.  Returns weights dict
    and diagnostics."""
    common = component_levels.index.intersection(pcepi_levels.index)
    X = component_levels.loc[common].values
    y = pcepi_levels.loc[common].values
    # OLS with intercept
    Xb = np.column_stack([np.ones_like(y), X])
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    alpha = float(coef[0])
    w_raw = coef[1:].astype(float)
    # Normalize so weights sum to 1 (renormalize α absorbing the residual)
    pred = Xb @ coef
    resid = y - pred
    return {
        "alpha": alpha,
        "weights": dict(zip(component_levels.columns, w_raw.tolist())),
        "sum_weights": float(w_raw.sum()),
        "fit_rmse_level": float(np.sqrt((resid ** 2).mean())),
        "n_obs": len(common),
    }


def compose_level(component_levels: pd.DataFrame,
                    weights: dict[str, float],
                    alpha: float = 0.0) -> pd.Series:
    """Linear composition. composite[t] = α + Σ w_c · level_c[t]."""
    composite = pd.Series(alpha, index=component_levels.index)
    for col, w in weights.items():
        composite = composite + w * component_levels[col]
    return composite


# ─── Composition validation ──────────────────────────────────────────────


def validate_composition(component_levels: pd.DataFrame,
                            weights: dict[str, float],
                            alpha: float,
                            pcepi_yoy: pd.Series) -> dict:
    composed = compose_level(component_levels, weights, alpha)
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


# ─── Per-component AR(1) on log-MoM ──────────────────────────────────────


def forecast_component_level_one_step(level_history: pd.Series,
                                          calib_window: int = CALIB_WINDOW,
                                          n_samples: int = 0,
                                          rng: np.random.Generator | None = None,
                                          ) -> tuple[float, np.ndarray]:
    """Returns (point_level, samples). Density via bootstrap of AR(1) residuals."""
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
                          weights: dict[str, float],
                          alpha_intercept: float,
                          pcepi_yoy: pd.Series,
                          origin: pd.Timestamp,
                          n_samples: int = N_SAMPLES,
                          seed: int = 0) -> dict:
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

    denom_next = next_month - pd.DateOffset(years=1)
    denom_next_target = pd.Timestamp(year=denom_next.year,
                                            month=denom_next.month,
                                            day=1) + pd.offsets.MonthEnd(0)
    if denom_next_target not in history.index:
        avail = history.index[history.index <= denom_next_target]
        denom_next_target = avail[-1]
    composed_denom = alpha_intercept + sum(
        weights[col] * history.loc[denom_next_target, col]
        for col in history.columns)
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

    if origin in pcepi_yoy.index:
        actual_at_origin = float(pcepi_yoy.loc[origin])
        anchor_offset = actual_at_origin - composed_yoy_at_origin
    else:
        actual_at_origin = float("nan")
        anchor_offset = 0.0

    yoy_point = composed_yoy_next + anchor_offset

    # Density: compose sample paths through linear composition → YoY → bands
    bands: dict = {}
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
    print("PCE-native CBDF — Next BEA PCE Headline YoY release forecast")
    print("=" * 78)

    component_levels = load_pce_component_levels()
    pcepi = load_pcepi_levels()
    pcepi_yoy = _yoy_from_levels(pcepi)

    print(f"\nPCE component panel: {component_levels.shape}, "
            f"{component_levels.index.min().date()} → {component_levels.index.max().date()}")
    print(f"PCEPI history: {len(pcepi)} months, latest = {pcepi.index.max().date()} "
            f"@ level {pcepi.iloc[-1]:.4f}")

    # Fit empirical weights
    fit = fit_weights_ols(component_levels, pcepi)
    print(f"\nWeight calibration (OLS, n={fit['n_obs']}):")
    print(f"  α (intercept) = {fit['alpha']:+.4f}")
    for c, w in fit["weights"].items():
        print(f"  w[{c:<18s}] = {w:+.5f}")
    print(f"  Σ weights = {fit['sum_weights']:.5f}  (should be ≈ 1.000)")
    print(f"  Level RMSE = {fit['fit_rmse_level']:.4f}")

    # Composition validation
    val = validate_composition(component_levels, fit["weights"], fit["alpha"],
                                    pcepi_yoy)
    print(f"\nComposition validation (composed YoY vs actual PCEPI YoY):")
    print(f"  n = {val['n']}, mean residual = {val['mean_resid']:+.4f} pp, "
            f"median = {val['median_resid']:+.4f} pp, sd = {val['sd_resid']:.4f}")
    print(f"  |residual| max = {val['abs_max']:.4f} pp, "
            f"within 0.1 pp = {val['within_0.1pp']*100:.1f}%, "
            f"within 0.3 pp = {val['within_0.3pp']*100:.1f}%")

    # Forecast for the next PCE release
    latest_month = pcepi.index.max()
    latest_yoy = float(pcepi_yoy.loc[latest_month])
    next_release_month = latest_month + pd.offsets.MonthEnd(1)
    print(f"\nLatest PCEPI: {latest_month.date()}  YoY = {latest_yoy:.4f}%")
    print(f"Next release target: {next_release_month.date()}")

    print(f"\n── PCE-native CBDF forecast at origin {latest_month.date()} ──")
    fc = forecast_next_yoy(component_levels, fit["weights"], fit["alpha"],
                                pcepi_yoy, latest_month, n_samples=N_SAMPLES,
                                seed=int(latest_month.value % 1_000_000))
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

    # Persist
    out_path = OUT_DIR / (
        f"pce_native_{next_release_month.date()}_forecast_{date.today()}.json")
    out_path.write_text(json.dumps({
        "target": "bea_headline_pce_yoy",
        "method": "pce_native_cbdf",
        "as_of_date": str(date.today()),
        "latest_pcepi_month": str(latest_month.date()),
        "latest_pcepi_yoy": latest_yoy,
        "next_release_month": str(next_release_month.date()),
        "forward_next_release_forecast": fc,
        "composition_validation": val,
        "weight_calibration": fit,
    }, indent=2, default=str))
    print(f"\nSaved: {out_path}")

    # ── Summary across forecasters ────────────────────────────────
    print()
    print("=" * 78)
    print("SUMMARY — PCE forecasters head-to-head")
    print("=" * 78)
    print(f"  Latest PCEPI print ({latest_month.date()}):  {latest_yoy:.4f}%")
    print(f"  Next release target:                                   "
            f"{next_release_month.date()}")
    print()

    # Read existing PCE standalone JSON
    pce_standalone_path = OUT_DIR / (
        f"pce_{next_release_month.date()}_forecast_{date.today()}.json")
    if pce_standalone_path.exists():
        pcs = json.loads(pce_standalone_path.read_text())
        print(f"  PCE standalone (headline-only AR(1)):      "
                f"{pcs['pce_standalone']['point']:.4f}%")
        print(f"    80% band: [{pcs['pce_standalone']['band_80'][0]:.4f}, "
                f"{pcs['pce_standalone']['band_80'][1]:.4f}]")
        print(f"  Cleveland Fed PCE nowcast:                  "
                f"{pcs['cleveland_fed']['point']:.4f}%")
    print(f"  PCE-native CBDF (3 BEA components):         {fc['point']:.4f}%")


if __name__ == "__main__":
    main()
