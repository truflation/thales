"""Truflation US CPI YoY — bottom-up multi-horizon forecaster (Phase 1 / 1.5).

End-product spec:

  - Trained on Truflation data only (12 top-level OR 58 leaf-level CPI
    per-component daily streams, 2010-01-01 → today; ~5,800 daily
    observations per stream).
  - Predicts Truflation US CPI YoY at h ∈ {1, 7, 14, 30, 90} days.
  - Bottom-up architecture: per-component forecaster → CBDF / weight
    composition → headline YoY. Academically supported by the bottom-up
    inflation forecasting literature (Hubrich 2005, Marcellino-Stock-
    Watson 2003, Martínez-Rivera 2025).
  - Density forecasts via bootstrap of per-component residuals propagated
    through the composer.
  - --crosswalk-level top12 (default) reproduces Phase 1; --crosswalk-level
    leaves58 runs Phase 1.5 with the deeper sub-component panel that
    closes the parent/child double-count (58 leaves cover exactly 100.000%
    of the v2 Truflation weight sheet).

Method (matches `composition_check.py` Method 2 — the validated 0.000 pp
median residual approach):

  1. For each top-level component k (12 total), forecast its level at
     horizon h. Forecasters per component:
       - persistence (level_pred[T+h] = level[T])
       - AR(1) on log-returns (predict daily log-return, integrate)
  2. Compose: composed_level[T+h] = Σ_k w_k · (level_pred_k[T+h] /
     level_k[base]) · 100.
  3. Compute headline YoY at horizon h using the composed forecasted
     level vs. the *known* composed level 365 days back:
       headline_yoy[T+h] = composed_level[T+h] / composed_level[T+h-365] − 1.
     (When T+h-365 < base, fall back to the actual published Truflation
     headline for the denominator.)
  4. Density: per-component rolling-conformal residuals → bootstrap
     forecast paths → compose → headline samples. CRPS-scoreable.
  5. Walk-forward eval over 2018-01-01 → present, after 8-year warm-up.

Outputs:
  - results/truflation_cpi_forecast/<origin>.json (point + bands + samples)
  - results/truflation_cpi_forecast/walk_forward_summary.csv
  - results/truflation_cpi_forecast/FINDINGS.md

Run:
  uv run python scripts/forecast_truflation_cpi_bottomup.py
  uv run python scripts/forecast_truflation_cpi_bottomup.py --origin 2026-05-01
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.weights import (    # noqa: E402
    build_crosswalk,
    get_top_level_weights,
    load_category_tree,
)

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
STREAMS_CSV = ROOT / "data" / "truflation" / "streams_catalog.csv"
OUT_DIR = ROOT / "results" / "truflation_cpi_forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEED_API_KEY = "egiZc8oSTWdHr7V6lwYZxYil1wF0ezgF"
FEED_BASE = "https://api.truflation.com/api/v1/feed/truflation/macro-data-us"
HEADLINE_FROZEN_COL = "truflation_us_cpi_frozen_yoy"

# Multi-horizon spec (daily horizons; long horizons handled in Phase 2)
HORIZONS_DAYS = [1, 7, 14, 30, 90]

# Calibration: per-component rolling-conformal residual window
CALIB_WINDOW_DAYS = 30
N_SAMPLES = 200


# ─── Data loading ────────────────────────────────────────────────────────


def _resolve_leaf_set(cw: pd.DataFrame) -> set[int]:
    """Find the leaf-set within a crosswalk: categories with no descendant
    that's also present in the same crosswalk. Pruning parents whose
    children are in the set avoids double-counting under leaf-level
    composition.
    """
    cat_set = set(cw["category_id"].dropna().astype(int))
    tree = load_category_tree()
    children_lookup: dict[int, list[int]] = {}
    for cid, pid in zip(tree["category_id"].astype(int),
                          tree["parent_id"]):
        if pd.notna(pid):
            children_lookup.setdefault(int(pid), []).append(int(cid))

    def has_descendant_in_set(cid: int) -> bool:
        for c in children_lookup.get(cid, []):
            if c in cat_set or has_descendant_in_set(c):
                return True
        return False

    return {cid for cid in cat_set if not has_descendant_in_set(cid)}


def _leaf_weights_pct(as_of: str = "2026-05-01") -> dict[int, float]:
    """Build leaf-level weight lookup: category_id → weight percentage.

    The Truflation weight sheet uses two row types:
      * subcategory_id == 0 → weight goes to category_id (top-level)
      * subcategory_id != 0 → weight goes to subcategory_id (children)

    For Phase 1.5 we read both row types into a single lookup keyed by
    category_id at every depth, so a leaf at any depth resolves correctly.
    """
    from datetime import date as _date
    if pd.Timestamp(as_of).date() >= pd.Timestamp("2026-01-01").date():
        wt = pd.read_csv(ROOT / "data" / "truflation" / "weights"
                                / "categories-tables-v2.csv")
    else:
        wt = pd.read_csv(ROOT / "data" / "truflation" / "weights"
                                / "categories-tables-v1.csv")
    out: dict[int, float] = {}
    base = wt[wt["source_id"] == 0]
    for _, r in base.iterrows():
        if int(r["subcategory_id"]) == 0:
            out[int(r["category_id"])] = float(r["relative_importance"])
        else:
            out[int(r["subcategory_id"])] = float(r["relative_importance"])
    return out


def load_component_levels(con: duckdb.DuckDBPyConnection,
                            crosswalk_level: str = "top12",
                            ) -> tuple[pd.DataFrame, dict[int, float]]:
    """Load CPI component daily level series + matching weights.

    crosswalk_level:
      * "top12"   — 12 top-level Truflation categories (Phase 1)
      * "leaves58" — leaf-set of the 80 ingested CPI streams (Phase 1.5)

    Returns (panel, weights_pct):
      panel: wide DataFrame, one column per category_id, daily ffill.
      weights_pct: dict {category_id: weight_pct, sum ≈ 100.0}.
    """
    catalog = pd.read_csv(STREAMS_CSV)
    cpi = catalog[catalog["humanized_name"].str.startswith("us_")]
    cw = build_crosswalk(cpi["raw_name"]).dropna(subset=["category_id"]).copy()
    cw["category_id"] = cw["category_id"].astype(int)

    if crosswalk_level == "top12":
        top12 = get_top_level_weights("2026-05-01")
        keep_ids = set(top12["category_id"].astype(int))
        weights_pct = dict(zip(top12["category_id"].astype(int),
                                  top12["weight"].astype(float)))
    elif crosswalk_level == "leaves58":
        keep_ids = _resolve_leaf_set(cw)
        all_w = _leaf_weights_pct("2026-05-01")
        weights_pct = {cid: all_w[cid] for cid in keep_ids if cid in all_w}
        missing = keep_ids - set(weights_pct)
        if missing:
            raise RuntimeError(
                f"leaf categories without weight: {sorted(missing)}")
    else:
        raise ValueError(f"unknown crosswalk_level={crosswalk_level!r}")

    cw_keep = cw[cw["category_id"].isin(keep_ids)].copy()

    frames = []
    for _, row in cw_keep.iterrows():
        raw_name = row["raw_name"]
        cat_id = int(row["category_id"])
        rows = con.execute(
            "SELECT reference_date, value FROM vintage "
            "WHERE series_id = ? AND source = 'truf_network' "
            "AND (series_id, reference_date, as_of_date) IN ("
            "  SELECT series_id, reference_date, MAX(as_of_date) "
            "  FROM vintage WHERE series_id = ? AND source = 'truf_network' "
            "  GROUP BY series_id, reference_date) "
            "ORDER BY reference_date",
            [raw_name, raw_name],
        ).fetchall()
        s = pd.Series(
            [r[1] for r in rows],
            index=pd.to_datetime([r[0] for r in rows]),
            name=cat_id,
        )
        frames.append(s)

    panel = pd.concat(frames, axis=1).sort_index()
    # Resample to daily frequency, forward-fill any gaps
    panel = panel.resample("D").ffill().dropna()
    return panel, weights_pct


def load_truflation_headline_yoy() -> pd.Series:
    """Pull current Truflation US CPI frozen YoY series from Feed API."""
    r = requests.get(f"{FEED_BASE}/{HEADLINE_FROZEN_COL}",
                       headers={"Authorization": FEED_API_KEY},
                       timeout=30)
    r.raise_for_status()
    data = r.json()
    s = pd.Series(data[HEADLINE_FROZEN_COL],
                       index=pd.to_datetime(data["index"])).sort_index().dropna()
    s.name = "truflation_us_cpi_yoy"
    return s


# ─── Composition (M2 method — validated by composition_check.py) ────────


def compose_level(component_levels: pd.DataFrame,
                    weights_pct: dict[int, float],
                    base_date: pd.Timestamp,
                    ) -> pd.Series:
    """Composite level series (Method M2): Σ_k w_k · (level_k[t] /
    level_k[base]) · 100, normalised so base = 100.

    weights_pct: dict mapping category_id → weight percentage (sums to 100).
    """
    base_levels = component_levels.loc[base_date]
    composite = pd.Series(0.0, index=component_levels.index, name="composed_level")
    for cat_id in component_levels.columns:
        w = weights_pct[int(cat_id)]
        composite += w * (component_levels[cat_id] / base_levels[cat_id]) * 100.0
    composite /= 100.0    # normalise so base = 100
    return composite


def headline_yoy_from_composed_level(composed_level: pd.Series,
                                          actual_headline_yoy: pd.Series | None = None,
                                          ) -> pd.Series:
    """Compute YoY % change from composed level. Drops first 365 days."""
    yoy = (composed_level / composed_level.shift(365) - 1.0) * 100.0
    return yoy.dropna()


# ─── Per-component forecasters ───────────────────────────────────────────


def fit_log_returns_ar1(level_history: pd.Series
                          ) -> tuple[float, float, np.ndarray]:
    """Fit AR(1) on log-returns of level history. Returns (alpha, phi,
    residuals).

      log_ret[t] = α + φ · log_ret[t-1] + ε[t]
    """
    log_ret = np.log(level_history) - np.log(level_history.shift(1))
    log_ret = log_ret.dropna()
    if len(log_ret) < 30:
        return 0.0, 0.0, np.array([])
    x = log_ret.values[:-1]
    y = log_ret.values[1:]
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, phi = float(coef[0]), float(coef[1])
    residuals = y - (alpha + phi * x)
    return alpha, phi, residuals


def forecast_component_level(level_history: pd.Series,
                                 horizons: list[int],
                                 method: str = "ar1_log_returns",
                                 n_samples: int = N_SAMPLES,
                                 calib_window: int = CALIB_WINDOW_DAYS,
                                 seed: int = 0,
                                 ) -> dict[int, dict]:
    """Forecast component level at multiple horizons.

    Returns {h: {"point": float, "samples": np.ndarray of shape (n_samples,)}}.
    """
    rng = np.random.default_rng(seed)
    last_level = float(level_history.iloc[-1])
    out = {}

    if method == "persistence":
        for h in horizons:
            out[h] = {
                "point": last_level,
                "samples": np.full(n_samples, last_level),
            }
        return out

    # AR(1) on log-returns
    alpha, phi, residuals = fit_log_returns_ar1(level_history)
    if len(residuals) < 10:
        # Fallback to persistence
        for h in horizons:
            out[h] = {"point": last_level, "samples": np.full(n_samples, last_level)}
        return out

    # Use last calib_window residuals for bootstrap
    calib_residuals = residuals[-calib_window:]
    log_ret_now = np.log(level_history.iloc[-1]) - np.log(level_history.iloc[-2])

    for h in horizons:
        # Deterministic point: iterate AR(1) on log returns
        log_returns = []
        last_log_ret = log_ret_now
        for _ in range(h):
            last_log_ret = alpha + phi * last_log_ret
            log_returns.append(last_log_ret)
        cumulative_log_return = sum(log_returns)
        point_level = last_level * np.exp(cumulative_log_return)

        # Density: bootstrap S sample paths
        sample_levels = np.empty(n_samples)
        for s in range(n_samples):
            last_log_ret_s = log_ret_now
            cumret = 0.0
            for _ in range(h):
                eps = rng.choice(calib_residuals)
                last_log_ret_s = alpha + phi * last_log_ret_s + eps
                cumret += last_log_ret_s
            sample_levels[s] = last_level * np.exp(cumret)

        out[h] = {"point": float(point_level), "samples": sample_levels}
    return out


# ─── Forecast composition + YoY computation ─────────────────────────────


def compose_forecast_yoy(component_forecasts: dict[int, dict],
                            component_levels_history: pd.DataFrame,
                            weights_pct: dict[int, float],
                            base_date: pd.Timestamp,
                            origin: pd.Timestamp,
                            horizons: list[int],
                            n_samples: int = N_SAMPLES,
                            anchor_yoy: float | None = None,
                            ) -> dict[int, dict]:
    """For each horizon h, compose per-component forecasts to a headline
    YoY forecast.

    component_forecasts: {category_id: {h: {"point", "samples"}}}
    Returns {h: {"point", "lo80", "hi80", "lo95", "hi95", "samples"}}.
    """
    base_levels = component_levels_history.loc[base_date]

    # Compute composed YoY at origin (for anchor calibration)
    if anchor_yoy is not None and origin in component_levels_history.index:
        origin_levels = component_levels_history.loc[origin]
        denom_origin = origin - pd.Timedelta(days=365)
        if denom_origin in component_levels_history.index:
            denom_origin_levels = component_levels_history.loc[denom_origin]
        else:
            available = component_levels_history.index[
                component_levels_history.index <= denom_origin]
            denom_origin_levels = (component_levels_history.loc[available[-1]]
                                       if len(available) else None)
        if denom_origin_levels is not None:
            composed_origin = sum(
                weights_pct[int(cid)] * (origin_levels[cid] / base_levels[cid]) * 100.0
                for cid in component_levels_history.columns) / 100.0
            composed_origin_denom = sum(
                weights_pct[int(cid)] * (denom_origin_levels[cid] / base_levels[cid]) * 100.0
                for cid in component_levels_history.columns) / 100.0
            composed_yoy_at_origin = (composed_origin / composed_origin_denom - 1.0) * 100.0
            anchor_offset = anchor_yoy - composed_yoy_at_origin
        else:
            anchor_offset = 0.0
    else:
        anchor_offset = 0.0

    out = {}
    for h in horizons:
        target = origin + pd.Timedelta(days=h)
        denom_date = target - pd.Timedelta(days=365)

        # Composed level at base, target, and denom_date
        # composed_level[t] = sum_k w_k × (level_k[t] / level_k[base]) (then /100)

        # At denom_date (must be in the past, use historical component levels)
        if denom_date < component_levels_history.index.min():
            # Not enough history to compute YoY at this horizon
            out[h] = None
            continue
        if denom_date in component_levels_history.index:
            denom_levels = component_levels_history.loc[denom_date]
        else:
            # Find nearest available date
            available = component_levels_history.index[
                component_levels_history.index <= denom_date]
            if len(available) == 0:
                out[h] = None
                continue
            denom_levels = component_levels_history.loc[available[-1]]

        # Composed level at denom_date
        composed_denom = sum(
            weights_pct[int(cid)] * (denom_levels[cid] / base_levels[cid]) * 100.0
            for cid in component_levels_history.columns
        ) / 100.0

        # Composed level at target (forecast)
        composed_target_point = sum(
            weights_pct[int(cid)] * (component_forecasts[int(cid)][h]["point"]
                                          / base_levels[cid]) * 100.0
            for cid in component_levels_history.columns
        ) / 100.0

        # Sample paths: for each sample s, compose all 12 components' sample
        S = n_samples
        composed_target_samples = np.zeros(S)
        for cid in component_levels_history.columns:
            w = weights_pct[int(cid)]
            samples_k = component_forecasts[int(cid)][h]["samples"]
            composed_target_samples += w * (samples_k / base_levels[cid]) * 100.0
        composed_target_samples /= 100.0

        # Compute YoY for samples (deterministic point would be biased low at
        # long horizons due to Jensen's inequality on exp(·); use the
        # sample median as the point so it sits inside the bands)
        yoy_samples = (composed_target_samples / composed_denom - 1.0) * 100.0
        # Apply anchor offset: shift the entire sample distribution so that
        # the forecast at origin matches the actual published Truflation YoY.
        # Equivalent to: forecast = composed_dynamics + (actual_origin − composed_origin).
        # Removes the small constant offset between 12-stream composition and
        # Truflation's 80-stream published headline.
        yoy_samples = yoy_samples + anchor_offset
        yoy_point = float(np.median(yoy_samples))
        # Also retain the deterministic point for diagnostic purposes
        yoy_point_deterministic = (composed_target_point / composed_denom - 1.0) * 100.0 + anchor_offset

        # Quantile-based bands
        out[h] = {
            "point": float(yoy_point),
            "point_deterministic": float(yoy_point_deterministic),
            "lo80": float(np.quantile(yoy_samples, 0.10)),
            "hi80": float(np.quantile(yoy_samples, 0.90)),
            "lo95": float(np.quantile(yoy_samples, 0.025)),
            "hi95": float(np.quantile(yoy_samples, 0.975)),
            "samples": yoy_samples.tolist(),
            "target_date": str(target.date()),
            "denom_date": str(denom_date.date()),
            "anchor_offset_pp": float(anchor_offset),
        }
    return out


# ─── Walk-forward driver + scoring ───────────────────────────────────────


def walk_forward(component_levels: pd.DataFrame,
                   weights_pct: dict[int, float],
                   actual_headline_yoy: pd.Series,
                   start_date: str = "2018-01-01",
                   end_date: str | None = None,
                   step_days: int = 30,
                   horizons: list[int] = HORIZONS_DAYS,
                   ) -> pd.DataFrame:
    """Walk forward through history, generating forecasts at each origin
    and scoring against actuals when target date has data."""
    end_date = end_date or str(component_levels.index.max().date())
    base_date = component_levels.index.min()
    origins = pd.date_range(start_date, end_date, freq=f"{step_days}D")
    rows = []
    for origin in origins:
        if origin not in component_levels.index:
            continue
        # Use only data available up to origin
        history = component_levels.loc[component_levels.index <= origin]
        if len(history) < 365 + 30:
            continue

        # Per-component forecasts
        component_forecasts = {}
        for cid in component_levels.columns:
            level_hist = history[cid]
            component_forecasts[int(cid)] = forecast_component_level(
                level_hist, horizons, method="ar1_log_returns",
                n_samples=N_SAMPLES, seed=int(origin.value % 10_000))

        # Anchor to actual published YoY at origin
        anchor = (float(actual_headline_yoy.loc[origin])
                    if origin in actual_headline_yoy.index else None)

        # Compose to headline YoY at each horizon
        headline = compose_forecast_yoy(
            component_forecasts, history, weights_pct, base_date,
            origin, horizons, n_samples=N_SAMPLES,
            anchor_yoy=anchor)

        # Score against actual at each horizon (when actual is available)
        for h, fc in headline.items():
            if fc is None:
                continue
            target = pd.to_datetime(fc["target_date"])
            actual = (float(actual_headline_yoy.loc[target])
                      if target in actual_headline_yoy.index else None)
            err = (fc["point"] - actual) if actual is not None else None
            in_80 = (fc["lo80"] <= actual <= fc["hi80"]) if actual is not None else None
            in_95 = (fc["lo95"] <= actual <= fc["hi95"]) if actual is not None else None
            rows.append({
                "origin": origin,
                "horizon_days": h,
                "target_date": target,
                "point": fc["point"],
                "lo80": fc["lo80"], "hi80": fc["hi80"],
                "lo95": fc["lo95"], "hi95": fc["hi95"],
                "actual": actual,
                "error_pp": err,
                "in_80": in_80,
                "in_95": in_95,
                "width80_pp": fc["hi80"] - fc["lo80"],
                "width95_pp": fc["hi95"] - fc["lo95"],
            })
    return pd.DataFrame(rows)


# ─── Main ────────────────────────────────────────────────────────────────


def run_forecast_at_origin(component_levels: pd.DataFrame,
                              weights_pct: dict[int, float],
                              origin: pd.Timestamp,
                              horizons: list[int] = HORIZONS_DAYS,
                              anchor_yoy: float | None = None,
                              ) -> dict:
    """Single-origin forecast (for live deployment)."""
    base_date = component_levels.index.min()
    history = component_levels.loc[component_levels.index <= origin]
    component_forecasts = {}
    for cid in component_levels.columns:
        level_hist = history[cid]
        component_forecasts[int(cid)] = forecast_component_level(
            level_hist, horizons, method="ar1_log_returns",
            n_samples=N_SAMPLES, seed=0)
    headline = compose_forecast_yoy(
        component_forecasts, history, weights_pct, base_date,
        origin, horizons, n_samples=N_SAMPLES,
        anchor_yoy=anchor_yoy)
    return headline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=str, default=None,
                        help="Single-origin forecast (default: walk-forward)")
    parser.add_argument("--start", type=str, default="2018-01-01",
                        help="Walk-forward start date")
    parser.add_argument("--step-days", type=int, default=30,
                        help="Walk-forward origin spacing (default 30)")
    parser.add_argument("--crosswalk-level", type=str, default="top12",
                        choices=["top12", "leaves58"],
                        help="Composition granularity (Phase 1 = top12, "
                              "Phase 1.5 = leaves58)")
    parser.add_argument("--label", type=str, default=None,
                        help="Filename suffix for outputs "
                              "(default: derived from crosswalk-level)")
    args = parser.parse_args()
    label = args.label or args.crosswalk_level

    print("=" * 78)
    print("Truflation US CPI YoY — bottom-up multi-horizon forecaster")
    print("=" * 78)

    # Load data
    print(f"\nLoading CPI component levels from vintage store "
            f"(crosswalk={args.crosswalk_level})…")
    con = duckdb.connect(str(VINTAGE_DB), read_only=True)
    component_levels, weights_pct = load_component_levels(
        con, crosswalk_level=args.crosswalk_level)
    con.close()
    print(f"  Panel: {len(component_levels)} days × "
            f"{component_levels.shape[1]} components, "
            f"{component_levels.index.min().date()} → "
            f"{component_levels.index.max().date()}")
    print(f"  Weights: {len(weights_pct)} components, "
            f"sum = {sum(weights_pct.values()):.3f}%")

    # Headline YoY (for scoring)
    print("\nLoading actual Truflation US CPI YoY (Feed API)…")
    actual_yoy = load_truflation_headline_yoy()
    print(f"  Actual YoY: n={len(actual_yoy)}, "
            f"{actual_yoy.index.min().date()} → {actual_yoy.index.max().date()}")

    if args.origin:
        # Single-origin forecast
        origin = pd.to_datetime(args.origin)
        # Anchor to actual published YoY at origin if available
        anchor = (float(actual_yoy.loc[origin])
                    if origin in actual_yoy.index else None)
        print(f"\nForecasting from origin {origin.date()}"
                f" (anchor: {anchor:.4f}%)…" if anchor is not None
                else f"\nForecasting from origin {origin.date()} (no anchor)…")
        headline = run_forecast_at_origin(
            component_levels, weights_pct, origin, anchor_yoy=anchor)
        for h, fc in sorted(headline.items()):
            if fc is None:
                continue
            print(f"\n  h = {h:>3d} days  (target {fc['target_date']})")
            print(f"    point: {fc['point']:.4f}%")
            print(f"    80%:   [{fc['lo80']:.4f}, {fc['hi80']:.4f}]   "
                    f"width {fc['hi80'] - fc['lo80']:.4f} pp")
            print(f"    95%:   [{fc['lo95']:.4f}, {fc['hi95']:.4f}]   "
                    f"width {fc['hi95'] - fc['lo95']:.4f} pp")

        # Save
        out_path = OUT_DIR / f"forecast_{origin.date()}_{label}.json"
        payload = {
            "origin": str(origin.date()),
            "method": "bottom_up_top12_ar1_log_returns",
            "n_samples": N_SAMPLES,
            "horizons": {str(h): {**{k: v for k, v in fc.items() if k != 'samples'},
                                    "n_samples": len(fc["samples"]) if fc else 0}
                          for h, fc in headline.items() if fc is not None},
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nSaved: {out_path}")
        return

    # Walk-forward
    print(f"\nWalk-forward from {args.start}, step {args.step_days} days, "
            f"horizons {HORIZONS_DAYS}…")
    df = walk_forward(component_levels, weights_pct, actual_yoy,
                        start_date=args.start,
                        step_days=args.step_days,
                        horizons=HORIZONS_DAYS)
    print(f"  Generated {len(df)} forecast points across "
            f"{df['origin'].nunique()} origins")

    # Score
    if len(df):
        print("\nWalk-forward summary by horizon:")
        scored = df.dropna(subset=["actual"])
        agg = scored.groupby("horizon_days").agg(
            n=("actual", "count"),
            rmse=("error_pp", lambda x: float(np.sqrt(np.mean(x ** 2)))),
            mae=("error_pp", lambda x: float(np.mean(np.abs(x)))),
            mean_err=("error_pp", "mean"),
            cov_80=("in_80", "mean"),
            cov_95=("in_95", "mean"),
            width80=("width80_pp", "mean"),
            width95=("width95_pp", "mean"),
        ).reset_index()
        print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        out_csv = OUT_DIR / f"walk_forward_summary_{label}.csv"
        df.to_csv(out_csv, index=False)
        agg_csv = OUT_DIR / f"walk_forward_aggregate_{label}.csv"
        agg.to_csv(agg_csv, index=False)
        print(f"\nSaved walk-forward results: {out_csv}")
        print(f"Saved aggregate metrics:    {agg_csv}")


if __name__ == "__main__":
    main()
