"""Day-ahead forecast using the LIVE Truflation published YoY as target.

Key correction from the frozen-CSV version: target = live `truflation_us_cpi_yoy`
(what shows on truflation.com, not the frozen revision-pinned CSV). Live
data pulled from the Truflation Feed API in real time — has through today.

Pipeline:
  1. Pull full history of live truflation_us_cpi_yoy from Feed API
  2. Load 12 component index streams from vintage store (fwd-filled from Apr 16)
  3. Build training panel with live YoY as target
  4. Walk-forward backtest over last 90 origins → check calibration on LIVE
  5. Fit on all data through today, predict tomorrow, emit post
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation import metrics as M  # noqa: E402
from thales.evaluation.conformal import conformal_band_offsets  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402
from thales.weights import build_crosswalk, top_level_category_ids  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
FEED_API_KEY = "egiZc8oSTWdHr7V6lwYZxYil1wF0ezgF"  # Feed API key (separate from TN private key)
# Migrated 2026-04-26 from live (`truflation_us_cpi_yoy`) to FROZEN
# (`truflation_us_cpi_frozen_yoy`). Live continually revises (we observed
# ~50bp upward drift over 14 days for older points); frozen is pinned at
# first publication, never revised. Frozen is what the public chart
# displays, what backtests should score against, and what we want to
# predict tomorrow morning.
FEED_URL = "https://api.truflation.com/api/v1/feed/truflation/macro-data-us/truflation_us_cpi_frozen_yoy"
FEED_COL = "truflation_us_cpi_frozen_yoy"
TRAIN_START = pd.Timestamp("2022-01-01")
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
CALIB_DAYS = 30
MIN_RIDGE_EDGE_PCT = 2.0

OUT_DIR = ROOT / "results" / "daily_forecast_live"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_SERIES = "thales_daily_forecast_live"
TODAY_BASELINE_SERIES = "thales_daily_forecast_live_today_published"


def pull_live_yoy() -> pd.Series:
    """Pull the FROZEN Truflation YoY series — first-published, never
    revised. (Function name retained for compatibility; series is now
    frozen, not the continuously-revising live series.)"""
    r = requests.get(FEED_URL, headers={"Authorization": FEED_API_KEY}, timeout=30)
    r.raise_for_status()
    body = r.json()
    idx = pd.to_datetime(body["index"])
    vals = body[FEED_COL]
    s = pd.Series(vals, index=idx).dropna()
    s.name = "live_yoy"    # column name kept stable so downstream code is unaffected
    return s


def load_components(store: VintageStore,
                      raw_names: list[str]) -> pd.DataFrame:
    cols = {}
    as_of = date.today()
    for sid in raw_names:
        s = store.get_vintage(sid, as_of)
        if not s.empty:
            cols[sid] = s
    return pd.DataFrame(cols)


@dataclass
class BacktestRow:
    origin: date; target: date
    pred: float; lo80: float; hi80: float; lo95: float; hi95: float
    today: float; actual: float; error: float; naive_error: float
    pred_up: bool; actual_up: bool


def one_origin(panel: pd.DataFrame, feature_cols: list[str],
                origin: pd.Timestamp) -> BacktestRow | None:
    feat = panel[feature_cols].copy()
    feat["__target_tp1"] = panel["live_yoy"].shift(-1)

    calib_start = origin - pd.Timedelta(days=CALIB_DAYS)
    train = feat.loc[(feat.index >= TRAIN_START) &
                      (feat.index < calib_start)].dropna()
    if len(train) < 90:
        return None

    X_tr = train[feature_cols].values
    y_tr = train["__target_tp1"].values
    model = RidgeCV(alphas=list(RIDGE_ALPHAS)).fit(X_tr, y_tr)

    calib = feat.loc[(feat.index >= calib_start) & (feat.index < origin)].dropna()
    if len(calib) < 10:
        return None
    errs = calib["__target_tp1"].values - model.predict(calib[feature_cols].values)

    x_origin = feat.loc[origin, feature_cols]
    if x_origin.isna().any():
        return None
    point = float(model.predict(x_origin.values.reshape(1, -1))[0])

    target = origin + pd.Timedelta(days=1)
    if target not in panel.index:
        return None
    actual = panel.loc[target, "live_yoy"]
    if pd.isna(actual):
        return None
    today = float(panel.loc[origin, "live_yoy"])

    return BacktestRow(
        origin=origin.date(), target=target.date(),
        pred=point,
        lo80=point + float(np.percentile(errs, 10)),
        hi80=point + float(np.percentile(errs, 90)),
        lo95=point + float(np.percentile(errs, 2.5)),
        hi95=point + float(np.percentile(errs, 97.5)),
        today=today, actual=float(actual),
        error=point - float(actual),
        naive_error=today - float(actual),
        pred_up=point > today, actual_up=float(actual) > today,
    )


def main() -> None:
    # ── Data ────────────────────────────────────────────────────────────
    print("Pulling LIVE truflation_us_cpi_yoy from Feed API...")
    live = pull_live_yoy()
    print(f"  live range: {live.index.min():%Y-%m-%d} → "
          f"{live.index.max():%Y-%m-%d}  latest={live.iloc[-1]:.4f}%")

    streams_df = pd.read_csv(ROOT / "data" / "truflation" / "streams_catalog.csv")
    cw = build_crosswalk(streams_df["raw_name"])
    tops = top_level_category_ids()
    top_streams = cw[cw["category_id"].isin(tops)].copy()
    top_info = [(int(r.category_id), r.raw_name, r.category)
                 for _, r in top_streams.iterrows()]
    raw_names = [t[1] for t in top_info]

    with VintageStore(VINTAGE_DB, read_only=True) as store:
        components = load_components(store, raw_names)
    print(f"  components range: {components.index.min():%Y-%m-%d} → "
          f"{components.dropna(how='all').index.max():%Y-%m-%d}")

    # Merge + fwd-fill components
    end_date = live.index.max()
    idx = pd.date_range(components.index.min(), end_date, freq="D")
    components = components.reindex(idx).ffill()
    panel = components.copy()
    panel["live_yoy"] = live
    panel = panel.dropna(subset=raw_names + ["live_yoy"])
    print(f"  merged panel: {panel.shape}  range {panel.index.min():%Y-%m-%d} → "
          f"{panel.index.max():%Y-%m-%d}")

    feature_cols = raw_names + ["live_yoy"]

    # ── Backtest on LIVE ────────────────────────────────────────────────
    print("\nBacktesting 90 origins on LIVE target...")
    origins = [o for o in panel.index[-95:] if (o + pd.Timedelta(days=1)) in panel.index]
    origins = origins[-90:]
    rows = []
    for o in origins:
        r = one_origin(panel, feature_cols, pd.Timestamp(o))
        if r is not None:
            rows.append(r)
    df = pd.DataFrame([r.__dict__ for r in rows])

    rmse_m = M.rmse(df["pred"].values, df["actual"].values)
    rmse_n = M.rmse(df["today"].values, df["actual"].values)
    mae_m = M.mae(df["pred"].values, df["actual"].values)
    cov80 = ((df["actual"] >= df["lo80"]) & (df["actual"] <= df["hi80"])).mean()
    cov95 = ((df["actual"] >= df["lo95"]) & (df["actual"] <= df["hi95"])).mean()
    dir_hit = (df["pred_up"] == df["actual_up"]).mean()
    dir_up = df["actual_up"].mean()
    rmse_red = (1 - rmse_m / rmse_n) * 100 if rmse_n > 0 else float("nan")

    print(f"  n={len(df)}  window {df['origin'].min()} → {df['origin'].max()}")
    print(f"  RMSE model/naive: {rmse_m:.4f} / {rmse_n:.4f}  ({rmse_red:+.2f}%)")
    print(f"  MAE model:        {mae_m:.4f} pp")
    print(f"  80% coverage:     {cov80:.1%}  (nominal 80%)")
    print(f"  95% coverage:     {cov95:.1%}  (nominal 95%)")
    print(f"  Directional acc:  {dir_hit:.1%}  (base-rate up: {dir_up:.1%})")

    selected_point_model = "ridge" if rmse_red >= MIN_RIDGE_EDGE_PCT else "persistence"
    ship = "SHIP" if (
        selected_point_model == "ridge" and
        abs(cov80 - 0.80) < 0.07 and abs(cov95 - 0.95) < 0.04
    ) else "BASELINE"
    print(f"  {ship}")
    print(f"  selected point model: {selected_point_model} "
          f"(Ridge edge threshold {MIN_RIDGE_EDGE_PCT:.1f}%)")

    # ── Live forecast for tomorrow ───────────────────────────────────────
    origin = panel.index.max()
    target = origin + pd.Timedelta(days=1)
    print(f"\nLive forecast: latest published reference={origin:%Y-%m-%d} "
          f"→ predicting next reference {target:%Y-%m-%d}")

    calib_start = origin - pd.Timedelta(days=CALIB_DAYS)
    feat = panel[feature_cols].copy()
    feat["__target_tp1"] = panel["live_yoy"].shift(-1)

    train = feat.loc[(feat.index >= TRAIN_START) &
                      (feat.index < calib_start)].dropna()
    calib = feat.loc[(feat.index >= calib_start) & (feat.index < origin)].dropna()

    today_val = float(panel.loc[origin, "live_yoy"])

    model = RidgeCV(alphas=list(RIDGE_ALPHAS)).fit(
        train[feature_cols].values, train["__target_tp1"].values)
    x_origin = feat.loc[origin, feature_cols].values.reshape(1, -1)
    ridge_raw_point = float(model.predict(x_origin)[0])

    if selected_point_model == "ridge":
        errs = (calib["__target_tp1"].values -
                 model.predict(calib[feature_cols].values))
        raw_point = ridge_raw_point
        point = raw_point
        lo80_off, hi80_off = conformal_band_offsets(errs, alpha=0.20)
        lo95_off, hi95_off = conformal_band_offsets(errs, alpha=0.05)
    else:
        # The component streams are currently stale/fwd-filled and Ridge is
        # not clearing persistence. Use the honest day-ahead baseline:
        # tomorrow's YoY equals today's YoY, with signed conformal offsets
        # from recent live one-day changes.
        y_live = panel["live_yoy"].dropna()
        cal_origins = [o for o in y_live.index[y_live.index < origin]
                       if o + pd.Timedelta(days=1) in y_live.index]
        cal_origins = cal_origins[-CALIB_DAYS:]
        errs = np.array([
            float(y_live.loc[o + pd.Timedelta(days=1)] - y_live.loc[o])
            for o in cal_origins
        ])
        raw_point = today_val
        point = today_val
        lo80_off, hi80_off = conformal_band_offsets(errs, alpha=0.20)
        lo95_off, hi95_off = conformal_band_offsets(errs, alpha=0.05)

    bias = 0.0
    lo80 = point + lo80_off
    hi80 = point + hi80_off
    lo95 = point + lo95_off
    hi95 = point + hi95_off

    # Attribution: β_i × 7-day move
    week_ago = origin - pd.Timedelta(days=7)
    if week_ago not in panel.index:
        week_ago = panel.index[panel.index.get_indexer([week_ago], method="nearest")[0]]
    contribs = []
    for i, (cid, raw, cat_name) in enumerate(top_info):
        today_idx = float(panel.loc[origin, raw])
        wa_idx = float(panel.loc[week_ago, raw])
        beta = float(model.coef_[i])
        mv = (today_idx - wa_idx) / wa_idx * 100 if wa_idx else 0.0
        contrib = beta * (today_idx - wa_idx)
        contribs.append((cid, cat_name, mv, contrib))
    contribs.sort(key=lambda c: abs(c[-1]), reverse=True)

    delta = point - today_val
    arrow = "↑" if delta > 0.0005 else ("↓" if delta < -0.0005 else "→")
    print()
    print("=" * 72)
    print("Thales — Day-ahead Truflation US CPI YoY (LIVE) forecast")
    print("=" * 72)
    print(f"{target:%b %d, %Y}:  {point:.4f}%  {arrow}")
    print(f"  80% band:  [{lo80:.4f}%, {hi80:.4f}%]  (width {hi80-lo80:.4f} pp)")
    print(f"  95% band:  [{lo95:.4f}%, {hi95:.4f}%]")
    print(f"  Latest ref:{today_val:.4f}%  ({origin:%Y-%m-%d})")
    print(f"  Δ vs today:{delta:+.4f} pp")
    print()
    print(f"Backtest over last 90 origins on LIVE target:")
    print(f"  80% coverage: {cov80:.1%}  |  MAE: {mae_m:.4f} pp  |  "
          f"direction: {dir_hit:.1%}")
    print()
    if selected_point_model == "ridge":
        print("Top 3 drivers (7-day component moves):")
        for cid, name, mv, c in contribs[:3]:
            ar = "↑" if c > 0.0001 else ("↓" if c < -0.0001 else "→")
            print(f"  {ar} {name:<32s}  {mv:+.4f}% → {c:+.4f} pp")
    else:
        print("Top drivers: skipped because production point is calibrated persistence.")
    print()
    print("DEBUG:")
    print(f"  selected point model: {selected_point_model}")
    print(f"  Ridge α: {model.alpha_}  lag coef (φ): {model.coef_[-1]:.6f}")
    print(f"  Ridge raw point: {ridge_raw_point:.6f}%")
    print(f"  production raw point: {raw_point:.6f}%")
    print(f"  bias correction: {bias:+.6f} pp")
    print(f"  production point: {point:.6f}%")
    print(f"  n_train: {len(train)}  n_calib: {len(calib)}")

    # ── Persist ─────────────────────────────────────────────────────────
    dump = {
        "origin_date": str(origin.date()),
        "target_date": str(target.date()),
        "point_yoy_pct": point,
        "selected_point_model": selected_point_model,
        "raw_point_pre_correction_pct": raw_point,
        "ridge_raw_point_pct": ridge_raw_point,
        "bias_correction_pp": bias,
        "today_published_yoy": today_val,
        "delta_vs_today_pp": delta,
        "band_80": [lo80, hi80],
        "band_95": [lo95, hi95],
        "band_80_width_pp": hi80 - lo80,
        "band_95_width_pp": hi95 - lo95,
        "ridge_alpha": float(model.alpha_),
        "intercept": float(model.intercept_),
        "lag_coef": float(model.coef_[-1]),
        "n_train": len(train),
        "n_calib": len(calib),
        "backtest": {
            "n_origins": len(df),
            "window_start": str(df["origin"].min()),
            "window_end": str(df["origin"].max()),
            "rmse_model": float(rmse_m),
            "rmse_naive": float(rmse_n),
            "rmse_reduction_pct": float(rmse_red),
            "mae_model": float(mae_m),
            "coverage_80": float(cov80),
            "coverage_95": float(cov95),
            "directional_accuracy": float(dir_hit),
            "base_rate_up": float(dir_up),
            "ship_verdict": "SHIP" if (abs(cov80 - 0.80) < 0.07 and
                                          abs(cov95 - 0.95) < 0.04 and
                                          selected_point_model == "ridge") else "BASELINE",
        },
        "drivers": [
            {"category_id": cid, "category_name": name,
             "move_pct_7d": mv, "contribution_pp": c}
            for cid, name, mv, c in contribs
        ] if selected_point_model == "ridge" else [],
    }
    out_path = OUT_DIR / f"forecast_live_{origin.date()}.json"
    out_path.write_text(json.dumps(dump, indent=2, default=str))
    print(f"\nSaved: {out_path}")

    with VintageStore(VINTAGE_DB) as store:
        store.ingest(
            series_id=FORECAST_SERIES,
            observations=[(target.date(), point)],
            as_of_date=origin.date(),
            source=FORECAST_SERIES,
        )
        store.ingest(
            series_id=TODAY_BASELINE_SERIES,
            observations=[(origin.date(), today_val)],
            as_of_date=origin.date(),
            source=FORECAST_SERIES,
        )
    print(f"Logged to vintage store: {FORECAST_SERIES} as_of={origin.date()} target={target.date()}")


if __name__ == "__main__":
    main()
