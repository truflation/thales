"""One-off day-ahead forecast using the fresh Truflation CSV.

Context:
  - TN Network frozen-index component streams are stuck at 2026-04-16
  - Truflation published-frozen YoY CSV (provided manually) goes to 2026-04-23
  - Today is 2026-04-24; the post would run for target 2026-04-24 (publishes
    on 2026-04-25)

Approach:
  1. Read the CSV for published YoY (our target series)
  2. Pull 12 top-level component streams from the vintage store
  3. Forward-fill components from their 2026-04-16 last observation through
     2026-04-23 (7 days; components are near-constant day-to-day, and the
     Ridge coefficient on the published-YoY lag is 0.996, so the prediction
     is dominated by that feature anyway)
  4. RidgeCV on training (< T − 30 days) + split-conformal calibration on
     the last 30 days → point + bands for T+1
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.vintage import VintageStore  # noqa: E402
from thales.weights import build_crosswalk, top_level_category_ids  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
CSV_PATH = Path("/Users/kluless/Downloads/Truflation_US_CPI_Data_(Frozen) (1).csv")
TRAIN_START = pd.Timestamp("2022-01-01")
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
CALIB_DAYS = 30


def main() -> None:
    # ── Load fresh CSV ──────────────────────────────────────────────────
    csv = pd.read_csv(CSV_PATH)
    csv["date"] = pd.to_datetime(csv["date"])
    csv = csv.set_index("date").sort_index()
    published_yoy = csv["inflation"].dropna()
    print(f"CSV range: {published_yoy.index.min():%Y-%m-%d} → "
          f"{published_yoy.index.max():%Y-%m-%d}  "
          f"({len(published_yoy)} obs)")

    # ── Load components from vintage store ──────────────────────────────
    streams_df = pd.read_csv(ROOT / "data" / "truflation" / "streams_catalog.csv")
    cw = build_crosswalk(streams_df["raw_name"])
    tops = top_level_category_ids()
    top_streams = cw[cw["category_id"].isin(tops)].copy()
    top_info = [(int(r.category_id), r.raw_name, r.category)
                 for _, r in top_streams.iterrows()]
    raw_names = [t[1] for t in top_info]

    with VintageStore(VINTAGE_DB, read_only=True) as store:
        cols = {}
        as_of = date.today()
        for sid in raw_names:
            s = store.get_vintage(sid, as_of)
            if not s.empty:
                cols[sid] = s
    components = pd.DataFrame(cols)
    components_last_obs = components.dropna(how="all").index.max()
    print(f"Components range: {components.index.min():%Y-%m-%d} → "
          f"{components_last_obs:%Y-%m-%d}  (fwd-filled to latest CSV date)")

    # ── Align panel, forward-fill components through the CSV's end ──────
    idx = pd.date_range(components.index.min(), published_yoy.index.max(), freq="D")
    components = components.reindex(idx).ffill()
    panel = components.copy()
    panel["__published_yoy"] = published_yoy
    panel = panel.dropna(subset=raw_names + ["__published_yoy"])

    # ── Split-conformal: train / calibrate / predict ────────────────────
    origin = panel.index.max()     # "today" = 2026-04-23
    target = origin + pd.Timedelta(days=1)
    print(f"Origin (today): {origin:%Y-%m-%d}  → forecast for {target:%Y-%m-%d}")

    feature_cols = raw_names + ["__published_yoy"]
    feat = panel[feature_cols].copy()
    feat["__target_tp1"] = panel["__published_yoy"].shift(-1)

    calib_start = origin - pd.Timedelta(days=CALIB_DAYS)
    train = feat.loc[(feat.index >= TRAIN_START) &
                      (feat.index < calib_start)].dropna()
    calib = feat.loc[(feat.index >= calib_start) &
                      (feat.index < origin)].dropna()
    print(f"Train n = {len(train)}  (through {train.index.max():%Y-%m-%d})")
    print(f"Calib n = {len(calib)}  ({calib.index.min():%Y-%m-%d} → "
          f"{calib.index.max():%Y-%m-%d})")

    X_tr = train[feature_cols].values
    y_tr = train["__target_tp1"].values
    model = RidgeCV(alphas=list(RIDGE_ALPHAS)).fit(X_tr, y_tr)
    print(f"Ridge α = {model.alpha_}  (coefs: lag={model.coef_[-1]:.4f})")

    # Calibration OOS errors
    preds_cal = model.predict(calib[feature_cols].values)
    errs_cal = calib["__target_tp1"].values - preds_cal
    print(f"Calib abs-err p50/p80/p95: "
          f"{np.percentile(np.abs(errs_cal),50):.4f} / "
          f"{np.percentile(np.abs(errs_cal),80):.4f} / "
          f"{np.percentile(np.abs(errs_cal),95):.4f}  pp")

    # Predict
    x_origin = feat.loc[origin, feature_cols]
    point = float(model.predict(x_origin.values.reshape(1, -1))[0])
    today_val = float(feat.loc[origin, "__published_yoy"])

    lo80 = point + float(np.percentile(errs_cal, 10))
    hi80 = point + float(np.percentile(errs_cal, 90))
    lo95 = point + float(np.percentile(errs_cal, 2.5))
    hi95 = point + float(np.percentile(errs_cal, 97.5))

    # ── Attribution from Ridge coefs + 7-day component moves ────────────
    week_ago = origin - pd.Timedelta(days=7)
    if week_ago not in panel.index:
        week_ago = panel.index[panel.index.get_indexer([week_ago], method="nearest")[0]]
    contribs = []
    for i, (cid, raw, name) in enumerate(top_info):
        today_idx = float(panel.loc[origin, raw])
        weekago_idx = float(panel.loc[week_ago, raw])
        beta = float(model.coef_[i])
        move_pct = (today_idx - weekago_idx) / weekago_idx * 100 if weekago_idx else 0.0
        contrib_pp = beta * (today_idx - weekago_idx)
        contribs.append((cid, name, today_idx, weekago_idx, move_pct, beta, contrib_pp))
    contribs.sort(key=lambda r: abs(r[-1]), reverse=True)

    # ── Print post ──────────────────────────────────────────────────────
    delta = point - today_val
    arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "→")
    print()
    print("=" * 72)
    print("Thales — Day-ahead Truflation US CPI YoY forecast")
    print("=" * 72)
    print(f"{target:%b %d, %Y}:  {point:.3f}%  {arrow}")
    print(f"  80% band:  [{lo80:.3f}%, {hi80:.3f}%]  (width {hi80-lo80:.3f} pp)")
    print(f"  95% band:  [{lo95:.3f}%, {hi95:.3f}%]")
    print(f"  Today:     {today_val:.3f}%  (as of {origin:%Y-%m-%d})")
    print()
    print("Top 3 drivers (7-day component moves, β × Δ):")
    for cid, name, today_idx, week_idx, mv, b, c in contribs[:3]:
        ar = "↑" if c > 0.001 else ("↓" if c < -0.001 else "→")
        print(f"  {ar} {name:<32s}  {mv:+.2f}% → {c:+.3f} pp")
    print()
    print(f"Method: RidgeCV on 12 top-level Truflation component index values + "
          f"published-YoY lag. Bands via split-conformal calibration on the "
          f"most-recent {CALIB_DAYS} days of out-of-sample errors. "
          f"Per the {CALIB_DAYS}-day calibration set, 80%% of realized errors "
          f"fell in this band width historically.")
    print()
    print("=" * 72)
    print("DEBUG")
    print("=" * 72)
    print(f"Ridge α: {model.alpha_}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Lag coef (φ): {model.coef_[-1]:.4f}")
    print(f"Train residual SD: {np.std(y_tr - model.predict(X_tr)):.4f} pp")
    print(f"Calib errors mean: {np.mean(errs_cal):+.4f} pp  "
          f"SD: {np.std(errs_cal):.4f} pp")
    print(f"Note: components are forward-filled from {components_last_obs:%Y-%m-%d} "
          f"({(origin - pd.Timestamp(components_last_obs)).days} days). "
          f"Forecast is dominated by the published-YoY lag term because components "
          f"have been constant during the fwd-fill window.")


if __name__ == "__main__":
    main()
