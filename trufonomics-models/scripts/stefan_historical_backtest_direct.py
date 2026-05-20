"""Historical backtest of the DIRECT-target Stefan forecaster.

Sister script to `stefan_historical_backtest.py`. Uses
`direct_target_forecast` instead of composite-based forecasting, so the
target of the regression IS the published frozen YoY — no composition drift
leak into the bands. Direction metric also fixed: compares actual vs
published-today, not vs reconstructed-today.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation import metrics as M  # noqa: E402
from thales.models.direct_forecaster import (  # noqa: E402
    DEFAULT_TRAIN_START,
    RIDGE_ALPHAS,
    build_training_panel,
)
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "daily_forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ORIGINS = 90
N_BOOT = 500


@dataclass
class BacktestRow:
    origin: date
    target: date
    pred: float              # predicted published YoY at T+1
    lo80: float
    hi80: float
    lo95: float
    hi95: float
    today: float             # published YoY at T (the TRUE naive baseline)
    actual: float            # published YoY at T+1 (target)
    error: float
    naive_error: float
    pred_up: bool
    actual_up: bool


CALIBRATION_WINDOW_DAYS = 30  # held-out window for split-conformal band calibration


def backtest_one_origin(panel: pd.DataFrame, raw_names: list[str],
                         origin: pd.Timestamp,
                         train_start: pd.Timestamp,
                         rng: np.random.Generator,
                         ) -> BacktestRow | None:
    """Split-conformal walk-forward at one origin.

    1) Train Ridge on data in [train_start, origin − CALIBRATION_WINDOW_DAYS)
    2) Run model on each day in the calibration window → collect OOS errors
    3) Predict at origin; derive bands from calibration quantiles
    """
    feature_cols = raw_names + ["__published_yoy"]
    feat = panel[feature_cols].copy()
    feat["__target_tp1"] = panel["__published_yoy"].shift(-1)

    calib_end = origin
    calib_start = origin - pd.Timedelta(days=CALIBRATION_WINDOW_DAYS)

    train = feat.loc[(feat.index >= train_start) &
                      (feat.index < calib_start)].dropna()
    if len(train) < 90:
        return None

    X_tr = train[feature_cols].values
    y_tr = train["__target_tp1"].values
    model = RidgeCV(alphas=list(RIDGE_ALPHAS)).fit(X_tr, y_tr)

    # Split-conformal calibration errors on the held-out window
    calib = feat.loc[(feat.index >= calib_start) &
                      (feat.index < calib_end)].dropna()
    if len(calib) < 10:
        return None
    preds_cal = model.predict(calib[feature_cols].values)
    errs_cal = calib["__target_tp1"].values - preds_cal   # signed errors

    x_origin = feat.loc[origin, feature_cols]
    if x_origin.isna().any():
        return None
    point = float(model.predict(x_origin.values.reshape(1, -1))[0])

    target_date = origin + pd.Timedelta(days=1)
    if target_date not in panel.index:
        return None
    actual = panel.loc[target_date, "__published_yoy"]
    if pd.isna(actual):
        return None
    today = float(panel.loc[origin, "__published_yoy"])

    # Conformal bands from calibration errors. For α-coverage take the
    # signed α/2 and (1-α/2) percentiles of errs_cal; add to point.
    lo80 = point + float(np.percentile(errs_cal, 10))
    hi80 = point + float(np.percentile(errs_cal, 90))
    lo95 = point + float(np.percentile(errs_cal, 2.5))
    hi95 = point + float(np.percentile(errs_cal, 97.5))
    return BacktestRow(
        origin=origin.date(),
        target=target_date.date(),
        pred=point,
        lo80=lo80, hi80=hi80,
        lo95=lo95, hi95=hi95,
        today=today,
        actual=float(actual),
        error=point - float(actual),
        naive_error=today - float(actual),
        pred_up=point > today,
        actual_up=float(actual) > today,
    )


def main() -> None:
    print("Loading panel...")
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        panel, top_info = build_training_panel(store)
    raw_names = [t[1] for t in top_info]
    print(f"Panel: {panel.shape}  range {panel.index.min():%Y-%m-%d} → "
          f"{panel.index.max():%Y-%m-%d}")

    # Pick origins where both origin and origin+1 are in panel
    available = panel.index
    origins = [o for o in available[-(N_ORIGINS + 5):]
                if (o + pd.Timedelta(days=1)) in available]
    origins = origins[-N_ORIGINS:]
    print(f"Backtest window: {origins[0].date()} → {origins[-1].date()}  "
          f"({len(origins)} origins)")

    rng = np.random.default_rng(0)
    rows: list[BacktestRow] = []
    for i, origin in enumerate(origins, 1):
        r = backtest_one_origin(panel, raw_names, pd.Timestamp(origin),
                                  DEFAULT_TRAIN_START, rng)
        if r is None:
            continue
        rows.append(r)
        if i % 15 == 0 or i == len(origins):
            in_80 = "✓" if r.lo80 <= r.actual <= r.hi80 else "✗"
            print(f"  {i}/{len(origins)}  {r.origin}  "
                  f"today={r.today:.3f} pred={r.pred:.3f} "
                  f"[{r.lo80:.3f},{r.hi80:.3f}] actual={r.actual:.3f} {in_80}")

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv(OUT_DIR / "historical_backtest_direct.csv", index=False)

    # Metrics
    rmse_model = M.rmse(df["pred"].values, df["actual"].values)
    rmse_naive = M.rmse(df["today"].values, df["actual"].values)
    mae_model = M.mae(df["pred"].values, df["actual"].values)
    mae_naive = M.mae(df["today"].values, df["actual"].values)
    rmse_red = (1 - rmse_model / rmse_naive) * 100 if rmse_naive > 0 else float("nan")
    mae_red = (1 - mae_model / mae_naive) * 100 if mae_naive > 0 else float("nan")

    cov80 = ((df["actual"] >= df["lo80"]) & (df["actual"] <= df["hi80"])).mean()
    cov95 = ((df["actual"] >= df["lo95"]) & (df["actual"] <= df["hi95"])).mean()
    width80 = (df["hi80"] - df["lo80"]).mean()
    width95 = (df["hi95"] - df["lo95"]).mean()
    dir_hit = (df["pred_up"] == df["actual_up"]).mean()
    dir_actual_up_rate = df["actual_up"].mean()

    last30 = df.sort_values("origin").tail(30)
    rmse30 = M.rmse(last30["pred"].values, last30["actual"].values)
    cov80_30 = ((last30["actual"] >= last30["lo80"]) &
                 (last30["actual"] <= last30["hi80"])).mean()
    dir_hit30 = (last30["pred_up"] == last30["actual_up"]).mean()

    print()
    print("=" * 72)
    print("DIRECT-TARGET FORECASTER — HISTORICAL BACKTEST")
    print(f"Window: {df['origin'].min()} → {df['origin'].max()}  (n={len(df)})")
    print("=" * 72)
    print(f"  RMSE model / naive:     {rmse_model:.4f} / {rmse_naive:.4f}  "
          f"({rmse_red:+.2f}%)")
    print(f"  MAE  model / naive:     {mae_model:.4f} / {mae_naive:.4f}  "
          f"({mae_red:+.2f}%)")
    print(f"  80% coverage:           {cov80:.1%}  (nominal 80%)   "
          f"mean width {width80:.3f} pp")
    print(f"  95% coverage:           {cov95:.1%}  (nominal 95%)   "
          f"mean width {width95:.3f} pp")
    print(f"  Directional acc:        {dir_hit:.1%}  "
          f"(base rate up: {dir_actual_up_rate:.1%})")
    print()
    print(f"Last 30 origins:")
    print(f"  RMSE:                   {rmse30:.4f} pp")
    print(f"  80% coverage:           {cov80_30:.1%}")
    print(f"  Directional acc:        {dir_hit30:.1%}")

    # Ship verdict
    calibrated = abs(cov80 - 0.80) < 0.07 and abs(cov95 - 0.95) < 0.04
    ship = "✅ SHIP" if calibrated and rmse_red > -10 else "❌ HOLD"
    print(f"\n{ship} — 80% calibration {'within ±7pp' if abs(cov80-0.80) < 0.07 else 'OFF'}, "
          f"RMSE {'competitive' if rmse_red > -10 else 'much worse than naive'}")
    print()

    # Findings markdown
    md = [
        "# Stefan Direct-Target Forecaster — Historical Backtest",
        "",
        f"**Date:** {date.today()}",
        f"**Method:** RidgeCV on 12 component index values + published-YoY lag → predict published YoY[T+1]",
        f"**Script:** `scripts/stefan_historical_backtest_direct.py`",
        f"**Window:** {df['origin'].min()} → {df['origin'].max()} ({len(df)} origins)",
        "",
        "## Headline metrics",
        "",
        "| Metric | Direct-target | Naive `y[T+1]=y[T]` |",
        "|---|---|---|",
        f"| RMSE | {rmse_model:.4f} pp | {rmse_naive:.4f} pp |",
        f"| MAE | {mae_model:.4f} pp | {mae_naive:.4f} pp |",
        f"| RMSE reduction vs naive | **{rmse_red:+.2f}%** | — |",
        f"| Directional accuracy | **{dir_hit:.1%}** | — (base rate up: {dir_actual_up_rate:.1%}) |",
        f"| 80% band coverage | **{cov80:.1%}** (nominal 80%) | — |",
        f"| 95% band coverage | **{cov95:.1%}** (nominal 95%) | — |",
        f"| Mean 80% band width | {width80:.4f} pp | — |",
        f"| Mean 95% band width | {width95:.4f} pp | — |",
        "",
        "## Last 30 origins",
        "",
        f"- RMSE: {rmse30:.4f} pp",
        f"- 80% coverage: {cov80_30:.1%}",
        f"- Directional accuracy: {dir_hit30:.1%}",
        "",
        "## Comparison to composite-based method",
        "",
        "- Composite method: 80% coverage was **2.4%** (bands 40× too narrow because per-component residuals missed the 0.3 pp composition drift vs published).",
        f"- Direct method: 80% coverage is **{cov80:.1%}** because the Ridge residuals are computed against the actual target series.",
        "",
        "## Ship verdict",
        "",
        "**" + ship + "**",
        "",
        "- 80% calibration: " + ("within ±7pp of nominal" if abs(cov80-0.80) < 0.07 else f"OFF ({cov80:.1%} vs 80%)"),
        "- 95% calibration: " + ("within ±4pp of nominal" if abs(cov95-0.95) < 0.04 else f"OFF ({cov95:.1%} vs 95%)"),
        f"- Point accuracy: RMSE {rmse_red:+.2f}% vs naive",
        "",
        "## Caveats",
        "",
        "- **Bootstrap bands assume iid residuals**. Residuals are daily, so some autocorrelation is likely. Block bootstrap would tighten this if we cared about formal CIs, but for band coverage the empirical resampling is adequate.",
        "- **Ridge alpha selected per-origin** — different origins may use different regularization strengths. Stable enough in practice (all origins fall back to one of 5 preset alphas).",
        "- **n=" + str(len(df)) + " is a modest sample.** Rolling 200+ days would tighten the coverage estimate.",
        "",
        "## Artifacts",
        "",
        "- `results/daily_forecast/historical_backtest_direct.csv` — per-origin predictions",
    ]
    (OUT_DIR / "historical_backtest_direct_findings.md").write_text("\n".join(md))
    print(f"Saved: {OUT_DIR / 'historical_backtest_direct_findings.md'}")


if __name__ == "__main__":
    main()
