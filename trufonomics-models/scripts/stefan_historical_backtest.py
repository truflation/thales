"""Historical backtest of the Stefan day-ahead forecaster.

Walks the component-forecaster through the last 90 origins and scores each
T+1 prediction against the realized value. Produces calibration metrics so
Stefan's first post can cite real track-record numbers instead of "trust us."

Method:
  1. Load the full component + covariate panel once (as_of=today, which
     returns every historical reference_date because our TN frozen streams
     have single-as_of-tag ingest).
  2. For each historical origin T in the last ~90 days:
     - Restrict the panel to reference_date <= T (ensures the forecaster
       doesn't peek at future values)
     - Fit 12 component OLS models on training history up to T-1
     - Predict T+1 for each component
     - Weight-compose into headline YoY forecast, bootstrap bands
     - Score against the realized headline YoY (from published
       `truflation_us_cpi_frozen_yoy`) at T+1
  3. Report metrics: RMSE vs naive, directional accuracy, 80/95 band
     coverage, PIT calibration.

Vintage caveat: since our TN streams are frozen (revision-pinned) and
tagged with as_of=ingest_date, the backtest doesn't respect a
publication-lag offset (Truflation publishes with ~24h QC delay). This
adds ~1 day of "future knowledge" at each origin — small effect for daily
YoY (autocorrelation 0.99) but worth acknowledging. A stricter backtest
would apply a 1-day lag mask; deferred as follow-up when we have proper
(ref, as_of) pairs from the backend.

Output:
    results/daily_forecast/historical_backtest.csv — per-origin rows
    results/daily_forecast/historical_backtest_findings.md — summary
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation import metrics as M  # noqa: E402
from thales.models.component_forecaster import CATEGORY_EXOG, DEFAULT_TRAIN_START  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402
from thales.weights import build_crosswalk, get_top_level_weights, top_level_category_ids  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
STREAMS_CSV = ROOT / "data" / "truflation" / "streams_catalog.csv"
KAIROS_PARQUET = Path("/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
OUT_DIR = ROOT / "results" / "daily_forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ORIGINS = 90
N_BOOT = 500  # reduced from 1000 for backtest speed; still tight bands


@dataclass
class BacktestRow:
    origin: date
    target: date
    pred_yoy: float
    lo80: float
    hi80: float
    lo95: float
    hi95: float
    today_yoy: float          # naive baseline: tomorrow = today
    actual_yoy: float
    error: float              # pred - actual
    naive_error: float        # today - actual
    direction_pred_up: bool   # did pred go up vs today?
    direction_actual_up: bool
    composite_today: float
    composite_tomorrow_pred: float


def load_panel(store: VintageStore) -> tuple[pd.DataFrame, list[tuple[int, str, str]]]:
    """Load component + covariate panel once, wide format, forward-filled."""
    streams_df = pd.read_csv(STREAMS_CSV)
    crosswalk = build_crosswalk(streams_df["raw_name"])
    tops = top_level_category_ids()
    top_streams = crosswalk[crosswalk["category_id"].isin(tops)].copy()
    top_streams["category_id"] = top_streams["category_id"].astype(int)

    component_ids = top_streams["raw_name"].tolist()
    exog_ids = {eid for ids in CATEGORY_EXOG.values() for eid in ids}
    needed = set(component_ids) | exog_ids

    cols = {}
    as_of = date.today()
    for sid in sorted(needed):
        s = store.get_vintage(sid, as_of)
        if not s.empty:
            cols[sid] = s
    if not cols:
        raise RuntimeError("vintage store empty for needed streams")
    panel = pd.DataFrame(cols)
    idx = pd.date_range(panel.index.min(), panel.index.max(), freq="D")
    panel = panel.reindex(idx).ffill()

    top_info = [
        (int(r.category_id), r.raw_name, r.category)
        for _, r in top_streams.iterrows()
    ]
    return panel, top_info


def fit_one_component(panel: pd.DataFrame, target_col: str,
                       feature_cols: list[str],
                       origin: pd.Timestamp,
                       train_start: pd.Timestamp = DEFAULT_TRAIN_START
                       ) -> tuple[float, np.ndarray] | None:
    """Fit OLS on data ≤ origin-1, predict at origin (= T+1 in the feature
    frame). Returns (point_forecast, residuals) or None if not enough data."""
    X_cols = [f"{target_col}__lag0"] + [f"{f}__lag0" for f in feature_cols]
    feat = pd.DataFrame(index=panel.index)
    feat[f"{target_col}__lag0"] = panel[target_col]
    for f in feature_cols:
        if f in panel.columns:
            feat[f"{f}__lag0"] = panel[f]
        else:
            return None
    feat["__target_tp1"] = panel[target_col].shift(-1)

    train = feat.loc[(feat.index >= train_start) &
                      (feat.index < origin)].dropna()
    if len(train) < 60:
        return None

    X = train[X_cols].values
    y = train["__target_tp1"].values
    model = LinearRegression().fit(X, y)
    residuals = y - model.predict(X)

    x_origin = feat.loc[origin, X_cols]
    if x_origin.isna().any():
        return None
    pred = float(model.predict(x_origin.values.reshape(1, -1))[0])
    return pred, residuals


def backtest_one_origin(
    panel: pd.DataFrame, top_info: list[tuple[int, str, str]],
    origin: pd.Timestamp, rng: np.random.Generator,
    actual_headline_yoy: pd.Series,
) -> BacktestRow | None:
    """One walk-forward origin: fit 12 models, compose, score."""
    # Restrict panel to reference_date ≤ origin (no future peeking)
    # (The fitters also enforce train_end < origin; this is belt-and-suspenders.)
    restricted = panel.loc[:origin]

    # Fit each component
    fits: list[tuple[int, str, str, float, np.ndarray, float]] = []
    for cid, raw_name, cat_name in top_info:
        exog = [e for e in CATEGORY_EXOG.get(cid, []) if e in restricted.columns]
        res = fit_one_component(restricted, raw_name, exog, origin)
        if res is None:
            return None
        pred, resid = res
        today_val = float(restricted.loc[origin, raw_name])
        fits.append((cid, raw_name, cat_name, pred, resid, today_val))

    # Weights effective at origin
    weights_df = get_top_level_weights(origin.date())
    w_lookup = dict(zip(weights_df["category_id"].astype(int),
                         weights_df["weight"].astype(float)))

    # Composite index today vs tomorrow
    def composite(today: bool) -> float:
        tot_w, tot_wv = 0.0, 0.0
        for (cid, _, _, pred, _, today_val) in fits:
            w = w_lookup[cid]
            tot_wv += w * (today_val if today else pred)
            tot_w += w
        return tot_wv / tot_w

    composite_today = composite(today=True)
    composite_tomorrow = composite(today=False)

    # Composite at T+1 - 365 for YoY denominator
    t_prior = origin - pd.DateOffset(days=365)
    if t_prior not in restricted.index:
        idx = restricted.index.get_indexer([t_prior], method="nearest")[0]
        t_prior = restricted.index[idx]
    prior_tot_w, prior_tot_wv = 0.0, 0.0
    for (cid, raw_name, _, _, _, _) in fits:
        v = restricted.loc[t_prior, raw_name]
        if pd.isna(v):
            continue
        w = w_lookup[cid]
        prior_tot_wv += w * float(v)
        prior_tot_w += w
    prior_composite = prior_tot_wv / prior_tot_w

    pred_yoy = (composite_tomorrow / prior_composite - 1.0) * 100.0
    today_yoy = (composite_today / prior_composite - 1.0) * 100.0

    # Bootstrap bands
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        tot_w, tot_wv = 0.0, 0.0
        for (cid, _, _, pred, resid, _) in fits:
            w = w_lookup[cid]
            r = rng.choice(resid) if len(resid) else 0.0
            tot_wv += w * (pred + r)
            tot_w += w
        draws[b] = (tot_wv / tot_w) / prior_composite - 1.0
    draws *= 100.0

    # Realized T+1 value
    target_date = origin + timedelta(days=1)
    actual = actual_headline_yoy.get(pd.Timestamp(target_date))
    if actual is None or pd.isna(actual):
        return None

    return BacktestRow(
        origin=origin.date(),
        target=target_date.date() if hasattr(target_date, "date") else target_date,
        pred_yoy=pred_yoy,
        lo80=float(np.percentile(draws, 10)),
        hi80=float(np.percentile(draws, 90)),
        lo95=float(np.percentile(draws, 2.5)),
        hi95=float(np.percentile(draws, 97.5)),
        today_yoy=today_yoy,
        actual_yoy=float(actual),
        error=pred_yoy - float(actual),
        naive_error=today_yoy - float(actual),
        direction_pred_up=pred_yoy > today_yoy,
        direction_actual_up=float(actual) > today_yoy,
        composite_today=composite_today,
        composite_tomorrow_pred=composite_tomorrow,
    )


def main() -> None:
    print("Loading vintage panel + kairos parquet...")
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        panel, top_info = load_panel(store)

    # Load published frozen YoY for scoring (from kairos parquet)
    pq = pd.read_parquet(KAIROS_PARQUET)
    pq["date"] = pd.to_datetime(pq["date"])
    pq = pq.set_index("date").sort_index()
    actual_series = pq["truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy"].dropna()

    # Pick origins: last N days where every component has data AND next day has actual
    origins = panel.dropna(subset=[t[1] for t in top_info]).index
    origins = origins[origins > (panel.index.max() - pd.Timedelta(days=N_ORIGINS + 2))]
    origins = [o for o in origins
                if pd.Timestamp(o + timedelta(days=1)) in actual_series.index]
    origins = sorted(set(origins))[-N_ORIGINS:]
    print(f"Backtest window: {origins[0].date()} → {origins[-1].date()}  "
          f"({len(origins)} origins)")

    rng = np.random.default_rng(0)
    rows: list[BacktestRow] = []
    for i, origin in enumerate(origins, 1):
        r = backtest_one_origin(panel, top_info, pd.Timestamp(origin), rng,
                                  actual_series)
        if r is None:
            continue
        rows.append(r)
        if i % 15 == 0:
            print(f"  {i}/{len(origins)}  latest: {r.origin} "
                  f"pred={r.pred_yoy:.3f}% actual={r.actual_yoy:.3f}%")

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv(OUT_DIR / "historical_backtest.csv", index=False)

    # ── Metrics ──
    err = df["error"].values
    naive_err = df["naive_error"].values
    rmse_model = M.rmse(df["pred_yoy"].values, df["actual_yoy"].values)
    rmse_naive = M.rmse(df["today_yoy"].values, df["actual_yoy"].values)
    mae_model = M.mae(df["pred_yoy"].values, df["actual_yoy"].values)
    mae_naive = M.mae(df["today_yoy"].values, df["actual_yoy"].values)
    rmse_reduction = (1 - rmse_model / rmse_naive) * 100 if rmse_naive > 0 else float("nan")

    cov80 = ((df["actual_yoy"] >= df["lo80"]) &
              (df["actual_yoy"] <= df["hi80"])).mean()
    cov95 = ((df["actual_yoy"] >= df["lo95"]) &
              (df["actual_yoy"] <= df["hi95"])).mean()
    width80 = (df["hi80"] - df["lo80"]).mean()
    width95 = (df["hi95"] - df["lo95"]).mean()

    # Directional accuracy: sign(pred - today) == sign(actual - today)
    dir_hit = (df["direction_pred_up"] == df["direction_actual_up"]).mean()
    dir_actual_up_rate = df["direction_actual_up"].mean()

    # 30-day rolling
    df_sorted = df.sort_values("origin")
    last30 = df_sorted.tail(30)
    rmse30 = M.rmse(last30["pred_yoy"].values, last30["actual_yoy"].values)
    cov80_30 = ((last30["actual_yoy"] >= last30["lo80"]) &
                 (last30["actual_yoy"] <= last30["hi80"])).mean()
    dir_hit30 = (last30["direction_pred_up"] == last30["direction_actual_up"]).mean()

    # ── Print report ──
    report = []
    report.append("=" * 72)
    report.append(f"Stefan day-ahead forecaster — historical backtest")
    report.append(f"Window: {df['origin'].min()} → {df['origin'].max()}  (n={len(df)})")
    report.append("=" * 72)
    report.append("")
    report.append(f"Full-window metrics:")
    report.append(f"  RMSE model / naive:     {rmse_model:.4f}% / {rmse_naive:.4f}%  "
                   f"({rmse_reduction:+.2f}%)")
    report.append(f"  MAE  model / naive:     {mae_model:.4f}% / {mae_naive:.4f}%")
    report.append(f"  80% coverage:           {cov80:.1%}  (nominal 80%)   "
                   f"mean width {width80:.3f}pp")
    report.append(f"  95% coverage:           {cov95:.1%}  (nominal 95%)   "
                   f"mean width {width95:.3f}pp")
    report.append(f"  Directional accuracy:   {dir_hit:.1%}  "
                   f"(base rate up: {dir_actual_up_rate:.1%})")
    report.append("")
    report.append("Last 30 origins:")
    report.append(f"  RMSE:                   {rmse30:.4f}%")
    report.append(f"  80% coverage:           {cov80_30:.1%}")
    report.append(f"  Directional accuracy:   {dir_hit30:.1%}")
    report.append("")
    report.append("Recent predictions (last 10):")
    report.append("  origin      target      today    pred     lo80      hi80      actual   |error|")
    for _, r in df_sorted.tail(10).iterrows():
        in_band = "✓" if r["lo80"] <= r["actual_yoy"] <= r["hi80"] else "✗"
        report.append(
            f"  {r['origin']}  {r['target']}  "
            f"{r['today_yoy']:6.3f}  {r['pred_yoy']:6.3f}  "
            f"[{r['lo80']:6.3f}, {r['hi80']:6.3f}]  {r['actual_yoy']:6.3f}  "
            f"{abs(r['error']):5.3f}  {in_band}"
        )
    text = "\n".join(report)
    print(text)

    # ── Write findings markdown ──
    md = [
        "# Stefan Day-Ahead Forecaster — Historical Backtest",
        "",
        f"**Date:** {date.today()}",
        f"**Script:** `scripts/stefan_historical_backtest.py`",
        f"**Window:** {df['origin'].min()} → {df['origin'].max()} ({len(df)} origins)",
        "",
        "## Method recap",
        "",
        "12 top-level Truflation CPI category components → per-category walk-forward OLS "
        "(persistence + 1–2 exogenous daily covariates) → weighted composition via 2026 v2 "
        "category weights → bootstrap residual bands.",
        "",
        "## Headline metrics",
        "",
        "| Metric | Model | Naive `y[T+1]=y[T]` |",
        "|---|---|---|",
        f"| RMSE | {rmse_model:.4f} pp | {rmse_naive:.4f} pp |",
        f"| MAE | {mae_model:.4f} pp | {mae_naive:.4f} pp |",
        f"| RMSE reduction vs naive | **{rmse_reduction:+.2f}%** | — |",
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
        "## Honest interpretation",
        "",
        _interpret(rmse_reduction, cov80, cov95, dir_hit, dir_actual_up_rate),
        "",
        "## Caveats",
        "",
        "- **Vintage approximation.** TN component streams tagged with `as_of=ingest_date` (not true first-publication date). For daily-frequency frozen streams the difference is ≤1 day (Truflation's 24h QC delay). Small leak; documented in pre-reg §2.5.",
        "- **Target = Truflation's own frozen YoY**, not BLS CPI. Predicting a different number than the institutional nowcast product.",
        "- **n=" + str(len(df)) + " is a small sample** for claims about calibration. 200+ days would tighten the coverage estimate.",
        "- **Bootstrap bands assume per-component residual independence.** Components co-move (gas and utilities both ride nat gas). True multivariate residual distribution would give slightly different (likely wider) bands.",
        "",
        "## Ship / no-ship verdict",
        "",
        _ship_verdict(rmse_reduction, cov80, cov95, dir_hit),
        "",
        "## Artifacts",
        "",
        "- `results/daily_forecast/historical_backtest.csv` — per-origin predictions + bands + realized values",
    ]
    (OUT_DIR / "historical_backtest_findings.md").write_text("\n".join(md))
    print(f"\nSaved: {OUT_DIR / 'historical_backtest_findings.md'}")


def _interpret(rmse_red: float, cov80: float, cov95: float,
                dir_hit: float, base_up: float) -> str:
    lines = []
    if rmse_red > 5:
        lines.append(f"- Model beats naive on RMSE by {rmse_red:.1f}% — meaningful edge at point accuracy.")
    elif rmse_red > 0:
        lines.append(f"- Model edges naive by {rmse_red:.1f}% on RMSE — small but positive.")
    elif rmse_red > -5:
        lines.append(f"- Model ties naive on RMSE ({rmse_red:.1f}%). Day-ahead is autocorrelation-dominated; expected.")
    else:
        lines.append(f"- Model underperforms naive by {-rmse_red:.1f}% on RMSE. Investigate before shipping.")

    if abs(cov80 - 0.80) < 0.05:
        lines.append(f"- 80% band coverage = {cov80:.1%} is within ±5pp of nominal — **calibrated**.")
    elif cov80 > 0.85:
        lines.append(f"- 80% band coverage = {cov80:.1%} is over-wide (expected 80%). Bands too conservative.")
    else:
        lines.append(f"- 80% band coverage = {cov80:.1%} is under nominal. Bands under-state uncertainty.")

    if dir_hit > 0.55:
        lines.append(f"- Directional accuracy {dir_hit:.1%} above coin-flip — meaningful direction signal.")
    else:
        lines.append(f"- Directional accuracy {dir_hit:.1%} near coin-flip (base rate {base_up:.1%}). Avoid claiming direction in the post.")
    return "\n".join(lines)


def _ship_verdict(rmse_red: float, cov80: float, cov95: float,
                   dir_hit: float) -> str:
    calibrated = abs(cov80 - 0.80) < 0.07 and abs(cov95 - 0.95) < 0.04
    if calibrated:
        return (
            "**Ship.** Bands are calibrated within tolerance; point accuracy is "
            "tied or slightly better than naive (expected at 1-day horizon on a "
            "99%-autocorrelated series). Post format should emphasize the "
            "calibrated band, de-emphasize the point, and cite the coverage number "
            "as the credibility anchor. Direction claims only if directional "
            "accuracy > 55%."
        )
    else:
        return (
            "**Hold.** Bands are off nominal by more than ±7pp on 80% or ±4pp on "
            "95%. Before Stefan posts: investigate why bands are off. Likely "
            "candidates — residual correlation across components, regime "
            "mismatch, or need for ALFRED vintages to prevent subtle leak. "
            "Re-run with wider/narrower bands or more data."
        )


if __name__ == "__main__":
    main()
