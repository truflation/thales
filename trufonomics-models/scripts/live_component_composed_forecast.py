"""Component-composed day-ahead Truflation CPI YoY forecast.

This is the accounting-first version of the daily forecaster:

  * use the 80-stream catalog, but keep only the 58 weighted leaf streams
    whose v2 leaf weights sum to 100%;
  * compose component index levels into a headline index;
  * forecast tomorrow's component index levels with simple local rules;
  * convert the composed index forecast into YoY using the known year-ago
    composed index;
  * select the component candidate only if it beats headline persistence in
    recent validation.

The script can use the local vintage store or pull directly from TRUF Network.
The TRUF path is deliberately bounded with per-stream timeouts and a smoke
`--limit`; do not run a full 58-stream refresh casually from an interactive
session.

Usage:
    uv run python scripts/live_component_composed_forecast.py
    uv run python scripts/live_component_composed_forecast.py --component-source truf --limit 3 --smoke-only
    uv run python scripts/live_component_composed_forecast.py --component-source truf --start 2024-10-01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation import metrics as M  # noqa: E402
from thales.evaluation.conformal import conformal_band_offsets  # noqa: E402
from thales.ingest.truf_network import call_worker  # noqa: E402
from thales.ingest.truf_network import DEFAULT_GATEWAY, DEFAULT_PROVIDER  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402
from thales.weights import build_crosswalk  # noqa: E402


STREAMS_CSV = ROOT / "data" / "truflation" / "streams_catalog.csv"
WEIGHTS_DIR = ROOT / "data" / "truflation" / "weights"
VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "daily_forecast_components"

FEED_API_KEY = os.environ.get(
    "TRUFLATION_FEED_API_KEY", "egiZc8oSTWdHr7V6lwYZxYil1wF0ezgF")
FEED_URL = (
    "https://api.truflation.com/api/v1/feed/truflation/"
    "macro-data-us/truflation_us_cpi_yoy"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    method: Literal["headline", "component_persist", "component_median_return"]
    lookback_days: int = 0


CANDIDATES = (
    Candidate("headline_persistence", "headline"),
    Candidate("component_index_persistence", "component_persist"),
    Candidate("component_median_return_7d", "component_median_return", 7),
    Candidate("component_median_return_30d", "component_median_return", 30),
)


def active_leaf_streams(as_of: date) -> pd.DataFrame:
    """Return weighted leaf streams: raw_name, tn_stream_id, category_id, weight."""
    streams = pd.read_csv(STREAMS_CSV)
    crosswalk = build_crosswalk(streams["raw_name"])
    crosswalk = crosswalk.merge(streams, on="raw_name", how="left")

    weights_csv = "categories-tables-v2.csv" if as_of >= date(2026, 1, 1) else "categories-tables-v1.csv"
    weights = pd.read_csv(WEIGHTS_DIR / weights_csv)
    leaf_rows = weights[weights["source_id"].fillna(0) != 0].copy()

    # Multiple source rows feed one leaf category. For our 80 stream catalog,
    # the stream is the leaf aggregate, so sum source weights to leaf weight.
    leaf_weight = (
        leaf_rows.groupby("subcategory_id")["relative_importance"]
        .sum()
        .rename("weight")
    )
    active = crosswalk.join(leaf_weight, on="category_id")
    active = active[active["weight"].notna()].copy()
    active["weight"] = active["weight"].astype(float)
    active = active.sort_values("weight", ascending=False).reset_index(drop=True)
    return active[["raw_name", "humanized_name", "tn_stream_id",
                   "category_id", "category", "weight"]]


def load_live_headline() -> pd.Series:
    r = requests.get(FEED_URL, headers={"Authorization": FEED_API_KEY}, timeout=30)
    r.raise_for_status()
    body = r.json()
    s = pd.Series(
        body["truflation_us_cpi_yoy"],
        index=pd.to_datetime(body["index"]),
        name="live_headline_yoy",
    ).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_components_from_vintage(active: pd.DataFrame,
                                 as_of: date) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        for raw_name in active["raw_name"]:
            s = store.get_vintage(str(raw_name), as_of)
            if not s.empty:
                cols[str(raw_name)] = s
    return pd.DataFrame(cols).sort_index()


def load_components_from_truf(active: pd.DataFrame,
                              start_date: str,
                              timeout_per_stream: int,
                              gateway: str,
                              provider: str,
                              limit: int | None = None) -> pd.DataFrame:
    start_ts = int(datetime.fromisoformat(start_date).replace(
        tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.now(timezone.utc).timestamp())
    selected = active.head(limit) if limit else active

    cols: dict[str, pd.Series] = {}
    for i, row in enumerate(selected.itertuples(index=False), 1):
        raw_name = str(row.raw_name)
        print(f"  [{i:02d}/{len(selected):02d}] fetching {raw_name}", flush=True)
        try:
            records = call_worker(
                str(row.tn_stream_id),
                start_ts,
                end_ts,
                gateway=gateway,
                provider=provider,
                timeout_s=timeout_per_stream,
            )
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        if not records:
            print("    no records", flush=True)
            continue
        s = pd.Series(
            [v for _, v in records],
            index=pd.to_datetime([datetime.fromtimestamp(t, tz=timezone.utc).date()
                                  for t, _ in records]),
            name=raw_name,
            dtype=float,
        )
        cols[raw_name] = s[~s.index.duplicated(keep="last")].sort_index()
        print(f"    ok n={len(cols[raw_name])} "
              f"{cols[raw_name].index.min():%Y-%m-%d}->"
              f"{cols[raw_name].index.max():%Y-%m-%d}", flush=True)
    return pd.DataFrame(cols).sort_index()


def complete_component_panel(panel: pd.DataFrame,
                             active: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    present = [c for c in active["raw_name"] if c in panel.columns]
    used = active[active["raw_name"].isin(present)].copy()
    panel = panel[present].copy()
    return panel, used


def composite_index(panel: pd.DataFrame, used: pd.DataFrame) -> pd.Series:
    weights = used.set_index("raw_name")["weight"].astype(float)
    aligned = panel[weights.index].dropna(how="any")
    comp = aligned.mul(weights, axis=1).sum(axis=1) / weights.sum()
    comp.name = "component_composite_index"
    return comp


def _component_prediction(panel: pd.DataFrame,
                          origin: pd.Timestamp,
                          candidate: Candidate) -> pd.Series:
    row = panel.loc[origin]
    if candidate.method == "component_persist":
        return row.astype(float)
    if candidate.method == "component_median_return":
        hist = panel.loc[:origin].astype(float)
        log_ret = np.log(hist / hist.shift(1))
        drift = log_ret.tail(candidate.lookback_days).median()
        return row * np.exp(drift.fillna(0.0))
    raise ValueError(f"candidate {candidate.name} is not component-based")


def component_yoy_forecast(panel: pd.DataFrame,
                           used: pd.DataFrame,
                           composite: pd.Series,
                           origin: pd.Timestamp,
                           candidate: Candidate) -> float | None:
    target = origin + pd.Timedelta(days=1)
    denom_date = target - pd.Timedelta(days=365)
    if denom_date not in composite.index:
        return None
    weights = used.set_index("raw_name")["weight"].astype(float)
    pred_components = _component_prediction(panel[weights.index], origin, candidate)
    pred_index = float((pred_components * weights).sum() / weights.sum())
    denom = float(composite.loc[denom_date])
    if not np.isfinite(denom) or denom == 0:
        return None
    return (pred_index / denom - 1.0) * 100.0


def prediction_frame(panel: pd.DataFrame,
                     used: pd.DataFrame,
                     headline: pd.Series,
                     start: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    comp = composite_index(panel, used)
    rows: list[dict] = []
    first_origin = max(start, panel.index.min() + pd.Timedelta(days=365))
    for origin in panel.loc[first_origin:].index:
        target = origin + pd.Timedelta(days=1)
        if origin not in headline.index or target not in headline.index:
            continue
        if panel.loc[origin, used["raw_name"]].isna().any():
            continue
        row = {
            "origin": origin,
            "target": target,
            "today": float(headline.loc[origin]),
            "actual": float(headline.loc[target]),
            "headline_persistence": float(headline.loc[origin]),
        }
        for cand in CANDIDATES:
            if cand.method == "headline":
                continue
            pred = component_yoy_forecast(panel, used, comp, origin, cand)
            row[cand.name] = pred
        rows.append(row)
    return pd.DataFrame(rows), comp


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_rmse = M.rmse(df["headline_persistence"].values, df["actual"].values)
    for cand in CANDIDATES:
        if cand.name not in df.columns:
            continue
        sub = df.dropna(subset=[cand.name])
        if sub.empty:
            continue
        pred = sub[cand.name].values
        actual = sub["actual"].values
        rmse = M.rmse(pred, actual)
        rows.append({
            "model": cand.name,
            "n": int(len(sub)),
            "rmse": float(rmse),
            "mae": float(M.mae(pred, actual)),
            "rmse_reduction_vs_headline_persistence_pct":
                float(100 * (1.0 - rmse / base_rmse)) if base_rmse > 0 else np.nan,
            "directional_accuracy": float(
                M.directional_accuracy(pred, actual, reference=sub["today"].values)),
        })
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def select_model(summary: pd.DataFrame, min_edge_pct: float) -> str:
    best = summary.iloc[0]
    if (best["model"] != "headline_persistence" and
            float(best["rmse_reduction_vs_headline_persistence_pct"]) >= min_edge_pct):
        return str(best["model"])
    return "headline_persistence"


def forecast_latest(panel: pd.DataFrame,
                    used: pd.DataFrame,
                    headline: pd.Series,
                    comp: pd.Series,
                    selected: str,
                    pred_df: pd.DataFrame,
                    calib_days: int) -> dict:
    headline_origin = headline.index.max()
    component_origin = panel.dropna(subset=used["raw_name"]).index.max()
    origin = min(headline_origin, component_origin)
    target = origin + pd.Timedelta(days=1)

    if selected == "headline_persistence":
        point = float(headline.loc[origin])
    else:
        cand = next(c for c in CANDIDATES if c.name == selected)
        maybe = component_yoy_forecast(panel, used, comp, origin, cand)
        point = float(maybe) if maybe is not None else float(headline.loc[origin])
        if maybe is None:
            selected = "headline_persistence"

    calib = pred_df.dropna(subset=[selected]).tail(calib_days)
    if len(calib) < 20:
        raise RuntimeError(f"Need at least 20 calibration rows; got {len(calib)}")
    errors = calib["actual"].values - calib[selected].values
    lo80, hi80 = conformal_band_offsets(errors, alpha=0.20)
    lo95, hi95 = conformal_band_offsets(errors, alpha=0.05)

    return {
        "origin_date": str(origin.date()),
        "target_date": str(target.date()),
        "selected_model": selected,
        "point_yoy_pct": point,
        "today_headline_yoy_pct": float(headline.loc[origin]),
        "component_origin_date": str(component_origin.date()),
        "headline_origin_date": str(headline_origin.date()),
        "component_data_stale_days_vs_headline": int((headline_origin - component_origin).days),
        "band_80": [point + lo80, point + hi80],
        "band_95": [point + lo95, point + hi95],
        "calibration_n": int(len(calib)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component-source", choices=("vintage", "truf"),
                    default="vintage")
    ap.add_argument("--start", default="2024-10-01",
                    help="TRUF fetch start date and earliest validation context")
    ap.add_argument("--backtest-start", default="2025-01-01")
    ap.add_argument("--selection-days", type=int, default=180)
    ap.add_argument("--calib-days", type=int, default=90)
    ap.add_argument("--min-edge-pct", type=float, default=2.0)
    ap.add_argument("--timeout-per-stream", type=int, default=25)
    ap.add_argument("--gateway", default=DEFAULT_GATEWAY)
    ap.add_argument("--provider", default=DEFAULT_PROVIDER)
    ap.add_argument("--limit", type=int, default=None,
                    help="Fetch/use only the largest N weighted streams; smoke only")
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    as_of = date.today()
    active = active_leaf_streams(as_of)
    if args.limit:
        active = active.head(args.limit).copy()

    print("Active weighted component streams")
    print(f"  n={len(active)}  weight_sum={active['weight'].sum():.3f}%")
    if args.limit:
        print("  WARNING: --limit is for smoke checks only; not production composition.")

    print(f"\nLoading components via {args.component_source}...")
    if args.component_source == "vintage":
        panel_raw = load_components_from_vintage(active, as_of)
    else:
        panel_raw = load_components_from_truf(
            active, args.start, args.timeout_per_stream,
            gateway=args.gateway, provider=args.provider, limit=args.limit)

    panel, used = complete_component_panel(panel_raw, active)
    print(f"  loaded columns={panel.shape[1]}/{len(active)}  "
          f"used_weight={used['weight'].sum():.3f}%")
    if panel.empty and args.smoke_only:
        print("\nSmoke-only requested and no component data loaded. "
              "This confirms the fetch path failed before composition.")
        return
    if panel.empty:
        raise RuntimeError("No component data loaded")
    print(f"  component range {panel.index.min():%Y-%m-%d} -> {panel.index.max():%Y-%m-%d}")

    if args.smoke_only:
        print("\nSmoke-only requested; stopping before composition/backtest.")
        return
    if used["weight"].sum() < 99.0:
        raise RuntimeError(
            f"Loaded component weight coverage is only {used['weight'].sum():.2f}%; "
            "need near-100% for headline composition.")

    print("\nLoading live headline...")
    headline = load_live_headline()
    print(f"  headline range {headline.index.min():%Y-%m-%d} -> "
          f"{headline.index.max():%Y-%m-%d} latest={headline.iloc[-1]:.6f}%")

    pred_df, comp = prediction_frame(
        panel, used, headline, pd.Timestamp(args.backtest_start))
    if pred_df.empty:
        raise RuntimeError("No scored component predictions produced")

    select_df = pred_df.tail(args.selection_days)
    summary = summarize(select_df)
    selected = select_model(summary, args.min_edge_pct)
    forecast = forecast_latest(
        panel, used, headline, comp, selected, pred_df, args.calib_days)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUT_DIR / "component_candidate_backtest.csv"
    summary_path = OUT_DIR / "component_candidate_summary.csv"
    json_path = OUT_DIR / f"component_forecast_{forecast['origin_date']}.json"
    pred_df.to_csv(pred_path, index=False)
    summary.to_csv(summary_path, index=False)

    payload = {
        "component_source": args.component_source,
        "active_streams": int(len(active)),
        "used_streams": int(len(used)),
        "used_weight_sum": float(used["weight"].sum()),
        "candidate_summary": summary.to_dict(orient="records"),
        "selected_model": selected,
        "forecast": forecast,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    print("\nCandidate validation")
    print(f"  window {select_df['origin'].min():%Y-%m-%d} -> "
          f"{select_df['origin'].max():%Y-%m-%d} n={len(select_df)}")
    for _, r in summary.iterrows():
        print(f"  {r['model']:<32s} RMSE={r['rmse']:.5f} "
              f"MAE={r['mae']:.5f} "
              f"vs_persist={r['rmse_reduction_vs_headline_persistence_pct']:+.2f}% "
              f"dir={r['directional_accuracy']:.1%}")

    print("\nForecast")
    print(f"  selected: {forecast['selected_model']}")
    print(f"  {forecast['origin_date']} -> {forecast['target_date']}: "
          f"{forecast['point_yoy_pct']:.6f}%")
    print(f"  80% band: [{forecast['band_80'][0]:.6f}%, "
          f"{forecast['band_80'][1]:.6f}%]")
    print(f"  95% band: [{forecast['band_95'][0]:.6f}%, "
          f"{forecast['band_95'][1]:.6f}%]")
    if forecast["component_data_stale_days_vs_headline"] > 0:
        print(f"  WARNING: component data lags headline by "
              f"{forecast['component_data_stale_days_vs_headline']} day(s); "
              "not a live production forecast.")

    print(f"\nSaved: {pred_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
