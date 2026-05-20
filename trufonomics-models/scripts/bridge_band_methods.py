"""Compare band methods on the SAME-MONTH BRIDGE family — Fix #1b/c.

Holds point forecasts identical across runs (the band method does not
change the point) and reports calibration:

  * Gaussian (legacy)
  * In-sample conformal
  * Rolling-conformal (production-recommended)

Three forecasters tested:

  * SameMonthBridgeNowcaster      — 1 truf feature
  * MultiComponentBridgeNowcaster — 12 truf features (overfits on point)
  * CompressedMultiComponentBridge — PCA-3

For each, the comparison metric is **coverage at 80% / 95% nominal** —
ideally these sit closer to nominal under conformal than Gaussian.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation.harness import attach_actuals, score, walk_forward  # noqa: E402
from thales.models.same_month_nowcaster import (  # noqa: E402
    CompressedMultiComponentBridge,
    MultiComponentBridgeNowcaster,
    SameMonthBridgeNowcaster,
)
from thales.weights import build_crosswalk, get_top_level_weights  # noqa: E402
from thales import targets as T  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
KAIROS_PARQUET = Path("/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
TRUFL_HEADLINE_COL = "truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy"
OUT_DIR = ROOT / "results" / "baseline_eval"


def _load_panel() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Returns (panel_long, panel_short, component_cols).
    panel_long has 1 truf feature (long history); panel_short has 12.
    """
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        bls_yoy = T.load_target_yoy(store, "cpi", as_of=date.today())

    parq = pd.read_parquet(KAIROS_PARQUET)
    parq["date"] = pd.to_datetime(parq["date"])
    parq = parq.set_index("date").sort_index()
    truf_yoy = (parq[TRUFL_HEADLINE_COL].dropna()
                   .resample("ME").last())

    panel_long = pd.concat({
        "bls_yoy": bls_yoy,
        "truf_yoy": truf_yoy,
    }, axis=1).dropna()

    w_df = get_top_level_weights("2026-04-25")
    streams_df = pd.read_csv(ROOT / "data" / "truflation" / "streams_catalog.csv")
    cw = build_crosswalk(streams_df["raw_name"])
    cid_to_raw = {str(int(r.category_id)): r.raw_name
                    for _, r in cw[cw["category_id"].astype(int).isin(
                        w_df["category_id"].astype(int).tolist())].iterrows()}
    truf_components: dict[str, pd.Series] = {}
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        for cid, raw in cid_to_raw.items():
            s = store.get_vintage(raw, date.today()).dropna()
            monthly = s.resample("ME").last()
            yoy = ((monthly / monthly.shift(12) - 1.0) * 100.0).dropna()
            truf_components[f"truf_c{cid}"] = yoy

    panel_short = pd.concat({
        "bls_yoy": bls_yoy,
        "truf_yoy": truf_yoy,
        **truf_components,
    }, axis=1).dropna()
    component_cols = sorted([c for c in panel_short.columns
                                if c.startswith("truf_c")])

    return panel_long, panel_short, component_cols


def _eval(forecaster, panel: pd.DataFrame, label: str) -> dict:
    origins = panel.index[36:]
    forecasts = walk_forward(forecaster, panel, "bls_yoy",
                                  origins, horizon=0)
    today = panel["bls_yoy"].shift(1)
    df = attach_actuals(forecasts, panel["bls_yoy"], today_baseline=today)
    if df.empty:
        return {"label": label, "n": 0}
    block = score(df)
    return {
        "label": label,
        "n": block.n,
        "rmse": block.rmse,
        "cov80": block.cov80,
        "cov95": block.cov95,
        "width80": block.width80,
        "width95": block.width95,
        "dir_hit": block.dir_hit,
    }


def main() -> None:
    print("=" * 78)
    print("Bridge band-method comparison — Fix #1b/c")
    print("=" * 78)

    panel_long, panel_short, comp_cols = _load_panel()

    band_methods = ["gaussian", "in_sample", "rolling_conformal"]

    results: list[dict] = []

    # ── SameMonthBridgeNowcaster ────────────────────────────────────
    print("\n── SameMonthBridgeNowcaster (1 feat, long panel) ──────────")
    for bm in band_methods:
        fc = SameMonthBridgeNowcaster(
            train_window_months=36, train_min=12,
            band_method=bm, calib_months=24,
            model_id=f"same_month_bridge_{bm}")
        r = _eval(fc, panel_long, f"same_month_{bm}")
        results.append(r)
        print(f"  {bm:<22s}  cov80={r['cov80']:.1%}  cov95={r['cov95']:.1%}  "
                f"w80={r['width80']:.4f}  w95={r['width95']:.4f}  "
                f"RMSE={r['rmse']:.4f}")

    # ── MultiComponentBridgeNowcaster ────────────────────────────────
    print("\n── MultiComponentBridgeNowcaster (12 feat, short panel) ──")
    for bm in band_methods:
        fc = MultiComponentBridgeNowcaster(
            truf_component_cols=comp_cols,
            train_window_months=36, train_min=24, ridge_alpha=10.0,
            band_method=bm, calib_months=24,
            model_id=f"multi_raw_{bm}")
        r = _eval(fc, panel_short, f"multi_raw_{bm}")
        results.append(r)
        print(f"  {bm:<22s}  cov80={r['cov80']:.1%}  cov95={r['cov95']:.1%}  "
                f"w80={r['width80']:.4f}  w95={r['width95']:.4f}  "
                f"RMSE={r['rmse']:.4f}")

    # ── CompressedMultiComponentBridge (PCA-3) ────────────────────────
    print("\n── CompressedMultiComponentBridge PCA-3 (short panel) ────")
    for bm in band_methods:
        fc = CompressedMultiComponentBridge(
            truf_component_cols=comp_cols,
            feature_compression="pca", n_components=3,
            train_window_months=36, train_min=24,
            band_method=bm, calib_months=24,
            model_id=f"pca3_{bm}")
        r = _eval(fc, panel_short, f"pca3_{bm}")
        results.append(r)
        print(f"  {bm:<22s}  cov80={r['cov80']:.1%}  cov95={r['cov95']:.1%}  "
                f"w80={r['width80']:.4f}  w95={r['width95']:.4f}  "
                f"RMSE={r['rmse']:.4f}")

    # Persist
    out = OUT_DIR / "bridge_band_methods.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("Coverage summary — distance from nominal (smaller is better)")
    print("=" * 78)
    print(f"  {'model_band':<32s}  {'cov80 dev':>10s}  {'cov95 dev':>10s}  "
            f"{'width80':>9s}")
    print("  " + "-" * 70)
    for r in results:
        if r.get("n", 0) == 0:
            continue
        dev80 = r["cov80"] - 0.80
        dev95 = r["cov95"] - 0.95
        print(f"  {r['label']:<32s}  {dev80:+>10.1%}  {dev95:+>10.1%}  "
                f"{r['width80']:>9.4f}")


if __name__ == "__main__":
    main()
