"""Compressed multi-component bridge — Fix #2 evaluation.

Compares six bridge variants on the same SAME-MONTH NOWCAST window where
the 12 per-component Truflation series are available (panel_short).

Variants:

  1. last_release_v1                  — naive floor (BLS_yoy[T] = BLS_yoy[T-1])
  2. same_month_bridge_v1             — α + β·BLS_lag + γ·truf_yoy[T] (1 feature)
  3. multi_raw_v1                     — α + β·BLS_lag + Σ γ_r·truf_r[T] (12 features, OVERFIT)
  4. compressed_pca_3                 — top-3 principal components (3 features)
  5. compressed_pls_3                 — top-3 PLS directions (3 features)
  6. compressed_grouped_5             — 5 weight-grouped macro-buckets (5 features)

All evaluated on identical n=26 origins (2024-01-31 → 2026-03-31). The
question: does compression recover the predictive signal lost to
overfitting in the 12-feature raw multi?
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
    LastReleaseBaseline,
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Five economic macro-buckets — composition the user could explain to a
# customer in one sentence each. Weights summed from each bucket's
# top-level Truflation weights at 2026-04-25.
#
#   energy_transport  = utilities + transport               (25.7%)
#   housing           = housing                              (23.1%)
#   sticky_services   = health + education + comms + rec    (19.9%)
#   goods_flexible    = food + clothing + household_durables (26.0%)
#   other             = alcohol + other                      (5.2%)
ECON_GROUPS_BY_CID: dict[str, list[str]] = {
    "energy_transport": ["81", "80"],
    "housing":          ["79"],
    "sticky_services":  ["82", "87", "86", "88"],
    "goods_flexible":   ["78", "85", "83"],
    "other":            ["84", "89"],
}


def main() -> None:
    print("=" * 78)
    print("Compressed multi-component bridge — Fix #2 comparison")
    print("=" * 78)

    # ── Load data ────────────────────────────────────────────────────────
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        bls_yoy = T.load_target_yoy(store, "cpi", as_of=date.today())

    parq = pd.read_parquet(KAIROS_PARQUET)
    parq["date"] = pd.to_datetime(parq["date"])
    parq = parq.set_index("date").sort_index()
    truf_yoy = (parq[TRUFL_HEADLINE_COL].dropna()
                   .resample("ME").last())

    # 12 top-level component YoYs
    w_df = get_top_level_weights("2026-04-25")
    streams_df = pd.read_csv(ROOT / "data" / "truflation" / "streams_catalog.csv")
    cw = build_crosswalk(streams_df["raw_name"])
    cid_to_raw = {str(int(r.category_id)): r.raw_name
                    for _, r in cw[cw["category_id"].astype(int).isin(
                        w_df["category_id"].astype(int).tolist())].iterrows()}
    cid_to_weight = {str(int(r.category_id)): float(r.weight)
                       for _, r in w_df.iterrows()}

    truf_components: dict[str, pd.Series] = {}
    component_weights: dict[str, float] = {}
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        for cid, raw in cid_to_raw.items():
            s = store.get_vintage(raw, date.today()).dropna()
            monthly = s.resample("ME").last()
            yoy = ((monthly / monthly.shift(12) - 1.0) * 100.0).dropna()
            col = f"truf_c{cid}"
            truf_components[col] = yoy
            component_weights[col] = cid_to_weight.get(cid, 0.0)

    # Build short panel with 12 components
    panel = pd.concat({
        "bls_yoy": bls_yoy,
        "truf_yoy": truf_yoy,
        **truf_components,
    }, axis=1).dropna()

    component_cols = sorted([c for c in panel.columns if c.startswith("truf_c")])
    print(f"\nPanel: n={len(panel)}  range "
          f"{panel.index.min():%Y-%m-%d} → {panel.index.max():%Y-%m-%d}")
    print(f"Components: {len(component_cols)}  → {component_cols}")

    # Origins: skip first 36 obs to leave training history
    origins = panel.index[36:]
    print(f"Origins: {len(origins)}  "
          f"({origins[0]:%Y-%m} → {origins[-1]:%Y-%m})")

    today_baseline = panel["bls_yoy"].shift(1)

    # Build grouped-by-cid → grouped-by-col mapping using ECON_GROUPS_BY_CID
    econ_groups: dict[str, list[str]] = {}
    for g, cids in ECON_GROUPS_BY_CID.items():
        cols = [f"truf_c{c}" for c in cids if f"truf_c{c}" in component_cols]
        if cols:
            econ_groups[g] = cols

    print("\nGrouping (5 macro-buckets):")
    for g, cols in econ_groups.items():
        wt = sum(component_weights[c] for c in cols)
        print(f"  {g:<18s}  {wt:>5.2f}%   {cols}")

    # ── Forecasters ─────────────────────────────────────────────────────
    forecasters = [
        ("last_release_v1",
            LastReleaseBaseline(target_col="bls_yoy")),
        ("same_month_bridge_v1",
            SameMonthBridgeNowcaster(train_window_months=36)),
        ("multi_raw_v1",
            MultiComponentBridgeNowcaster(
                truf_component_cols=component_cols,
                train_window_months=36,
                ridge_alpha=10.0)),
        ("compressed_pca_3",
            CompressedMultiComponentBridge(
                truf_component_cols=component_cols,
                feature_compression="pca", n_components=3,
                train_window_months=36, train_min=24)),
        ("compressed_pca_5",
            CompressedMultiComponentBridge(
                truf_component_cols=component_cols,
                feature_compression="pca", n_components=5,
                train_window_months=36, train_min=24)),
        ("compressed_pls_3",
            CompressedMultiComponentBridge(
                truf_component_cols=component_cols,
                feature_compression="pls", n_components=3,
                train_window_months=36, train_min=24,
                model_id="compressed_pls_3")),
        ("compressed_grouped_5",
            CompressedMultiComponentBridge(
                truf_component_cols=component_cols,
                feature_compression="grouped",
                component_groups=econ_groups,
                component_weights=component_weights,
                train_window_months=36, train_min=24,
                model_id="compressed_grouped_5")),
    ]

    # ── Run + score ──────────────────────────────────────────────────────
    # Same-month nowcast frame ⇒ target = origin ⇒ horizon=0 in the harness.
    # (Earlier version called horizon=1 then mutated f.target — that silently
    # dropped the last origin. Use horizon=0 directly so n=26 is honest.)
    blocks = {}
    rows_for_csv = []
    n_features_by_model: dict[str, int] = {}
    for name, fc in forecasters:
        forecasts = walk_forward(fc, panel, "bls_yoy", origins, horizon=0)
        if forecasts and forecasts[0].metadata:
            n_features_by_model[name] = int(
                forecasts[0].metadata.get("n_features", 0)) or 0
        df = attach_actuals(forecasts, panel["bls_yoy"],
                                today_baseline=today_baseline)
        if df.empty:
            print(f"\n  [{name}] no scored rows — skipping")
            continue
        block = score(df)
        blocks[name] = block
        df["model_id"] = name
        rows_for_csv.append(df)
        print(f"\n── {name} ─────────────────────────────────")
        print("  " + block.summary().replace("\n", "\n  "))

    # ── Compact comparison ───────────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"Comparison on n={blocks['last_release_v1'].n} same-month nowcast origins")
    print("=" * 78)

    last_rmse = blocks["last_release_v1"].rmse
    print()
    header = (f"  {'Model':<26s}  {'n_feat':>6s}  {'RMSE':>7s}  "
                f"{'Δ vs floor':>11s}  {'cov80':>6s}  {'cov95':>6s}  "
                f"{'dir hit':>8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    # Hardcoded for non-multi forecasters; metadata-derived for the rest.
    static_n_feat = {"last_release_v1": 0, "same_month_bridge_v1": 3}
    for name, block in blocks.items():
        if name in static_n_feat:
            n_feat = str(static_n_feat[name])
        else:
            n_feat = str(n_features_by_model.get(name, "?"))
        delta = (1 - block.rmse / last_rmse) * 100
        cov80 = f"{block.cov80:.1%}" if block.cov80 is not None else "n/a"
        cov95 = f"{block.cov95:.1%}" if block.cov95 is not None else "n/a"
        dh = f"{block.dir_hit:.1%}" if block.dir_hit is not None else "n/a"
        print(f"  {name:<26s}  {n_feat:>6s}  {block.rmse:>7.4f}  "
                f"{delta:+>10.2f}%  {cov80:>6s}  {cov95:>6s}  {dh:>8s}")

    # Persist results
    out_csv = OUT_DIR / "compressed_bridge_comparison.csv"
    pd.concat(rows_for_csv).to_csv(out_csv, index=False)
    print(f"\nPer-row predictions: {out_csv}")


if __name__ == "__main__":
    main()
