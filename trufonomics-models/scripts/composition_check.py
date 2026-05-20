"""Composition sanity check — reconstruct Truflation's daily US CPI YoY from
the 80 component streams + weights, compare against the published aggregate.

This is the trust check before any CBDF work. If the 12 top-level streams
weight-combined reproduce `truflation_us_cpi_yoy` within a tight tolerance,
we know (a) our ingested data is correct, (b) our weight interpretation is
correct, (c) Truflation's own aggregation math is what the architecture doc
claims. If the reconstruction drifts from the published aggregate, we have
a targeted question for backend dev rather than a vague ask.

Approach — two methods, report both:

  Method 1 — weighted-sum of YoYs
      For each day, compute yoy_i from each top-level stream's index vs
      its 365-day-prior value, weight-average using the effective weights
      at that day, compare to `truflation_us_cpi_yoy`.

  Method 2 — aggregate-then-YoY
      Build a composite index from weighted-sum of component indexes,
      then compute YoY on the composite, compare to `truflation_us_cpi_yoy`.

Both should match closely. Method 2 is the "Truflation own math"
reconstruction; Method 1 is the first-order approximation.

Output:
    results/composition_check/headline_residuals.csv
    results/composition_check/summary.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.vintage import VintageStore
from thales.weights import (
    build_crosswalk,
    get_top_level_weights,
    top_level_category_ids,
    V2_EFFECTIVE_FROM,
)

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
STREAMS_CSV = ROOT / "data" / "truflation" / "streams_catalog.csv"
KAIROS_PARQUET = Path("/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
OUT_DIR = ROOT / "results" / "composition_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def effective_weights_for_date(d: pd.Timestamp) -> pd.DataFrame:
    """Top-level weights for a specific date (picks v1 or v2)."""
    return get_top_level_weights(d.date())


def load_component_panel(store: VintageStore,
                          stream_ids: list[str]) -> pd.DataFrame:
    """Wide dataframe: rows = date, columns = stream_id, values = index level.

    Uses get_vintage with as_of=today (our ingest tagged everything with today).
    """
    as_of = pd.Timestamp.today().date()
    cols = {}
    for sid in stream_ids:
        s = store.get_vintage(sid, as_of)
        if not s.empty:
            cols[sid] = s
    return pd.DataFrame(cols).sort_index()


def main() -> None:
    print("Loading cross-walk...")
    streams_df = pd.read_csv(STREAMS_CSV)
    crosswalk = build_crosswalk(streams_df["raw_name"])
    tops = top_level_category_ids()
    top_streams = crosswalk[crosswalk["category_id"].isin(tops)]
    print(f"  top-level streams in catalog: {len(top_streams)}/{len(tops)}")
    for _, r in top_streams.iterrows():
        print(f"    id={int(r.category_id):3d}  {r.raw_name:45s}  {r.category}")

    if len(top_streams) < len(tops):
        missing_ids = set(tops) - set(top_streams["category_id"].astype(int))
        tree_lookup = crosswalk.set_index("category_id")["category"].to_dict()
        print(f"\n  MISSING top-level category IDs:")
        for mid in missing_ids:
            # Find name from earlier top-level data
            w = get_top_level_weights("2026-04-24")
            name = w[w["category_id"] == mid]["category"].iloc[0]
            print(f"    {mid}: {name}")
        print()

    # Load panel for the top-level streams
    print("\nLoading vintage panel for top-level streams...")
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        panel = load_component_panel(store,
                                       top_streams["raw_name"].tolist())
    print(f"  panel shape: {panel.shape}")
    print(f"  range: {panel.index.min():%Y-%m-%d} → {panel.index.max():%Y-%m-%d}")

    # Rename columns from raw_name → category_id for cleaner weight lookup
    rename = dict(zip(top_streams["raw_name"],
                       top_streams["category_id"].astype(int).map(str)))
    panel = panel.rename(columns=rename)

    # Load Truflation's published headline for comparison
    print("\nLoading published headline truflation_us_cpi_yoy + cpi_index...")
    pq = pd.read_parquet(KAIROS_PARQUET)
    pq["date"] = pd.to_datetime(pq["date"])
    pq = pq.set_index("date").sort_index()
    # Apples-to-apples: our component streams are FROZEN (revision-pinned),
    # so we compare against the FROZEN aggregate, not the live one.
    published_yoy_live = pq["truflation_us_cpi_yoy/truflation_us_cpi_yoy"].dropna()
    published_yoy_frozen = pq["truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy"].dropna()
    published_yoy = published_yoy_frozen     # headline comparison series
    print(f"  frozen yoy range: {published_yoy.index.min():%Y-%m-%d} → "
          f"{published_yoy.index.max():%Y-%m-%d}")
    print(f"  live yoy range:   {published_yoy_live.index.min():%Y-%m-%d} → "
          f"{published_yoy_live.index.max():%Y-%m-%d}")

    # Align panel to published series index
    common_index = panel.index.intersection(published_yoy.index)
    panel = panel.loc[common_index]
    pub_yoy = published_yoy.loc[common_index]
    print(f"  common date range: {common_index.min():%Y-%m-%d} → {common_index.max():%Y-%m-%d}")

    # ── Method 1 — weighted-sum of YoYs ──
    # yoy_i_t = (index_i_t / index_i_{t-365}) - 1 (expressed as percent)
    # Weight by top-level weight effective at t.
    print("\n[Method 1] weighted-sum of YoYs")
    shifted = panel.shift(freq="365D")
    yoy_panel = (panel / shifted.reindex(panel.index) - 1.0) * 100.0  # in percent

    reconstructed_m1 = pd.Series(index=panel.index, dtype=float)
    for t in panel.index:
        w = effective_weights_for_date(t)
        w_lookup = dict(zip(w["category_id"].astype(int).map(str),
                             w["weight"].astype(float)))
        yoy_row = yoy_panel.loc[t]
        acc = 0.0
        wsum = 0.0
        for cid_str, yoy_val in yoy_row.items():
            if pd.isna(yoy_val) or cid_str not in w_lookup:
                continue
            w_i = w_lookup[cid_str]
            acc += w_i * yoy_val
            wsum += w_i
        if wsum > 0:
            reconstructed_m1.loc[t] = acc / wsum
    m1_residual = reconstructed_m1 - pub_yoy

    # ── Method 2 — aggregate-then-YoY ──
    # composite_index_t = sum(w_i * index_i_t) / sum(w_i)
    # yoy_agg_t = (composite_t / composite_{t-365}) - 1
    print("[Method 2] aggregate-then-YoY")
    composite = pd.Series(index=panel.index, dtype=float)
    for t in panel.index:
        w = effective_weights_for_date(t)
        w_lookup = dict(zip(w["category_id"].astype(int).map(str),
                             w["weight"].astype(float)))
        row = panel.loc[t]
        acc = 0.0
        wsum = 0.0
        for cid_str, val in row.items():
            if pd.isna(val) or cid_str not in w_lookup:
                continue
            w_i = w_lookup[cid_str]
            acc += w_i * val
            wsum += w_i
        if wsum > 0:
            composite.loc[t] = acc / wsum
    shifted_c = composite.shift(freq="365D").reindex(composite.index)
    reconstructed_m2 = (composite / shifted_c - 1.0) * 100.0
    m2_residual = reconstructed_m2 - pub_yoy

    # ── Summary stats on the eval window (where YoY is defined) ──
    eval_mask = reconstructed_m1.notna() & pub_yoy.notna()
    n = int(eval_mask.sum())
    print(f"\nEval window: {n} days")

    def stats(resid: pd.Series, label: str) -> dict:
        r = resid.dropna()
        return {
            "method":       label,
            "n":            len(r),
            "mean":         float(r.mean()),
            "median":       float(r.median()),
            "sd":           float(r.std()),
            "abs_max":      float(r.abs().max()),
            "abs_p95":      float(r.abs().quantile(0.95)),
            "abs_p99":      float(r.abs().quantile(0.99)),
            "within_0.1pp": float((r.abs() < 0.1).mean()),
            "within_0.5pp": float((r.abs() < 0.5).mean()),
        }

    rows = [stats(m1_residual, "M1_weighted_yoy"),
             stats(m2_residual, "M2_aggregate_then_yoy")]
    summary = pd.DataFrame(rows)
    print("\nResidual statistics (reconstructed − published, in pp):")
    print(summary.to_string(index=False))

    # Save
    outfile = OUT_DIR / "headline_residuals.csv"
    out = pd.DataFrame({
        "published_yoy_frozen":  pub_yoy,
        "published_yoy_live":    published_yoy_live.reindex(pub_yoy.index),
        "reconstructed_m1":      reconstructed_m1,
        "reconstructed_m2":      reconstructed_m2,
        "residual_m1":           m1_residual,
        "residual_m2":           m2_residual,
    })
    out.to_csv(outfile)
    print(f"\nSaved: {outfile}")

    # Also report vs LIVE for reference
    m1_vs_live = reconstructed_m1 - published_yoy_live.reindex(reconstructed_m1.index)
    m2_vs_live = reconstructed_m2 - published_yoy_live.reindex(reconstructed_m2.index)
    print("\nResiduals vs LIVE aggregate (for reference — frozen was the apples-to-apples comparison):")
    print(f"  M1 vs live:  mean={m1_vs_live.mean():+.4f}  median={m1_vs_live.median():+.4f}  sd={m1_vs_live.std():.4f}")
    print(f"  M2 vs live:  mean={m2_vs_live.mean():+.4f}  median={m2_vs_live.median():+.4f}  sd={m2_vs_live.std():.4f}")

    # Diagnostic — latest 10 days
    print("\nLast 10 days (all series):")
    print(out.tail(10).to_string())


if __name__ == "__main__":
    main()
