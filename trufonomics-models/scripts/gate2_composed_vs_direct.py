"""Gate-2 lite — composed Thales forecast vs direct headline persistence.

Runs both through the Phase 0.7 harness on real Truflation data:

  Composed:  per-component persistence on each of the 12 Truflation
             top-level categories → CBDFComposer with real 2026 v2
             weights → composed Truflation headline forecast
  Direct:    persistence on Truflation published headline YoY directly

Target:      Truflation US CPI frozen YoY (the published headline)
Window:      All months where the panel + headline are aligned

This isolates *the contribution of the composition layer*. By
accounting identity, composed persistence on per-component YoY weighted
by weights ≈ direct persistence on weighted-sum YoY. Any divergence is
informative about composition residuals.

Then drops both fits into the scoring DB so they can be queried
side-by-side with the baseline_eval scoreboard.

Future extensions: replace per-component PersistenceBaseline with the
real archetypes (BSTS-LL on Recreation/Food-away, commodity TVP on
Utilities, pure MS+persistence-by-regime on Health, etc.) — that's
gate-2 proper. Tonight is the integration plumbing test.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation.harness import (  # noqa: E402
    attach_actuals, score, walk_forward,
)
from thales.evaluation.scoring_db import open_scoring_db  # noqa: E402
from thales.models.baselines import PersistenceBaseline  # noqa: E402
from thales.models.composition.cbdf import CBDFComposer  # noqa: E402
from thales.models.composition.composed_forecaster import (  # noqa: E402
    ComposedForecaster,
)
from thales.vintage import VintageStore  # noqa: E402
from thales.weights import build_crosswalk, get_top_level_weights  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
KAIROS_PARQUET = Path("/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
TRUFL_HEADLINE_COL = "truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy"
SCORING_DB = ROOT / "results" / "baseline_eval" / "scoring.duckdb"


def _to_monthly_yoy(daily: pd.Series) -> pd.Series:
    monthly = daily.resample("ME").last().dropna()
    yoy = (monthly / monthly.shift(12) - 1.0) * 100.0
    return yoy.dropna()


def main() -> None:
    print("=" * 78)
    print("Gate-2 Lite — Composed vs Direct Persistence on Truflation Headline")
    print("=" * 78)

    # ── Top-level weights ────────────────────────────────────────────────
    w_df = get_top_level_weights("2026-04-25")
    weights = {str(int(row.category_id)): float(row.weight) / 100.0
                 for row in w_df.itertuples()}
    weights = {k: v for k, v in weights.items() if v > 0}
    print(f"\n12 top-level weights (sum = {sum(weights.values()):.4f})")

    # ── Component series → monthly YoY panel ─────────────────────────────
    streams_df = pd.read_csv(ROOT / "data" / "truflation" / "streams_catalog.csv")
    cw = build_crosswalk(streams_df["raw_name"])
    cid_to_raw = {str(int(r.category_id)): r.raw_name
                    for _, r in cw[cw["category_id"].astype(str).isin(weights)].iterrows()}

    print("\nLoading 12 component daily series → monthly YoY...")
    yoy_dict = {}
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        for cid, raw in cid_to_raw.items():
            s = store.get_vintage(raw, date.today())
            yoy_dict[cid] = _to_monthly_yoy(s)

    common_idx = sorted(set.intersection(*[set(s.index) for s in yoy_dict.values()]))
    common_idx = pd.DatetimeIndex(common_idx)
    panel = pd.DataFrame({cid: s.reindex(common_idx) for cid, s in yoy_dict.items()})

    # ── Direct headline ──────────────────────────────────────────────────
    parq = pd.read_parquet(KAIROS_PARQUET)
    parq["date"] = pd.to_datetime(parq["date"])
    parq = parq.set_index("date").sort_index()
    headline_yoy = (parq[TRUFL_HEADLINE_COL].dropna()
                       .resample("ME").last().dropna())
    headline_yoy = headline_yoy.reindex(common_idx).dropna()
    panel = panel.loc[headline_yoy.index]
    panel["headline"] = headline_yoy
    panel = panel.dropna()

    print(f"Aligned panel: {panel.shape}  range "
          f"{panel.index.min():%Y-%m-%d} → {panel.index.max():%Y-%m-%d}")

    # ── Composer + ComposedForecaster ────────────────────────────────────
    composer = CBDFComposer(weights=weights, weight_sum_tol=5e-3,
                                n_mc_samples=500, seed=0)
    components = {cid: PersistenceBaseline(target_col=cid)
                    for cid in weights}
    composed = ComposedForecaster(components=components, composer=composer,
                                       model_id="composed_persistence_v1")
    direct = PersistenceBaseline(target_col="headline",
                                       model_id="direct_persistence_v1")

    # ── Walk-forward ─────────────────────────────────────────────────────
    origins = panel.index[24: -1]
    print(f"\nWalk-forward over {len(origins)} origins...")

    composed_fcs = walk_forward(composed, panel, "headline", origins, horizon=1)
    direct_fcs = walk_forward(direct, panel, "headline", origins, horizon=1)

    composed_df = attach_actuals(composed_fcs, panel["headline"])
    direct_df = attach_actuals(direct_fcs, panel["headline"])

    composed_block = score(composed_df)
    direct_block = score(direct_df)

    # ── Reports ──────────────────────────────────────────────────────────
    print()
    print("─── composed_persistence_v1 ───")
    print("  " + composed_block.summary().replace("\n", "\n  "))

    print()
    print("─── direct_persistence_v1 ───")
    print("  " + direct_block.summary().replace("\n", "\n  "))

    # Composition residual
    merged = composed_df.merge(direct_df, on=["origin", "target", "actual"],
                                  suffixes=("_composed", "_direct"))
    composition_residual = (merged["point_composed"] - merged["point_direct"])
    print()
    print("Composed point − Direct point (composition residual):")
    print(f"  median  = {composition_residual.median():+.4f} pp")
    print(f"  p10/p90 = {composition_residual.quantile(0.10):+.4f} / "
          f"{composition_residual.quantile(0.90):+.4f} pp")

    # ── Persist to scoring DB ────────────────────────────────────────────
    print()
    print(f"Persisting both runs to scoring DB: {SCORING_DB}")
    with open_scoring_db(SCORING_DB) as db:
        for fc in composed_fcs:
            db.insert_forecast("composed_persistence_v1",
                                  "trufl_headline_yoy", fc)
        for _, row in composed_df.iterrows():
            db.attach_actual("composed_persistence_v1", "trufl_headline_yoy",
                                pd.Timestamp(row["target"]).date(),
                                actual=row["actual"], today_baseline=row["today"])
        for fc in direct_fcs:
            db.insert_forecast("direct_persistence_v1",
                                  "trufl_headline_yoy", fc)
        for _, row in direct_df.iterrows():
            db.attach_actual("direct_persistence_v1", "trufl_headline_yoy",
                                pd.Timestamp(row["target"]).date(),
                                actual=row["actual"], today_baseline=row["today"])
        inv = db.list_models()
        print()
        print("Scoreboard inventory:")
        print(inv.to_string(index=False))


if __name__ == "__main__":
    main()
