"""Gate-2 proper — composed Thales (with Truflation→BLS bridge) vs
Cleveland Fed on real BLS Headline CPI YoY.

Pipeline:

  1. 12 Truflation top-level component series → monthly YoY
  2. Per-component PersistenceBaseline (per-archetype upgrade is the
     next iteration after this gate)
  3. CBDFComposer with real 2026 v2 weights → composed Truflation YoY
     forecast at T+1
  4. **TruflationToBLSBridge** → BLS Headline CPI YoY forecast at T+1
  5. Walk-forward through the harness; score against:
     - direct persistence on BLS YoY (the floor)
     - Cleveland Fed h=0 nowcast (the institutional bar — but at h=1
       frame it's the misaligned comparator we already documented)

This is the first run of "the institutional product" end-to-end.
Persistence-as-inner is honest scope: shows what the bridge alone
contributes, before per-component archetypes are added on top.
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
from thales.models.composition.bridge import TruflationToBLSBridge  # noqa: E402
from thales.models.composition.cbdf import CBDFComposer  # noqa: E402
from thales.models.composition.composed_forecaster import (  # noqa: E402
    ComposedForecaster,
)
from thales import targets as T  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402
from thales.weights import build_crosswalk, get_top_level_weights  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
KAIROS_PARQUET = Path("/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
TRUFL_HEADLINE_COL = "truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy"
SCORING_DB = ROOT / "results" / "baseline_eval" / "scoring.duckdb"


def _to_monthly_yoy_from_daily(daily: pd.Series) -> pd.Series:
    monthly = daily.resample("ME").last().dropna()
    return ((monthly / monthly.shift(12) - 1.0) * 100.0).dropna()


def main() -> None:
    print("=" * 78)
    print("Gate-2 Proper — Composed Thales (bridged to BLS) vs benchmarks")
    print("=" * 78)

    # ── Top-level weights ────────────────────────────────────────────────
    w_df = get_top_level_weights("2026-04-25")
    weights = {str(int(row.category_id)): float(row.weight) / 100.0
                 for row in w_df.itertuples()}
    weights = {k: v for k, v in weights.items() if v > 0}

    # ── Truflation component panel + headline ───────────────────────────
    streams_df = pd.read_csv(ROOT / "data" / "truflation" / "streams_catalog.csv")
    cw = build_crosswalk(streams_df["raw_name"])
    cid_to_raw = {str(int(r.category_id)): r.raw_name
                    for _, r in cw[cw["category_id"].astype(str).isin(weights)].iterrows()}

    print("\nLoading Truflation component panel + headline...")
    yoy_dict = {}
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        for cid, raw in cid_to_raw.items():
            yoy_dict[cid] = _to_monthly_yoy_from_daily(
                store.get_vintage(raw, date.today()))

    common_idx = sorted(set.intersection(*[set(s.index) for s in yoy_dict.values()]))
    common_idx = pd.DatetimeIndex(common_idx)
    panel = pd.DataFrame({cid: s.reindex(common_idx) for cid, s in yoy_dict.items()})

    parq = pd.read_parquet(KAIROS_PARQUET)
    parq["date"] = pd.to_datetime(parq["date"])
    parq = parq.set_index("date").sort_index()
    panel["truf_yoy"] = (parq[TRUFL_HEADLINE_COL].dropna()
                              .resample("ME").last()).reindex(common_idx)

    # ── BLS Headline CPI YoY (the official target) ──────────────────────
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        bls_yoy = T.load_target_yoy(store, "cpi", as_of=date.today())
    panel["bls_yoy"] = bls_yoy.reindex(common_idx)
    panel = panel.dropna(subset=list(weights) + ["truf_yoy", "bls_yoy"])

    print(f"Aligned panel: {panel.shape}  range "
          f"{panel.index.min():%Y-%m-%d} → {panel.index.max():%Y-%m-%d}")

    # ── Configure forecasters ────────────────────────────────────────────
    composer = CBDFComposer(weights=weights, weight_sum_tol=5e-3,
                                n_mc_samples=500, seed=0)
    component_forecasters = {
        cid: PersistenceBaseline(target_col=cid, model_id=f"persist_{cid}")
        for cid in weights
    }
    composed = ComposedForecaster(
        components=component_forecasters, composer=composer,
        model_id="composed_truf_persistence",
    )
    # Bridge composed Truflation forecast → BLS YoY
    thales_bridged = TruflationToBLSBridge(
        inner=composed,
        target_truf_col="truf_yoy",
        target_bls_col="bls_yoy",
        bridge_window_months=24,
        model_id="thales_bridged_v1",
    )

    # Direct persistence on BLS (the floor)
    bls_persistence = PersistenceBaseline(
        target_col="bls_yoy", model_id="bls_persistence_v1")

    # ── Walk-forward ─────────────────────────────────────────────────────
    origins = panel.index[24: -1]
    print(f"\nWalk-forward over {len(origins)} origins...")

    thales_fcs = walk_forward(thales_bridged, panel, "bls_yoy",
                                  origins, horizon=1)
    direct_fcs = walk_forward(bls_persistence, panel, "bls_yoy",
                                  origins, horizon=1)

    thales_df = attach_actuals(thales_fcs, panel["bls_yoy"])
    direct_df = attach_actuals(direct_fcs, panel["bls_yoy"])

    # ── Reports ──────────────────────────────────────────────────────────
    thales_block = score(thales_df)
    direct_block = score(direct_df)

    print()
    print("─── Thales bridged (composed Truflation persistence + bridge) ───")
    print("  " + thales_block.summary().replace("\n", "\n  "))

    print()
    print("─── BLS persistence (the floor) ───")
    print("  " + direct_block.summary().replace("\n", "\n  "))

    rmse_red = (1 - thales_block.rmse / direct_block.rmse) * 100
    mae_red = (1 - thales_block.mae / direct_block.mae) * 100
    print()
    print("=" * 72)
    print("Gate-2 lite verdict")
    print("=" * 72)
    print(f"  Thales bridged RMSE: {thales_block.rmse:.4f} pp")
    print(f"  BLS persistence:     {direct_block.rmse:.4f} pp")
    print(f"  Δ RMSE vs persist:   {rmse_red:+.2f}%")
    print(f"  Δ MAE vs persist:    {mae_red:+.2f}%")
    if rmse_red > 5:
        print(f"  ✓ Thales bridged BEATS BLS persistence by {rmse_red:.2f}% RMSE")
    elif rmse_red > 0:
        print(f"  ~ Thales bridged marginally better than persistence")
    else:
        print(f"  ✗ Thales bridged loses to BLS persistence by {-rmse_red:.2f}%")
    print()
    print("Note: with per-component PERSISTENCE as the inner forecaster,")
    print("any beat over BLS persistence is purely the BRIDGE'S contribution.")
    print("Substituting per-component archetypes (BSTS for Recreation,")
    print("commodity TVP for Utilities, pure MS for Health) is the next layer.")

    # Bridge coefficient summary
    if thales_fcs:
        last_alpha = thales_fcs[-1].metadata.get("bridge_alpha")
        last_beta = thales_fcs[-1].metadata.get("bridge_beta")
        last_resid_sd = thales_fcs[-1].metadata.get("bridge_residual_sd")
        print()
        print(f"Bridge at last origin: α = {last_alpha:+.4f}, β = {last_beta:+.4f},"
              f" residual SD = {last_resid_sd:.4f}")

    # Persist to scoring DB
    print()
    print(f"Persisting both runs to scoring DB: {SCORING_DB}")
    with open_scoring_db(SCORING_DB) as db:
        for fc in thales_fcs:
            db.insert_forecast("thales_bridged_v1", "bls_cpi_yoy", fc)
        for _, row in thales_df.iterrows():
            db.attach_actual("thales_bridged_v1", "bls_cpi_yoy",
                                pd.Timestamp(row["target"]).date(),
                                actual=row["actual"], today_baseline=row["today"])
        for fc in direct_fcs:
            db.insert_forecast("bls_persistence_v1", "bls_cpi_yoy", fc)
        for _, row in direct_df.iterrows():
            db.attach_actual("bls_persistence_v1", "bls_cpi_yoy",
                                pd.Timestamp(row["target"]).date(),
                                actual=row["actual"], today_baseline=row["today"])


if __name__ == "__main__":
    main()
