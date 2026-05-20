"""End-to-end eval: persistence + AR(1) baselines on official inflation
targets, scored against the Cleveland Fed nowcast comparator.

This is the first time the repo evaluates models against official BLS
CPI / BEA PCE — Truflation-flavored daily forecasters do NOT speak to
this question. The point is to establish:

  1. The naive floor (persistence YoY[T+1] = YoY[T]) — Stock-Watson 2007
     says nobody beats this on near-term CPI YoY level.
  2. The AR(1) floor — slightly more sophisticated autoregressive base.
  3. Cleveland Fed nowcast performance on the same window — the hurdle
     every archetype model has to clear to justify the rebuild.

Outputs:

  * Per-target ScoreBlock printed to stdout.
  * Forecasts + scoring rows persisted to
    ``results/baseline_eval/scoring.duckdb`` under model_ids
    ``persistence_v1``, ``ar1_v1``, ``clevfed_v1`` for cross-model SQL.
  * Per-target CSV in ``results/baseline_eval/<target>_predictions.csv``.

Usage:
    uv run python scripts/eval_official_baselines.py
    uv run python scripts/eval_official_baselines.py --target cpi --start 2020-01-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation.harness import (  # noqa: E402
    Forecast,
    attach_actuals,
    score,
    walk_forward,
)
from thales.evaluation.scoring_db import open_scoring_db  # noqa: E402
from thales.models.baselines import (  # noqa: E402
    AR1Baseline,
    PathAForecaster,
    PersistenceBaseline,
)
from thales.models.mom_composed import MoMComposedForecaster  # noqa: E402
from thales import targets as T  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "baseline_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCORING_DB = OUT_DIR / "scoring.duckdb"

KAIROS_PARQUET = Path("/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
TRUFLATION_FROZEN_YOY_COL = "truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy"


def load_truflation_yoy_monthly() -> pd.Series:
    """Truflation US CPI YoY (frozen / revision-pinned), aligned to month-end.

    Frozen series chosen for vintage discipline: the value at month-end M
    is what was published on that date and never revised — the proper
    'what was knowable at origin' input.
    """
    df = pd.read_parquet(KAIROS_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    daily = df[TRUFLATION_FROZEN_YOY_COL].dropna()
    monthly = daily.resample("ME").last()
    monthly.name = "truf_yoy"
    return monthly


class ClevFedComparator:
    """Wrap Cleveland Fed nowcast as a Forecaster — no fitting, just look up
    the comparator's value at origin and use it as the prediction for
    target. Bands collapse (we don't have density from the scrape)."""
    model_id = "clevfed_v1"

    def __init__(self, clev_series: pd.Series):
        self.clev = clev_series

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        if origin not in self.clev.index:
            raise ValueError(f"clevfed has no value at origin {origin}")
        point = float(self.clev.loc[origin])
        return Forecast(origin=origin, target=target, point=point,
                          metadata={"baseline": "clevfed_lookup"})


def evaluate_target(target_name: str, start: pd.Timestamp,
                       horizon: int = 1) -> None:
    print()
    print("=" * 72)
    print(f"  Target: {target_name}    horizon: +{horizon} month(s)    "
          f"start: {start:%Y-%m-%d}")
    print("=" * 72)

    with VintageStore(VINTAGE_DB, read_only=True) as store:
        panel = T.load_panel(store, target_name, as_of=date.today())

    truf = load_truflation_yoy_monthly()
    panel["truf_yoy"] = truf

    panel = panel.loc[panel.index >= start].copy()
    panel = panel.dropna(subset=["y"])  # need actual target

    print(f"  panel: {panel.shape}  range "
          f"{panel.index.min():%Y-%m-%d} → {panel.index.max():%Y-%m-%d}")
    print(f"    y notna:        {panel['y'].notna().sum():>4d}")
    print(f"    clevfed notna:  {panel['clevfed'].notna().sum():>4d}")
    print(f"    truf_yoy notna: {panel['truf_yoy'].notna().sum():>4d}")

    # Origins: every month where we have y at origin AND y at origin+h
    available = panel.index
    origins = []
    for o in available:
        pos = available.get_loc(o)
        if pos + horizon < len(available):
            origins.append(o)
    if not origins:
        print("  no eligible origins; skipping")
        return

    # ── Run baselines through the harness ────────────────────────────────
    # In-sample variants for sanity + rolling-conformal variants for
    # production-grade calibrated bands. Persistence skipped from
    # conformal since its bands ARE the empirical diff distribution — no
    # model fit to debias.
    forecasters = [
        PersistenceBaseline(target_col="y", horizon=horizon),
        AR1Baseline(target_col="y", horizon=horizon,
                      band_method="in_sample"),
        AR1Baseline(target_col="y", horizon=horizon,
                      calib_months=24,
                      band_method="rolling_conformal",
                      model_id="ar1_rolling_v1"),
        # Fix #5 MoM-composed AR(1): forecast monthly log-MoM, compose
        # to YoY via the closed-form identity. Empirically beats
        # AR(1)-on-YoY by ~36% RMSE on Headline CPI (see
        # results/baseline_eval/MOM_COMPOSED_FINDINGS.md).
        MoMComposedForecaster(
            inner=AR1Baseline(target_col="bls_mom",
                                  horizon=1, train_min=24,
                                  calib_months=24,
                                  band_method="rolling_conformal",
                                  model_id="ar1_mom_inner"),
            bls_level_col="level",
            bls_yoy_col="y",
            mom_col="bls_mom",
            log_mom=True,
            horizon=1,
            model_id="ar1_mom_composed_v1",
        ),
        PathAForecaster(target_col="y", truflation_col="truf_yoy",
                          horizon=horizon, band_method="in_sample"),
        PathAForecaster(target_col="y", truflation_col="truf_yoy",
                          horizon=horizon, calib_months=24,
                          band_method="rolling_conformal",
                          model_id="patha_rolling_v1"),
    ]
    runs: dict[str, list[Forecast]] = {}
    for fc in forecasters:
        # PathA needs both y and truf_yoy at origin — restrict origins where
        # the Truflation feature exists. (Truflation goes back to 2010
        # so this is rarely binding except at the panel edges.)
        if isinstance(fc, PathAForecaster):
            valid_origins = [o for o in origins
                                if o in panel.index
                                and pd.notna(panel.loc[o, "truf_yoy"])]
        else:
            valid_origins = origins
        forecasts = walk_forward(fc, panel, "y", valid_origins,
                                    horizon=horizon)
        runs[fc.model_id] = forecasts

    # Cleveland Fed: only score on origins where it has a value
    clev_series = panel["clevfed"].dropna()
    clev_origins = [o for o in origins if o in clev_series.index]
    if clev_origins:
        runs["clevfed_v1"] = walk_forward(
            ClevFedComparator(clev_series), panel, "y",
            clev_origins, horizon=horizon)

    # ── Score each model + persist ───────────────────────────────────────
    blocks: dict[str, object] = {}
    with open_scoring_db(SCORING_DB) as db:
        for model_id, forecasts in runs.items():
            df = attach_actuals(forecasts, panel["y"])
            if df.empty:
                print(f"\n  [{model_id}] no scored rows — skipping")
                continue
            block = score(df)
            blocks[model_id] = block
            df.to_csv(OUT_DIR / f"{target_name}_{model_id}.csv", index=False)

            for f in forecasts:
                db.insert_forecast(model_id, f"{target_name}_yoy", f)
            for _, row in df.iterrows():
                db.attach_actual(model_id, f"{target_name}_yoy",
                                    pd.Timestamp(row["target"]).date(),
                                    actual=row["actual"],
                                    today_baseline=row["today"])

            print()
            print(f"  ── {model_id} ─────────────────────────────────")
            print("  " + block.summary().replace("\n", "\n  "))

    # ── Compact comparison table ─────────────────────────────────────────
    print()
    print("  ┌─────────────────────┬─────┬────────┬────────┬─────────┬─────────┬──────────┐")
    print("  │ model               │  n  │ RMSE   │ MAE    │ cov80   │ cov95   │ dir hit  │")
    print("  ├─────────────────────┼─────┼────────┼────────┼─────────┼─────────┼──────────┤")
    for model_id, block in blocks.items():
        cov80 = f"{block.cov80:.1%}" if block.cov80 is not None else "n/a"
        cov95 = f"{block.cov95:.1%}" if block.cov95 is not None else "n/a"
        dh = f"{block.dir_hit:.1%}" if block.dir_hit is not None else "n/a"
        print(f"  │ {model_id:<19s} │ {block.n:>3d} │ {block.rmse:>6.4f} │ "
              f"{block.mae:>6.4f} │ {cov80:>7s} │ {cov95:>7s} │ {dh:>8s} │")
    print("  └─────────────────────┴─────┴────────┴────────┴─────────┴─────────┴──────────┘")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(T.TARGETS) + ["all"], default="all")
    ap.add_argument("--start", default="2014-01-01",
                     help="Start of evaluation window (default 2014 — Cleveland "
                           "Fed nowcast starts mid-2013)")
    ap.add_argument("--horizon", type=int, default=1,
                     help="Forecast horizon in months (default 1)")
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    targets = list(T.TARGETS) if args.target == "all" else [args.target]
    for tgt in targets:
        try:
            evaluate_target(tgt, start, horizon=args.horizon)
        except Exception as e:   # noqa: BLE001
            print(f"\n  [{tgt}] FAILED: {type(e).__name__}: {e}")

    print()
    print(f"Persisted to {SCORING_DB}")
    print(f"Per-model CSVs in {OUT_DIR}")


if __name__ == "__main__":
    main()
