"""Density evaluation across the Thales model zoo — multi-horizon.

Runs walk-forward over BLS Headline CPI YoY at h ∈ {1, 3, 6, 9, 12} with
samples emission enabled on every Forecaster, then scores both point
AND density metrics in a single ScoreBlock per horizon. Produces a
comparison table with:

  * RMSE, MAE, RMSE-Δ% vs naive persistence
  * CRPS — the proper-scoring-rule analog of MAE for densities
  * PIT KS p-value — calibrated iff p > 0.05
  * Empirical 80%/95% coverage from the sample matrix
  * Sharpness (mean band width) at 80%/95%

Forecasters scored:

  * persistence_v1 — naive floor
  * ar1_yoy_v1 — AR(1) on YoY
  * ar1_mom_composed_v1 — AR(1) on MoM, composed via closed-form identity
    (multi-horizon via AR(1) chain bootstrap)
  * patha_v1 — Cleveland-style 2-feature OLS
  * dfm_stock_watson_v1 — single-factor DFM (Stock-Watson 2002 baseline)
  * bridged_cbdf_v1 — CBDF → BLS via OLS bridge (h=1 only; the bridge
    regression is one-step by construction)

Outputs in ``results/baseline_eval/``:

  * ``density_eval_multihorizon.csv`` — one row per (model, horizon)
  * Per-horizon summaries printed to stdout

Run::

    uv run python -m scripts.density_eval
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.evaluation.harness import (    # noqa: E402
    Forecast, attach_actuals, score, walk_forward,
)
from thales.models.baselines import (    # noqa: E402
    AR1Baseline, PathAForecaster, PersistenceBaseline,
)
from thales.models.composition.bridged_cbdf import (    # noqa: E402
    BridgedCBDFForecaster,
)
from thales.models.composition.cbdf import CBDFComposer    # noqa: E402
from thales.models.dfm import StockWatsonDFMForecaster    # noqa: E402
from thales.models.mom_composed import (    # noqa: E402
    MoMComposedForecaster,
    mom_from_level,
)
from thales.weights import build_crosswalk, get_top_level_weights  # noqa: E402
from thales import targets as T    # noqa: E402
from thales.vintage import VintageStore    # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "baseline_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6, 9, 12]


# ─── Panel construction ──────────────────────────────────────────────────


def _load_panel() -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Build a monthly panel: BLS CPI YoY + level + clev + 12 component YoYs.

    Same shape as the head-to-head script.
    """
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        base = T.load_panel(store, "cpi", as_of=date.today())

        w_df = get_top_level_weights("2026-04-25")
        streams_df = pd.read_csv(
            ROOT / "data" / "truflation" / "streams_catalog.csv")
        cw = build_crosswalk(streams_df["raw_name"])
        cw = cw.dropna(subset=["category_id"]).copy()
        cid_to_raw = {
            str(int(r.category_id)): r.raw_name
            for _, r in cw[cw["category_id"].astype(int).isin(
                w_df["category_id"].astype(int).tolist())].iterrows()
        }
        comp_yoys: dict[str, pd.Series] = {}
        comp_levels: dict[str, pd.Series] = {}
        for cid, raw in cid_to_raw.items():
            s = store.get_vintage(raw, date.today()).dropna()
            monthly = s.resample("ME").last()
            yoy = ((monthly / monthly.shift(12) - 1.0) * 100.0).dropna()
            comp_yoys[f"comp_{cid}"] = yoy
            comp_levels[f"comp_{cid}"] = monthly

    panel = pd.concat({**{c: base[c] for c in base.columns},
                            **comp_yoys}, axis=1)
    component_cols = sorted([c for c in panel.columns
                                  if c.startswith("comp_")])
    cid_to_raw_short = {f"comp_{cid}": raw for cid, raw in cid_to_raw.items()}
    return panel, component_cols, cid_to_raw_short


def _attach_truflation_yoy(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the frozen Truflation YoY headline series at monthly resolution."""
    KAIROS_PARQUET = Path(
        "/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
    TRUF_HEADLINE = (
        "truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy")
    parq = pd.read_parquet(KAIROS_PARQUET)
    parq["date"] = pd.to_datetime(parq["date"])
    parq = parq.set_index("date").sort_index()
    truf_yoy = parq[TRUF_HEADLINE].dropna().resample("ME").last()
    panel = panel.copy()
    panel["truf_yoy"] = truf_yoy.reindex(panel.index)
    return panel


def _build_cbdf_archetype_predictions(panel_full: pd.DataFrame,
                                          component_cols: list[str],
                                          weights: dict[str, float],
                                          origins,
                                          horizon: int = 1,
                                          ) -> pd.Series:
    """For each origin, run per-component MoM-composed AR(1) and compose
    via CBDF. Returns a Series indexed at ORIGIN with the CBDF prediction
    for the next-period BLS target — the prediction-for-tomorrow form
    BridgedCBDFForecaster expects in its ``inner_pred_col``.
    """
    composer = CBDFComposer(weights=weights, weight_sum_tol=5e-3,
                                  n_mc_samples=200, seed=0)
    out: dict[pd.Timestamp, float] = {}
    for origin in origins:
        try:
            target = panel_full.index[
                panel_full.index.get_loc(origin) + horizon]
        except IndexError:
            continue
        per_comp: dict[str, Forecast] = {}
        for cid in component_cols:
            yoy_series = panel_full[cid].dropna()
            yoy_series = yoy_series.loc[yoy_series.index <= origin]
            if len(yoy_series) < 24:
                continue
            mom = yoy_series.diff().dropna()
            if len(mom) < 13:
                continue
            x = mom.values[:-1]
            y = mom.values[1:]
            X = np.column_stack([np.ones_like(x), x])
            cf, *_ = np.linalg.lstsq(X, y, rcond=None)
            mom_pred = float(cf[0] + cf[1] * mom.values[-1])
            yoy_T = float(yoy_series.iloc[-1])
            mom_T_minus_11 = (float(mom.iloc[-11])
                                  if len(mom) >= 11 else 0.0)
            yoy_pred = yoy_T + mom_pred - mom_T_minus_11
            per_comp[cid] = Forecast(origin=origin, target=target,
                                              point=float(yoy_pred))
        if len(per_comp) < len(weights):
            continue
        composed = composer.compose(per_comp, origin, target)
        out[origin] = composed.point
    return pd.Series(out, name="cbdf_pred")


# ─── Main ────────────────────────────────────────────────────────────────


def _build_forecasters_for_horizon(panel: pd.DataFrame,
                                       panel_full: pd.DataFrame,
                                       component_cols: list[str],
                                       horizon: int,
                                       train_min: int = 36,
                                       ) -> dict[str, tuple[object, pd.DataFrame]]:
    """Construct the forecaster zoo for a given horizon.

    Bridged-CBDF is excluded for h>1 — the bridge regression
    `BLS[t] ~ α + β·BLS[t-1] + γ·CBDF[t-1]` is one-step by
    construction, and extending it to multi-step requires a different
    bridge formulation outside the scope of this multi-horizon eval.
    """
    forecasters: dict[str, tuple[object, pd.DataFrame]] = {
        f"persistence_v1": (
            PersistenceBaseline(target_col="y", horizon=horizon,
                                      model_id=f"persistence_h{horizon}"),
            panel,
        ),
        f"ar1_yoy_v1": (
            AR1Baseline(target_col="y", horizon=horizon, calib_months=24,
                          band_method="rolling_conformal",
                          model_id=f"ar1_yoy_h{horizon}"),
            panel,
        ),
        f"ar1_mom_composed_v1": (
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
                horizon=horizon,
                model_id=f"ar1_mom_composed_h{horizon}",
            ),
            panel,
        ),
        f"patha_v1": (
            PathAForecaster(
                target_col="y", truflation_col="truf_yoy",
                horizon=horizon, calib_months=24,
                band_method="rolling_conformal",
                model_id=f"patha_h{horizon}"),
            panel,
        ),
        f"dfm_stock_watson_v1": (
            StockWatsonDFMForecaster(
                component_cols=component_cols, target_col="y",
                horizon=horizon, train_min=train_min,
                model_id=f"dfm_stock_watson_h{horizon}"),
            panel_full,
        ),
    }
    if horizon == 1:
        forecasters["bridged_cbdf_v1"] = (
            BridgedCBDFForecaster(
                target_bls_col="y",
                inner_pred_col="cbdf_pred",
                calib_window=24,
                train_min=12,
                band_method="rolling_conformal",
                model_id="bridged_cbdf_v1"),
            panel_full,
        )
    return forecasters


def main() -> None:
    print("=" * 78)
    print("Density evaluation — multi-horizon RMSE + CRPS + PIT + coverage")
    print("=" * 78)

    panel, component_cols, _cid_to_raw = _load_panel()
    panel = panel.dropna(subset=["y", "level"])
    panel = panel.loc[panel.index >= "2017-01-01"]
    panel = _attach_truflation_yoy(panel)

    panel_full = panel.dropna(subset=component_cols).copy()

    print(f"\nPanel: n={len(panel)}  range "
          f"{panel.index.min():%Y-%m-%d} → {panel.index.max():%Y-%m-%d}")
    print(f"DFM/CBDF panel: n={len(panel_full)}  range "
          f"{panel_full.index.min():%Y-%m-%d} → "
          f"{panel_full.index.max():%Y-%m-%d}")

    train_min = 36
    today = panel["y"].shift(1)

    # ── CBDF inner predictions for h=1 only (Bridged-CBDF is h=1 only) ─
    print("\nBuilding CBDF inner predictions (h=1) for Bridged-CBDF…")
    weights_df = get_top_level_weights("2026-04-25")
    weights = {f"comp_{int(r.category_id)}": float(r.weight) / 100.0
                  for _, r in weights_df.iterrows()}
    origins_full_h1 = panel_full.index[train_min:-1]
    cbdf_pred_series = _build_cbdf_archetype_predictions(
        panel_full, component_cols, weights, origins_full_h1, horizon=1)
    panel_full = panel_full.copy()
    panel_full["cbdf_pred"] = cbdf_pred_series
    print(f"  CBDF predictions: n={len(cbdf_pred_series)}")

    rows: list[dict] = []
    for h in HORIZONS:
        print("\n" + "=" * 78)
        print(f"Horizon h = {h}")
        print("=" * 78)

        origins = panel.index[train_min:-h]
        origins_full = panel_full.index[train_min:-h]

        forecasters = _build_forecasters_for_horizon(
            panel, panel_full, component_cols, horizon=h,
            train_min=train_min)

        for name, (fc, use_panel) in forecasters.items():
            use_origins = (origins_full
                              if use_panel is panel_full else origins)
            forecasts = walk_forward(fc, use_panel, "y", use_origins,
                                          horizon=h)
            if not forecasts:
                print(f"\n  [h={h} {name}] no forecasts — skipping")
                continue
            df = attach_actuals(forecasts, use_panel["y"],
                                  today_baseline=today)
            if df.empty:
                print(f"\n  [h={h} {name}] empty df — skipping")
                continue
            block = score(df)
            print(f"\n── h={h}  {name} ──")
            print("  " + block.summary().replace("\n", "\n  "))
            rows.append({
                "model": name,
                "horizon": h,
                "n": block.n,
                "rmse": block.rmse,
                "mae": block.mae,
                "rmse_red_pct": block.rmse_reduction_pct,
                "crps": block.crps,
                "pit_ks_pvalue": block.pit_ks_pvalue,
                "cov80": block.cov80_density,
                "cov95": block.cov95_density,
                "sharp80": block.sharp80_density,
                "sharp95": block.sharp95_density,
                "n_density": block.n_density,
            })

    # ── Output ───────────────────────────────────────────────────────
    summary = pd.DataFrame(rows)
    out_csv = OUT_DIR / "density_eval_multihorizon.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\n[write] {out_csv}")

    print("\n" + "=" * 78)
    print("Multi-horizon density-eval summary")
    print("=" * 78)
    cols = ["model", "horizon", "n", "rmse", "rmse_red_pct", "crps",
            "pit_ks_pvalue", "cov80", "sharp80"]
    avail = [c for c in cols if c in summary.columns]
    print(summary[avail].to_string(index=False, float_format="%.4f"))

    # Pivot table by horizon for the headline view.
    print("\n" + "=" * 78)
    print("RMSE by model × horizon")
    print("=" * 78)
    rmse_pivot = summary.pivot(index="model", columns="horizon",
                                  values="rmse")
    print(rmse_pivot.to_string(float_format="%.4f"))

    print("\nCRPS by model × horizon")
    print("=" * 78)
    crps_pivot = summary.pivot(index="model", columns="horizon",
                                  values="crps")
    print(crps_pivot.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
