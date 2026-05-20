"""O'Keeffe-Petrova head-to-head — CBDF vs Standard DFM on inflation.

The flagship Phase 2.1 ablation. Reproduces O'Keeffe & Petrova 2025
NY Fed SR 1152 (CBDF on GDP) on the inflation target Thales is built
for, using Truflation's 12 top-level component panel as input.

Models compared (all walk-forward, h=1, BLS Headline CPI YoY target):

  1. **persistence_v1**           — naive floor (y[T+1] = y[T])
  2. **ar1_yoy_v1**               — AR(1) on YoY (Stock-Watson 2007 baseline)
  3. **ar1_mom_composed_v1**      — Fix #5: AR(1) on MoM, compose to YoY
  4. **patha_v1**                 — clev + truf 2-feature OLS
  5. **dfm_stock_watson_v1**      — single-factor DFM on 12 components
                                    (the O'Keeffe-Petrova baseline)
  6. **cbdf_persistence_v1**      — CBDF with persistence-per-component
  7. **cbdf_archetype_v1**        — CBDF with MoM-composed AR(1) per
                                    component (drift fix D)
  8. **clevfed_v1**               — Cleveland Fed nowcast alone
  9. **clev_plus_thales_v1**      — actual ~ α + β·clev + γ·thales_nowcast
                                    (the operational ensemble — Fix #3)

Reports: RMSE, MAE, CRPS (where density available), 80%/95%
coverage, direction hit, plus DM tests pairwise vs DFM and vs
Cleveland Fed.

The two key claims we're validating:

  (a) **CBDF vs DFM**: O'Keeffe-Petrova report 15% RMSE / 20% density
      improvement on GDP. We expect similar magnitude on inflation.
  (b) **Thales + Clev vs Clev**: Fix #3 finding — adding Thales to
      Cleveland's public nowcast gives ~15% additional RMSE
      reduction.
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
from thales.evaluation.tests import diebold_mariano    # noqa: E402
from thales.models.baselines import (    # noqa: E402
    AR1Baseline, PathAForecaster, PersistenceBaseline,
)
from thales.models.composition.cbdf import CBDFComposer    # noqa: E402
from thales.models.dfm import StockWatsonDFMForecaster    # noqa: E402
from thales.models.mom_composed import MoMComposedForecaster    # noqa: E402
from thales.weights import build_crosswalk, get_top_level_weights    # noqa: E402
from thales import targets as T    # noqa: E402
from thales.vintage import VintageStore    # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "baseline_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_panel() -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Build a monthly panel: BLS CPI YoY + level + clev + 12 component YoYs.

    Returns (panel, component_cols, cid_to_raw_name).
    """
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        base = T.load_panel(store, "cpi", as_of=date.today())

        # 12 top-level Truflation component YoYs (compute from level)
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
        for cid, raw in cid_to_raw.items():
            s = store.get_vintage(raw, date.today()).dropna()
            monthly = s.resample("ME").last()
            yoy = ((monthly / monthly.shift(12) - 1.0) * 100.0).dropna()
            comp_yoys[f"comp_{cid}"] = yoy

    panel = pd.concat({**{c: base[c] for c in base.columns},
                            **comp_yoys}, axis=1)
    component_cols = sorted([c for c in panel.columns if c.startswith("comp_")])
    # Map component_col → raw_name for level lookup later
    cid_to_raw_short = {f"comp_{cid}": raw
                              for cid, raw in cid_to_raw.items()}
    return panel, component_cols, cid_to_raw_short


def _attribute_clev_combo(panel: pd.DataFrame, thales_pred_col: str,
                              origin_min: pd.Timestamp,
                              calib_window_months: int = 36) -> dict:
    """Walk-forward 'clev + thales' linear combo using rolling regression.

    For each origin T, fit `actual ~ α + β·clev + γ·thales` on the
    trailing `calib_window_months` of data; predict actual[T+1] using
    clev[T+1] and thales[T+1] (both known by harness convention).
    Returns a dict that mimics what `attach_actuals + score` produce.
    """
    rows = []
    cols_required = ["y", "clevfed", thales_pred_col]
    sub = panel[cols_required].dropna()
    sub = sub.loc[sub.index >= origin_min]
    for i in range(calib_window_months, len(sub) - 1):
        train = sub.iloc[i - calib_window_months: i]
        target_idx = sub.index[i + 1]
        # Fit OLS on training window
        X_tr = np.column_stack([
            np.ones(len(train)),
            train["clevfed"].values,
            train[thales_pred_col].values,
        ])
        y_tr = train["y"].values
        coef, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        # Predict at target index using its (clev, thales) values
        clev_t = float(sub.loc[target_idx, "clevfed"])
        thales_t = float(sub.loc[target_idx, thales_pred_col])
        pred = float(coef[0] + coef[1] * clev_t + coef[2] * thales_t)
        actual = float(sub.loc[target_idx, "y"])
        rows.append({"origin": sub.index[i], "target": target_idx,
                          "point": pred, "actual": actual})
    return rows


def main() -> None:
    print("=" * 78)
    print("O'Keeffe head-to-head — CBDF vs DFM on inflation")
    print("=" * 78)

    panel, component_cols, cid_to_raw = _load_panel()
    panel = panel.dropna(subset=["y", "level"])
    panel = panel.loc[panel.index >= "2017-01-01"]    # need component history
    print(f"\nPanel: n={len(panel)}  range "
          f"{panel.index.min():%Y-%m-%d} → {panel.index.max():%Y-%m-%d}")
    print(f"Components: {len(component_cols)}")
    print(f"Components NaN%: "
          f"{(panel[component_cols].isna().sum().sum() / panel[component_cols].size * 100):.1f}%")

    # Drop rows where any component is NaN — needed for DFM/CBDF
    panel_full = panel.dropna(subset=component_cols).copy()
    print(f"Panel for DFM/CBDF: n={len(panel_full)}  range "
          f"{panel_full.index.min():%Y-%m-%d} → "
          f"{panel_full.index.max():%Y-%m-%d}")

    # ── Truflation YoY column for PathA + bridge ──────────────────
    KAIROS_PARQUET = Path(
        "/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
    TRUF_HEADLINE = "truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy"
    parq = pd.read_parquet(KAIROS_PARQUET)
    parq["date"] = pd.to_datetime(parq["date"])
    parq = parq.set_index("date").sort_index()
    truf_yoy = parq[TRUF_HEADLINE].dropna().resample("ME").last()
    panel["truf_yoy"] = truf_yoy
    panel_full["truf_yoy"] = truf_yoy.reindex(panel_full.index)

    train_min = 36
    origins = panel.index[train_min:-1]
    origins_full = panel_full.index[train_min:-1]
    today = panel["y"].shift(1)

    # ── Forecaster zoo ────────────────────────────────────────────
    forecasters_full_panel = {
        "persistence_v1": PersistenceBaseline(target_col="y", horizon=1,
                                                          model_id="persistence_v1"),
        "ar1_yoy_v1": AR1Baseline(target_col="y", horizon=1,
                                              calib_months=24,
                                              band_method="rolling_conformal",
                                              model_id="ar1_yoy_v1"),
        "ar1_mom_composed_v1": MoMComposedForecaster(
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
        "patha_v1": PathAForecaster(
            target_col="y", truflation_col="truf_yoy",
            horizon=1, calib_months=24,
            band_method="rolling_conformal",
            model_id="patha_v1"),
        "dfm_stock_watson_v1": StockWatsonDFMForecaster(
            component_cols=component_cols, target_col="y",
            horizon=1, train_min=train_min,
            model_id="dfm_stock_watson_v1"),
    }

    blocks: dict[str, object] = {}
    pred_dfs: dict[str, pd.DataFrame] = {}

    for name, fc in forecasters_full_panel.items():
        # Use the full panel for DFM (needs all components); use the
        # broader panel for the others (don't need components).
        use_panel = (panel_full
                          if "dfm" in name or "cbdf" in name
                          else panel)
        use_origins = (origins_full
                            if "dfm" in name or "cbdf" in name
                            else origins)
        forecasts = walk_forward(fc, use_panel, "y", use_origins, horizon=1)
        if not forecasts:
            print(f"\n  [{name}] no forecasts — skipping")
            continue
        df = attach_actuals(forecasts, use_panel["y"], today_baseline=today)
        if df.empty:
            print(f"\n  [{name}] empty df — skipping")
            continue
        block = score(df)
        blocks[name] = block
        pred_dfs[name] = df
        print(f"\n── {name} ──")
        print("  " + block.summary().replace("\n", "\n  "))

    # ── CBDF (persistence per component) ──────────────────────────
    print("\n── cbdf_persistence_v1 ──")
    weights_df = get_top_level_weights("2026-04-25")
    weights = {f"comp_{int(r.category_id)}": float(r.weight) / 100.0
                  for _, r in weights_df.iterrows()}
    composer = CBDFComposer(weights=weights, weight_sum_tol=5e-3,
                                   n_mc_samples=300, seed=0)
    cbdf_persist_rows = []
    for origin in origins_full:
        try:
            target = panel_full.index[panel_full.index.get_loc(origin) + 1]
        except IndexError:
            continue
        per_comp = {
            cid: Forecast(origin=origin, target=target,
                              point=float(panel_full.loc[origin, cid]))
            for cid in weights
        }
        composed = composer.compose(per_comp, origin, target)
        cbdf_persist_rows.append({
            "origin": origin, "target": target,
            "point": composed.point, "actual": float(panel_full.loc[target, "y"]),
            "today": float(today.loc[origin]) if origin in today.index else np.nan,
        })
    cbdf_persist_df = pd.DataFrame(cbdf_persist_rows).dropna(subset=["actual"])
    pred_dfs["cbdf_persistence_v1"] = cbdf_persist_df
    rmse_cbdf_p = float(np.sqrt(np.mean(
        (cbdf_persist_df["point"] - cbdf_persist_df["actual"]) ** 2)))
    mae_cbdf_p = float(np.mean(
        np.abs(cbdf_persist_df["point"] - cbdf_persist_df["actual"])))
    print(f"  n={len(cbdf_persist_df)}  RMSE={rmse_cbdf_p:.4f}  MAE={mae_cbdf_p:.4f}")

    # ── CBDF (MoM-composed AR(1) archetype per component) ─────────
    # For each component, build a MoM forecast using AR(1) on its MoM,
    # composed back to YoY via the closed-form identity.
    # Optimization: precompute MoM per component, run rolling AR(1)
    # forecast.
    print("\n── cbdf_archetype_v1 ──")
    print("  (per-component MoM-composed AR(1) → CBDFComposer)")
    # Pre-compute MoM for each component
    comp_levels: dict[str, pd.Series] = {}
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        for cid_short in weights:
            raw_id = cid_to_raw[cid_short]
            s = store.get_vintage(raw_id, date.today()).dropna()
            comp_levels[cid_short] = np.log(s.resample("ME").last()).dropna()

    cbdf_arch_rows = []
    for origin in origins_full:
        try:
            target = panel_full.index[panel_full.index.get_loc(origin) + 1]
        except IndexError:
            continue
        per_comp: dict[str, Forecast] = {}
        skip_origin = False
        for cid in weights:
            level_s = comp_levels[cid]
            level_train = level_s.loc[level_s.index <= origin]
            if len(level_train) < 24:
                skip_origin = True
                break
            # Recent component MoMs (in pp)
            mom = (level_train.diff().dropna() * 100.0)
            if len(mom) < 14:
                skip_origin = True
                break
            # AR(1) on MoM
            x = mom.values[:-1]
            y = mom.values[1:]
            X = np.column_stack([np.ones_like(x), x])
            cf, *_ = np.linalg.lstsq(X, y, rcond=None)
            mom_pred = float(cf[0] + cf[1] * mom.values[-1])
            # Compose to YoY
            yoy_T = float(panel_full.loc[origin, cid])
            mom_T_minus_11 = float(mom.iloc[-12])    # MoM 11 months back
            yoy_pred = yoy_T + mom_pred - mom_T_minus_11
            per_comp[cid] = Forecast(origin=origin, target=target,
                                              point=yoy_pred)
        if skip_origin:
            continue
        composed = composer.compose(per_comp, origin, target)
        cbdf_arch_rows.append({
            "origin": origin, "target": target,
            "point": composed.point, "actual": float(panel_full.loc[target, "y"]),
            "today": float(today.loc[origin]) if origin in today.index else np.nan,
        })
    cbdf_arch_df = pd.DataFrame(cbdf_arch_rows).dropna(subset=["actual"])
    pred_dfs["cbdf_archetype_v1"] = cbdf_arch_df
    rmse_cbdf_a = float(np.sqrt(np.mean(
        (cbdf_arch_df["point"] - cbdf_arch_df["actual"]) ** 2)))
    mae_cbdf_a = float(np.mean(
        np.abs(cbdf_arch_df["point"] - cbdf_arch_df["actual"])))
    print(f"  n={len(cbdf_arch_df)}  RMSE={rmse_cbdf_a:.4f}  MAE={mae_cbdf_a:.4f}")

    # ── Bridged CBDF: CBDF nowcast → BLS via OLS bridge ──────────
    # The proper Thales-CBDF-on-BLS architecture: per-component CBDF
    # forecasts produce a Truflation-headline-like nowcast; a rolling
    # OLS bridge maps that to BLS YoY. Mirrors PathA but with CBDF as
    # the Truflation signal instead of raw `truf_yoy`. This is what
    # the head-to-head SHOULD compare to DFM, since the direct-CBDF
    # output is on a Truflation scale, not BLS scale.
    def _bridge_cbdf(cbdf_df: pd.DataFrame,
                          calib_window: int = 24) -> pd.DataFrame:
        """At each origin, fit `BLS[t] ~ α + β·BLS[t-1] + γ·CBDF_pred[t]`
        on the trailing calib_window of (BLS, CBDF_pred) pairs, then
        predict BLS[T+1] using BLS[T] and CBDF_pred[T+1]."""
        # Align cbdf predictions with BLS targets
        cbdf_indexed = cbdf_df.set_index("target")[["point", "actual"]]
        cbdf_indexed = cbdf_indexed.rename(columns={"point": "cbdf_pred"})
        # We need BLS_lag = actual at the origin (= BLS[T-1] for target T)
        cbdf_sorted = cbdf_indexed.sort_index()
        cbdf_sorted["bls_lag"] = cbdf_sorted["actual"].shift(1)
        cbdf_sorted = cbdf_sorted.dropna()

        rows = []
        for i in range(calib_window, len(cbdf_sorted) - 1):
            train = cbdf_sorted.iloc[i - calib_window: i]
            target_idx = cbdf_sorted.index[i + 1]
            X = np.column_stack([
                np.ones(len(train)),
                train["bls_lag"].values,
                train["cbdf_pred"].values,
            ])
            y = train["actual"].values
            cf, *_ = np.linalg.lstsq(X, y, rcond=None)
            # At target_idx: bls_lag = actual at idx[i], cbdf_pred = cbdf at i+1
            bls_lag_t = float(cbdf_sorted.iloc[i]["actual"])
            cbdf_t = float(cbdf_sorted.loc[target_idx, "cbdf_pred"])
            pred = float(cf[0] + cf[1] * bls_lag_t + cf[2] * cbdf_t)
            rows.append({
                "origin": cbdf_sorted.index[i],
                "target": target_idx,
                "point": pred,
                "actual": float(cbdf_sorted.loc[target_idx, "actual"]),
                "today": bls_lag_t,
                "cbdf_input": cbdf_t,
                "alpha": float(cf[0]),
                "beta_lag": float(cf[1]),
                "gamma_cbdf": float(cf[2]),
            })
        return pd.DataFrame(rows)

    print("\n── bridged_cbdf_persistence_v1 ──")
    bridged_persist = _bridge_cbdf(cbdf_persist_df, calib_window=12)
    pred_dfs["bridged_cbdf_persistence_v1"] = bridged_persist
    if not bridged_persist.empty:
        rmse_bp = float(np.sqrt(np.mean(
            (bridged_persist["point"] - bridged_persist["actual"]) ** 2)))
        mae_bp = float(np.mean(np.abs(
            bridged_persist["point"] - bridged_persist["actual"])))
        print(f"  n={len(bridged_persist)}  RMSE={rmse_bp:.4f}  MAE={mae_bp:.4f}")

    print("\n── bridged_cbdf_archetype_v1 ──")
    bridged_arch = _bridge_cbdf(cbdf_arch_df, calib_window=12)
    pred_dfs["bridged_cbdf_archetype_v1"] = bridged_arch
    if not bridged_arch.empty:
        rmse_ba = float(np.sqrt(np.mean(
            (bridged_arch["point"] - bridged_arch["actual"]) ** 2)))
        mae_ba = float(np.mean(np.abs(
            bridged_arch["point"] - bridged_arch["actual"])))
        print(f"  n={len(bridged_arch)}  RMSE={rmse_ba:.4f}  MAE={mae_ba:.4f}")

    # ── Cleveland Fed alone ─────────────────────────────────────
    print("\n── clevfed_v1 ──")
    clev_rows = []
    for origin in origins:
        try:
            target = panel.index[panel.index.get_loc(origin) + 1]
        except IndexError:
            continue
        if pd.isna(panel.loc[origin, "clevfed"]):
            continue
        clev_rows.append({
            "origin": origin, "target": target,
            "point": float(panel.loc[origin, "clevfed"]),
            "actual": float(panel.loc[target, "y"]),
            "today": float(today.loc[origin]) if origin in today.index else np.nan,
        })
    clev_df = pd.DataFrame(clev_rows).dropna(subset=["actual"])
    pred_dfs["clevfed_v1"] = clev_df
    rmse_clev = float(np.sqrt(np.mean(
        (clev_df["point"] - clev_df["actual"]) ** 2)))
    mae_clev = float(np.mean(np.abs(clev_df["point"] - clev_df["actual"])))
    print(f"  n={len(clev_df)}  RMSE={rmse_clev:.4f}  MAE={mae_clev:.4f}")

    # ── Cleveland + Thales (use MoM-composed AR(1) as the Thales signal) ──
    # The MoM-composed forecaster has a longer history than CBDF (no
    # component-data limitation), giving us a usable calibration window
    # for the linear combo.
    print("\n── clev_plus_thales_v1 ──")
    thales_panel = panel.copy()
    if "ar1_mom_composed_v1" in pred_dfs:
        thales_signal = pred_dfs["ar1_mom_composed_v1"].set_index("target")["point"]
        thales_panel["thales_pred"] = thales_signal.reindex(thales_panel.index)
    else:
        thales_panel["thales_pred"] = pd.Series(dtype=float)
    rows_combo = _attribute_clev_combo(
        thales_panel, "thales_pred",
        origin_min=thales_panel.index.min(),
        calib_window_months=24)
    combo_df = pd.DataFrame(rows_combo)
    if not combo_df.empty:
        combo_df = combo_df.dropna(subset=["actual"])
    pred_dfs["clev_plus_thales_v1"] = combo_df
    if len(combo_df) > 0:
        rmse_combo = float(np.sqrt(np.mean(
            (combo_df["point"] - combo_df["actual"]) ** 2)))
        mae_combo = float(np.mean(
            np.abs(combo_df["point"] - combo_df["actual"])))
        print(f"  n={len(combo_df)}  RMSE={rmse_combo:.4f}  MAE={mae_combo:.4f}")
    else:
        print("  empty — no overlap window")

    # ── Comparison table ─────────────────────────────────────────
    print()
    print("=" * 78)
    print("Comparison table")
    print("=" * 78)
    print()
    print(f"  {'model':<26s}  {'n':>4s}  {'RMSE':>7s}  {'MAE':>6s}  "
            f"{'Δ% vs persist':>14s}  {'Δ% vs DFM':>11s}  {'Δ% vs Clev':>11s}")
    print("  " + "-" * 95)

    persist_rmse = (blocks["persistence_v1"].rmse
                          if "persistence_v1" in blocks else None)
    dfm_rmse = (blocks["dfm_stock_watson_v1"].rmse
                    if "dfm_stock_watson_v1" in blocks else None)
    clev_rmse_total = rmse_clev

    summary_rows = []
    for name in ["persistence_v1", "ar1_yoy_v1", "ar1_mom_composed_v1",
                      "patha_v1", "dfm_stock_watson_v1",
                      "cbdf_persistence_v1", "cbdf_archetype_v1",
                      "bridged_cbdf_persistence_v1", "bridged_cbdf_archetype_v1",
                      "clevfed_v1", "clev_plus_thales_v1"]:
        if name in blocks:
            r = blocks[name].rmse
            m = blocks[name].mae
            n = blocks[name].n
        elif name in pred_dfs and not pred_dfs[name].empty:
            df = pred_dfs[name]
            r = float(np.sqrt(np.mean((df["point"] - df["actual"]) ** 2)))
            m = float(np.mean(np.abs(df["point"] - df["actual"])))
            n = len(df)
        else:
            continue
        d_persist = ((1 - r / persist_rmse) * 100
                          if persist_rmse else float("nan"))
        d_dfm = (1 - r / dfm_rmse) * 100 if dfm_rmse else float("nan")
        d_clev = (1 - r / clev_rmse_total) * 100
        print(f"  {name:<26s}  {n:>4d}  {r:>7.4f}  {m:>6.4f}  "
                f"{d_persist:+>13.2f}%  {d_dfm:+>10.2f}%  {d_clev:+>10.2f}%")
        summary_rows.append({
            "model": name, "n": n, "rmse": r, "mae": m,
            "rmse_red_vs_persist_pct": d_persist,
            "rmse_red_vs_dfm_pct": d_dfm,
            "rmse_red_vs_clev_pct": d_clev,
        })

    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / "okeefe_headtohead_summary.csv", index=False)

    # ── DM tests vs DFM and vs Clev for the key models ───────────
    print()
    print("=" * 78)
    print("Diebold-Mariano pairwise tests (squared-error loss, h=1)")
    print("=" * 78)
    print()

    def common_window(a: pd.DataFrame, b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a_s = a.set_index("target")["point"]
        b_s = b.set_index("target")["point"]
        actual_s = a.set_index("target")["actual"]
        common = a_s.index.intersection(b_s.index)
        return (a_s.loc[common].values, b_s.loc[common].values,
                actual_s.loc[common].values)

    def dm_pair(label: str, key_a: str, key_b: str) -> None:
        """DM test: errors of A vs errors of B, where positive stat
        means A has larger loss → B is more accurate."""
        if key_a not in pred_dfs or key_b not in pred_dfs:
            return
        ap, bp, act = common_window(pred_dfs[key_a], pred_dfs[key_b])
        if len(act) < 6:
            return
        errors_a = ap - act
        errors_b = bp - act
        res = diebold_mariano(errors_a, errors_b, loss="squared")
        print(f"  {label:<35s}  DM={res.statistic:+.3f}  "
                f"p(two-sided)={res.pvalue:.4f}  n={len(act)}")

    dm_pair("CBDF-archetype vs DFM",          "dfm_stock_watson_v1", "cbdf_archetype_v1")
    dm_pair("CBDF-persist vs DFM",            "dfm_stock_watson_v1", "cbdf_persistence_v1")
    dm_pair("Bridged-CBDF-archetype vs DFM",  "dfm_stock_watson_v1", "bridged_cbdf_archetype_v1")
    dm_pair("Bridged-CBDF-persist vs DFM",    "dfm_stock_watson_v1", "bridged_cbdf_persistence_v1")
    dm_pair("Bridged-CBDF-archetype vs MoM-composed",
              "ar1_mom_composed_v1", "bridged_cbdf_archetype_v1")
    dm_pair("Clev+Thales vs Clev alone",
              "clevfed_v1", "clev_plus_thales_v1")
    dm_pair("AR(1)-MoM vs AR(1)-YoY (Fix #5)",
              "ar1_yoy_v1", "ar1_mom_composed_v1")
    dm_pair("MoM-composed AR(1) vs DFM",
              "dfm_stock_watson_v1", "ar1_mom_composed_v1")

    print(f"\nSaved → {OUT_DIR}/okeefe_headtohead_summary.csv")


if __name__ == "__main__":
    main()
