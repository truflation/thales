"""Johansen-gated VECM on real data — Fix #4 evaluation.

Compares four configurations on Truflation Clothing × BLS Apparel CPI
(the canonical cointegrated pair) plus a non-cointegrated control:

  1. Forced VECM (no gate)             — always uses VECM regardless
  2. Johansen-gated → ARDL fallback    — main production candidate
  3. Johansen-gated → bridge fallback  — alternative
  4. Johansen-gated → AR(1) fallback   — most conservative

For each config we report:
  * which branch was taken at each origin
  * out-of-sample RMSE / MAE / coverage / direction-hit
  * % of origins where Johansen detected cointegration

The control panel uses BLS Apparel paired with a known-uncorrelated
series (FRED energy or similar) to verify the gate correctly fires
the fallback when no cointegration exists.
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
from thales.models.archetypes.johansen_gated_vecm import (  # noqa: E402
    JohansenGatedVECM,
    johansen_test,
)
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "real_data_archetypes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_clothing_pair() -> pd.DataFrame:
    """Returns log-level monthly panel: log_truf_clothing × log_bls_apparel."""
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        truf_daily = store.get_vintage("clothing_and_footwear",
                                              date.today()).dropna()
        bls = store.get_vintage("CUSR0000SAA",
                                       date.today()).dropna()

    truf_monthly = truf_daily.resample("ME").last()
    bls_m = bls.copy()
    bls_m.index = bls_m.index.to_period("M").to_timestamp("M")

    df = pd.concat({
        "log_truf": np.log(truf_monthly),
        "log_bls":  np.log(bls_m),
    }, axis=1).dropna()
    return df


def _load_uncorrelated_pair() -> pd.DataFrame:
    """Returns log-level panel of two known-uncorrelated series for
    gate-validation control. We use BLS Apparel × natural-gas Henry Hub
    spot — different economic concepts, no theory-implied cointegration."""
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        bls = store.get_vintage("CUSR0000SAA",
                                       date.today()).dropna()
        try:
            henry = store.get_vintage("DHHNGSP", date.today()).dropna()
        except Exception:
            return pd.DataFrame()    # control panel optional
    bls.index = bls.index.to_period("M").to_timestamp("M")
    henry_m = henry.resample("ME").last()
    df = pd.concat({
        "log_bls": np.log(bls),
        "log_henry": np.log(henry_m),
    }, axis=1).dropna()
    return df


def _eval_panel(panel: pd.DataFrame, target: str, paired: str,
                  label: str, train_min: int = 36,
                  train_window: int = 60) -> dict[str, dict]:
    """Run the four configs on a panel; return labeled summary blocks."""
    print(f"\n{'='*78}")
    print(f"  {label}")
    print(f"  panel: {len(panel)} obs  range "
            f"{panel.index.min():%Y-%m} → {panel.index.max():%Y-%m}")
    print("=" * 78)

    # Static Johansen on full panel (informational)
    j = johansen_test(panel[target].values, panel[paired].values)
    print(f"\nFull-panel Johansen: trace_stat[r=0]={j['trace_stat'][0]:.2f} "
            f"vs CV(95%)={j['cv'][0]:.2f}  "
            f"→  {'COINTEGRATED' if j['cointegrated'] else 'NO COINTEGRATION'}")

    origins = panel.index[train_min + 1:]    # leave gap for horizon=1 target
    if len(origins) < 12:
        print(f"  too few origins ({len(origins)}) — skipping")
        return {}

    configs = [
        ("forced_vecm",
            JohansenGatedVECM(target_col=target, paired_col=paired,
                                  train_window_months=train_window,
                                  train_min=train_min,
                                  significance_level=0.10,    # very lax → almost always VECM
                                  fallback="ardl",
                                  band_method="rolling_conformal",
                                  calib_months=24,
                                  model_id="forced_vecm")),
        ("gated_ardl",
            JohansenGatedVECM(target_col=target, paired_col=paired,
                                  train_window_months=train_window,
                                  train_min=train_min,
                                  significance_level=0.05,
                                  fallback="ardl",
                                  band_method="rolling_conformal",
                                  calib_months=24,
                                  model_id="gated_ardl")),
        ("gated_bridge",
            JohansenGatedVECM(target_col=target, paired_col=paired,
                                  train_window_months=train_window,
                                  train_min=train_min,
                                  significance_level=0.05,
                                  fallback="bridge",
                                  band_method="rolling_conformal",
                                  calib_months=24,
                                  model_id="gated_bridge")),
        ("gated_ar1",
            JohansenGatedVECM(target_col=target, paired_col=paired,
                                  train_window_months=train_window,
                                  train_min=train_min,
                                  significance_level=0.05,
                                  fallback="ar1",
                                  band_method="rolling_conformal",
                                  calib_months=24,
                                  model_id="gated_ar1")),
    ]

    results = {}
    for name, fc in configs:
        forecasts = walk_forward(fc, panel, target, origins, horizon=1)
        if not forecasts:
            print(f"\n  [{name}] no forecasts — skipping")
            continue

        branches = [f.metadata["branch"] for f in forecasts]
        n_vecm = sum(b == "vecm" for b in branches)
        n_fb = len(branches) - n_vecm
        coint_pct = 100 * sum(f.metadata["cointegrated"] for f in forecasts) / len(forecasts)

        df = attach_actuals(forecasts, panel[target])
        block = score(df)

        print(f"\n── {name} ──────────────────────────")
        print(f"  branch counts: VECM={n_vecm}  fallback={n_fb}  "
                f"coint%={coint_pct:.1f}")
        print(f"  RMSE: {block.rmse:.6f}   MAE: {block.mae:.6f}")
        if block.cov80 is not None:
            print(f"  cov80: {block.cov80:.1%}   cov95: {block.cov95:.1%}")
        if block.dir_hit is not None:
            print(f"  direction hit: {block.dir_hit:.1%}")

        results[name] = {
            "n": block.n,
            "rmse": block.rmse,
            "mae": block.mae,
            "cov80": block.cov80,
            "cov95": block.cov95,
            "dir_hit": block.dir_hit,
            "n_vecm": n_vecm,
            "n_fallback": n_fb,
            "coint_pct": coint_pct,
        }

    return results


def main() -> None:
    print("=" * 78)
    print("Johansen-gated VECM evaluation — Fix #4")
    print("=" * 78)

    # ── Cointegrated pair: Truflation Clothing × BLS Apparel ────────
    df_cloth = _load_clothing_pair()
    res_cloth = _eval_panel(
        df_cloth, target="log_bls", paired="log_truf",
        label="Truflation Clothing × BLS Apparel CPI (cointegrated)")

    # ── Control: BLS Apparel × Henry Hub (presumed not cointegrated) ──
    df_ctrl = _load_uncorrelated_pair()
    res_ctrl = {}
    if not df_ctrl.empty:
        res_ctrl = _eval_panel(
            df_ctrl, target="log_bls", paired="log_henry",
            label="BLS Apparel × Henry Hub gas (control / not cointegrated)")

    # Persist
    rows = []
    for name, r in res_cloth.items():
        rows.append({"panel": "clothing", "model": name, **r})
    for name, r in res_ctrl.items():
        rows.append({"panel": "control", "model": name, **r})
    pd.DataFrame(rows).to_csv(
        OUT_DIR / "johansen_gated_vecm_results.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'johansen_gated_vecm_results.csv'}")


if __name__ == "__main__":
    main()
