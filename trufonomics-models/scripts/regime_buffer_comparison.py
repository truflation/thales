"""Regime-transition-buffer comparison — Fix #6.

Compares the three buffer methods on the same Headline CPI same-month
nowcast frame:

  1. ``filtered`` — original behavior. σ̂ uses smoothed P(high) at T.
  2. ``transition`` — one-step-ahead Markov forecast (Fix #6 default).
  3. ``transition_max`` — pin to max(σ_low, σ_high) within the
     transition zone (most conservative).

Reports:
  * Coverage at 80% / 95% nominal (calibration target)
  * Mean band width (the cost of widening)
  * Behavior at near-known regime transitions (COVID 2020, peak-
    inflation 2022)
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
    RegimeConditionalBridgeNowcaster,
)
from thales import targets as T  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
KAIROS_PARQUET = Path("/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
TRUFL_HEADLINE_COL = "truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy"
OUT_DIR = ROOT / "results" / "baseline_eval"


def _load_panel() -> pd.DataFrame:
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        bls_yoy = T.load_target_yoy(store, "cpi", as_of=date.today())
    parq = pd.read_parquet(KAIROS_PARQUET)
    parq["date"] = pd.to_datetime(parq["date"])
    parq = parq.set_index("date").sort_index()
    truf_yoy = (parq[TRUFL_HEADLINE_COL].dropna()
                   .resample("ME").last())
    return pd.concat({"bls_yoy": bls_yoy, "truf_yoy": truf_yoy},
                          axis=1).dropna()


def main() -> None:
    print("=" * 78)
    print("Regime-transition buffer comparison — Fix #6")
    print("=" * 78)

    panel = _load_panel()
    print(f"\nPanel: n={len(panel)}  range "
          f"{panel.index.min():%Y-%m} → {panel.index.max():%Y-%m}")

    # Need ≥50 residuals before MS fits; train_window=60 → first
    # eligible origin is at index 60.
    origins = panel.index[60:]
    today = panel["bls_yoy"].shift(1)

    rows: list[dict] = []
    per_origin_records: list[dict] = []

    for buffer_method in ("filtered", "transition", "transition_max"):
        fc = RegimeConditionalBridgeNowcaster(
            target_bls_col="bls_yoy",
            truf_col="truf_yoy",
            train_window_months=60,
            train_min=24,
            buffer_method=buffer_method,
            transition_threshold=0.20,
            model_id=f"regime_cond_{buffer_method}")
        forecasts = walk_forward(fc, panel, "bls_yoy", origins, horizon=0)

        # Force same-month frame: target = origin (as in the legacy
        # gate-2 script). walk_forward(horizon=0) already does this.
        df = attach_actuals(forecasts, panel["bls_yoy"], today_baseline=today)
        if df.empty:
            print(f"\n  [{buffer_method}] no scored rows — skipping")
            continue
        block = score(df)

        # Per-origin records for downstream analysis
        for f in forecasts:
            per_origin_records.append({
                "origin": f.origin,
                "buffer": buffer_method,
                "p_high": f.metadata.get("p_high"),
                "p_high_eff": f.metadata.get("p_high_eff"),
                "sigma": f.metadata.get("sigma_conditional"),
                "width80": (f.hi80 - f.lo80) if f.has_bands else None,
                "width95": (f.hi95 - f.lo95) if f.has_bands else None,
            })

        print()
        print(f"── buffer_method = {buffer_method} ──")
        print("  " + block.summary().replace("\n", "\n  "))

        rows.append({
            "buffer_method": buffer_method,
            "n": block.n,
            "rmse": block.rmse,
            "cov80": block.cov80,
            "cov95": block.cov95,
            "width80": block.width80,
            "width95": block.width95,
            "dir_hit": block.dir_hit,
        })

    # Comparison
    print()
    print("=" * 78)
    print("Comparison")
    print("=" * 78)
    print()
    print(f"  {'buffer_method':<22s}  {'cov80':>7s}  {'cov95':>7s}  "
            f"{'width80':>9s}  {'width95':>9s}")
    print("  " + "-" * 60)
    for r in rows:
        print(f"  {r['buffer_method']:<22s}  {r['cov80']:>6.1%}  "
                f"{r['cov95']:>6.1%}  {r['width80']:>9.4f}  "
                f"{r['width95']:>9.4f}")

    # Highlight regime-transition periods (COVID 2020-Q2, peak inflation 2022-Q3)
    print()
    print("=" * 78)
    print("Width at known-regime-change months — does buffer anticipate?")
    print("=" * 78)
    print()
    targets = ["2020-04-30", "2020-05-31",     # COVID shock
                  "2021-04-30", "2021-05-31",     # post-COVID burst
                  "2022-06-30", "2022-07-31",     # peak inflation
                  "2024-09-30", "2024-10-31"]     # disinflation tail
    rec_df = pd.DataFrame(per_origin_records)
    rec_df["origin"] = pd.to_datetime(rec_df["origin"])
    print(f"  {'origin':<12s}  {'filtered w80':>13s}  "
            f"{'transition w80':>15s}  {'tx_max w80':>11s}  "
            f"{'p_h':>6s}  {'p_h_eff':>8s}")
    print("  " + "-" * 78)
    for tgt in targets:
        try:
            ts = pd.Timestamp(tgt)
            sub = rec_df[rec_df["origin"] == ts]
            if sub.empty:
                continue
            row_filt = sub[sub["buffer"] == "filtered"]
            row_trans = sub[sub["buffer"] == "transition"]
            row_max = sub[sub["buffer"] == "transition_max"]
            if row_filt.empty or row_trans.empty:
                continue
            p_h = row_trans["p_high"].iloc[0]
            p_h_eff = row_trans["p_high_eff"].iloc[0]
            print(f"  {ts:%Y-%m-%d}  {row_filt['width80'].iloc[0]:>13.4f}  "
                    f"{row_trans['width80'].iloc[0]:>15.4f}  "
                    f"{row_max['width80'].iloc[0]:>11.4f}  "
                    f"{p_h:>6.2f}  {p_h_eff:>8.2f}")
        except Exception:    # noqa: BLE001
            continue

    # Persist
    out = OUT_DIR / "regime_buffer_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    out_per = OUT_DIR / "regime_buffer_per_origin.csv"
    rec_df.to_csv(out_per, index=False)
    print(f"\nSaved: {out} and {out_per}")


if __name__ == "__main__":
    main()
