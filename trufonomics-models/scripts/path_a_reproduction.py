"""Faithful Path A v1 reproduction + 3 experiments.

Per `~/kairos/docs/PRE_REGISTRATION.md` (locked 2026-04-20):

  Variant D headline:
    target  = bls_cpi_yoy/bls_cpi_yoy at month M
    features = [BLS[M-1], Truflation_daily[day=25, M]] + intercept
    OLS, walk-forward, train ≥ 2021-01, min 24 months, first eval 2023-01
    Snapshot day = D = 25

This script reproduces Variant D + Variant D+energy, then runs three
experiments:

  Exp A — Path A reproduction at day=25 with LIVE Truflation feed
  Exp B — Day-of-month value curve at days 5, 10, 15, 20, 25, end
  Exp C — FRED-extended bridge (persistence + Truflation + gasoline +
          WTI + 30yr mortgage rate at day=25)

All scored vs persistence floor + cleveland fed h=0 comparator.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales import targets as T  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
KAIROS_PARQUET = Path("/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
OUT_DIR = ROOT / "results" / "baseline_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Path A pre-reg constants
TRAIN_START = pd.Timestamp("2021-01-01")
EVAL_START = pd.Timestamp("2023-01-01")
MIN_TRAIN = 24


def _day_of_month_snapshot(daily: pd.Series, target_month_end: pd.Timestamp,
                              day: int) -> float:
    """Return Truflation value on `day` of the month containing target_month_end.

    Per pre-reg §2.1: 'or the latest available value on or before the
    25th if the 25th falls on a weekend / holiday'.
    """
    month_start = target_month_end.replace(day=1)
    snap_target = month_start + pd.Timedelta(days=day - 1)
    available = daily.loc[daily.index <= snap_target]
    if available.empty:
        return float("nan")
    return float(available.iloc[-1])


def build_panel(snapshot_day: int = 25) -> pd.DataFrame:
    """Build the monthly panel per Path A pre-reg.

    Columns:
      * bls_yoy            target (from kairos parquet)
      * bls_lag1           BLS[M-1]
      * truf_d{day}        Truflation YoY at day=snapshot_day of month M
      * gasoline_d{day}    gasoline YoY at day=snapshot_day of month M
      * wti_d{day}         WTI at day=snapshot_day of month M
      * mortgage_d{day}    30yr mortgage at day=snapshot_day of month M
    """
    parq = pd.read_parquet(KAIROS_PARQUET)
    parq["date"] = pd.to_datetime(parq["date"])
    parq = parq.set_index("date").sort_index()

    # BLS YoY (monthly, period-start indexed in parquet → resnap to month-end)
    bls = parq["bls_cpi_yoy/bls_cpi_yoy"].dropna()
    bls.index = bls.index.to_period("M").to_timestamp("M")
    bls = bls.sort_index()

    # Daily LIVE Truflation YoY
    truf_daily = parq["truflation_us_cpi_yoy/truflation_us_cpi_yoy"].dropna()
    # Daily gasoline YoY
    gas_daily = parq["us_gasoline_yoy/us_gasoline_yoy"].dropna()

    # FRED covariates
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        wti = store.get_vintage("DCOILWTICO", date.today()).dropna()
        mort = store.get_vintage("MORTGAGE30US", date.today()).dropna()

    # For each month-end target, take day-`snapshot_day` snapshot of each
    # daily series
    rows = []
    for month_end in bls.index:
        rec = {
            "date": month_end,
            "bls_yoy": float(bls.loc[month_end]),
            f"truf_d{snapshot_day}": _day_of_month_snapshot(
                truf_daily, month_end, snapshot_day),
            f"gasoline_d{snapshot_day}": _day_of_month_snapshot(
                gas_daily, month_end, snapshot_day),
            f"wti_d{snapshot_day}": _day_of_month_snapshot(
                wti, month_end, snapshot_day),
            f"mortgage_d{snapshot_day}": _day_of_month_snapshot(
                mort, month_end, snapshot_day),
        }
        rows.append(rec)
    panel = pd.DataFrame(rows).set_index("date")
    panel["bls_lag1"] = panel["bls_yoy"].shift(1)
    return panel


def walk_forward_ols(panel: pd.DataFrame, target_col: str,
                       feature_cols: list[str],
                       train_start: pd.Timestamp = TRAIN_START,
                       eval_start: pd.Timestamp = EVAL_START,
                       min_train: int = MIN_TRAIN,
                       ) -> pd.DataFrame:
    """Walk-forward OLS with intercept. Returns DataFrame with columns
    (origin, target, point, actual, error)."""
    rows = []
    eligible = panel.dropna(subset=[target_col] + feature_cols)
    for origin in eligible.index:
        if origin < eval_start:
            continue
        train = eligible.loc[(eligible.index >= train_start)
                                & (eligible.index < origin)]
        if len(train) < min_train:
            continue
        X_tr = np.column_stack([np.ones(len(train)),
                                  train[feature_cols].values])
        y_tr = train[target_col].values
        coefs, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        x_test = np.concatenate([[1.0],
                                    eligible.loc[origin, feature_cols].values])
        pred = float(x_test @ coefs)
        actual = float(eligible.loc[origin, target_col])
        rows.append({"origin": origin, "target": origin,
                       "point": pred, "actual": actual,
                       "error": pred - actual})
    return pd.DataFrame(rows)


def persistence_predict(panel: pd.DataFrame,
                          target_col: str = "bls_yoy",
                          eval_start: pd.Timestamp = EVAL_START,
                          ) -> pd.DataFrame:
    """Predict y[T] = y[T-1]."""
    rows = []
    for origin in panel.index:
        if origin < eval_start:
            continue
        if pd.isna(panel.loc[origin, "bls_lag1"]):
            continue
        actual = float(panel.loc[origin, target_col])
        if pd.isna(actual):
            continue
        pred = float(panel.loc[origin, "bls_lag1"])
        rows.append({"origin": origin, "target": origin,
                       "point": pred, "actual": actual,
                       "error": pred - actual})
    return pd.DataFrame(rows)


def score_block(name: str, df: pd.DataFrame,
                  persist_df: pd.DataFrame) -> dict:
    """Compute RMSE, MAE, vs-persistence reduction, direction (vs lag-1)."""
    if df.empty:
        return {"name": name, "n": 0}
    # Align with persistence on same origins
    merged = df.merge(persist_df, on="origin",
                          suffixes=("", "_persist"))
    rmse = np.sqrt(np.mean(merged["error"] ** 2))
    rmse_persist = np.sqrt(np.mean(merged["error_persist"] ** 2))
    mae = np.mean(np.abs(merged["error"]))
    rmse_red = (1 - rmse / rmse_persist) * 100 if rmse_persist > 0 else float("nan")
    # Direction: pred_up = point > today (=BLS_lag1), actual_up = actual > today
    today = merged["point_persist"]   # persistence prediction = BLS_lag1
    pred_up = merged["point"] > today
    actual_up = merged["actual"] > today
    dir_acc = (pred_up == actual_up).mean()
    return {
        "name": name, "n": len(merged),
        "rmse": rmse, "mae": mae,
        "rmse_red_vs_persist": rmse_red,
        "direction_acc": dir_acc,
    }


def main() -> None:
    print("=" * 80)
    print("Faithful Path A reproduction + experiments")
    print("=" * 80)

    # ── Exp A: Path A reproduction at day=25 ─────────────────────────────
    print("\n--- Exp A: Path A reproduction at day=25 ---")
    panel = build_panel(snapshot_day=25)
    print(f"Panel: {panel.shape}  range "
          f"{panel.index.min():%Y-%m-%d} → {panel.index.max():%Y-%m-%d}")
    print(f"  bls_yoy notna: {panel['bls_yoy'].notna().sum()}")
    print(f"  truf_d25 notna: {panel['truf_d25'].notna().sum()}")
    print(f"  gasoline_d25 notna: {panel['gasoline_d25'].notna().sum()}")

    persist_df = persistence_predict(panel)
    persist_block = score_block("persistence (BLS[M-1])", persist_df, persist_df)

    path_a_df = walk_forward_ols(panel, "bls_yoy",
                                       ["bls_lag1", "truf_d25"])
    path_a_block = score_block("Path A (persist + truf_d25)",
                                  path_a_df, persist_df)

    path_a_energy_df = walk_forward_ols(panel, "bls_yoy",
                                              ["bls_lag1", "truf_d25",
                                               "gasoline_d25"])
    path_a_energy_block = score_block("Path A + energy",
                                            path_a_energy_df, persist_df)

    persist_energy_df = walk_forward_ols(panel, "bls_yoy",
                                               ["bls_lag1", "gasoline_d25"])
    persist_energy_block = score_block("persist + energy",
                                              persist_energy_df, persist_df)

    print()
    print(f"  {'Variant':<32s}  {'n':>4s}  {'RMSE':>7s}  "
          f"{'vs persist':>10s}  {'Direction':>10s}")
    print("  " + "-" * 75)
    for b in [persist_block, persist_energy_block, path_a_block,
                  path_a_energy_block]:
        if b.get("n", 0) == 0:
            continue
        print(f"  {b['name']:<32s}  {b['n']:>4d}  "
              f"{b['rmse']:>7.4f}  "
              f"{b.get('rmse_red_vs_persist', 0):+>9.2f}%  "
              f"{b.get('direction_acc', 0):>10.1%}")

    # ── Exp C: FRED-extended ────────────────────────────────────────────
    print("\n--- Exp C: FRED-extended bridge (5 features) ---")
    fred_ext_df = walk_forward_ols(
        panel, "bls_yoy",
        ["bls_lag1", "truf_d25", "gasoline_d25",
         "wti_d25", "mortgage_d25"])
    fred_ext_block = score_block("Path A + gas + WTI + mortgage",
                                      fred_ext_df, persist_df)
    print(f"  {fred_ext_block['name']:<35s}  n={fred_ext_block['n']}  "
          f"RMSE={fred_ext_block['rmse']:.4f}  "
          f"Δ persist={fred_ext_block['rmse_red_vs_persist']:+.2f}%  "
          f"dir={fred_ext_block['direction_acc']:.1%}")

    # ── Exp B: Day-of-month value curve ─────────────────────────────────
    print("\n--- Exp B: Day-of-month value curve (Path A + energy) ---")
    print(f"  {'Day':>5s}  {'n':>4s}  {'RMSE':>7s}  {'vs persist':>10s}  {'Direction':>10s}")
    print("  " + "-" * 50)
    curve_rows = []
    for d in [5, 10, 15, 20, 25, 30]:
        panel_d = build_panel(snapshot_day=d)
        persist_d = persistence_predict(panel_d)
        path_a_d_df = walk_forward_ols(
            panel_d, "bls_yoy",
            ["bls_lag1", f"truf_d{d}", f"gasoline_d{d}"])
        block = score_block(f"Path A+E day={d}", path_a_d_df, persist_d)
        curve_rows.append({"day": d, **block})
        if block.get("n", 0) > 0:
            print(f"  {d:>5d}  {block['n']:>4d}  "
                  f"{block['rmse']:>7.4f}  "
                  f"{block.get('rmse_red_vs_persist', 0):+>9.2f}%  "
                  f"{block.get('direction_acc', 0):>10.1%}")

    # ── Cleveland Fed comparator ─────────────────────────────────────────
    print("\n--- Cleveland Fed h=0 (institutional comparator) ---")
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        clev = T.load_nowcast_comparator(store, "cpi", as_of=date.today())
    bls_aligned = panel["bls_yoy"].dropna()
    common = bls_aligned.index.intersection(clev.index)
    common = common[common >= EVAL_START]
    if len(common) > 0:
        clev_df = pd.DataFrame({
            "origin": common,
            "target": common,
            "point": clev.loc[common].values,
            "actual": bls_aligned.loc[common].values,
        })
        clev_df["error"] = clev_df["point"] - clev_df["actual"]
        clev_block = score_block("Cleveland Fed h=0", clev_df, persist_df)
        print(f"  n={clev_block['n']}  RMSE={clev_block['rmse']:.4f}  "
              f"Δ persist={clev_block.get('rmse_red_vs_persist', 0):+.2f}%  "
              f"dir={clev_block.get('direction_acc', 0):.1%}")

    # Save artifacts
    summary = pd.DataFrame([
        persist_block, persist_energy_block, path_a_block,
        path_a_energy_block, fred_ext_block, clev_block,
    ])
    summary.to_csv(OUT_DIR / "path_a_reproduction_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(
        OUT_DIR / "path_a_day_of_month_curve.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/path_a_reproduction_summary.csv")
    print(f"       {OUT_DIR}/path_a_day_of_month_curve.csv")


if __name__ == "__main__":
    main()
