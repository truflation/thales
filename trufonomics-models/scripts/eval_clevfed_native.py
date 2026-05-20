"""Cleveland Fed nowcast evaluated in its NATIVE same-month frame.

The earlier `eval_official_baselines.py` run used `clev[T]` as a +1m
forecast for `y[T+1]`. That's a frame mismatch — Cleveland Fed's product
is a same-month nowcast (an estimate of `y[T]` built up through month T,
finalized at end of T, with BLS publishing T's actual ~13 days into T+1).

Here we score it natively:

    pred[T] = clev[T]      target = y[T]      (h = 0)

The natural information-equivalent baseline at end of month T is
**"last published BLS print"** = `y[T-1]`. That's what a person reading
news headlines without any model would say.

  pred_lastrel[T] = y[T-1]  (lag-1 persistence applied at h=0)

This matches the question Cleveland Fed's product actually answers: "is
the nowcast better than just remembering last month's number?"

A second, weaker baseline is **`y[T-12]`** — same-month-prior-year. Tells
you whether the nowcast is materially capturing the YoY drift, not just
the seasonal level.

Output: results/baseline_eval/clevfed_native_FINDINGS.md
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
from thales.evaluation import metrics as M  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT = ROOT / "results" / "baseline_eval"
OUT.mkdir(parents=True, exist_ok=True)


def evaluate(target_name: str, start: pd.Timestamp) -> dict:
    """Return a row of metrics for one target."""
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        panel = T.load_panel(store, target_name, as_of=date.today())

    panel = panel.loc[panel.index >= start].copy()
    panel = panel.dropna(subset=["y", "clevfed"])
    if len(panel) < 24:
        return {"target": target_name, "n": 0, "note": "insufficient data"}

    # Native frame: score clev[t] vs y[t]
    actual = panel["y"].values
    clev_pred = panel["clevfed"].values

    # Last-release baseline: at end of month t we know y[t-1]
    last_rel = panel["y"].shift(1).values

    # Year-ago baseline
    year_ago = panel["y"].shift(12).values

    # Mask: drop rows where any baseline is NaN (year_ago drops first 12 obs)
    mask = ~(np.isnan(actual) | np.isnan(clev_pred) |
              np.isnan(last_rel) | np.isnan(year_ago))
    actual = actual[mask]
    clev_pred = clev_pred[mask]
    last_rel = last_rel[mask]
    year_ago = year_ago[mask]
    n = len(actual)

    rmse_clev = M.rmse(clev_pred, actual)
    rmse_last = M.rmse(last_rel, actual)
    rmse_ya = M.rmse(year_ago, actual)
    mae_clev = M.mae(clev_pred, actual)
    mae_last = M.mae(last_rel, actual)

    # Direction: did "this month vs last published" go up?
    # actual_up = y[t] > y[t-1]; pred_up = clev[t] > y[t-1]
    actual_up = actual > last_rel
    clev_up = clev_pred > last_rel
    dir_clev = (actual_up == clev_up).mean()

    # Last-release baseline predicts no change ⇒ pred_up always False
    dir_last = (~actual_up).mean()

    return {
        "target": target_name,
        "n": n,
        "window_start": str(panel.index[mask][0].date()),
        "window_end": str(panel.index[mask][-1].date()),
        "rmse_clev": float(rmse_clev),
        "rmse_last_release": float(rmse_last),
        "rmse_year_ago": float(rmse_ya),
        "rmse_clev_vs_last_pct": (1 - rmse_clev / rmse_last) * 100,
        "rmse_clev_vs_yearago_pct": (1 - rmse_clev / rmse_ya) * 100,
        "mae_clev": float(mae_clev),
        "mae_last_release": float(mae_last),
        "dir_clev": float(dir_clev),
        "dir_last_release": float(dir_last),
        "base_rate_up": float(actual_up.mean()),
    }


def main() -> None:
    start = pd.Timestamp("2014-01-01")
    rows = [evaluate(t, start) for t in T.TARGETS]
    df = pd.DataFrame(rows)

    print()
    print("=" * 86)
    print("Cleveland Fed nowcast — NATIVE same-month frame eval")
    print(f"Window:  {df['window_start'].min()} → {df['window_end'].max()}")
    print("=" * 86)
    print()
    print(f"{'target':<10s}  {'n':>3s}  {'RMSE clev':>9s}  "
          f"{'RMSE last-rel':>13s}  {'RMSE Δ vs last':>14s}  "
          f"{'dir clev':>9s}  {'base up':>7s}")
    print("-" * 86)
    for _, r in df.iterrows():
        print(f"{r['target']:<10s}  {r['n']:>3d}  "
              f"{r['rmse_clev']:>9.4f}  "
              f"{r['rmse_last_release']:>13.4f}  "
              f"{r['rmse_clev_vs_last_pct']:>+13.2f}%  "
              f"{r['dir_clev']:>9.1%}  {r['base_rate_up']:>7.1%}")

    out_csv = OUT / "clevfed_native_results.csv"
    df.to_csv(out_csv, index=False)
    print()
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
