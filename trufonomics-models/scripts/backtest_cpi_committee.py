"""Backtest the canonical CPI committee — avg(BLS-native CBDF,
Truflation-weighted CPI CBDF) — across historical BLS prints.

Procedure (mirrors the production scripts exactly):
  * For each month-end origin from 2018-01-31 onward:
    - Call BLS-native CBDF `forecast_next_yoy` at that origin → point + 80% band
    - Call Truflation-weighted CPI CBDF `forecast_next_yoy` at that origin → point + 80% band
    - Committee point = simple average of the two
    - Committee band (union) = min of lo80s, max of hi80s
    - Persistence baseline = actual BLS YoY at origin
    - Score against actual BLS YoY at origin + 1 month

  * Aggregate:
    - RMSE / MAE / mean bias for each method
    - Diebold-Mariano test (committee vs each individual + persistence)
    - Coverage of committee 80% band
    - Per-origin CSV with all numbers

The forecasters use a 24-month rolling AR(1) calibration window and
anchor-correct to actual BLS YoY at origin (point-in-time — origin's
value is known by definition when forecasting next-month). No
peek-ahead: forecast at T uses only data ≤ T.

Run::

    uv run python scripts/backtest_cpi_committee.py
    uv run python scripts/backtest_cpi_committee.py --start 2020-01-31
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "results" / "next_release_forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_script_as_module(name: str, path: Path):
    """Load a sibling script as a module so we can call its functions
    without triggering its __main__."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


blsnative = _load_script_as_module(
    "blsnative_fwd",
    ROOT / "scripts" / "forecast_next_bls_cpi_blsnative.py")
trufw = _load_script_as_module(
    "trufweights_fwd",
    ROOT / "scripts" / "forecast_next_bls_cpi_trufweights.py")


# ─── Diebold-Mariano (Newey-West HAC, lag 1) ─────────────────────────────


def _newey_west_var(d: np.ndarray, lag: int = 1) -> float:
    d = d - d.mean()
    n = len(d)
    g0 = (d * d).sum() / n
    s = g0
    for k in range(1, lag + 1):
        gk = (d[k:] * d[:-k]).sum() / n
        s += 2 * (1 - k / (lag + 1)) * gk
    return s / n


def diebold_mariano(err_a: np.ndarray, err_b: np.ndarray,
                       lag: int = 1) -> tuple[float, float]:
    """Returns (DM statistic, two-sided p-value).

    Positive statistic ⇒ A has larger loss ⇒ B is more accurate.
    """
    la = err_a ** 2
    lb = err_b ** 2
    d = la - lb
    mask = ~np.isnan(d)
    d = d[mask]
    if len(d) < 3:
        return float("nan"), float("nan")
    var = _newey_west_var(d, lag)
    if var <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(var)
    # Two-sided p
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(stat)))
    return float(stat), float(p)


# ─── Main backtest loop ──────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2018-01-31",
                        help="First origin (month-end)")
    parser.add_argument("--end", type=str, default=None,
                        help="Last origin (month-end); default = latest "
                              "BLS month with a known next-month actual")
    args = parser.parse_args()

    print("=" * 78)
    print("CPI Committee backtest — avg(BLS-native, Truflation-weighted CPI CBDF)")
    print("=" * 78)

    # ── Load data ────────────────────────────────────────────────────
    con = duckdb.connect(str(ROOT / "data" / "vintage_store" / "thales.duckdb"),
                            read_only=True)
    component_levels = blsnative.load_bls_component_levels(con)
    bls_yoy = blsnative.load_bls_headline_yoy(con)
    con.close()
    bls_weights = blsnative.load_bls_weights()
    trufw_weights = trufw.load_truflation_cpi_weights()

    start = pd.Timestamp(args.start)
    end = (pd.Timestamp(args.end) if args.end
              else bls_yoy.index[-2])    # need next-month actual to score
    print(f"\nBLS component panel: {component_levels.shape}")
    print(f"BLS headline YoY: {len(bls_yoy)} months, "
            f"{bls_yoy.index.min().date()} → {bls_yoy.index.max().date()}")
    print(f"Backtest window: {start.date()} → {end.date()}")
    print(f"BLS weights:       sum = {sum(bls_weights.values()):.3f}%")
    print(f"Truflation weights: sum = {sum(trufw_weights.values()):.3f}%")

    # Iterate over month-end origins
    origins = [o for o in component_levels.index
                  if start <= o <= end
                  and o.is_month_end
                  and o in bls_yoy.index]

    print(f"\nWalking {len(origins)} origins…")

    rows: list[dict] = []
    for i, origin in enumerate(origins):
        target = origin + pd.offsets.MonthEnd(1)
        if target not in bls_yoy.index:
            continue
        actual = float(bls_yoy.loc[target])
        persistence = float(bls_yoy.loc[origin])

        # Forecast at this origin from each CBDF
        history = component_levels.loc[component_levels.index <= origin]
        hist_yoy = bls_yoy.loc[bls_yoy.index <= origin]
        try:
            fc_bls = blsnative.forecast_next_yoy(
                history, bls_weights, hist_yoy, origin,
                n_samples=200, seed=int(origin.value % 1_000_000))
            fc_trf = trufw.forecast_next_yoy(
                history, trufw_weights, hist_yoy, origin,
                n_samples=200, seed=int(origin.value % 1_000_000))
        except Exception as e:    # noqa: BLE001
            print(f"  [skip] {origin.date()}: {type(e).__name__}: {e}")
            continue

        committee = (fc_bls["point"] + fc_trf["point"]) / 2.0
        # Committee 80% band = union
        lo80 = min(fc_bls.get("lo80", np.nan), fc_trf.get("lo80", np.nan))
        hi80 = max(fc_bls.get("hi80", np.nan), fc_trf.get("hi80", np.nan))

        rows.append({
            "origin":              origin,
            "target":              target,
            "actual":              actual,
            "persistence":         persistence,
            "bls_native_cbdf":     fc_bls["point"],
            "truf_weighted_cbdf":  fc_trf["point"],
            "committee":           committee,
            "committee_lo80":      lo80,
            "committee_hi80":      hi80,
            "err_persistence_bp":  (persistence - actual) * 100,
            "err_bls_native_bp":   (fc_bls["point"] - actual) * 100,
            "err_trufw_bp":        (fc_trf["point"] - actual) * 100,
            "err_committee_bp":    (committee - actual) * 100,
            "actual_in_80":        bool(lo80 <= actual <= hi80),
        })

        if (i + 1) % 24 == 0:
            print(f"  [{i+1:>3d}/{len(origins)}] origin={origin.date()}  "
                    f"committee err={(committee - actual)*100:+.2f} bp")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No scoreable origins.")
        return

    # Save per-origin CSV
    out_csv = OUT_DIR / "backtest_cpi_committee.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-origin: {out_csv}")

    # ── Aggregate metrics ──────────────────────────────────────────
    def stats(errs_bp: np.ndarray) -> dict:
        errs_pp = errs_bp / 100.0
        return {
            "n":         int(np.isfinite(errs_bp).sum()),
            "rmse_pp":   float(np.sqrt(np.nanmean(errs_pp ** 2))),
            "mae_pp":    float(np.nanmean(np.abs(errs_pp))),
            "mean_bp":   float(np.nanmean(errs_bp)),
            "median_bp": float(np.nanmedian(errs_bp)),
            "p95_abs_bp": float(np.nanpercentile(np.abs(errs_bp), 95)),
        }

    print()
    print("=" * 78)
    print(f"AGGREGATE — {len(df)} OOS prints {df['origin'].iloc[0].date()} → "
            f"{df['origin'].iloc[-1].date()}")
    print("=" * 78)
    print(f"{'method':<25s}  {'n':>4s}  {'RMSE (pp)':>10s}  "
            f"{'MAE (pp)':>10s}  {'mean (bp)':>10s}  {'|err| p95 (bp)':>14s}")
    print("  " + "-" * 75)
    methods = {
        "persistence":         "err_persistence_bp",
        "bls_native_cbdf":     "err_bls_native_bp",
        "truf_weighted_cbdf":  "err_trufw_bp",
        "COMMITTEE (canonical)": "err_committee_bp",
    }
    aggs = {}
    for name, col in methods.items():
        s = stats(df[col].values)
        aggs[name] = s
        print(f"  {name:<25s}  {s['n']:>4d}  {s['rmse_pp']:>9.4f}  "
                f"{s['mae_pp']:>9.4f}  {s['mean_bp']:>+9.2f}  {s['p95_abs_bp']:>13.2f}")

    # Coverage
    cov80 = float(df["actual_in_80"].mean())
    print()
    print(f"Committee 80% band empirical coverage: {cov80*100:.1f}%  "
            f"(nominal 80%)")

    # DM tests
    print()
    print("Diebold-Mariano (committee vs each, Newey-West lag 1, two-sided):")
    print(f"  Positive DM stat ⇒ alternative has LARGER loss "
            f"⇒ committee wins.")
    err_committee = df["err_committee_bp"].values / 100.0
    for alt_name, alt_col in [
        ("persistence",         "err_persistence_bp"),
        ("bls_native_cbdf",     "err_bls_native_bp"),
        ("truf_weighted_cbdf",  "err_trufw_bp"),
    ]:
        err_alt = df[alt_col].values / 100.0
        stat, p = diebold_mariano(err_alt, err_committee, lag=1)
        verdict = ("committee" if stat > 0 and p < 0.10 else
                     alt_name if stat < 0 and p < 0.10 else "tie")
        print(f"  committee vs {alt_name:<22s}  stat={stat:>+6.3f}  "
                f"p={p:.4f}  → {verdict}")

    # RMSE reduction vs persistence
    persist_rmse = aggs["persistence"]["rmse_pp"]
    cmte_rmse = aggs["COMMITTEE (canonical)"]["rmse_pp"]
    bls_rmse = aggs["bls_native_cbdf"]["rmse_pp"]
    trf_rmse = aggs["truf_weighted_cbdf"]["rmse_pp"]
    print()
    print("RMSE reduction vs persistence:")
    print(f"  bls_native_cbdf:    {(1 - bls_rmse/persist_rmse)*100:+.1f}%")
    print(f"  truf_weighted_cbdf: {(1 - trf_rmse/persist_rmse)*100:+.1f}%")
    print(f"  COMMITTEE:          {(1 - cmte_rmse/persist_rmse)*100:+.1f}%")

    # Top/bottom 5 origins by committee error
    print()
    print("Worst 5 origins (by absolute committee error):")
    worst = df.reindex(df["err_committee_bp"].abs().sort_values(ascending=False).index).head(5)
    for _, r in worst.iterrows():
        print(f"  {r['target'].date()}  actual {r['actual']:>5.2f}%  "
                f"committee {r['committee']:>5.2f}%  "
                f"err {r['err_committee_bp']:>+7.2f} bp")

    print()
    print("Best 5 origins (by absolute committee error):")
    best = df.reindex(df["err_committee_bp"].abs().sort_values().index).head(5)
    for _, r in best.iterrows():
        print(f"  {r['target'].date()}  actual {r['actual']:>5.2f}%  "
                f"committee {r['committee']:>5.2f}%  "
                f"err {r['err_committee_bp']:>+7.2f} bp")


if __name__ == "__main__":
    main()
