"""Replicate BLS's seasonally-adjusted CPI from the NSA series via X-13ARIMA-SEATS.

This is the verification exercise — confirm we can stand up the same
seasonal-adjustment methodology BLS uses to publish ``CUSR0000SA0`` from
the NSA input ``CUUR0000SA0``.

Methodology
-----------
1. Pull NSA headline (``CUUR0000SA0``) and NSA core (``CUUR0000SA0L1E``)
   level series from the vintage store.
2. Pull BLS's published SA equivalents (``CUSR0000SA0`` from BLS, which
   equals ``CPIAUCSL`` on FRED) for comparison.
3. Run ``statsmodels.tsa.x13.x13_arima_analysis`` on each NSA series.
   That wrapper shells out to the ``x13as`` binary from Census, which
   needs to be on ``PATH`` (or pointed at via ``X13PATH`` env var, or
   passed as ``x12path=``).
4. Compare our SA output against BLS's published SA on three metrics:
   level RMSE, MoM RMSE, YoY RMSE. Report the residual structure (mean
   bias, max abs diff, distribution of |diff|).
5. Persist results to ``results/x13_replication/`` for the methodology
   doc.

Pre-requisite
-------------
The ``x13as`` binary must be installed. On macOS::

    cd /tmp
    curl -O https://www2.census.gov/software/x-13arima-seats/x13as/\
unix-linux/program-archives/x13as_asciisrc-v1-1-b62.tar.gz
    tar xzf x13as_asciisrc-v1-1-b62.tar.gz
    cd x13assrc_V1.1_B62
    sed -i '' 's/-static//g' makefile.gf       # strip macOS-incompatible flag
    make -f makefile.gf FC=gfortran CC=clang
    sudo cp x13as_ascii /usr/local/bin/x13as

Statsmodels auto-discovers the binary via ``PATH`` or ``X13PATH``.

Run
---
::

    uv run python scripts/x13_replicate_bls_sa.py
    uv run python scripts/x13_replicate_bls_sa.py --series CUUR0000SA0L1E
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "x13_replication"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# NSA → SA series mapping. BLS publishes both versions for each major CPI
# aggregate; CUUR (NSA) is the X-13 input, CUSR (SA) is the published
# output we compare against.
SERIES_MAP = {
    "CUUR0000SA0":    "CUSR0000SA0",     # headline
    "CUUR0000SA0L1E": "CUSR0000SA0L1E",  # core
}


def _find_x13_binary() -> str | None:
    for candidate in ["x13as", "x13as_ascii"]:
        path = shutil.which(candidate)
        if path:
            return path
    env_path = os.environ.get("X13PATH")
    if env_path:
        for candidate in ["x13as", "x13as_ascii"]:
            full = Path(env_path) / candidate
            if full.exists():
                return str(full)
    return None


def load_level_series(con: duckdb.DuckDBPyConnection,
                          series_id: str,
                          source: str) -> pd.Series:
    rows = con.execute(
        "SELECT reference_date, value FROM vintage "
        "WHERE series_id = ? AND source = ? "
        "AND (series_id, reference_date, as_of_date) IN ("
        "  SELECT series_id, reference_date, MAX(as_of_date) "
        "  FROM vintage WHERE series_id = ? AND source = ? "
        "  GROUP BY series_id, reference_date) "
        "ORDER BY reference_date",
        [series_id, source, series_id, source],
    ).fetchall()
    if not rows:
        raise RuntimeError(f"empty series {series_id} (source={source})")
    idx = [pd.Timestamp(r[0]) + pd.offsets.MonthEnd(0) for r in rows]
    return pd.Series([r[1] for r in rows], index=idx, name=series_id)


def yoy_from_levels(levels: pd.Series) -> pd.Series:
    out = {}
    for t in levels.index:
        denom = pd.Timestamp(year=t.year - 1, month=t.month, day=1) \
                    + pd.offsets.MonthEnd(0)
        if denom in levels.index:
            out[t] = (levels.loc[t] / levels.loc[denom] - 1.0) * 100.0
    return pd.Series(out).sort_index()


def mom_from_levels(levels: pd.Series) -> pd.Series:
    return (levels / levels.shift(1) - 1.0) * 100.0


def run_x13(nsa: pd.Series, x13_path: str) -> pd.Series:
    """Run X-13ARIMA-SEATS on an NSA series; return the SA output."""
    from statsmodels.tsa.x13 import x13_arima_analysis

    # X-13 needs the series indexed by a pd.PeriodIndex or DatetimeIndex
    # with a monthly frequency. We pass MonthBegin so the wrapper
    # recognises it; the SA values come back at the same dates.
    x = nsa.copy()
    x.index = pd.PeriodIndex(x.index, freq="M")
    result = x13_arima_analysis(
        endog=x,
        x12path=x13_path,
        outlier=True,                   # auto-detect outliers (BLS does this)
        prefer_x13=True,                # use SEATS, not legacy X-11
        log=None,                       # let x13 pick log vs additive
        retspec=True,
    )
    sa = result.seasadj
    sa.index = (sa.index.to_timestamp(how="end").floor("D")
                  + pd.offsets.MonthEnd(0))
    return sa.rename(f"our_x13_sa")


def compare(our_sa: pd.Series, bls_sa: pd.Series) -> dict:
    common = our_sa.index.intersection(bls_sa.index)
    if len(common) < 24:
        raise RuntimeError(
            f"not enough overlap between our SA and BLS SA "
            f"({len(common)} months)")
    a, b = our_sa.loc[common], bls_sa.loc[common]

    # Level metrics
    diff_level = a - b
    pct_diff_level = (a / b - 1.0) * 100.0

    # MoM metrics
    a_mom = mom_from_levels(a)
    b_mom = mom_from_levels(b)
    diff_mom = (a_mom - b_mom).dropna()

    # YoY metrics
    a_yoy = yoy_from_levels(a)
    b_yoy = yoy_from_levels(b)
    diff_yoy_index = a_yoy.index.intersection(b_yoy.index)
    diff_yoy = (a_yoy.loc[diff_yoy_index] - b_yoy.loc[diff_yoy_index])

    def stats(d: pd.Series, unit: str) -> dict:
        return {
            f"n":                 int(len(d)),
            f"mean_{unit}":       float(d.mean()),
            f"median_{unit}":     float(d.median()),
            f"std_{unit}":        float(d.std()),
            f"rmse_{unit}":       float(np.sqrt((d ** 2).mean())),
            f"mae_{unit}":        float(d.abs().mean()),
            f"max_abs_{unit}":    float(d.abs().max()),
            f"p95_abs_{unit}":    float(d.abs().quantile(0.95)),
        }

    return {
        "overlap_months":   len(common),
        "window_start":     str(common.min().date()),
        "window_end":       str(common.max().date()),
        "level_index_pt":   stats(diff_level, "index_pt"),
        "level_pct":        stats(pct_diff_level, "pct"),
        "mom_pp":           stats(diff_mom, "pp"),
        "yoy_pp":           stats(diff_yoy, "pp"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default=None,
                        help="restrict to one NSA series id (default: all "
                              "in SERIES_MAP)")
    parser.add_argument("--start", default="2010-01",
                        help="earliest month to use (YYYY-MM)")
    args = parser.parse_args()

    print("=" * 78)
    print("X-13ARIMA-SEATS replication of BLS published SA")
    print("=" * 78)

    x13_path = _find_x13_binary()
    if not x13_path:
        print("\nERROR: x13as binary not found on PATH or X13PATH.")
        print("See script docstring for macOS install commands.")
        sys.exit(2)
    print(f"\nUsing x13 binary: {x13_path}")

    targets = ({args.series: SERIES_MAP[args.series]}
                  if args.series else SERIES_MAP)
    start = pd.Timestamp(args.start)

    con = duckdb.connect(str(VINTAGE_DB), read_only=True)
    summary = {}
    for nsa_id, sa_id in targets.items():
        print()
        print(f"── {nsa_id} (NSA) → {sa_id} (BLS published SA) ──")

        try:
            nsa = load_level_series(con, nsa_id, "bls_direct")
            bls_sa = load_level_series(con, sa_id, "bls_direct")
        except RuntimeError as e:
            print(f"  [skip] {e}")
            continue

        nsa = nsa.loc[nsa.index >= start]
        bls_sa = bls_sa.loc[bls_sa.index >= start]
        print(f"  NSA history:    {nsa.index.min().date()} → "
                f"{nsa.index.max().date()}  (n = {len(nsa)})")
        print(f"  BLS SA history: {bls_sa.index.min().date()} → "
                f"{bls_sa.index.max().date()}  (n = {len(bls_sa)})")

        print(f"  Running X-13ARIMA-SEATS on NSA…")
        our_sa = run_x13(nsa, x13_path)
        print(f"  Our SA output:  {our_sa.index.min().date()} → "
                f"{our_sa.index.max().date()}  (n = {len(our_sa)})")

        cmp = compare(our_sa, bls_sa)
        summary[nsa_id] = cmp

        lp = cmp["level_pct"]
        print(f"\n  Comparison (our X-13 SA vs BLS published SA):")
        print(f"    Overlap:           {cmp['overlap_months']} months "
                f"({cmp['window_start']} → {cmp['window_end']})")
        print(f"    Level % diff:      RMSE = {lp['rmse_pct']:.5f}%   "
                f"max |diff| = {lp['max_abs_pct']:.4f}%")
        mp = cmp["mom_pp"]
        print(f"    MoM diff (pp):     RMSE = {mp['rmse_pp']:.5f} pp   "
                f"max |diff| = {mp['max_abs_pp']:.4f} pp")
        yp = cmp["yoy_pp"]
        print(f"    YoY diff (pp):     RMSE = {yp['rmse_pp']:.5f} pp   "
                f"max |diff| = {yp['max_abs_pp']:.4f} pp")

        # Persist per-series CSV with our SA, BLS SA, the deltas, and the
        # seasonal factors X-13 applied each month. Multiplicative form:
        # factor = NSA / SA. factor > 1 ⇒ that month's reading was "high"
        # vs the deseasonalized trend; factor < 1 ⇒ "low".
        out_df = pd.DataFrame({
            "nsa_input":         nsa,
            "our_x13_sa":        our_sa,
            "bls_published_sa":  bls_sa,
        }).dropna()
        out_df["our_seasonal_factor"] = out_df["nsa_input"] / out_df["our_x13_sa"]
        out_df["our_seasonal_pct"]    = (out_df["our_seasonal_factor"] - 1.0) * 100
        out_df["bls_seasonal_factor"] = out_df["nsa_input"] / out_df["bls_published_sa"]
        out_df["bls_seasonal_pct"]    = (out_df["bls_seasonal_factor"] - 1.0) * 100
        out_df["our_minus_bls_level"] = (
            out_df["our_x13_sa"] - out_df["bls_published_sa"])
        out_df["our_minus_bls_pct"] = (
            (out_df["our_x13_sa"] / out_df["bls_published_sa"] - 1.0) * 100)
        csv_path = OUT_DIR / f"x13_replication_{nsa_id}.csv"
        out_df.to_csv(csv_path)
        print(f"    Saved per-month CSV: {csv_path}")

        # Typical seasonal pattern by calendar month for both our X-13 and
        # BLS's published — readable side-by-side check.
        pattern_df = (out_df.assign(cal_month=lambda d: d.index.month)
                              .groupby("cal_month")[["our_seasonal_pct",
                                                          "bls_seasonal_pct"]]
                              .mean()
                              .round(4))
        pattern_df.index = pd.Index([pd.Timestamp(2026, m, 1).strftime("%b")
                                        for m in pattern_df.index], name="month")
        pattern_path = OUT_DIR / f"x13_replication_pattern_{nsa_id}.csv"
        pattern_df.to_csv(pattern_path)
        print(f"    Saved seasonal pattern: {pattern_path}")

    con.close()

    # Aggregate JSON
    summary_path = OUT_DIR / f"x13_replication_summary_{date.today()}.json"
    summary_path.write_text(json.dumps({
        "as_of_date":   str(date.today()),
        "x13_binary":   x13_path,
        "by_series":    summary,
    }, indent=2, default=str))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
