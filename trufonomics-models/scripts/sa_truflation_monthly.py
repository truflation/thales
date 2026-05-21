"""Produce a seasonally-adjusted version of Truflation's daily index.

Applies the same X-13ARIMA-SEATS pipeline validated against BLS published
SA in ``scripts/x13_replicate_bls_sa.py`` to:

  1. Each top-level Truflation category (housing, transport, etc.).
  2. A composed Truflation headline built by weighting the categories
     with the v2 ``relative_importance`` weights.

For each series the script reports:

  * Whether X-13 detects seasonality (M7 stat; QS test on the regARIMA
    residuals — both come out of X-13's printout).
  * Volatility reduction: SD of NSA MoM vs SD of SA MoM. If seasonality
    is real, SA MoM should be quieter than NSA MoM.
  * Mean absolute SA-vs-NSA gap in level percent (smaller = less
    seasonality in the underlying series).

Output:
  * ``results/x13_replication/sa_truflation_<series>.csv`` per series
  * ``results/x13_replication/sa_truflation_summary_<date>.json`` —
    aggregate diagnostics

Run::

    uv run python scripts/sa_truflation_monthly.py
"""

from __future__ import annotations

import json
import shutil
import sys
import warnings
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
WEIGHTS_CSV = ROOT / "data" / "truflation" / "weights" / "categories-tables-v2.csv"
OUT_DIR = ROOT / "results" / "x13_replication"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Top-level Truflation streams + their category_id in the v2 weights file.
TRUFLATION_CATEGORIES = {
    "food_and_non_alcoholic_beverages": 78,
    "housing":                          79,
    "transport":                        80,
    "utilities":                        81,
    "health":                           82,
    "household_durables_and_daily_use_items": 83,
    "alcohol_and_tobacco":              84,
    "clothing_and_footwear":            85,
    "education":                        86,
    "communications":                   87,
    "recreation_and_culture":           88,
    "other":                            89,
}


def _find_x13_binary() -> str | None:
    for name in ["x13as", "x13as_ascii"]:
        p = shutil.which(name)
        if p:
            return p
    return None


def load_daily_series(con: duckdb.DuckDBPyConnection,
                          series_id: str,
                          source: str = "truf_network") -> pd.Series:
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
        return pd.Series(dtype=float, name=series_id)
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
    return pd.Series([r[1] for r in rows], index=idx, name=series_id)


def daily_to_monthly(daily: pd.Series) -> pd.Series:
    """Aggregate a daily series to monthly mean, indexed at month-end."""
    monthly = daily.resample("ME").mean()
    return monthly.dropna()


def load_truflation_weights() -> dict[int, float]:
    """Truflation's own headline weights (`relative_importance`)."""
    return _load_weights_column("relative_importance")


def load_bls_weights() -> dict[int, float]:
    """BLS-CPI-equivalent weights (`bls_relative_importance`) applied to
    Truflation streams. Composing here gives a Truflation aggregation
    that is directly comparable in scope to BLS published CPI."""
    return _load_weights_column("bls_relative_importance")


def load_pce_weights() -> dict[int, float]:
    """BEA-PCE-equivalent weights (`pce_relative_importance`) applied to
    Truflation streams. Composing here gives a Truflation aggregation
    that is directly comparable in scope to BEA published PCE."""
    return _load_weights_column("pce_relative_importance")


def _load_weights_column(col: str) -> dict[int, float]:
    wt = pd.read_csv(WEIGHTS_CSV)
    top = wt[(wt["subcategory_id"] == 0) & (wt["source_id"] == 0)].copy()
    top["category_id"] = top["category_id"].astype(int)
    return dict(zip(top["category_id"], top[col].astype(float)))


WEIGHT_SCHEMES = {
    "truflation_us_cpi": ("relative_importance",     load_truflation_weights),
    "truflation_bls_cpi": ("bls_relative_importance", load_bls_weights),
    "truflation_bea_pce": ("pce_relative_importance", load_pce_weights),
}


def run_x13(monthly: pd.Series, x13_path: str) -> dict:
    """Run X-13 on a monthly series. Returns dict with SA series, factors,
    M7 (seasonality strength), and a 'seasonality_detected' flag."""
    from statsmodels.tsa.x13 import x13_arima_analysis

    x = monthly.copy()
    x.index = pd.PeriodIndex(x.index, freq="M")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = x13_arima_analysis(
            endog=x,
            outlier=True,
            prefer_x13=True,
            x12path=x13_path,
        )
    sa = result.seasadj.copy()
    sa.index = sa.index.to_timestamp(how="end").floor("D") + pd.offsets.MonthEnd(0)
    trend = result.trend.copy()
    trend.index = (trend.index.to_timestamp(how="end").floor("D")
                     + pd.offsets.MonthEnd(0))

    # Parse diagnostics from raw X-13 stdout. M7 is X-11-only and won't be
    # present when prefer_x13=True (SEATS). Fall back to the empirical
    # volatility test: if SA log-MoM SD is materially lower than NSA log-MoM
    # SD, there was seasonal structure to remove. The threshold below
    # (>=5% volatility reduction) is loose; categories with strong
    # seasonality remove 20-60% on this dataset.
    raw = getattr(result, "results", "") or ""
    m7 = _parse_m7(raw)
    qs_p = _parse_qs_p(raw)
    mom_nsa_sd = float((np.log(monthly) - np.log(monthly.shift(1))).dropna().std() * 100)
    mom_sa_sd = float((np.log(sa) - np.log(sa.shift(1))).dropna().std() * 100)
    vol_reduction_pct = ((mom_nsa_sd - mom_sa_sd) / mom_nsa_sd) * 100
    seasonality_detected = (
        (m7 is not None and m7 < 1.0)
        or (qs_p is not None and qs_p < 0.01)
        or (vol_reduction_pct >= 5.0)
    )

    return {
        "sa":                    sa,
        "trend":                 trend,
        "m7":                    m7,
        "qs_p_value":            qs_p,
        "seasonality_detected":  seasonality_detected,
    }


def _parse_m7(raw: str) -> float | None:
    """Pull the M7 diagnostic from X-13's stdout. M7 = combined test for
    identifiable seasonality. Values < 1.0 indicate seasonality is
    identifiable; >= 1.0 suggests no identifiable seasonality.
    """
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("M7 "):
            try:
                return float(s.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _parse_qs_p(raw: str) -> float | None:
    """Pull the QS test p-value from the regARIMA residuals printout."""
    # QS test result typically prints as 'QS (...) = stat   p = 0.xxx'
    for line in raw.splitlines():
        s = line.strip()
        if "QS" in s and "p =" in s:
            try:
                return float(s.split("p =")[1].strip().split()[0])
            except (IndexError, ValueError):
                return None
    return None


def analyze_series(name: str, monthly: pd.Series,
                       x13_path: str) -> dict:
    print(f"\n── {name} ──  ({len(monthly)} months, "
            f"{monthly.index.min().date()} → {monthly.index.max().date()})")
    if len(monthly) < 60:
        print(f"  [skip] insufficient history (need ≥60 months)")
        return {"name": name, "status": "skipped_short"}

    try:
        res = run_x13(monthly, x13_path)
    except Exception as e:    # noqa: BLE001
        print(f"  [skip] X-13 failed: {type(e).__name__}: {e}")
        return {"name": name, "status": "failed", "error": str(e)}

    sa, m7, qs_p, detected = res["sa"], res["m7"], res["qs_p_value"], res["seasonality_detected"]

    # Volatility comparison on log-MoM
    mom_nsa = (np.log(monthly) - np.log(monthly.shift(1))).dropna() * 100
    mom_sa = (np.log(sa) - np.log(sa.shift(1))).dropna() * 100
    sd_nsa, sd_sa = float(mom_nsa.std()), float(mom_sa.std())
    sd_reduction_pct = ((sd_nsa - sd_sa) / sd_nsa) * 100

    # Mean absolute level gap (SA - NSA) / NSA × 100
    common = sa.index.intersection(monthly.index)
    abs_pct_gap = float(((sa.loc[common] / monthly.loc[common] - 1.0).abs()
                              * 100).mean())

    print(f"  Seasonality detected:        {detected}")
    if m7 is not None:
        print(f"  M7 stat (X-13 combined):     {m7:.3f}   (< 1 ⇒ identifiable)")
    if qs_p is not None:
        print(f"  QS p-value (residuals):      {qs_p:.4f}  (< 0.01 ⇒ rejects no-seasonality)")
    print(f"  NSA log-MoM SD:              {sd_nsa:.3f} pp")
    print(f"  SA  log-MoM SD:              {sd_sa:.3f} pp  "
            f"({sd_reduction_pct:+.1f}% vs NSA)")
    print(f"  Mean |SA − NSA| / NSA × 100: {abs_pct_gap:.3f}%")

    # Per-month seasonal factor: multiplicative form, factor = NSA / SA.
    # Reading: factor > 1 ⇒ that month's NSA reading is "high" relative
    # to the deseasonalized trend; factor < 1 ⇒ "low".
    seasonal_factor = monthly / sa
    seasonal_pct = (seasonal_factor - 1.0) * 100

    # Persist per-series CSV
    out_df = pd.DataFrame({
        "nsa":              monthly,
        "sa":               sa,
        "trend":            res["trend"],
        "seasonal_factor":  seasonal_factor,
        "seasonal_pct":     seasonal_pct,
    })
    out_df["sa_minus_nsa_pct"] = (out_df["sa"] / out_df["nsa"] - 1.0) * 100
    csv_path = OUT_DIR / f"sa_truflation_{name}.csv"
    out_df.to_csv(csv_path)

    # Typical seasonal pattern by calendar month (averaged across years).
    # Gives a 12-row summary of how the series behaves seasonally.
    by_month = (out_df.dropna(subset=["seasonal_pct"])
                       .assign(cal_month=lambda d: d.index.month)
                       .groupby("cal_month")["seasonal_pct"]
                       .agg(["mean", "std", "count"])
                       .round(4))
    by_month.index = pd.Index([pd.Timestamp(2026, m, 1).strftime("%b")
                                  for m in by_month.index], name="month")
    pattern_path = OUT_DIR / f"sa_truflation_pattern_{name}.csv"
    by_month.to_csv(pattern_path)

    return {
        "name":                  name,
        "status":                "ok",
        "n_months":              len(monthly),
        "window_start":          str(monthly.index.min().date()),
        "window_end":            str(monthly.index.max().date()),
        "m7":                    m7,
        "qs_p_value":            qs_p,
        "seasonality_detected":  bool(detected),
        "nsa_mom_sd_pp":         sd_nsa,
        "sa_mom_sd_pp":          sd_sa,
        "sd_reduction_pct":      float(sd_reduction_pct),
        "mean_abs_sa_nsa_gap_pct": abs_pct_gap,
        "csv_path":              str(csv_path.relative_to(ROOT)),
        "pattern_path":          str(pattern_path.relative_to(ROOT)),
        "monthly_pattern_pct":   {idx: float(row["mean"])
                                       for idx, row in by_month.iterrows()},
    }


def compose_headline(monthly_per_cat: dict[str, pd.Series],
                          weights: dict[int, float]) -> pd.Series:
    """Compose a daily-Truflation-headline-style aggregate from monthly
    categories using the v2 ``relative_importance`` weights. Output is a
    Laspeyres-style fixed-base weighted sum (base = first month with all
    categories present)."""
    df = pd.DataFrame(monthly_per_cat).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    base = df.iloc[0]
    weight_total = sum(weights[cid] for name, cid in TRUFLATION_CATEGORIES.items()
                          if name in df.columns)
    composite = pd.Series(0.0, index=df.index)
    for name, cid in TRUFLATION_CATEGORIES.items():
        if name not in df.columns:
            continue
        w = weights[cid] / weight_total * 100.0
        composite += w * (df[name] / base[name])
    composite /= 100.0
    return composite.rename("truflation_headline")


def main() -> None:
    print("=" * 78)
    print("Seasonally-Adjusted Truflation via X-13ARIMA-SEATS")
    print("=" * 78)

    x13_path = _find_x13_binary()
    if not x13_path:
        print("\nERROR: x13as binary not found on PATH.")
        sys.exit(2)
    print(f"\nUsing x13 binary: {x13_path}")

    weights = load_truflation_weights()
    print(f"\nTop-level Truflation weights (v2 relative_importance):")
    for name, cid in TRUFLATION_CATEGORIES.items():
        w = weights.get(cid)
        print(f"  {name:<40s} id={cid:>3d}  weight={w:.3f}%"
                if w is not None else
                f"  {name:<40s} id={cid:>3d}  weight=NOT FOUND")

    con = duckdb.connect(str(VINTAGE_DB), read_only=True)
    print(f"\nPulling daily Truflation streams from vintage store…")
    daily_per_cat: dict[str, pd.Series] = {}
    for name in TRUFLATION_CATEGORIES:
        s = load_daily_series(con, name)
        if s.empty:
            print(f"  [skip] {name}: no rows in vintage store")
            continue
        daily_per_cat[name] = s
        print(f"  {name:<40s} {len(s):>5d} daily obs  "
                f"{s.index.min().date()} → {s.index.max().date()}")
    con.close()

    # Aggregate to monthly
    monthly_per_cat = {name: daily_to_monthly(s)
                            for name, s in daily_per_cat.items()}

    # Per-category X-13
    print("\n" + "=" * 78)
    print("PER-CATEGORY X-13 ANALYSIS")
    print("=" * 78)
    per_cat_results = []
    for name, monthly in monthly_per_cat.items():
        per_cat_results.append(analyze_series(name, monthly, x13_path))

    # Composed headlines under three weight schemes
    # (Truflation own / BLS-CPI-equivalent / BEA-PCE-equivalent)
    print("\n" + "=" * 78)
    print("COMPOSED TRUFLATION HEADLINES — three weight schemes")
    print("=" * 78)
    headline_results: list[dict] = []
    for scheme_name, (weight_col, loader) in WEIGHT_SCHEMES.items():
        print(f"\n──  {scheme_name}  (weights: {weight_col})  ──")
        scheme_weights = loader()
        scheme_total = sum(scheme_weights[cid]
                                for name, cid in TRUFLATION_CATEGORIES.items()
                                if cid in scheme_weights)
        print(f"  weights sum (12 top-level): {scheme_total:.3f}")
        composed = compose_headline(monthly_per_cat, scheme_weights)
        if composed.empty:
            print(f"  [skip] composed headline empty")
            headline_results.append({"name": f"{scheme_name}_composed",
                                            "status": "empty"})
            continue
        print(f"  composed from {len(monthly_per_cat)} categories "
                f"({composed.index.min().date()} → "
                f"{composed.index.max().date()})")
        headline_results.append(
            analyze_series(f"{scheme_name}_composed", composed, x13_path))
    headline_result = headline_results[0]    # truflation_us_cpi_composed

    # Official published Truflation headline (from truf_network_published)
    print("\n" + "=" * 78)
    print("OFFICIAL TRUFLATION HEADLINE (truflation_us_cpi_frozen_index)")
    print("=" * 78)
    con2 = duckdb.connect(str(VINTAGE_DB), read_only=True)
    official_results: list[dict] = []
    for off_id in ["truflation_us_cpi_frozen_index", "truflation_us_cpi_index"]:
        s = load_daily_series(con2, off_id, source="truf_network_published")
        if s.empty:
            print(f"  [skip] {off_id}: not in vintage store yet "
                    f"(run ingest_truflation_official_headline.py first)")
            continue
        monthly = daily_to_monthly(s)
        official_results.append(analyze_series(off_id, monthly, x13_path))
    con2.close()

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY — seasonality detected by series")
    print("=" * 78)
    ok_results = [r for r in per_cat_results + headline_results + official_results
                      if r.get("status") == "ok"]
    print(f"\n{'series':<40s}  {'season?':>8s}  "
            f"{'M7':>6s}  {'NSA SD':>8s}  {'SA SD':>8s}  {'Δ%':>7s}")
    print("  " + "-" * 78)
    for r in ok_results:
        m7 = f"{r['m7']:.3f}" if r['m7'] is not None else " n/a "
        print(f"  {r['name']:<38s}  {str(r['seasonality_detected']):>8s}  "
                f"{m7:>6s}  {r['nsa_mom_sd_pp']:>6.3f} pp  "
                f"{r['sa_mom_sd_pp']:>6.3f} pp  {r['sd_reduction_pct']:>+5.1f}%")

    # Save summary JSON
    summary_path = OUT_DIR / f"sa_truflation_summary_{date.today()}.json"
    summary_path.write_text(json.dumps({
        "as_of_date":      str(date.today()),
        "x13_binary":      x13_path,
        "weights_used":    weights,
        "per_category":    per_cat_results,
        "composed":        headline_result,
        "official":        official_results,
    }, indent=2, default=str))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
