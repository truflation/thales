"""IASA experiment — does adding hand-curated intervention dummies tighten
the BLS replication?

Background: the baseline replication (``scripts/x13_replicate_bls_sa.py``)
runs X-13 with ``outlier=True`` only, which lets X-13 auto-detect outliers
algorithmically. BLS layers a hand-curated intervention list on top
(Intervention Analysis Seasonal Adjustment, IASA): known events like
COVID-2020 and the 2022 energy spike are pre-specified as additive
outliers or level shifts and passed into the regARIMA component.

This script tests whether passing equivalent intervention dummies via
``exog=`` to ``x13_arima_analysis`` tightens the residual against BLS's
published SA. Four nested variants:

  A. Baseline               — auto-outlier-detection only (no exog).
  B. + COVID                — Mar 2020 + Apr 2020 + May 2020 (AOs).
  C. B + 2022 energy spike  — Mar 2022 + Jun 2022 (AOs).
  D. C + 2025 disinflation  — Mar 2025 + Apr 2025 (AOs).

Output: per-variant level / MoM / YoY RMSE against the published BLS SA
series, for both headline and core CPI.

Run::

    uv run python scripts/x13_iasa_experiment.py
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
OUT_DIR = ROOT / "results" / "x13_replication"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# NSA → SA series mapping (same as the baseline replication)
SERIES_MAP = {
    "CUUR0000SA0":    "CUSR0000SA0",
    "CUUR0000SA0L1E": "CUSR0000SA0L1E",
}

# Curated intervention events. Mirrors the style of the BLS IASA list:
# additive outliers (AO) at event months. Level shifts (LS) and
# temporary changes (TC) are supported too but a tight AO-only baseline
# is what BLS reports for most CPI series.
EVENTS_COVID = [
    {"name": "covid_2020m03", "type": "AO", "date": "2020-03"},
    {"name": "covid_2020m04", "type": "AO", "date": "2020-04"},
    {"name": "covid_2020m05", "type": "AO", "date": "2020-05"},
]
EVENTS_ENERGY_2022 = [
    {"name": "energy_2022m03", "type": "AO", "date": "2022-03"},
    {"name": "energy_2022m06", "type": "AO", "date": "2022-06"},
]
EVENTS_DISINFL_2025 = [
    {"name": "disinfl_2025m03", "type": "AO", "date": "2025-03"},
    {"name": "disinfl_2025m04", "type": "AO", "date": "2025-04"},
]

VARIANTS = [
    ("A_baseline",         []),
    ("B_covid",            EVENTS_COVID),
    ("C_covid_energy",     EVENTS_COVID + EVENTS_ENERGY_2022),
    ("D_covid_energy_25",  EVENTS_COVID + EVENTS_ENERGY_2022 + EVENTS_DISINFL_2025),
]


def find_x13_binary() -> str | None:
    for name in ["x13as", "x13as_ascii"]:
        p = shutil.which(name)
        if p:
            return p
    return None


def load_level_series(con: duckdb.DuckDBPyConnection,
                          series_id: str) -> pd.Series:
    rows = con.execute(
        "SELECT reference_date, value FROM vintage "
        "WHERE series_id = ? AND source = 'bls_direct' "
        "AND (series_id, reference_date, as_of_date) IN ("
        "  SELECT series_id, reference_date, MAX(as_of_date) "
        "  FROM vintage WHERE series_id = ? AND source = 'bls_direct' "
        "  GROUP BY series_id, reference_date) "
        "ORDER BY reference_date",
        [series_id, series_id],
    ).fetchall()
    idx = [pd.Timestamp(r[0]) + pd.offsets.MonthEnd(0) for r in rows]
    return pd.Series([r[1] for r in rows], index=idx, name=series_id)


def make_intervention_exog(period_index: pd.PeriodIndex,
                              events: list[dict],
                              forecast_periods: int = 12) -> pd.DataFrame | None:
    """Build a DataFrame of intervention dummies indexed by PeriodIndex.

    X-13 extends the series with `forecast_periods` months of ARIMA
    forecasts to handle endpoint filter behaviour; when user-defined
    regressors are supplied via exog, they must cover the forecast
    horizon as well. We extend the index 12 months past the data end
    and fill exog with zeros there (no intervention in the future).
    """
    if not events:
        return None
    last = period_index[-1]
    future = pd.period_range(start=last + 1, periods=forecast_periods, freq="M")
    full_idx = period_index.append(future)
    cols: dict[str, pd.Series] = {}
    for ev in events:
        target = pd.Period(ev["date"], freq="M")
        ev_type = ev["type"]
        s = pd.Series(0.0, index=full_idx, name=ev["name"])
        if ev_type == "AO":
            if target in full_idx:
                s.loc[target] = 1.0
        elif ev_type == "LS":
            mask = full_idx.asfreq("M") >= target
            s.loc[mask] = 1.0
        elif ev_type == "TC":
            decay = 0.7
            for k, d in enumerate(full_idx):
                if d >= target:
                    s.iloc[k] = decay ** (k - list(full_idx).index(target))
        cols[ev["name"]] = s
    return pd.DataFrame(cols)


def run_x13(nsa: pd.Series, x13_path: str,
                exog: pd.DataFrame | None = None) -> pd.Series:
    from statsmodels.tsa.x13 import x13_arima_analysis

    x = nsa.copy()
    x.index = pd.PeriodIndex(x.index, freq="M")
    # exog is built by make_intervention_exog with the forecast horizon
    # already appended; do NOT reset its index here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = x13_arima_analysis(
            endog=x,
            exog=exog,
            outlier=True,
            prefer_x13=True,
            x12path=x13_path,
        )
    sa = result.seasadj.copy()
    sa.index = (sa.index.to_timestamp(how="end").floor("D")
                  + pd.offsets.MonthEnd(0))
    return sa


def score(our_sa: pd.Series, bls_sa: pd.Series) -> dict:
    common = our_sa.index.intersection(bls_sa.index)
    a, b = our_sa.loc[common], bls_sa.loc[common]
    pct_diff = (a / b - 1.0) * 100
    mom_a = (np.log(a) - np.log(a.shift(1))).dropna() * 100
    mom_b = (np.log(b) - np.log(b.shift(1))).dropna() * 100
    common_m = mom_a.index.intersection(mom_b.index)
    diff_mom = (mom_a.loc[common_m] - mom_b.loc[common_m])
    return {
        "n":                int(len(common)),
        "level_rmse_pct":   float(np.sqrt((pct_diff ** 2).mean())),
        "level_max_pct":    float(pct_diff.abs().max()),
        "mom_rmse_pp":      float(np.sqrt((diff_mom ** 2).mean())),
        "mom_max_pp":       float(diff_mom.abs().max()),
    }


def main() -> None:
    print("=" * 78)
    print("IASA experiment — auto-outlier vs hand-curated interventions")
    print("=" * 78)

    x13_path = find_x13_binary()
    if not x13_path:
        print("\nERROR: x13as binary not found")
        sys.exit(2)
    print(f"\nUsing x13 binary: {x13_path}")

    con = duckdb.connect(str(VINTAGE_DB), read_only=True)
    nsa_series: dict[str, pd.Series] = {}
    bls_sa_series: dict[str, pd.Series] = {}
    for nsa_id, sa_id in SERIES_MAP.items():
        nsa_series[nsa_id] = load_level_series(con, nsa_id)
        bls_sa_series[nsa_id] = load_level_series(con, sa_id)
    con.close()

    summary: dict = {}
    for nsa_id, sa_id in SERIES_MAP.items():
        print()
        print("=" * 78)
        print(f"{nsa_id}  (NSA)  →  {sa_id}  (BLS published SA)")
        print("=" * 78)
        nsa = nsa_series[nsa_id]
        bls_sa = bls_sa_series[nsa_id]
        period_idx = pd.PeriodIndex(nsa.index, freq="M")
        print(f"n = {len(nsa)} months  "
                f"({nsa.index.min().date()} → {nsa.index.max().date()})")

        rows = []
        for variant_name, events in VARIANTS:
            exog = make_intervention_exog(period_idx, events)
            n_events = len(events) if events else 0
            print(f"\n── {variant_name}  ({n_events} interventions) ──")
            if events:
                for ev in events:
                    print(f"    {ev['type']} @ {ev['date']}    ({ev['name']})")
            try:
                our_sa = run_x13(nsa, x13_path, exog=exog)
            except Exception as e:    # noqa: BLE001
                print(f"  [FAILED] {type(e).__name__}: {e}")
                rows.append({"variant": variant_name, "status": "failed",
                                "error": str(e)})
                continue
            s = score(our_sa, bls_sa)
            print(f"  level RMSE = {s['level_rmse_pct']*100:.2f} bp   "
                    f"max = {s['level_max_pct']*100:.1f} bp")
            print(f"  MoM   RMSE = {s['mom_rmse_pp']*100:.2f} bp   "
                    f"max = {s['mom_max_pp']*100:.1f} bp")
            rows.append({
                "variant":         variant_name,
                "status":          "ok",
                "n_events":        n_events,
                "level_rmse_pct":  s["level_rmse_pct"],
                "level_max_pct":   s["level_max_pct"],
                "mom_rmse_pp":     s["mom_rmse_pp"],
                "mom_max_pp":      s["mom_max_pp"],
            })
        summary[nsa_id] = rows

    print()
    print("=" * 78)
    print("SUMMARY — level RMSE (bp = basis points)")
    print("=" * 78)
    print(f"\n{'variant':<22s}  {'headline (bp)':>14s}  {'core (bp)':>14s}  "
            f"{'Δ vs A (bp)':>14s}")
    print("  " + "-" * 70)
    base = {nid: next((r for r in rows if r["variant"] == "A_baseline"
                          and r["status"] == "ok"), None)
                for nid, rows in summary.items()}
    for variant_name, _ in VARIANTS:
        cells = []
        delta = []
        for nid in SERIES_MAP:
            r = next((x for x in summary[nid] if x["variant"] == variant_name
                          and x["status"] == "ok"), None)
            if r is None:
                cells.append("   n/a   "); delta.append("        ")
            else:
                bp = r["level_rmse_pct"] * 100
                cells.append(f"  {bp:>6.2f}      ")
                b = base[nid]
                if b is not None and variant_name != "A_baseline":
                    d = (r["level_rmse_pct"] - b["level_rmse_pct"]) * 100
                    delta.append(f"{d:>+5.2f}        ")
                else:
                    delta.append("        ")
        print(f"  {variant_name:<20s}  {cells[0]} {cells[1]} "
                f"{delta[0] if delta else ''}{delta[1] if len(delta) > 1 else ''}")

    out_path = OUT_DIR / f"iasa_experiment_{date.today()}.json"
    out_path.write_text(json.dumps({
        "as_of_date":  str(date.today()),
        "x13_binary":  x13_path,
        "variants":    {n: e for n, e in VARIANTS},
        "results":     summary,
    }, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
