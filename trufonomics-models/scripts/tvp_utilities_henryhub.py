"""Real-data fit of Phase 1.1 commodity TVP archetype.

Pair:  Truflation Utilities daily index  ↔  Henry Hub natural gas spot

Captures the time-varying pass-through coefficient `β_t` of natural gas
prices into Utilities CPI. Production-grade utility-pricing economists
know this β has shifted over the last 5 years (gas-to-electricity
transitions, weather extremes, regional pricing reforms).

This is the first real-data application of any of our Phase 1
synthetic-validated archetypes. The synthetic recovery test already
proved the ESTIMATION CORE works (Pearson 0.999 on β path). This script
asks the next question: does β recovery on REAL data tell a coherent
story?
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.commodity import fit_tvp_commodity  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "real_data_archetypes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 72)
    print("Real-data TVP commodity fit — Utilities × Henry Hub natural gas")
    print("=" * 72)

    with VintageStore(VINTAGE_DB, read_only=True) as store:
        util = store.get_vintage("utilities", date.today()).dropna()
        hh = store.get_vintage("DHHNGSP", date.today()).dropna()

    # Align both to a common daily index
    common = util.index.intersection(hh.index)
    util = util.loc[common]
    hh = hh.loc[common]
    print(f"\nAligned panel: n={len(common)}  range "
          f"{common.min():%Y-%m-%d} → {common.max():%Y-%m-%d}")

    # Drop any zero/negative values before log
    valid = (util > 0) & (hh > 0)
    util = util.loc[valid]
    hh = hh.loc[valid]

    log_retail = np.log(util.values)
    log_commodity = np.log(hh.values)

    print(f"  log(Utilities) range: [{log_retail.min():.3f}, "
          f"{log_retail.max():.3f}]  std = {log_retail.std():.3f}")
    print(f"  log(Henry Hub) range: [{log_commodity.min():.3f}, "
          f"{log_commodity.max():.3f}]  std = {log_commodity.std():.3f}")

    # Fit TVP commodity model
    print("\nFitting TVP commodity (Kalman + RTS, MLE on α, σ_ε, σ_β)...")
    fit = fit_tvp_commodity(log_commodity, log_retail, beta_0=0.05,
                                P_0=0.5)
    print(f"  α̂        = {fit.alpha:+.4f}")
    print(f"  σ̂_ε      = {fit.sigma_eps:.5f}")
    print(f"  σ̂_β      = {fit.sigma_beta:.5f}")
    print(f"  log-lik  = {fit.log_likelihood:.1f}  iter = {fit.n_iter}")

    # Static OLS for comparison
    X = np.column_stack([np.ones_like(log_commodity), log_commodity])
    coefs, *_ = np.linalg.lstsq(X, log_retail, rcond=None)
    ols_intercept, ols_beta = float(coefs[0]), float(coefs[1])
    ols_resid = log_retail - X @ coefs
    print(f"\n  Static OLS β̂ = {ols_beta:.4f}  (residual SD = {ols_resid.std():.5f})")

    print(f"\n  TVP β path:")
    print(f"    range  = [{fit.beta_smoothed.min():.4f}, "
          f"{fit.beta_smoothed.max():.4f}]")
    print(f"    mean   = {fit.beta_smoothed.mean():.4f}")
    print(f"    std    = {fit.beta_smoothed.std():.4f}")

    # Year-by-year mean β
    print()
    print("Year-by-year mean β (TVP smoothed):")
    df = pd.DataFrame({
        "date": util.index,
        "log_utilities": log_retail,
        "log_henryhub": log_commodity,
        "beta_filtered": fit.beta_filtered,
        "beta_smoothed": fit.beta_smoothed,
    })
    df["year"] = df["date"].dt.year
    by_year = df.groupby("year")["beta_smoothed"].agg(["mean", "std", "count"])
    print(by_year.round(4).to_string())

    # Output
    out_path = OUT_DIR / "tvp_utilities_henryhub.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Sanity check: identify regimes where β changed materially
    rolling_60d = pd.Series(fit.beta_smoothed,
                                index=util.index).rolling(60).mean()
    rolling_diffs = rolling_60d.diff(60).abs()
    big_shifts = rolling_diffs.nlargest(5)
    print()
    print("Top 5 dates of largest 60-day β shift (production interpretability):")
    for dt, v in big_shifts.items():
        b_now = rolling_60d.loc[dt]
        print(f"  {dt:%Y-%m-%d}  Δβ_60d = {v:+.4f}  β = {b_now:.4f}")


if __name__ == "__main__":
    main()
