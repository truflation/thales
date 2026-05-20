"""Real-data BSTS fit — Truflation Recreation & Culture.

Second real-data archetype validation (after commodity TVP × Utilities ×
Henry Hub). Tests both BSTS variants per the empirical per-transform
rule from `bsts_recovery_FINDINGS.md`:

  * **LLT on the monthly LEVEL** — secular drift expected, seasonality
    visible at 12-month period
  * **LL on the monthly YoY** — already differenced, no level walk
    needed; seasonal should be small (12-month diff cancels yearly cycle)

Compares the two fits and documents the trend + seasonal recovery on
real data.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.bsts import (  # noqa: E402
    fit_bsts,
    fit_bsts_local_level,
)
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "real_data_archetypes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 72)
    print("Real-data BSTS — Truflation Recreation & Culture")
    print("=" * 72)

    with VintageStore(VINTAGE_DB, read_only=True) as store:
        daily = store.get_vintage("recreation_and_culture",
                                       date.today()).dropna()

    monthly_level = daily.resample("ME").last()
    monthly_yoy = (monthly_level / monthly_level.shift(12) - 1.0) * 100.0
    monthly_yoy = monthly_yoy.dropna()

    print(f"\nMonthly level: n={len(monthly_level)}  range "
          f"{monthly_level.index.min():%Y-%m-%d} → "
          f"{monthly_level.index.max():%Y-%m-%d}")
    print(f"  level range: [{monthly_level.min():.2f}, "
          f"{monthly_level.max():.2f}]")
    print(f"\nMonthly YoY:   n={len(monthly_yoy)}  range "
          f"{monthly_yoy.index.min():%Y-%m-%d} → "
          f"{monthly_yoy.index.max():%Y-%m-%d}")
    print(f"  YoY range: [{monthly_yoy.min():.3f}%, "
          f"{monthly_yoy.max():.3f}%]")

    # ── LLT on monthly level ─────────────────────────────────────────────
    print()
    print("-" * 72)
    print("Variant A — BSTS LLT on monthly LEVEL (period=12)")
    print("-" * 72)
    fit_llt = fit_bsts(monthly_level.values, period=12)
    print(f"  σ̂_μ        = {fit_llt.sigma_mu:.5f}")
    print(f"  σ̂_δ        = {fit_llt.sigma_delta:.5f}")
    print(f"  σ̂_seasonal = {fit_llt.sigma_seasonal:.5f}")
    print(f"  σ̂_ε        = {fit_llt.sigma_eps:.5f}")
    print(f"  iter       = {fit_llt.n_iter}    log-lik = {fit_llt.log_likelihood:.1f}")
    print(f"  trend range:    [{fit_llt.trend_smoothed.min():.2f}, "
          f"{fit_llt.trend_smoothed.max():.2f}]")
    print(f"  seasonal range: [{fit_llt.seasonal_smoothed.min():.3f}, "
          f"{fit_llt.seasonal_smoothed.max():.3f}]")
    seasonal_amp_llt = (fit_llt.seasonal_smoothed.max()
                          - fit_llt.seasonal_smoothed.min())
    print(f"  seasonal amplitude (peak-to-peak): {seasonal_amp_llt:.3f}")

    # Decomposition R² on level
    fitted_level = fit_llt.trend_smoothed + fit_llt.seasonal_smoothed
    burn = 24
    truth = monthly_level.values[burn:]
    fitted = fitted_level[burn:]
    ss_res = np.sum((fitted - truth) ** 2)
    ss_tot = np.sum((truth - truth.mean()) ** 2)
    r2_llt = 1 - ss_res / ss_tot
    print(f"  Reconstruction R² (post burn-in 24): {r2_llt:.4f}")

    # ── LL on monthly YoY ────────────────────────────────────────────────
    print()
    print("-" * 72)
    print("Variant B — BSTS Local-Level on monthly YoY (period=12)")
    print("-" * 72)
    if len(monthly_yoy) < 2 * 12 + 10:
        print(f"  ⚠️ insufficient data for period=12 (need ≥34, have "
              f"{len(monthly_yoy)})")
        return

    fit_ll = fit_bsts_local_level(monthly_yoy.values, period=12)
    print(f"  σ̂_μ        = {fit_ll.sigma_mu:.5f}")
    print(f"  σ̂_seasonal = {fit_ll.sigma_seasonal:.5f}")
    print(f"  σ̂_ε        = {fit_ll.sigma_eps:.5f}")
    print(f"  iter       = {fit_ll.n_iter}    log-lik = {fit_ll.log_likelihood:.1f}")
    print(f"  trend range:    [{fit_ll.trend_smoothed.min():.3f}%, "
          f"{fit_ll.trend_smoothed.max():.3f}%]")
    print(f"  seasonal range: [{fit_ll.seasonal_smoothed.min():.3f}%, "
          f"{fit_ll.seasonal_smoothed.max():.3f}%]")
    seasonal_amp_ll = (fit_ll.seasonal_smoothed.max()
                          - fit_ll.seasonal_smoothed.min())
    print(f"  seasonal amplitude (peak-to-peak): {seasonal_amp_ll:.3f}%")

    # Decomposition R² on YoY (smaller burn since fewer total obs)
    fitted_yoy = fit_ll.trend_smoothed + fit_ll.seasonal_smoothed
    burn_yoy = 24
    if len(monthly_yoy) > burn_yoy:
        truth_yoy = monthly_yoy.values[burn_yoy:]
        fitted_y = fitted_yoy[burn_yoy:]
        ss_res = np.sum((fitted_y - truth_yoy) ** 2)
        ss_tot = np.sum((truth_yoy - truth_yoy.mean()) ** 2)
        r2_ll = 1 - ss_res / ss_tot
        print(f"  Reconstruction R² (post burn-in {burn_yoy}): {r2_ll:.4f}")

    # ── Save artifacts ───────────────────────────────────────────────────
    df_level = pd.DataFrame({
        "date": monthly_level.index,
        "level": monthly_level.values,
        "trend_llt": fit_llt.trend_smoothed,
        "slope_llt": fit_llt.slope_smoothed,
        "seasonal_llt": fit_llt.seasonal_smoothed,
    })
    df_yoy = pd.DataFrame({
        "date": monthly_yoy.index,
        "yoy_pct": monthly_yoy.values,
        "trend_ll": fit_ll.trend_smoothed,
        "seasonal_ll": fit_ll.seasonal_smoothed,
    })
    df_level.to_csv(OUT_DIR / "bsts_recreation_level_llt.csv", index=False)
    df_yoy.to_csv(OUT_DIR / "bsts_recreation_yoy_ll.csv", index=False)
    print()
    print(f"Saved: {OUT_DIR / 'bsts_recreation_level_llt.csv'}")
    print(f"Saved: {OUT_DIR / 'bsts_recreation_yoy_ll.csv'}")

    # ── Cross-check: per-transform rule ──────────────────────────────────
    print()
    print("=" * 72)
    print("Per-transform check")
    print("=" * 72)
    print(f"  Level seasonal amplitude:  {seasonal_amp_llt:.3f} index points")
    print(f"  YoY   seasonal amplitude:  {seasonal_amp_ll:.3f} percentage points")
    print()
    if seasonal_amp_llt > 1.5 * seasonal_amp_ll:
        print("  ✓ Level shows materially MORE seasonality than YoY — "
              "consistent with")
        print("    YoY differencing absorbing the yearly cycle "
              "(per-transform rule confirmed)")
    else:
        print("  ⚠ Level and YoY seasonal amplitudes comparable — "
              "differencing may not")
        print("    have fully removed the yearly cycle on this series")


if __name__ == "__main__":
    main()
