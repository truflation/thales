"""Real-data VECM fit — Truflation Clothing × BLS Apparel CPI.

Both are inflation indices for the same economic concept (clothing
prices). Theory predicts they should be cointegrated with β ≈ (1, -1):
the spread between Truflation's daily-aggregated clothing index and
BLS's monthly clothing CPI should be approximately stationary, modulo
methodology differences.

Tariff regime dummy = 1 starting April 2025 (Trump-era tariff regime
return). Tests whether the cointegrating relationship between Truflation
and BLS clothing measures shifted under the new tariff regime.

This validates the Phase 1.4 VECM model on real cointegrated economic
data — the canonical use case for VECM.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.vecm import fit_vecm  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "real_data_archetypes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 72)
    print("Real-data VECM — Truflation Clothing × BLS Apparel CPI")
    print("=" * 72)

    with VintageStore(VINTAGE_DB, read_only=True) as store:
        truf_daily = store.get_vintage("clothing_and_footwear",
                                              date.today()).dropna()
        bls = store.get_vintage("CUSR0000SAA", date.today()).dropna()

    # Resample Truflation to monthly + log
    truf_monthly = truf_daily.resample("ME").last()
    log_truf = np.log(truf_monthly).rename("log_truf")
    log_bls = np.log(bls).rename("log_bls")
    # Re-index BLS to month-end
    log_bls.index = log_bls.index.to_period("M").to_timestamp("M")

    # Inner join
    df = pd.concat([log_truf, log_bls], axis=1).dropna()
    print(f"\nAligned monthly panel: n={len(df)}  range "
          f"{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")
    print(f"  log(truf) range: [{df['log_truf'].min():.3f}, "
          f"{df['log_truf'].max():.3f}]")
    print(f"  log(bls)  range: [{df['log_bls'].min():.3f}, "
          f"{df['log_bls'].max():.3f}]")

    # Build tariff regime dummy: 1 from April 2025 onward
    tariff_start = pd.Timestamp("2025-04-01")
    df["regime"] = (df.index >= tariff_start).astype(int)
    n_pre = (df["regime"] == 0).sum()
    n_post = (df["regime"] == 1).sum()
    print(f"\nTariff regime dummy: pre={n_pre} months, post={n_post} months")

    if n_post < 6:
        print(f"  ⚠️ only {n_post} post-tariff months — θ will be noisy")

    # Sanity: spread y1-y2 should be roughly stationary if cointegrated
    spread = df["log_truf"] - df["log_bls"]
    print(f"\nSpread log(truf) - log(bls):")
    print(f"  range = [{spread.min():.4f}, {spread.max():.4f}]")
    print(f"  pre-tariff mean  = {spread[df['regime']==0].mean():+.4f}")
    print(f"  post-tariff mean = {spread[df['regime']==1].mean():+.4f}")
    spread_shift = (spread[df['regime']==1].mean()
                       - spread[df['regime']==0].mean())
    print(f"  shift            = {spread_shift:+.4f}")

    # Fit VECM
    print("\nFitting VECM (per-equation OLS, β=(1,-1) known)...")
    fit = fit_vecm(df["log_truf"].values, df["log_bls"].values,
                       df["regime"].values)

    print()
    print(f"Eq 1: Δlog(truf) = α_1 z_{{t-1}} + c_1 + γ_1 D_{{t-1}} + ε_1")
    print(f"  α_1 = {fit.alpha_1:+.5f}    "
          f"(truf adjusts toward equilibrium at this rate)")
    print(f"  c_1 = {fit.c_1:+.5f}")
    print(f"  γ_1 = {fit.gamma_1:+.5f}")
    print(f"  σ_1 = {fit.sigma_1:.5f}")
    print()
    print(f"Eq 2: Δlog(bls) = α_2 z_{{t-1}} + c_2 + γ_2 D_{{t-1}} + ε_2")
    print(f"  α_2 = {fit.alpha_2:+.5f}    "
          f"(bls adjusts at this rate)")
    print(f"  c_2 = {fit.c_2:+.5f}")
    print(f"  γ_2 = {fit.gamma_2:+.5f}")
    print(f"  σ_2 = {fit.sigma_2:.5f}")
    print()
    print(f"Implied structural parameters:")
    print(f"  μ_0  (eq 1) = {fit.mu_1:+.5f}    (eq 2) = {fit.mu_2:+.5f}    "
          f"avg = {(fit.mu_1 + fit.mu_2)/2:+.5f}")
    print(f"  θ    (eq 1) = {fit.theta_1:+.5f}    (eq 2) = {fit.theta_2:+.5f}")
    print(f"  ρ    (residual corr) = {fit.rho:+.4f}")
    print(f"  n_train = {fit.n_train}")

    # Save artifact
    df.to_csv(OUT_DIR / "vecm_clothing_real.csv")
    print(f"\nSaved: {OUT_DIR / 'vecm_clothing_real.csv'}")

    # Interpretation
    print()
    print("=" * 72)
    print("Interpretation")
    print("=" * 72)
    if fit.alpha_1 < 0 and fit.alpha_2 > 0:
        print("  ✓ Sign pattern: α_1 < 0 (truf falls when above equilibrium),")
        print("    α_2 > 0 (bls rises) — proper error-correction dynamics")
    elif fit.alpha_1 > 0 and fit.alpha_2 < 0:
        print("  ✓ Sign pattern: α_1 > 0, α_2 < 0 — error-correction with")
        print("    truf as the leading series (vs textbook clothing case)")
    else:
        print("  ⚠ Sign pattern: α_1 and α_2 same sign — non-standard "
              "cointegration; may not be cointegrated, or the cointegrating")
        print("    vector is not (1, -1) but something else")

    speed = abs(fit.alpha_1) + abs(fit.alpha_2)
    half_life_months = np.log(0.5) / np.log(1 - speed) if speed < 1 else np.nan
    print(f"  Speed of adjustment: |α_1|+|α_2| = {speed:.4f}")
    if not np.isnan(half_life_months):
        print(f"  Implied spread half-life: ~{abs(half_life_months):.1f} months")

    if abs(fit.theta_1 - fit.theta_2) < 0.05:
        print(f"  ✓ θ estimates agree across equations "
              f"({fit.theta_1:+.4f} vs {fit.theta_2:+.4f}) — model "
              "well-specified")
    else:
        print(f"  ⚠ θ disagreement {fit.theta_1:+.4f} vs {fit.theta_2:+.4f} — "
              "limited post-tariff data inflates standard errors")

    if abs(spread_shift) > 0.01 and abs((fit.theta_1 + fit.theta_2)/2) > 0.01:
        print(f"  Tariff regime: visible spread shift of {spread_shift:+.4f} "
              f"in raw data; model captures it as θ ≈ "
              f"{(fit.theta_1+fit.theta_2)/2:+.4f}")


if __name__ == "__main__":
    main()
