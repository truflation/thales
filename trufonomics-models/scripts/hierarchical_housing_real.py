"""Real-data Phase 1.5 hierarchical housing — BLS housing-cost panel.

Applies the hierarchical DFM (national factor + component-specific
AR(1) idiosyncratic + component-specific loadings) to a 4-component
panel of non-overlapping BLS housing-related series:

  * CUSR0000SEHA   Rent of primary residence
  * CUSR0000SEHC01 Owners' equivalent rent (OER)
  * CUSR0000SEHF01 Electricity
  * CUSR0000SEHF02 Utility (piped) gas service

These four cover most of the BLS Housing CPI hierarchy (shelter +
utilities). The hierarchical model should identify:

  * **National factor F_t**: shared housing-cost shock (rates,
    aggregate housing demand, energy markets driving utilities)
  * **High loadings on rent + OER** (shelter components track
    each other tightly)
  * **Lower / different loadings on electricity + gas** (driven by
    energy markets, not just housing)
  * **Idiosyncratic AR(1) per component**: persistent component-
    specific noise

Same code as `demo_hierarchical_housing.py`; just real BLS data
instead of synthetic. JAX-native; GPU-enabled when run with
``JAX_PLATFORMS=cuda``.

Usage on Vast:
    JAX_PLATFORMS=cuda uv run python scripts/hierarchical_housing_real.py
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.hierarchical_housing import (  # noqa: E402
    fit_hierarchical_housing,
)
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "real_data_archetypes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 4-component housing panel: shelter (rent + OER) + utilities (electricity + gas)
HOUSING_PANEL = {
    "rent_primary":      ("CUSR0000SEHA",   "Rent of primary residence"),
    "owners_eq_rent":    ("CUSR0000SEHC01", "Owners' equivalent rent (OER)"),
    "electricity":       ("CUSR0000SEHF01", "Electricity"),
    "utility_gas":       ("CUSR0000SEHF02", "Utility (piped) gas service"),
}


def _to_monthly_log(daily_or_monthly: pd.Series) -> pd.Series:
    """Resample to month-end + take log. BLS data is already monthly."""
    s = daily_or_monthly.dropna().sort_index()
    monthly = s.resample("ME").last().dropna()
    return np.log(monthly)


def main() -> None:
    print("=" * 78)
    print("Real-data Phase 1.5 — Hierarchical DFM on BLS Housing Panel")
    print("=" * 78)

    print(f"\n4-component housing panel:")
    for label, (sid, desc) in HOUSING_PANEL.items():
        print(f"  {sid:<20s} {desc:<40s}  ({label})")

    print(f"\nLoading from vintage store: {VINTAGE_DB}")
    series_dict = {}
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        for label, (sid, _desc) in HOUSING_PANEL.items():
            s = store.get_vintage(sid, date.today())
            if s.empty:
                print(f"  ⚠ {sid} not found in vintage store; skipping")
                continue
            series_dict[label] = _to_monthly_log(s)

    if len(series_dict) < 2:
        print("Insufficient data — at least 2 series needed.")
        return

    # Inner-join all series on common monthly index
    df = pd.DataFrame(series_dict).dropna()
    region_names = list(df.columns)
    y = df.values  # (T, R)
    print(f"\nAligned panel: T={y.shape[0]} months, R={y.shape[1]} components")
    print(f"  range {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")

    # Standardize each series (subtract mean, scale by std) — improves
    # numerical conditioning of the JAX optimizer when level magnitudes
    # vary across series
    means = y.mean(axis=0)
    stds = y.std(axis=0, ddof=1)
    y_std = (y - means) / stds

    print(f"\nFitting hierarchical DFM (JAX Kalman + LBFGS-ML)...")
    print(f"  (~10-30 sec on CPU; faster on GPU)")
    fit = fit_hierarchical_housing(y_std, region_names=region_names)

    print()
    print("Posterior summary:")
    print(f"  σ̂_F        = {fit.sigma_F:.4f}")
    print(f"  ρ̂          = {[f'{x:.3f}' for x in fit.rhos]}")
    print(f"  σ̂_λ        = {[f'{x:.3f}' for x in fit.sigma_lambdas]}")
    print(f"  β̂          = {[f'{x:.3f}' for x in fit.betas]}")
    print(f"  σ̂_ε        = {[f'{x:.3f}' for x in fit.sigma_eps]}")
    print(f"  iter       = {fit.n_iter}    log-lik = {fit.log_likelihood:.1f}")

    # Per-component loadings interpretation
    print()
    print("Per-component β loadings on national factor (after fitting):")
    for r, label in enumerate(region_names):
        print(f"  {label:<22s} β̂_{r+1} = {fit.betas[r]:+.4f}    "
              f"ρ̂_{r+1} = {fit.rhos[r]:.3f}    "
              f"σ̂_ε,{r+1} = {fit.sigma_eps[r]:.3f}")

    # Renormalize: divide by first β to make rent_primary = 1.0 by convention
    if abs(fit.betas[0]) > 1e-6:
        beta_anchor = fit.betas[0]
        norm_betas = fit.betas / beta_anchor
        print()
        print(f"After renormalizing β̂_{region_names[0]} := 1.0:")
        for r, label in enumerate(region_names):
            print(f"  {label:<22s} β̂ = {norm_betas[r]:+.4f}")

    # Save artifact
    df_out = pd.DataFrame({"date": df.index})
    df_out["F_smoothed"] = fit.F_smoothed
    for r, label in enumerate(region_names):
        df_out[f"y_{label}_log_std"] = y_std[:, r]
        df_out[f"lambda_{label}"] = fit.lambda_smoothed[:, r]
    out_path = OUT_DIR / "hierarchical_housing_real.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Reconstruction R²
    burn = 24
    y_hat = (fit.F_smoothed[burn:, None] * fit.betas[None, :]
              + fit.lambda_smoothed[burn:])
    y_truth = y_std[burn:]
    ss_res = np.sum((y_hat - y_truth) ** 2)
    ss_tot = np.sum((y_truth - y_truth.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"Reconstruction R² (post burn-in {burn}): {r2:.4f}")

    # F path interpretation
    f_first_yr = fit.F_smoothed[:12].mean()
    f_last_yr = fit.F_smoothed[-12:].mean()
    print()
    print(f"National housing factor F_t evolution (standardized units):")
    print(f"  first year mean = {f_first_yr:+.3f}")
    print(f"  last year mean  = {f_last_yr:+.3f}")
    print(f"  total drift     = {f_last_yr - f_first_yr:+.3f}")


if __name__ == "__main__":
    main()
