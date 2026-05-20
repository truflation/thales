"""Demonstrate VECM-tradables archetype recovery on synthetic data.

Generates a bivariate cointegrated path with a tariff regime shift,
fits the VECM, prints recovery for α_1, α_2, μ, θ, σ, ρ, saves a CSV.

Usage:
    uv run python scripts/demo_vecm_recovery.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.vecm import fit_vecm  # noqa: E402
from thales.synthetic.vecm_tariff import simulate_vecm_tariff  # noqa: E402

OUT_DIR = ROOT / "results" / "archetype_recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Generating VECM-tradables DGP (T={args.T}, seed={args.seed})...")
    dgp = simulate_vecm_tariff(
        T=args.T, alpha_1=-0.05, alpha_2=+0.10,
        mu_0=0.0, theta=5.0,
        sigma_1=0.4, sigma_2=0.6, rho=0.3,
        seed=args.seed,
    )
    print(f"  y1 range: [{dgp.y1.min():.2f}, {dgp.y1.max():.2f}]")
    print(f"  y2 range: [{dgp.y2.min():.2f}, {dgp.y2.max():.2f}]")
    print(f"  spread z: [{dgp.z.min():.2f}, {dgp.z.max():.2f}]  "
          f"mean={dgp.z.mean():.3f}")
    print(f"  regime onset at t={(dgp.regime == 0).sum()}, "
          f"regime obs n={dgp.regime.sum()}")

    print()
    print("Fitting VECM (per-equation OLS, β=(1,-1) known)...")
    fit = fit_vecm(dgp.y1, dgp.y2, dgp.regime)

    print(f"  α̂_1 = {fit.alpha_1:+.5f}   true {dgp.alpha_1:+.5f}")
    print(f"  α̂_2 = {fit.alpha_2:+.5f}   true {dgp.alpha_2:+.5f}")
    print(f"  μ̂_1 = {fit.mu_1:+.4f}    μ̂_2 = {fit.mu_2:+.4f}    "
          f"true {dgp.mu_0:+.4f}")
    print(f"  θ̂_1 = {fit.theta_1:+.4f}    θ̂_2 = {fit.theta_2:+.4f}    "
          f"true {dgp.theta:+.4f}")
    print(f"  σ̂_1 = {fit.sigma_1:.4f}    true {dgp.sigma_1:.4f}")
    print(f"  σ̂_2 = {fit.sigma_2:.4f}    true {dgp.sigma_2:.4f}")
    print(f"  ρ̂   = {fit.rho:+.4f}    true {dgp.rho:+.4f}")
    print(f"  n_train = {fit.n_train}")

    df = pd.DataFrame({
        "t": np.arange(args.T),
        "y1": dgp.y1, "y2": dgp.y2,
        "spread": dgp.z, "regime": dgp.regime,
    })
    out_path = OUT_DIR / f"vecm_recovery_seed{args.seed}_T{args.T}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
