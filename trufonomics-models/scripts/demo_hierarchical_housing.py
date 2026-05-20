"""Demo: Hierarchical-housing DFM recovery.

Generates 4-region synthetic path with known F_t and λ_{r,t} latents,
fits via JAX-LBFGS, prints recovery metrics + saves CSV.

JAX_PLATFORMS=cuda picks GPU if available; falls back to CPU.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.hierarchical_housing import (  # noqa: E402
    fit_hierarchical_housing,
)
from thales.synthetic.hierarchical_housing import (  # noqa: E402
    simulate_hierarchical_housing,
)

OUT_DIR = ROOT / "results" / "archetype_recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Generating hierarchical-housing DGP (T={args.T}, seed={args.seed})...")
    dgp = simulate_hierarchical_housing(
        T=args.T,
        region_names=("NE", "MW", "S", "W"),
        sigma_F=0.15,
        rhos=(0.85, 0.80, 0.90, 0.88),
        sigma_lambdas=(0.20, 0.15, 0.18, 0.22),
        betas=(1.0, 0.7, 1.1, 1.3),
        sigma_eps=(0.10, 0.10, 0.10, 0.10),
        seed=args.seed,
    )
    print(f"  R = 4 regions: {dgp.region_names}")
    print(f"  national F range: [{dgp.national_factor.min():+.2f}, "
          f"{dgp.national_factor.max():+.2f}]")
    print(f"  true betas:  {dgp.betas}")

    print()
    print("Fitting hierarchical DFM (JAX Kalman + LBFGS-ML)...")
    fit = fit_hierarchical_housing(dgp.y, region_names=dgp.region_names)

    print(f"  σ̂_F        = {fit.sigma_F:.4f}    (true {dgp.sigma_F:.4f})")
    print(f"  ρ̂          = {[f'{x:.3f}' for x in fit.rhos]}")
    print(f"     true ρ  = {[f'{x:.3f}' for x in dgp.rhos]}")
    print(f"  σ̂_λ        = {[f'{x:.3f}' for x in fit.sigma_lambdas]}")
    print(f"  β̂          = {[f'{x:.3f}' for x in fit.betas]}")
    print(f"     true β  = {[f'{x:.3f}' for x in dgp.betas]}")
    print(f"  σ̂_ε        = {[f'{x:.3f}' for x in fit.sigma_eps]}")
    print(f"  iter       = {fit.n_iter}    log-lik = {fit.log_likelihood:.1f}")

    burn = 30
    r_F = abs(np.corrcoef(fit.F_smoothed[burn:],
                              dgp.national_factor[burn:])[0, 1])
    print()
    print("Recovery metrics (post burn-in):")
    print(f"  |Pearson(F_smoothed, F_true)|     = {r_F:.4f}")

    correlations = []
    for r, region in enumerate(dgp.region_names):
        c = abs(np.corrcoef(fit.lambda_smoothed[burn:, r],
                                dgp.regional_idio[burn:, r])[0, 1])
        correlations.append(c)
        print(f"  |Pearson(λ_{region}_smoothed, true)| = {c:.4f}")

    # Reconstruction R²
    y_hat = (fit.F_smoothed[burn:, None] * fit.betas[None, :]
              + fit.lambda_smoothed[burn:])
    y_true = dgp.y[burn:]
    ss_res = np.sum((y_hat - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  Reconstruction R²                  = {r2:.4f}")

    df = pd.DataFrame({"t": np.arange(args.T)})
    df["F_true"] = dgp.national_factor
    df["F_smoothed"] = fit.F_smoothed
    for r, region in enumerate(dgp.region_names):
        df[f"y_{region}"] = dgp.y[:, r]
        df[f"lambda_{region}_true"] = dgp.regional_idio[:, r]
        df[f"lambda_{region}_smoothed"] = fit.lambda_smoothed[:, r]

    out_path = OUT_DIR / f"hierarchical_housing_seed{args.seed}_T{args.T}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
