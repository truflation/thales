"""Demonstrate the commodity archetype Kalman/RTS recovery on synthetic data.

Generates one synthetic path (default 2000 obs, β drifts, low SV), fits
the TVP-Commodity model, and prints recovery metrics + saves a CSV with
the true β path next to the smoothed estimate so the result can be
plotted/inspected later.

Usage:
    uv run python scripts/demo_commodity_recovery.py
    uv run python scripts/demo_commodity_recovery.py --seed 1 --T 3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.commodity import fit_tvp_commodity  # noqa: E402
from thales.synthetic.commodity_passthrough import (  # noqa: E402
    simulate_commodity_passthrough,
    static_ols_recovery,
)

OUT_DIR = ROOT / "results" / "archetype_recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beta_0", type=float, default=0.35)
    ap.add_argument("--beta_drift_sd", type=float, default=0.008)
    args = ap.parse_args()

    print(f"Generating synthetic commodity DGP (T={args.T}, seed={args.seed})...")
    dgp = simulate_commodity_passthrough(
        T=args.T, beta_0=args.beta_0,
        beta_drift_sd=args.beta_drift_sd,
        commodity_shock_sd=0.02,
        sv_log_mean=-7.0, sv_shock_sd=0.05, sv_persistence=0.99,
        seed=args.seed,
    )
    print(f"  true β: range [{dgp.true_beta.min():.3f}, "
          f"{dgp.true_beta.max():.3f}]  mean={dgp.true_beta.mean():.4f}")

    print("Fitting TVP-Commodity (Kalman + RTS, MLE on α, σ_ε, σ_β)...")
    fit = fit_tvp_commodity(dgp.commodity, dgp.retail, beta_0=0.5)
    print(f"  α̂={fit.alpha:+.4f}  σ̂_ε={fit.sigma_eps:.5f}  "
          f"σ̂_β={fit.sigma_beta:.5f}  iter={fit.n_iter}  "
          f"loglik={fit.log_likelihood:.1f}")

    # OLS for comparison
    _, ols_beta, ols_resid = static_ols_recovery(dgp)
    print(f"  static OLS β̂={ols_beta:.4f}")

    # Recovery metrics
    burn = 100
    true_b = dgp.true_beta[burn:]
    smoothed = fit.beta_smoothed[burn:]
    filtered = fit.beta_filtered[burn:]
    r_smoothed = np.corrcoef(smoothed, true_b)[0, 1]
    r_filtered = np.corrcoef(filtered, true_b)[0, 1]
    mae_smoothed = np.mean(np.abs(smoothed - true_b))
    mae_filtered = np.mean(np.abs(filtered - true_b))
    mae_ols = np.mean(np.abs(ols_beta - true_b))

    print()
    print("Recovery metrics (post-burn-in):")
    print(f"  Pearson(smoothed, true)  = {r_smoothed:.4f}")
    print(f"  Pearson(filtered, true)  = {r_filtered:.4f}")
    print(f"  MAE smoothed             = {mae_smoothed:.4f}")
    print(f"  MAE filtered             = {mae_filtered:.4f}")
    print(f"  MAE static OLS           = {mae_ols:.4f}  (constant β estimate)")
    print(f"  TVP improvement vs OLS   = {(1 - mae_smoothed / mae_ols) * 100:+.1f}%")

    # Save CSV
    df = pd.DataFrame({
        "t": np.arange(len(dgp.true_beta)),
        "log_commodity": dgp.commodity,
        "log_retail": dgp.retail,
        "true_beta": dgp.true_beta,
        "beta_filtered": fit.beta_filtered,
        "beta_smoothed": fit.beta_smoothed,
    })
    out_path = OUT_DIR / f"commodity_recovery_seed{args.seed}_T{args.T}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
