"""Demo: stochastic-volatility recovery via NumPyro NUTS.

Generates a SV path with known μ_h, φ, σ_h, fits via MCMC, prints
posterior summaries + h-path recovery metrics.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.sv import fit_sv  # noqa: E402
from thales.synthetic.regime_switching import simulate_sv  # noqa: E402

OUT_DIR = ROOT / "results" / "archetype_recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_warmup", type=int, default=500)
    ap.add_argument("--num_samples", type=int, default=1000)
    args = ap.parse_args()

    print(f"Generating SV DGP (T={args.T}, seed={args.seed})...")
    dgp = simulate_sv(T=args.T, mu_h=-1.5, phi=0.95, sigma_h=0.3,
                          seed=args.seed)
    print(f"  h_t range:  [{dgp.h_path.min():+.3f}, {dgp.h_path.max():+.3f}]  "
          f"mean={dgp.h_path.mean():+.3f}")
    print(f"  vol(σ_t) range: [{np.exp(dgp.h_path.min()/2):.3f}, "
          f"{np.exp(dgp.h_path.max()/2):.3f}]")

    print()
    print(f"Fitting via NumPyro NUTS (warmup={args.num_warmup}, "
          f"samples={args.num_samples})...")
    fit = fit_sv(dgp.y, num_warmup=args.num_warmup,
                    num_samples=args.num_samples, seed=args.seed)

    print(f"  μ̂_h     = {fit.mu_h:+.4f}    (true {dgp.mu_h:+.4f})")
    print(f"  φ̂       = {fit.phi:.4f}    (true {dgp.phi:.4f})")
    print(f"  σ̂_h     = {fit.sigma_h:.4f}    (true {dgp.sigma_h:.4f})")
    print(f"  divergences  = {fit.diverging} / {fit.n_samples}")

    burn = 50
    r = np.corrcoef(fit.h_smoothed[burn:], dgp.h_path[burn:])[0, 1]
    mae = np.mean(np.abs(fit.h_smoothed[burn:] - dgp.h_path[burn:]))
    coverage_90 = ((dgp.h_path[burn:] >= fit.h_q05[burn:]) &
                      (dgp.h_path[burn:] <= fit.h_q95[burn:])).mean()

    print()
    print("h-path recovery (post burn-in t > 50):")
    print(f"  Pearson(smoothed, true) = {r:.4f}")
    print(f"  MAE                     = {mae:.4f}")
    print(f"  90% band coverage       = {coverage_90:.1%}  (nominal 90%)")

    df = pd.DataFrame({
        "t": np.arange(args.T),
        "y": dgp.y,
        "h_true": dgp.h_path,
        "h_smoothed": fit.h_smoothed,
        "h_q05": fit.h_q05,
        "h_q95": fit.h_q95,
    })
    out_path = OUT_DIR / f"sv_recovery_seed{args.seed}_T{args.T}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
