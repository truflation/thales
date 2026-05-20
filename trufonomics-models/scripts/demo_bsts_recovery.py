"""Demonstrate BSTS-discretionary archetype recovery on synthetic data.

Generates one synthetic path with known trend + seasonal + noise, fits
BSTS, prints recovery metrics for each component, saves a CSV.

Usage:
    uv run python scripts/demo_bsts_recovery.py
    uv run python scripts/demo_bsts_recovery.py --seed 1 --T 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.bsts import (  # noqa: E402
    fit_bsts,
    fit_bsts_local_level,
)
from thales.synthetic.bsts_discretionary import (  # noqa: E402
    simulate_bsts_discretionary,
)

OUT_DIR = ROOT / "results" / "archetype_recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--period", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Generating BSTS DGP (T={args.T}, period={args.period}, "
          f"seed={args.seed})...")
    pattern = 3.0 * np.sin(2 * np.pi * np.arange(args.period) / args.period)
    dgp = simulate_bsts_discretionary(
        T=args.T, period=args.period,
        initial_level=100.0, initial_slope=0.05,
        seasonal_pattern=pattern,
        sigma_mu=0.05, sigma_delta=0.005,
        sigma_seasonal=0.1, sigma_eps=0.5,
        seed=args.seed,
    )
    print(f"  trend:    [{dgp.trend.min():.2f}, {dgp.trend.max():.2f}]  "
          f"mean={dgp.trend.mean():.2f}")
    print(f"  seasonal: [{dgp.seasonal.min():.3f}, "
          f"{dgp.seasonal.max():.3f}]  amp~3")
    print(f"  noise:    σ={dgp.sigma_eps:.3f}")

    print()
    print("=" * 72)
    print("VARIANT A — Local Linear Trend (LLT, with slope state δ)")
    print("=" * 72)
    fit = fit_bsts(dgp.y, period=args.period)
    print(f"  σ̂_μ={fit.sigma_mu:.5f}  (true {dgp.sigma_mu:.5f})")
    print(f"  σ̂_δ={fit.sigma_delta:.5f}  (true {dgp.sigma_delta:.5f})")
    print(f"  σ̂_s={fit.sigma_seasonal:.5f}  (true {dgp.sigma_seasonal:.5f})")
    print(f"  σ̂_ε={fit.sigma_eps:.5f}  (true {dgp.sigma_eps:.5f})")
    print(f"  iter={fit.n_iter}  loglik={fit.log_likelihood:.1f}")
    print()
    print("=" * 72)
    print("VARIANT B — Local Level only (no slope state)")
    print("=" * 72)
    fit_ll = fit_bsts_local_level(dgp.y, period=args.period)
    print(f"  σ̂_μ={fit_ll.sigma_mu:.5f}  (true {dgp.sigma_mu:.5f})")
    print(f"  σ̂_s={fit_ll.sigma_seasonal:.5f}  (true {dgp.sigma_seasonal:.5f})")
    print(f"  σ̂_ε={fit_ll.sigma_eps:.5f}  (true {dgp.sigma_eps:.5f})")
    print(f"  iter={fit_ll.n_iter}  loglik={fit_ll.log_likelihood:.1f}")

    burn = 24

    def _metrics(label, trend_sm, seas_sm):
        r_trend = np.corrcoef(trend_sm[burn:], dgp.trend[burn:])[0, 1]
        r_seas = np.corrcoef(seas_sm[burn:], dgp.seasonal[burn:])[0, 1]
        mae_trend = np.mean(np.abs(trend_sm[burn:] - dgp.trend[burn:]))
        mae_seas = np.mean(np.abs(seas_sm[burn:] - dgp.seasonal[burn:]))
        fitted = trend_sm[burn:] + seas_sm[burn:]
        truth = dgp.trend[burn:] + dgp.seasonal[burn:]
        ss_res = np.sum((fitted - truth) ** 2)
        ss_tot = np.sum((truth - truth.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"\n  Recovery metrics — {label} (post burn-in t > 24):")
        print(f"    Pearson(trend,    true) = {r_trend:.4f}")
        print(f"    Pearson(seasonal, true) = {r_seas:.4f}")
        print(f"    MAE  trend              = {mae_trend:.4f}")
        print(f"    MAE  seasonal           = {mae_seas:.4f}")
        print(f"    Decomposition R²        = {r2:.4f}")

    _metrics("LLT", fit.trend_smoothed, fit.seasonal_smoothed)
    _metrics("Local-level", fit_ll.trend_smoothed, fit_ll.seasonal_smoothed)

    df = pd.DataFrame({
        "t": np.arange(len(dgp.y)),
        "y": dgp.y,
        "trend_true": dgp.trend,
        "trend_smoothed_llt": fit.trend_smoothed,
        "trend_smoothed_ll": fit_ll.trend_smoothed,
        "seasonal_true": dgp.seasonal,
        "seasonal_smoothed_llt": fit.seasonal_smoothed,
        "seasonal_smoothed_ll": fit_ll.seasonal_smoothed,
        "noise_true": dgp.noise,
    })
    out_path = OUT_DIR / f"bsts_recovery_seed{args.seed}_T{args.T}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
