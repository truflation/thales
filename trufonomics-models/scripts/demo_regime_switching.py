"""Demo: Markov-switching variance recovery on synthetic data.

Generates a 2-state regime-switched series, fits Hamilton+Kim, prints
recovery metrics + saves a CSV with smoothed regime probabilities.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.regime_switching import (  # noqa: E402
    fit_hamilton_2state,
)
from thales.synthetic.regime_switching import simulate_markov_switching  # noqa: E402

OUT_DIR = ROOT / "results" / "archetype_recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Generating Markov-switching DGP (T={args.T}, seed={args.seed})...")
    dgp = simulate_markov_switching(
        T=args.T, mu=0.0, sigma_low=0.5, sigma_high=2.0,
        p_stay_low=0.95, p_stay_high=0.85, seed=args.seed,
    )
    high_share = dgp.regime.mean()
    print(f"  regime path: {(dgp.regime == 0).sum()} low, "
          f"{(dgp.regime == 1).sum()} high  ({high_share:.2%} high)")

    print()
    print("Fitting Hamilton 2-state filter (MLE on μ, σ_0, σ_1, p_00, p_11)...")
    fit = fit_hamilton_2state(dgp.y)

    print(f"  μ̂      = {fit.mu:+.4f}    (true {dgp.mu:+.4f})")
    print(f"  σ̂_low  = {fit.sigma_low:.4f}    (true {dgp.sigma_low:.4f})")
    print(f"  σ̂_high = {fit.sigma_high:.4f}    (true {dgp.sigma_high:.4f})")
    print(f"  p̂_00   = {fit.p_stay_low:.4f}    (true {dgp.p_stay_low:.4f})")
    print(f"  p̂_11   = {fit.p_stay_high:.4f}    (true {dgp.p_stay_high:.4f})")
    print(f"  iter   = {fit.n_iter}    log-lik = {fit.log_likelihood:.1f}")

    pred_smooth = (fit.smoothed_prob_high > 0.5).astype(int)
    pred_filt = (fit.filtered_prob_high > 0.5).astype(int)
    acc_smooth = (pred_smooth == dgp.regime).mean()
    acc_filt = (pred_filt == dgp.regime).mean()
    base_rate = max(high_share, 1 - high_share)

    print()
    print("Regime classification accuracy:")
    print(f"  Filtered:  {acc_filt:.4f}")
    print(f"  Smoothed:  {acc_smooth:.4f}")
    print(f"  Base rate (always-low): {base_rate:.4f}")

    df = pd.DataFrame({
        "t": np.arange(args.T),
        "y": dgp.y,
        "true_regime": dgp.regime,
        "filtered_prob_high": fit.filtered_prob_high,
        "smoothed_prob_high": fit.smoothed_prob_high,
    })
    out_path = OUT_DIR / f"regime_switching_recovery_seed{args.seed}_T{args.T}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
