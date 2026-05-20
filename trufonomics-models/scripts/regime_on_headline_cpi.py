"""Phase 2.2 — Apply UC+SV+MS regime model to real BLS Headline CPI YoY.

Loads CPIAUCSL via ALFRED, computes monthly YoY, fits the full
UC + SV + MS composed model from Phase 1.3. Outputs:

  * Posterior point estimates of σ_low, σ_high, p_00, p_11, σ_eta, φ, σ_h
  * Smoothed P(S_t = high-vol regime | y_{1:T}) over the entire history
  * Smoothed level μ_t and log-vol h_t paths
  * Visual validation: high-vol regime should align with known
    inflation shocks (2008 GFC, 2014 oil crash, 2020 COVID, 2022 surge)

Saves CSV with all latents for plotting.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.uc_sv_ms import fit_uc_sv_ms  # noqa: E402
from thales import targets as T  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "regime"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 72)
    print("Phase 2.2 — UC+SV+MS regime model on BLS Headline CPI YoY")
    print("=" * 72)

    # Load CPIAUCSL → YoY
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        yoy = T.load_target_yoy(store, "cpi", as_of=date.today())
    print(f"\nBLS Headline CPI YoY: n={len(yoy)}  range "
          f"{yoy.index.min():%Y-%m-%d} → {yoy.index.max():%Y-%m-%d}")
    print(f"  range = [{yoy.min():.2f}, {yoy.max():.2f}]  "
          f"mean = {yoy.mean():.2f}  std = {yoy.std():.2f}")

    y = yoy.values

    # Fit UC+SV+MS via NumPyro NUTS
    # Tight σ_η prior (0.05) — production setting for monthly CPI YoY,
    # forces level to be slow-moving so MS mechanism actually fires.
    # Synthetic-recovery default (0.5) lets the level absorb regimes.
    print("\nFitting UC + SV + MS via NumPyro NUTS (warmup=600, samples=600)...")
    print("(σ_η prior tightened to HalfNormal(0.05) — see comment in script)")
    print("(this takes a few minutes on CPU)")
    fit = fit_uc_sv_ms(y, num_warmup=600, num_samples=600, seed=42,
                          sigma_eta_prior_scale=0.05)

    print()
    print("Posterior summary:")
    print(f"  σ̂_eta    = {fit.sigma_eta:.4f}")
    print(f"  σ̂_low    = {fit.sigma_low:.4f}")
    print(f"  σ̂_high   = {fit.sigma_high:.4f}")
    print(f"  p̂_00     = {fit.p_stay_low:.4f}")
    print(f"  p̂_11     = {fit.p_stay_high:.4f}")
    print(f"  φ̂        = {fit.phi:.4f}")
    print(f"  σ̂_h      = {fit.sigma_h:.4f}")
    print(f"  divergences = {fit.diverging}/{fit.n_samples}")

    # Construct output frame
    df = pd.DataFrame({
        "date": yoy.index,
        "cpi_yoy": y,
        "level_smoothed": fit.mu_smoothed,
        "log_vol_smoothed": fit.h_smoothed,
        "prob_high_regime": fit.smoothed_prob_high,
    })
    df["regime_label"] = (df["prob_high_regime"] > 0.5).map(
        {True: "high-vol", False: "low-vol"})

    # Identify high-vol windows
    print()
    print("=" * 72)
    print("High-vol regime windows (P(high) > 0.5)")
    print("=" * 72)
    is_high = df["prob_high_regime"] > 0.5
    transitions = is_high.diff()
    starts = df.index[transitions == True].tolist()    # noqa: E712
    ends = df.index[transitions == False].tolist()      # noqa: E712
    if is_high.iloc[0]:
        starts.insert(0, df.index[0])
    if is_high.iloc[-1]:
        ends.append(df.index[-1] + 1)
    for s, e in zip(starts, ends):
        d_start = df.loc[s, "date"]
        d_end = df.loc[e - 1, "date"] if e <= len(df) else df["date"].iloc[-1]
        peak_p = df.loc[s: e - 1, "prob_high_regime"].max()
        print(f"  {d_start:%Y-%m} → {d_end:%Y-%m}  "
              f"({(e - s)} months, peak P(high) = {peak_p:.3f})")

    out_path = OUT_DIR / "regime_on_bls_headline_cpi.csv"
    df.to_csv(out_path, index=False)
    print()
    print(f"Saved: {out_path}")

    # Quick comparison: known shock dates vs identified windows
    print()
    print("=" * 72)
    print("Known shocks (sanity check)")
    print("=" * 72)
    known_shocks = [
        ("2008-09", "2009-06", "Global financial crisis"),
        ("2014-12", "2015-12", "Oil price collapse"),
        ("2020-03", "2020-08", "COVID-19 onset"),
        ("2021-06", "2023-12", "Post-COVID inflation surge"),
    ]
    for s_start, s_end, name in known_shocks:
        try:
            start_ts = pd.Timestamp(s_start) + pd.offsets.MonthEnd(0)
            end_ts = pd.Timestamp(s_end) + pd.offsets.MonthEnd(0)
            mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
            mean_p = df.loc[mask, "prob_high_regime"].mean()
            max_p = df.loc[mask, "prob_high_regime"].max()
            label = "✓" if mean_p > 0.5 else ("~" if mean_p > 0.3 else "✗")
            print(f"  {label}  {s_start} → {s_end}  {name:<40s}  "
                  f"mean P(high) = {mean_p:.3f}  max = {max_p:.3f}")
        except Exception as e:   # noqa: BLE001
            print(f"  ?  {name}: lookup error ({e})")


if __name__ == "__main__":
    main()
