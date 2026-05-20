"""Phase 2.2c — UC+SV+MS applied to CPI LEVEL (CPIAUCSL index value).

The Phase 2.2 finding showed UC+SV+MS over-fits monthly CPI YoY because
YoY is already differenced — there's no genuine level walk for the UC
layer to identify. Pure MS turned out to be the right architecture for
YoY.

This script tests the COMPLEMENTARY claim: on CPI LEVEL data
(CPIAUCSL index), the UC layer SHOULD work because CPI level trends
secularly over decades. If it fits properly, this validates the
"choose the architecture by data type" production rule.

We use 100 * log(CPIAUCSL) so the units are similar in magnitude to
the YoY series (around 100-300 over the full history). MCMC cost
scales with T; we use 2010-onward to keep T tractable (~194 obs).
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
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "regime"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 78)
    print("Phase 2.2c — UC+SV+MS on CPI LEVEL (CPIAUCSL index)")
    print("=" * 78)

    with VintageStore(VINTAGE_DB, read_only=True) as store:
        cpi_level = store.get_vintage("CPIAUCSL", date.today()).dropna()

    cpi_level.index = pd.to_datetime(cpi_level.index)
    cpi_level = cpi_level.sort_index()
    # Use 100 * log(CPI) so scale is similar to YoY in magnitude
    log_cpi = 100.0 * np.log(cpi_level)

    print(f"\nCPI level: n={len(log_cpi)}  range "
          f"{log_cpi.index.min():%Y-%m-%d} → {log_cpi.index.max():%Y-%m-%d}")
    print(f"  100*log(CPI) range: [{log_cpi.min():.2f}, {log_cpi.max():.2f}]")
    print(f"  total drift: {log_cpi.iloc[-1] - log_cpi.iloc[0]:+.2f}")

    print("\nFitting UC+SV+MS via NumPyro (warmup=600, samples=600)...")
    print("(level series — UC layer expected to identify secular drift cleanly)")
    fit = fit_uc_sv_ms(log_cpi.values, num_warmup=600, num_samples=600,
                          seed=42, sigma_eta_prior_scale=2.0)

    print()
    print("Posterior summary:")
    print(f"  σ̂_eta    = {fit.sigma_eta:.4f}    (level walk SD)")
    print(f"  σ̂_low    = {fit.sigma_low:.4f}    (residual SD calm regime)")
    print(f"  σ̂_high   = {fit.sigma_high:.4f}    (residual SD turbulent regime)")
    print(f"  p̂_00     = {fit.p_stay_low:.4f}")
    print(f"  p̂_11     = {fit.p_stay_high:.4f}")
    print(f"  φ̂        = {fit.phi:.4f}    (SV persistence)")
    print(f"  σ̂_h      = {fit.sigma_h:.4f}    (SV innovation SD)")
    print(f"  divergences = {fit.diverging}/{fit.n_samples}")

    # Diagnostic: does level absorb regime?
    level_smoothed = fit.mu_smoothed
    log_cpi_arr = log_cpi.values
    residual_after_level = log_cpi_arr - level_smoothed
    residual_std_post = residual_after_level.std()
    print()
    print(f"Diagnostic — variance allocation:")
    print(f"  level_smoothed correlation with raw: "
          f"{np.corrcoef(level_smoothed, log_cpi_arr)[0,1]:.4f}")
    print(f"  residual_after_level std: {residual_std_post:.4f}")
    print(f"  σ̂_low / residual_std: {fit.sigma_low/residual_std_post:.2f}")

    df = pd.DataFrame({
        "date": log_cpi.index,
        "log_cpi_x100": log_cpi.values,
        "level_smoothed": fit.mu_smoothed,
        "log_vol_smoothed": fit.h_smoothed,
        "prob_high_regime": fit.smoothed_prob_high,
    })

    print()
    print("=" * 78)
    print("High-vol regime windows (P(high) > 0.5)")
    print("=" * 78)
    is_high = (df["prob_high_regime"] > 0.5).astype(int).values
    out = []
    i = 0
    while i < len(is_high):
        if is_high[i] == 1:
            j = i
            while j < len(is_high) and is_high[j] == 1:
                j += 1
            d_start = df.iloc[i]["date"]
            d_end = df.iloc[j - 1]["date"]
            peak = df.iloc[i:j]["prob_high_regime"].max()
            out.append((d_start, d_end, j - i, peak))
            i = j
        else:
            i += 1
    if not out:
        print("  (none — UC layer absorbed regime variance, same finding as YoY)")
    for d_start, d_end, n_months, peak in out:
        print(f"  {d_start:%Y-%m} → {d_end:%Y-%m}  "
              f"({n_months} months, peak P={peak:.3f})")

    out_path = OUT_DIR / "uc_sv_ms_on_cpi_level.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Known shocks comparison
    print()
    print("=" * 78)
    print("Known shocks (sanity check — does UC+SV+MS catch them on level data?)")
    print("=" * 78)
    known = [
        ("2014-12", "2015-12", "Oil price collapse (CPI level fell briefly)"),
        ("2020-03", "2020-08", "COVID-19 onset"),
        ("2021-06", "2023-12", "Post-COVID inflation surge (level accelerated)"),
        ("2024-01", "2024-12", "Disinflation (level steady)"),
    ]
    for s_start, s_end, name in known:
        start_ts = pd.Timestamp(s_start) + pd.offsets.MonthEnd(0)
        end_ts = pd.Timestamp(s_end) + pd.offsets.MonthEnd(0)
        mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
        if mask.any():
            mean_p = df.loc[mask, "prob_high_regime"].mean()
            label = "✓" if mean_p > 0.5 else ("~" if mean_p > 0.3 else "✗")
            print(f"  {label}  {s_start} → {s_end}  {name:<45s}  "
                  f"mean P(high) = {mean_p:.3f}")


if __name__ == "__main__":
    main()
