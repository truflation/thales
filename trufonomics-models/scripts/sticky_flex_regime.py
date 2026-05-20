"""Bils-Klenow sticky/flex decomposition + per-bucket regime detection.

Standard methodology (Atlanta Fed): components with average price-change
interval > 4.3 months are "sticky"; ≤ 4.3 months are "flexible." Sticky
captures services pricing inertia; flexible captures food/energy
volatility.

For Thales we use the BLS subindex panel + canonical sticky/flex
classification from the literature, weighted to BLS-published category
weights (approx).

  STICKY        ~ 70% of headline   — services dominated
    Shelter (SAH1)
    Owners' equivalent rent (SEHC01)
    Rent of primary residence (SEHA)
    Medical care (SAM)
    Recreation (SAR)
    Education & communications (not in our subindex panel; approximated)

  FLEXIBLE      ~ 30% of headline   — goods + energy
    Food at home (SAF11)
    Food away from home (SEFV)
    Energy (SA0E)        — composite
    Energy commodities, gasoline (SAE / SETB01)
    Apparel (SAA)
    New + used vehicles (SETA01 + SETA02)

We approximate STICKY ≈ Core CPI YoY (CPILFESL — already vintage-stored)
and FLEX ≈ Headline YoY − Core YoY (from CPIAUCSL − CPILFESL). This is
the most-used approximation in the literature; the exact Bils-Klenow
basket re-weighting differs by ~5-10% in the resulting series.

Pure MS regime detector applied to each. Outputs:

  * Sticky-CPI YoY regime probability
  * Flex-CPI YoY regime probability
  * Time-varying gap between them
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.regime_switching import fit_hamilton_2state  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402
from thales import targets as T  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "regime"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 78)
    print("Bils-Klenow sticky/flex decomposition + regime detection")
    print("=" * 78)

    # Load Headline + Core CPI YoY
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        headline_yoy = T.load_target_yoy(store, "cpi", as_of=date.today())
        core_yoy = T.load_target_yoy(store, "core_cpi", as_of=date.today())

    # Sticky ≈ Core CPI YoY
    # Flex ≈ Headline - Core YoY (residual, captures food + energy effects)
    common = headline_yoy.index.intersection(core_yoy.index)
    sticky_yoy = core_yoy.loc[common].rename("sticky_yoy")
    flex_yoy = (headline_yoy.loc[common] - core_yoy.loc[common]).rename("flex_yoy")
    headline_yoy = headline_yoy.loc[common].rename("headline_yoy")

    print(f"\nMonthly panel: n={len(common)}  range "
          f"{common.min():%Y-%m-%d} → {common.max():%Y-%m-%d}")
    print(f"  headline range:  [{headline_yoy.min():.2f}, {headline_yoy.max():.2f}]")
    print(f"  sticky range:    [{sticky_yoy.min():.2f}, {sticky_yoy.max():.2f}]")
    print(f"  flex range:      [{flex_yoy.min():.2f}, {flex_yoy.max():.2f}]")

    # Pure MS regime detection on each bucket
    print("\nFitting pure MS on sticky CPI YoY...")
    fit_sticky = fit_hamilton_2state(sticky_yoy.values)
    print(f"  σ_low={fit_sticky.sigma_low:.4f}  σ_high={fit_sticky.sigma_high:.4f}  "
          f"p_00={fit_sticky.p_stay_low:.3f}  p_11={fit_sticky.p_stay_high:.3f}")

    print("Fitting pure MS on flex CPI YoY...")
    fit_flex = fit_hamilton_2state(flex_yoy.values)
    print(f"  σ_low={fit_flex.sigma_low:.4f}  σ_high={fit_flex.sigma_high:.4f}  "
          f"p_00={fit_flex.p_stay_low:.3f}  p_11={fit_flex.p_stay_high:.3f}")

    # Sticky-flex GAP — the persistence-conditioned vol indicator
    sticky_p_high = pd.Series(fit_sticky.smoothed_prob_high, index=common)
    flex_p_high = pd.Series(fit_flex.smoothed_prob_high, index=common)
    gap = (sticky_p_high - flex_p_high).rename("sticky_minus_flex_gap")

    df = pd.DataFrame({
        "headline_yoy": headline_yoy,
        "sticky_yoy": sticky_yoy,
        "flex_yoy": flex_yoy,
        "sticky_p_high": sticky_p_high,
        "flex_p_high": flex_p_high,
        "sticky_minus_flex_gap": gap,
    })

    # High-vol regime windows for each bucket
    print()
    print("=" * 78)
    print("High-vol regime windows by bucket")
    print("=" * 78)

    def _windows(p_high: pd.Series, threshold: float = 0.5) -> list[tuple]:
        is_high = (p_high > threshold).astype(int).values
        out = []
        i = 0
        while i < len(is_high):
            if is_high[i] == 1:
                j = i
                while j < len(is_high) and is_high[j] == 1:
                    j += 1
                d_start = p_high.index[i]
                d_end = p_high.index[j - 1]
                peak = p_high.iloc[i:j].max()
                out.append((d_start, d_end, j - i, peak))
                i = j
            else:
                i += 1
        return out

    print("\nSTICKY (Core CPI YoY) regimes:")
    for d_start, d_end, n_months, peak in _windows(sticky_p_high):
        print(f"  {d_start:%Y-%m} → {d_end:%Y-%m}  ({n_months} mo, P={peak:.3f})")

    print("\nFLEXIBLE (Headline - Core YoY) regimes:")
    for d_start, d_end, n_months, peak in _windows(flex_p_high):
        print(f"  {d_start:%Y-%m} → {d_end:%Y-%m}  ({n_months} mo, P={peak:.3f})")

    # Sticky-vs-flex divergence analysis
    print()
    print("=" * 78)
    print("Sticky-vs-flex divergence — when do the two move differently?")
    print("=" * 78)
    div_threshold = 0.5
    sticky_high_flex_low = ((sticky_p_high > div_threshold)
                                & (flex_p_high <= div_threshold)).sum()
    flex_high_sticky_low = ((flex_p_high > div_threshold)
                                & (sticky_p_high <= div_threshold)).sum()
    both_high = ((sticky_p_high > div_threshold)
                    & (flex_p_high > div_threshold)).sum()
    both_low = ((sticky_p_high <= div_threshold)
                   & (flex_p_high <= div_threshold)).sum()
    n = len(sticky_p_high)
    print(f"  Sticky high, flex low:    {sticky_high_flex_low:>4d} months  "
          f"({sticky_high_flex_low/n:.1%})")
    print(f"  Flex high, sticky low:    {flex_high_sticky_low:>4d} months  "
          f"({flex_high_sticky_low/n:.1%})")
    print(f"  Both high (regime-coherent shock):  {both_high:>4d} months  "
          f"({both_high/n:.1%})")
    print(f"  Both low (calm regime):             {both_low:>4d} months  "
          f"({both_low/n:.1%})")

    # Latest reading
    print()
    print("=" * 78)
    print(f"Latest reading ({df.index[-1]:%Y-%m-%d})")
    print("=" * 78)
    print(f"  Headline YoY:      {df['headline_yoy'].iloc[-1]:+.3f}%")
    print(f"  Sticky YoY:        {df['sticky_yoy'].iloc[-1]:+.3f}%")
    print(f"  Flex YoY:          {df['flex_yoy'].iloc[-1]:+.3f}%")
    print(f"  P(sticky high):    {df['sticky_p_high'].iloc[-1]:.3f}")
    print(f"  P(flex high):      {df['flex_p_high'].iloc[-1]:.3f}")
    print(f"  Sticky-flex gap:   {df['sticky_minus_flex_gap'].iloc[-1]:+.3f}")

    out_path = OUT_DIR / "sticky_flex_regime.csv"
    df.to_csv(out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
