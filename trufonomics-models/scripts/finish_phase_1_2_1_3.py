"""Finish remaining Phase 1.2 and 1.3 real-data extensions in one pass.

§1.2 BSTS additional categories (real data):
  - Food-away portion of Food (food_and_non_alcoholic_beverages_food_away_from_home)
  - All Other (other)

§1.3 Pure MS on sticky-services categories (real data):
  - Health
  - Education
  - Communications
  - Alcohol & Tobacco

For each category produce per-category CSV with the relevant smoothed
latents. Combined summary table at the end.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.bsts import fit_bsts_local_level  # noqa: E402
from thales.models.archetypes.regime_switching import fit_hamilton_2state  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "real_data_archetypes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_monthly_yoy(daily: pd.Series) -> pd.Series:
    monthly = daily.resample("ME").last()
    yoy = (monthly / monthly.shift(12) - 1.0) * 100.0
    return yoy.dropna()


def fit_bsts_real(label: str, raw_name: str,
                     store: VintageStore) -> dict:
    daily = store.get_vintage(raw_name, date.today()).dropna()
    yoy = _to_monthly_yoy(daily)
    if len(yoy) < 34:
        return {"label": label, "n": len(yoy), "skipped": "<34 obs"}
    fit = fit_bsts_local_level(yoy.values, period=12)
    burn = 24
    fitted = fit.trend_smoothed[burn:] + fit.seasonal_smoothed[burn:]
    truth = yoy.values[burn:]
    ss_res = np.sum((fitted - truth) ** 2)
    ss_tot = np.sum((truth - truth.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    seasonal_amp = (fit.seasonal_smoothed.max()
                      - fit.seasonal_smoothed.min())

    df = pd.DataFrame({
        "date": yoy.index,
        "yoy_pct": yoy.values,
        "trend_ll": fit.trend_smoothed,
        "seasonal_ll": fit.seasonal_smoothed,
    })
    df.to_csv(OUT_DIR / f"bsts_yoy_ll_{label}.csv", index=False)
    return {
        "label": label, "n": len(yoy),
        "sigma_mu": fit.sigma_mu, "sigma_seasonal": fit.sigma_seasonal,
        "sigma_eps": fit.sigma_eps,
        "trend_first": float(fit.trend_smoothed[0]),
        "trend_last": float(fit.trend_smoothed[-1]),
        "seasonal_amp_pp": float(seasonal_amp),
        "r2_post_burn": float(r2),
    }


def fit_pure_ms_real(label: str, raw_name: str,
                       store: VintageStore) -> dict:
    daily = store.get_vintage(raw_name, date.today()).dropna()
    yoy = _to_monthly_yoy(daily)
    if len(yoy) < 50:
        return {"label": label, "n": len(yoy), "skipped": "<50 obs"}
    fit = fit_hamilton_2state(yoy.values)

    df = pd.DataFrame({
        "date": yoy.index,
        "yoy_pct": yoy.values,
        "p_high": fit.smoothed_prob_high,
    })
    df.to_csv(OUT_DIR / f"pure_ms_yoy_{label}.csv", index=False)

    is_high = (fit.smoothed_prob_high > 0.5).astype(int)
    n_high = int(is_high.sum())
    return {
        "label": label, "n": len(yoy),
        "sigma_low": fit.sigma_low, "sigma_high": fit.sigma_high,
        "p_stay_low": fit.p_stay_low, "p_stay_high": fit.p_stay_high,
        "n_high_months": n_high,
        "frac_high": n_high / len(yoy),
    }


def main() -> None:
    print("=" * 78)
    print("Finishing Phase 1.2 BSTS + 1.3 Pure MS real-data extensions")
    print("=" * 78)

    bsts_targets = {
        "food_away": "food_and_non_alcoholic_beverages_food_away_from_home",
        "other": "other",
    }
    sticky_targets = {
        "health": "health",
        "education": "education",
        "communications": "communications",
        "alcohol_tobacco": "alcohol_and_tobacco",
    }

    bsts_rows = []
    ms_rows = []
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        # § 1.2 BSTS
        print("\n--- § 1.2 BSTS local-level on YoY ---")
        for label, raw_name in bsts_targets.items():
            print(f"  Fitting BSTS on {label} ({raw_name})...")
            r = fit_bsts_real(label, raw_name, store)
            bsts_rows.append(r)
            if "skipped" in r:
                print(f"    skipped: {r['skipped']}")
            else:
                print(f"    σ_eps={r['sigma_eps']:.3f}  "
                      f"seasonal_amp={r['seasonal_amp_pp']:.3f}pp  "
                      f"R²={r['r2_post_burn']:.4f}  "
                      f"trend {r['trend_first']:+.2f}% → {r['trend_last']:+.2f}%")

        # § 1.3 Pure MS on sticky services
        print("\n--- § 1.3 Pure MS regime detector on sticky services ---")
        for label, raw_name in sticky_targets.items():
            print(f"  Fitting pure MS on {label} ({raw_name})...")
            r = fit_pure_ms_real(label, raw_name, store)
            ms_rows.append(r)
            if "skipped" in r:
                print(f"    skipped: {r['skipped']}")
            else:
                print(f"    σ_low={r['sigma_low']:.3f}  "
                      f"σ_high={r['sigma_high']:.3f}  "
                      f"p_00={r['p_stay_low']:.3f}  "
                      f"p_11={r['p_stay_high']:.3f}  "
                      f"high frac={r['frac_high']:.1%}")

    # Combined summary
    print()
    print("=" * 78)
    print("Combined summary tables")
    print("=" * 78)
    bsts_df = pd.DataFrame(bsts_rows)
    ms_df = pd.DataFrame(ms_rows)
    print()
    print("BSTS LL on YoY:")
    print(bsts_df.to_string(index=False))
    print()
    print("Pure MS on YoY:")
    print(ms_df.to_string(index=False))

    bsts_df.to_csv(OUT_DIR / "bsts_yoy_ll_summary.csv", index=False)
    ms_df.to_csv(OUT_DIR / "pure_ms_yoy_summary.csv", index=False)
    print()
    print(f"Saved summary CSVs and per-category CSVs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
