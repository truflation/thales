"""Empirical test: is slope drift meaningful in real US inflation data?

Decides whether the LLT BSTS variant earns its extra σ_δ parameter on
the four official inflation target series (CPIAUCSL, CPILFESL, PCEPI,
PCEPILFE). Fits both variants on each series' level (where seasonal +
trend matter) and reports:

  * σ_δ from LLT — is it materially > 0?
  * log-likelihood gain LLT − LL
  * AIC: 2k − 2ll, lower = better
  * Sample-size-adjusted Δ AIC threshold:  2 (one extra parameter)
    means LLT must beat LL by Δ log-lik > 1 to be net positive.

If σ_δ collapses to zero on every series and Δ log-lik < 1, the LLT
flexibility doesn't pay for itself empirically and the local-level
variant is the recommended default.

Conversely, if σ_δ is consistently positive and Δ log-lik > 1, LLT
captures real structure that LL misses.

Output: per-series report + summary verdict.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.bsts import (  # noqa: E402
    fit_bsts,
    fit_bsts_local_level,
)
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"

# Headline + Core for both BLS CPI and BEA PCE (level series, period=12)
SERIES = [
    ("CPIAUCSL",  "BLS Headline CPI"),
    ("CPILFESL",  "BLS Core CPI"),
    ("PCEPI",     "BEA Headline PCE"),
    ("PCEPILFE",  "BEA Core PCE"),
]


def evaluate(series_id: str, label: str,
              transform: str = "level") -> dict:
    """Fit both variants on the chosen transform of the series.

    ``transform`` ∈ {"level", "yoy"}. YoY is computed as 12-month log
    change in %. The yearly seasonality is largely absorbed by the
    differencing, so BSTS seasonal estimates should be small.
    """
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        s = store.get_vintage(series_id, date.today())
    s = s.dropna().sort_index()

    if transform == "level":
        y = s.values
    elif transform == "yoy":
        yoy = (s / s.shift(12) - 1.0) * 100.0
        y = yoy.dropna().values
    else:
        raise ValueError(f"unknown transform {transform}")

    if len(y) < 50:
        return {"series_id": series_id, "label": label,
                "transform": transform,
                "note": f"insufficient data (n={len(y)})"}

    fit_llt = fit_bsts(y, period=12)
    fit_ll = fit_bsts_local_level(y, period=12)

    # AIC = 2k − 2ll. LLT has 4 hyperparameters; LL has 3.
    aic_llt = 2 * 4 - 2 * fit_llt.log_likelihood
    aic_ll = 2 * 3 - 2 * fit_ll.log_likelihood
    delta_ll = fit_llt.log_likelihood - fit_ll.log_likelihood
    delta_aic = aic_llt - aic_ll
    sigma_delta_ratio = (fit_llt.sigma_delta /
                            (fit_llt.sigma_mu + 1e-12))

    # Verdict per series
    if fit_llt.sigma_delta < 1e-6:
        verdict_per = "σ_δ collapsed to zero — LLT degenerates to LL"
    elif delta_ll < 1.0:
        verdict_per = "Δ log-lik < 1 — extra parameter not worth it (prefer LL)"
    else:
        verdict_per = "LLT meaningfully better"

    return {
        "series_id": series_id,
        "label": label,
        "transform": transform,
        "n": len(y),
        "log_lik_llt": fit_llt.log_likelihood,
        "log_lik_ll": fit_ll.log_likelihood,
        "delta_log_lik": delta_ll,
        "aic_llt": aic_llt,
        "aic_ll": aic_ll,
        "delta_aic": delta_aic,
        "sigma_mu_llt": fit_llt.sigma_mu,
        "sigma_delta_llt": fit_llt.sigma_delta,
        "sigma_delta_to_mu_ratio": sigma_delta_ratio,
        "sigma_mu_ll": fit_ll.sigma_mu,
        "verdict": verdict_per,
    }


def main() -> None:
    print()
    print("=" * 80)
    print("Empirical slope-drift test — LLT vs Local-Level BSTS on US inflation")
    print("=" * 80)

    rows = []
    for transform in ["level", "yoy"]:
        rows.extend(evaluate(sid, label, transform=transform)
                       for sid, label in SERIES)
    df = pd.DataFrame(rows)

    for transform in ["level", "yoy"]:
        sub = df[df["transform"] == transform]
        print()
        print(f"### {transform.upper()} series ###")
        print(f"{'series':<12s} {'label':<22s} {'σ_δ (LLT)':>12s} "
              f"{'σ_δ/σ_μ':>10s} {'Δ ll':>8s} {'Δ AIC':>8s}  verdict")
        print("-" * 110)
        for _, r in sub.iterrows():
            if "note" in r and pd.notna(r.get("note")):
                print(f"{r['series_id']:<12s} {r['label']:<22s} {r['note']}")
                continue
            print(f"{r['series_id']:<12s} {r['label']:<22s} "
                  f"{r['sigma_delta_llt']:>12.5f} "
                  f"{r['sigma_delta_to_mu_ratio']:>10.4f} "
                  f"{r['delta_log_lik']:>+8.3f} "
                  f"{r['delta_aic']:>+8.3f}  {r['verdict']}")

    out = ROOT / "results" / "archetype_recovery" / "bsts_slope_drift_real_cpi.csv"
    df.to_csv(out, index=False)
    print()
    print(f"Saved: {out}")

    # Aggregate verdict per transform
    print()
    print("=" * 80)
    print("Aggregate read")
    print("=" * 80)
    for transform in ["level", "yoy"]:
        sub = df[df["transform"] == transform]
        n_collapsed = (sub["sigma_delta_llt"] < 1e-6).sum()
        n_unworth = (sub["delta_log_lik"] < 1.0).sum()
        n_meaningful = (sub["delta_log_lik"] >= 1.0).sum()
        print(f"\n[{transform.upper()}]")
        print(f"  σ_δ collapsed to zero:        {n_collapsed} / {len(sub)}")
        print(f"  Δ log-lik < 1 (LL adequate):  {n_unworth} / {len(sub)}")
        print(f"  LLT meaningfully better:      {n_meaningful} / {len(sub)}")
        if n_meaningful == len(sub):
            print(f"  → LLT preferred default for {transform}")
        elif n_meaningful == 0:
            print(f"  → LL preferred default for {transform}")
        else:
            print(f"  → mixed — choose per-series for {transform}")


if __name__ == "__main__":
    main()
