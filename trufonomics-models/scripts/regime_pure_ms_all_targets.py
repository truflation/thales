"""Apply pure MS regime detector to all 4 official inflation targets.

For each of {BLS Headline CPI, BLS Core CPI, BEA PCE, BEA Core PCE} YoY:
  1. Load via thales.targets
  2. Fit fit_hamilton_2state
  3. Identify P(high) > 0.5 windows
  4. Tabulate cross-target coherence on known regime windows

Production-grade output: per-target CSV + a cross-target regime
co-occurrence table.
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
from thales import targets as T  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "regime"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_one(target_name: str, store: VintageStore) -> tuple:
    yoy = T.load_target_yoy(store, target_name, as_of=date.today())
    fit = fit_hamilton_2state(yoy.values)
    df = pd.DataFrame({
        "date": yoy.index,
        "yoy": yoy.values,
        "p_high": fit.smoothed_prob_high,
    })
    return target_name, fit, df


def find_windows(df: pd.DataFrame, threshold: float = 0.5) -> list[tuple]:
    """Find contiguous P(high) > threshold windows. Returns list of
    (start_date, end_date, n_months, peak_p_high)."""
    is_high = (df["p_high"] > threshold).astype(int).values
    out = []
    i = 0
    n = len(is_high)
    while i < n:
        if is_high[i] == 1:
            j = i
            while j < n and is_high[j] == 1:
                j += 1
            d_start = df.iloc[i]["date"]
            d_end = df.iloc[j - 1]["date"]
            peak = df.iloc[i:j]["p_high"].max()
            out.append((d_start, d_end, j - i, peak))
            i = j
        else:
            i += 1
    return out


def main() -> None:
    print("=" * 78)
    print("Pure MS regime detector on all 4 official inflation YoY targets")
    print("=" * 78)

    results = {}
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        for target in ["cpi", "core_cpi", "pce", "core_pce"]:
            name, fit, df = fit_one(target, store)
            results[name] = (fit, df)

    # Per-target summary
    for name, (fit, df) in results.items():
        print()
        print(f"── {name} (n={len(df)}) ──")
        print(f"  σ̂_low={fit.sigma_low:.4f}  σ̂_high={fit.sigma_high:.4f}  "
              f"p̂_00={fit.p_stay_low:.4f}  p̂_11={fit.p_stay_high:.4f}")
        windows = find_windows(df)
        if not windows:
            print("  (no high-vol regimes detected)")
        for d_start, d_end, n_months, peak in windows:
            print(f"  high-vol: {d_start:%Y-%m} → {d_end:%Y-%m}  "
                  f"({n_months} mo, peak P={peak:.3f})")

        out_path = OUT_DIR / f"regime_pure_ms_{name}_yoy.csv"
        df.to_csv(out_path, index=False)

    # Cross-target coherence on key windows
    print()
    print("=" * 78)
    print("Cross-target coherence — mean P(high) across known windows")
    print("=" * 78)
    known_windows = [
        ("2014-12", "2015-12", "Oil price collapse"),
        ("2020-03", "2020-08", "COVID-19 onset"),
        ("2021-06", "2023-12", "Post-COVID inflation surge"),
        ("2024-01", "2024-12", "Disinflation"),
    ]
    print()
    print(f"  {'window':<32s}  {'CPI':>6s}  {'Core CPI':>9s}  "
          f"{'PCE':>6s}  {'Core PCE':>9s}")
    print("  " + "-" * 74)
    for s_start, s_end, label in known_windows:
        start_ts = pd.Timestamp(s_start) + pd.offsets.MonthEnd(0)
        end_ts = pd.Timestamp(s_end) + pd.offsets.MonthEnd(0)
        row = [f"  {label:<30s}"]
        for name in ["cpi", "core_cpi", "pce", "core_pce"]:
            df = results[name][1]
            mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
            mean_p = df.loc[mask, "p_high"].mean() if mask.any() else float("nan")
            row.append(f"{mean_p:>9.3f}" if not np.isnan(mean_p) else f"{'n/a':>9s}")
        print("  ".join(row))

    print()
    print(f"Per-target CSVs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
