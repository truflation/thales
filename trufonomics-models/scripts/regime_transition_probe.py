"""Empirical probe — which signals would have flagged the high-error
months in the CPI committee backtest?

Tests 4 candidate regime-transition signals at each origin:

  1. MoM z-score: |MoM[origin]| / SD(trailing 12-month MoM)
     Detects when this month's MoM is far from typical recent magnitude.

  2. Volatility ratio: SD(trailing 3-month MoM) / SD(trailing 24-month MoM)
     Detects volatility-regime change (recent variance vs long-run).

  3. Component dispersion: cross-section SD of 11 BLS subindex YoYs at origin
     Detects fan-out in component behavior (suggests structural shift).

  4. Persistence-error elevation: SD(last 6 persistence errors) /
     SD(trailing 24 persistence errors)
     Detects "persistence has been wrong lately" — a meta-signal.

For each signal, computes correlation with |committee error| at origin+1m
and the AUC of "predicting band miss" (binary).

Output: per-origin CSV with all 4 signals + 80%-band-miss flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the loaders from BLS-native CBDF
import importlib.util
spec = importlib.util.spec_from_file_location(
    "blsnative", ROOT / "scripts" / "forecast_next_bls_cpi_blsnative.py")
blsnative = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blsnative)


def _component_yoy_panel(component_levels: pd.DataFrame) -> pd.DataFrame:
    """Compute YoY for each component at each date."""
    out = {}
    for col in component_levels.columns:
        s = component_levels[col]
        out[col] = pd.Series({
            t: (s.loc[t] / s.loc[pd.Timestamp(year=t.year-1, month=t.month, day=1)
                                  + pd.offsets.MonthEnd(0)] - 1.0) * 100.0
            for t in s.index
            if pd.Timestamp(year=t.year-1, month=t.month, day=1)
                 + pd.offsets.MonthEnd(0) in s.index
        })
    return pd.DataFrame(out).sort_index()


def main() -> None:
    print("=" * 78)
    print("Regime-transition signal probe")
    print("=" * 78)

    backtest_csv = ROOT / "results" / "next_release_forecast" / "backtest_cpi_committee.csv"
    bt = pd.read_csv(backtest_csv, parse_dates=["origin", "target"])
    print(f"\nLoaded backtest: {len(bt)} origins")

    # Load BLS headline level series for MoM signal
    con = duckdb.connect(str(ROOT / "data" / "vintage_store" / "thales.duckdb"),
                            read_only=True)
    bls_yoy = blsnative.load_bls_headline_yoy(con)
    component_levels = blsnative.load_bls_component_levels(con)
    con.close()

    # Reconstruct BLS headline LEVEL from the panel — we have the components
    # but for MoM we need the headline. Use composed level (it's very close
    # to actual headline due to Laspeyres identity).
    weights = blsnative.load_bls_weights()
    base = component_levels.index.min()
    base_levels = component_levels.loc[base]
    composite = pd.Series(0.0, index=component_levels.index)
    for col in component_levels.columns:
        composite += weights[col] * (component_levels[col] / base_levels[col]) * 100.0
    composite /= 100.0    # composite headline level

    log_mom = (np.log(composite) - np.log(composite.shift(1))) * 100.0  # in pp
    log_mom = log_mom.dropna()

    # Component YoY panel for dispersion signal
    comp_yoy = _component_yoy_panel(component_levels)

    rows = []
    for _, r in bt.iterrows():
        origin = r["origin"]
        if origin not in log_mom.index:
            continue

        # Signal 1: MoM z-score
        trailing12 = log_mom.loc[:origin].iloc[-12:-1]   # exclude current
        if len(trailing12) < 6:
            mom_z = float("nan")
        else:
            mu = trailing12.mean()
            sd = trailing12.std()
            mom_z = (log_mom.loc[origin] - mu) / sd if sd > 0 else 0.0

        # Signal 2: Volatility ratio
        trailing3 = log_mom.loc[:origin].iloc[-3:]
        trailing24 = log_mom.loc[:origin].iloc[-24:]
        if len(trailing24) < 12:
            vol_ratio = float("nan")
        else:
            sd_short = trailing3.std()
            sd_long = trailing24.std()
            vol_ratio = sd_short / sd_long if sd_long > 0 else 1.0

        # Signal 3: Component dispersion at origin
        if origin in comp_yoy.index:
            comp_disp = comp_yoy.loc[origin].std()
        else:
            comp_disp = float("nan")

        # Signal 4: Persistence-error elevation (rolling)
        # Persistence error = YoY[t] - YoY[t-1]
        yoy_diff = bls_yoy.diff().abs()
        recent6 = yoy_diff.loc[:origin].iloc[-6:]
        long24 = yoy_diff.loc[:origin].iloc[-24:]
        if len(long24) < 12:
            persist_err_elev = float("nan")
        else:
            sd6 = recent6.std()
            sd24 = long24.std()
            persist_err_elev = sd6 / sd24 if sd24 > 0 else 1.0

        rows.append({
            "origin":             origin,
            "target":             r["target"],
            "actual":             r["actual"],
            "committee":          r["committee"],
            "err_committee_bp":   r["err_committee_bp"],
            "abs_err_bp":         abs(r["err_committee_bp"]),
            "band_miss":          int(not r["actual_in_80"]),
            "sig1_mom_z":         mom_z,
            "sig2_vol_ratio":     vol_ratio,
            "sig3_comp_disp":     comp_disp,
            "sig4_persist_elev":  persist_err_elev,
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["sig1_mom_z", "sig2_vol_ratio",
                              "sig3_comp_disp", "sig4_persist_elev"])

    out_csv = ROOT / "results" / "next_release_forecast" / "regime_signals_backtest.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ── Correlations with committee error ─────────────────────────
    print()
    print("=" * 78)
    print(f"Pearson correlations with |committee error|  (n={len(df)})")
    print("=" * 78)
    for sig, label in [
        ("sig1_mom_z",        "|MoM z-score|"),
        ("sig2_vol_ratio",    "Volatility ratio (3m/24m SD)"),
        ("sig3_comp_disp",    "Component dispersion"),
        ("sig4_persist_elev", "Persistence-error elevation (6m/24m SD)"),
    ]:
        x = df[sig].abs() if sig == "sig1_mom_z" else df[sig]
        corr = x.corr(df["abs_err_bp"])
        print(f"  {label:<45s}  ρ = {corr:+.3f}")

    # ── Threshold analysis: what fraction of band-misses are flagged? ──
    print()
    print("=" * 78)
    print(f"Band-miss rate at signal thresholds  (overall band miss = "
            f"{df['band_miss'].mean()*100:.1f}%)")
    print("=" * 78)
    for sig, label, thresh in [
        ("sig1_mom_z",        "|MoM z-score| > 2",          2.0),
        ("sig1_mom_z",        "|MoM z-score| > 1.5",        1.5),
        ("sig2_vol_ratio",    "Vol ratio > 1.5",            1.5),
        ("sig2_vol_ratio",    "Vol ratio > 2.0",            2.0),
        ("sig3_comp_disp",    "Component dispersion > 3.0", 3.0),
        ("sig4_persist_elev", "Persistence-elev > 1.5",     1.5),
    ]:
        if sig == "sig1_mom_z":
            flagged = df[df[sig].abs() > thresh]
        else:
            flagged = df[df[sig] > thresh]
        nflag = len(flagged)
        if nflag == 0:
            print(f"  {label:<32s}  flagged={nflag:>3d}/{len(df)}  "
                    f"(no obs)")
            continue
        band_miss_in_flag = flagged["band_miss"].mean()
        recall = (flagged["band_miss"].sum()
                    / max(df["band_miss"].sum(), 1))
        print(f"  {label:<32s}  flagged={nflag:>3d}/{len(df)}  "
                f"band-miss-rate={band_miss_in_flag*100:>5.1f}%  "
                f"captures {recall*100:>5.1f}% of all band-misses")

    # ── Composite "transition score" — average of normalized signals ──
    print()
    print("=" * 78)
    print("Composite transition score (mean of 0-1 normalized signals)")
    print("=" * 78)
    def _norm(s, lo, hi):
        return ((s - lo) / (hi - lo)).clip(0, 1)
    df["transition_score"] = (
        _norm(df["sig1_mom_z"].abs(), 0, 3) +
        _norm(df["sig2_vol_ratio"], 0.5, 2.0) +
        _norm(df["sig3_comp_disp"], 1.0, 4.0) +
        _norm(df["sig4_persist_elev"], 0.5, 2.0)
    ) / 4.0

    print(f"  Correlation transition_score ↔ |error|: "
            f"ρ = {df['transition_score'].corr(df['abs_err_bp']):+.3f}")
    for thresh in [0.3, 0.4, 0.5]:
        flagged = df[df["transition_score"] > thresh]
        nflag = len(flagged)
        if nflag == 0:
            print(f"  transition_score > {thresh}  flagged=0/{len(df)}")
            continue
        miss_in_flag = flagged["band_miss"].mean()
        recall = (flagged["band_miss"].sum() / max(df["band_miss"].sum(), 1))
        print(f"  transition_score > {thresh}   flagged={nflag:>3d}/{len(df)}  "
                f"band-miss-rate={miss_in_flag*100:>5.1f}%  "
                f"captures {recall*100:>5.1f}% of band-misses")

    # ── Show the historical big-miss months and their signal values ──
    print()
    print("=" * 78)
    print("Big-miss months (|err| > 50 bp) and their concurrent signal values")
    print("=" * 78)
    big = df[df["abs_err_bp"] > 50.0].sort_values("origin")
    print(f"{'origin':<10s}  {'err (bp)':>10s}  {'momZ':>6s}  "
            f"{'volR':>5s}  {'cDisp':>6s}  {'persE':>6s}  {'trans':>6s}")
    for _, r in big.iterrows():
        print(f"{r['origin'].strftime('%Y-%m')}     "
                f"{r['err_committee_bp']:>+9.2f}   "
                f"{r['sig1_mom_z']:>+5.2f}   "
                f"{r['sig2_vol_ratio']:>4.2f}   "
                f"{r['sig3_comp_disp']:>5.2f}   "
                f"{r['sig4_persist_elev']:>5.2f}   "
                f"{r['transition_score']:>5.2f}")


if __name__ == "__main__":
    main()
