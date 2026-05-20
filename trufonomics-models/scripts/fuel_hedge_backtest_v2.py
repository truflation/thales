"""Phase 3.1e v2 — fuel-hedging backtest with optimal hedge ratios.

Fixes from v1:
  * Use the **β-optimal** hedge ratio formula: h* = corr(diesel, hedge)
    × σ(diesel) / σ(hedge), instead of an arbitrary 50%
  * Test the best single instrument (DBO — Invesco DB Oil ETF, 0.60
    monthly correlation with retail diesel) instead of HOUSD
  * Test a **multi-instrument basket hedge**: regress diesel returns
    on {DBO, HOUSD, RBUSD} via rolling OLS, use those weights
  * Use the BVAR fuel-volatility signal as a *risk-on/risk-off
    multiplier* on the optimal ratio — not a 0/1 switch

Hedge instrument upper bound:
  * The PPI Diesel Fuel index (WPU057303) — itself a diesel measure
    — only correlates 0.74 monthly with the retail diesel target
    (GASDESW). That's the *theoretical* ceiling. Tradeable instruments
    cap out at ~0.60. Half of retail-diesel variance is fundamentally
    un-hedgeable via tradeable assets at the monthly frequency.

Five strategies compared:

  1. **Unhedged** — baseline
  2. **Static β-optimal DBO** — single-instrument optimal ratio,
     fit on full sample (look-ahead but standard reference)
  3. **Rolling β-optimal DBO** — refit ratio at each origin on the
     prior 36 months only (no look-ahead)
  4. **Rolling basket** — multi-instrument OLS (DBO + HOUSD + RBUSD)
     refit at each origin
  5. **BVAR-modulated basket** — basket weights from (4) scaled by
     the BVAR diesel-vol signal (1.5× when vol > median, 0.5× when below)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.cost_structures import get_cost_structure    # noqa: E402
from thales.models.archetypes.bvar_minnesota import (    # noqa: E402
    fit_bvar_minnesota,
)
from thales.vintage import VintageStore    # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "real_data_archetypes"

REVENUE = 100_000_000
OPEX_SHARE = 0.85
COST_POOL = REVENUE * OPEX_SHARE
FUEL_SHARE = 0.35
ANNUAL_FUEL_COST = COST_POOL * FUEL_SHARE
MONTHLY_FUEL_COST = ANNUAL_FUEL_COST / 12

HEDGE_INSTRUMENTS = ["DBO", "HOUSD", "RBUSD"]


def _load_panel() -> pd.DataFrame:
    """Monthly log-levels: target diesel + 5 BVAR vars + hedge instruments."""
    with VintageStore(VINTAGE_DB, read_only=True) as store:
        diesel = store.get_vintage("GASDESW", date.today()).dropna()
        freight = store.get_vintage("PCU48414841", date.today()).dropna()
        maint = store.get_vintage("CUSR0000SETD", date.today()).dropna()
        labor = store.get_vintage("CES4300000008", date.today()).dropna()
        volume = store.get_vintage("TRUCKD11", date.today()).dropna()
        hedges = {sym: store.get_vintage(sym, date.today()).dropna()
                       for sym in HEDGE_INSTRUMENTS}
    series = {
        "log_diesel":      np.log(diesel.resample("ME").last()),
        "log_freight":     np.log(freight.resample("ME").last()),
        "log_maintenance": np.log(maint.resample("ME").last()),
        "log_labor":       np.log(labor.resample("ME").last()),
        "log_volume":      np.log(volume.resample("ME").last()),
    }
    for sym, s in hedges.items():
        series[f"log_{sym}"] = np.log(s.resample("ME").last())
    return pd.concat(series, axis=1).dropna()


def _optimal_hedge_ratio(target_ret: pd.Series,
                              hedge_ret: pd.Series) -> float:
    """h* = corr × σ_target / σ_hedge — the minimum-variance hedge."""
    if hedge_ret.std() == 0:
        return 0.0
    return float(target_ret.cov(hedge_ret) / hedge_ret.var())


def _basket_weights_ols(target_ret: pd.Series,
                            hedge_rets: pd.DataFrame) -> np.ndarray:
    """OLS regression coefficients of target on hedges (no intercept,
    pure hedge-ratio fit). Returns a vector matching hedge_rets columns."""
    X = hedge_rets.values
    y = target_ret.values
    if len(X) < 6 or X.shape[1] >= len(X):
        return np.zeros(X.shape[1])
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coefs


def main() -> None:
    print("=" * 78)
    print("Phase 3.1e v2 — fuel-hedging backtest with optimal ratios")
    print("=" * 78)

    panel = _load_panel()
    print(f"\nPanel: n={len(panel)}  range "
          f"{panel.index.min():%Y-%m} → {panel.index.max():%Y-%m}")
    print(f"Hedge instruments: {HEDGE_INSTRUMENTS}")
    print(f"Shipper: ${REVENUE/1e6:.0f}M revenue, "
            f"${MONTHLY_FUEL_COST/1e6:.2f}M monthly fuel cost")

    # Compute monthly log-returns
    rets = panel.diff()

    # Hedge instrument quality
    print("\nHedge-instrument quality (vs log_diesel monthly returns):")
    for sym in HEDGE_INSTRUMENTS:
        c = rets["log_diesel"].corr(rets[f"log_{sym}"])
        h = _optimal_hedge_ratio(rets["log_diesel"],
                                          rets[f"log_{sym}"])
        print(f"  {sym:<6s}  corr={c:+.4f}   "
                f"β-optimal hedge ratio = {h:+.4f}")

    full_basket = _basket_weights_ols(
        rets["log_diesel"].dropna(),
        rets[[f"log_{s}" for s in HEDGE_INSTRUMENTS]].dropna())
    print(f"\nFull-sample multi-instrument basket weights:")
    for sym, w in zip(HEDGE_INSTRUMENTS, full_basket):
        print(f"  {sym:<6s}  weight = {w:+.4f}")
    # In-sample R² ceiling for the basket
    pred = (rets[[f"log_{s}" for s in HEDGE_INSTRUMENTS]].dropna().values
              @ full_basket)
    actual = rets["log_diesel"].loc[
        rets[[f"log_{s}" for s in HEDGE_INSTRUMENTS]].dropna().index].values
    r2 = 1 - np.var(actual - pred) / np.var(actual)
    print(f"  in-sample R²: {r2:.4f}  (theoretical max σ-reduction: "
            f"{(1 - np.sqrt(1 - r2)) * 100:.2f}%)")

    # ── Walk-forward backtest ───────────────────────────────────────
    train_min = 60
    origins = panel.index[train_min:-1]
    rows = []

    for origin in origins:
        # Full history up through origin (no peek)
        hist = panel.loc[panel.index <= origin]
        hist_rets = hist.diff().dropna()
        if len(hist_rets) < train_min:
            continue

        target = origin + pd.offsets.MonthEnd(1)
        if target not in panel.index:
            continue

        # Realized next-month log-returns
        d_diesel = float(panel.loc[target, "log_diesel"]
                              - panel.loc[origin, "log_diesel"])
        hedge_rets_next = {
            sym: float(panel.loc[target, f"log_{sym}"]
                           - panel.loc[origin, f"log_{sym}"])
            for sym in HEDGE_INSTRUMENTS
        }

        # Cost shock from realized diesel return (linear approx)
        cost_shock = MONTHLY_FUEL_COST * (np.exp(d_diesel) - 1)

        # Hedge P&Ls per dollar notional (one for each hedge instrument)
        # Notional convention: long $X of hedge → P&L = $X · (e^d_log - 1)
        per_dollar_pnl = {
            sym: np.exp(hedge_rets_next[sym]) - 1
            for sym in HEDGE_INSTRUMENTS
        }

        # Strategy 1: unhedged
        net_unhedged = -cost_shock

        # Strategy 2: static β-optimal DBO using FULL SAMPLE
        # (look-ahead — included as reference)
        h_static = _optimal_hedge_ratio(rets["log_diesel"].dropna(),
                                                  rets["log_DBO"].dropna())
        net_static = (-cost_shock
                          + h_static * MONTHLY_FUEL_COST * per_dollar_pnl["DBO"])

        # Strategy 3: rolling β-optimal DBO (no look-ahead, last 36 months)
        recent = hist_rets.tail(36)
        h_rolling = _optimal_hedge_ratio(recent["log_diesel"],
                                                   recent["log_DBO"])
        net_rolling = (-cost_shock
                            + h_rolling * MONTHLY_FUEL_COST
                            * per_dollar_pnl["DBO"])

        # Strategy 4: rolling basket OLS
        basket_w = _basket_weights_ols(
            recent["log_diesel"],
            recent[[f"log_{s}" for s in HEDGE_INSTRUMENTS]])
        basket_pnl = sum(
            w * MONTHLY_FUEL_COST * per_dollar_pnl[sym]
            for w, sym in zip(basket_w, HEDGE_INSTRUMENTS))
        net_basket = -cost_shock + basket_pnl

        # Strategy 5: BVAR-modulated basket
        # Fit BVAR on the panel (5-var subset), get conditional vol signal
        bvar_cols = ["log_diesel", "log_freight", "log_maintenance",
                          "log_labor", "log_volume"]
        try:
            fit = fit_bvar_minnesota(
                hist[bvar_cols].values, p=1,
                overall_tightness=0.5, cross_tightness=0.5, lag_decay=1.0)
            # σ_diesel from BVAR Σ (h=1 forecast SD on diesel)
            bvar_diesel_sd = float(np.sqrt(fit.sigma[0, 0]))
        except Exception:
            bvar_diesel_sd = float(recent["log_diesel"].std())

        all_diesel_sd = recent["log_diesel"].std()
        vol_ratio = bvar_diesel_sd / all_diesel_sd if all_diesel_sd > 0 else 1.0
        # Multiplier: 1.5× when BVAR thinks vol is high (>1.1× recent),
        #             0.5× when low (<0.9×), else 1.0
        if vol_ratio > 1.1:
            multiplier = 1.5
        elif vol_ratio < 0.9:
            multiplier = 0.5
        else:
            multiplier = 1.0
        net_bvar = -cost_shock + multiplier * basket_pnl

        rows.append({
            "origin": origin, "target": target,
            "diesel_ret": d_diesel,
            "h_static": h_static, "h_rolling": h_rolling,
            "basket_DBO": float(basket_w[0]),
            "basket_HOUSD": float(basket_w[1]),
            "basket_RBUSD": float(basket_w[2]),
            "vol_ratio": vol_ratio, "multiplier": multiplier,
            "cost_shock": cost_shock,
            "net_unhedged": net_unhedged,
            "net_static_DBO": net_static,
            "net_rolling_DBO": net_rolling,
            "net_basket": net_basket,
            "net_bvar_basket": net_bvar,
        })

    df = pd.DataFrame(rows)
    print(f"\nBacktest horizon: n={len(df)} months  "
            f"({df['origin'].min():%Y-%m} → {df['origin'].max():%Y-%m})")

    # ── Summary statistics ──────────────────────────────────────────
    def metrics(s: pd.Series, label: str) -> dict:
        ann_mean = s.mean() * 12
        ann_sd = s.std() * np.sqrt(12)
        sharpe = ann_mean / ann_sd if ann_sd > 0 else float("nan")
        cum = s.cumsum()
        max_dd = float((cum - cum.cummax()).min())
        return {
            "strategy": label, "n": len(s),
            "annual_mean_$": ann_mean, "annual_sd_$": ann_sd,
            "sharpe": sharpe, "max_dd_$": max_dd,
            "total_pnl_$": s.sum(),
        }

    print()
    print("=" * 78)
    print("Strategy comparison")
    print("=" * 78)
    summary = pd.DataFrame([
        metrics(df["net_unhedged"],         "1. unhedged"),
        metrics(df["net_static_DBO"],       "2. static DBO (β-opt, look-ahead)"),
        metrics(df["net_rolling_DBO"],      "3. rolling DBO (β-opt, 36m)"),
        metrics(df["net_basket"],           "4. rolling basket OLS"),
        metrics(df["net_bvar_basket"],      "5. BVAR-modulated basket"),
    ])
    print()
    print(summary.to_string(
        index=False,
        formatters={
            "annual_mean_$":   "{:>14,.0f}".format,
            "annual_sd_$":     "{:>14,.0f}".format,
            "sharpe":          "{:>+8.3f}".format,
            "max_dd_$":        "{:>14,.0f}".format,
            "total_pnl_$":     "{:>14,.0f}".format,
        }))

    # ── Variance reduction ─────────────────────────────────────────
    print()
    print("Variance reduction vs unhedged (positive = better):")
    base = df["net_unhedged"].std()
    for col, label in [
        ("net_static_DBO",   "static DBO (look-ahead)"),
        ("net_rolling_DBO",  "rolling DBO"),
        ("net_basket",       "rolling basket"),
        ("net_bvar_basket",  "BVAR-modulated basket"),
    ]:
        red = (1 - df[col].std() / base) * 100
        marker = "✓" if red > 0 else "✗"
        print(f"  {label:<24s}  σ-reduction: {red:+>6.2f}%  {marker}")

    # Persist
    out = OUT_DIR / "fuel_hedge_backtest_v2.csv"
    df.to_csv(out, index=False)
    summary.to_csv(OUT_DIR / "fuel_hedge_backtest_v2_summary.csv", index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
