"""MTD (month-to-date) daily-updating nowcast — the forecast that sharpens toward print day.

The print engine produces ONE forecast per print (origin = last completed month). But Truflation
prints DAILY, so as the target month fills in we can re-nowcast every day and the band should
tighten toward print day. This module makes the engine a function of an AS-OF date.

Mechanism: every month-M Truflation input (headline / core-goods / housing / gasoline) is replaced
by its month-to-date partial average through the as-of date. Early in month M only a few days are
observed (noisy → wide band); by month-end the partial average == the full-month average the engine
already used (→ converges to the locked print-day forecast). The BLS structural backbone (origin =
M-1) is fixed and only used once the M-1 print is actually released (leak guard).

Outputs: (1) the tightening curve — OOS error by days-to-print, proving the forecast improves;
(2) a replay of the last completed print showing day-by-day convergence; (3) the live trajectory.

Run::  uv run python -W ignore scripts/mtd_nowcast.py
"""
from __future__ import annotations
import os
import importlib.util
import duckdb, numpy as np, pandas as pd
from pathlib import Path

from thales.evaluation.calibrated_density import calibrated_samples, predictive_quantiles

ROOT = Path(__file__).resolve().parents[1]
ME = pd.offsets.MonthEnd(1)
TEST_START = pd.Timestamp("2021-06-30")
TODAY = pd.Timestamp(os.environ["THALES_AS_OF"]) if os.environ.get("THALES_AS_OF") else pd.Timestamp.today().normalize()
TF_CSV = os.environ.get("THALES_TF_CSV", "")
GAS_SID = "transport_gasoline_other_fuels_and_motor_oil"
OFFSETS = [30, 25, 20, 15, 10, 5, 2]      # days before print day (BLS CPI releases ~11th of M+1)
LABELS = ["headline_cpi_yoy_nsa", "core_cpi_yoy_nsa", "headline_cpi_mom_sa", "core_cpi_mom_sa"]


def _load_engine(con):
    spec = importlib.util.spec_from_file_location("eng", ROOT / "scripts/print_nowcast_engine.py")
    E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
    return E, E.PrintNowcaster(con)


def print_day(target_me: pd.Timestamp) -> pd.Timestamp:
    """Approx BLS CPI release for reference month M (month-end target_me): ~11th of M+1."""
    return target_me + pd.Timedelta(days=11)


class MTDNowcaster:
    """Print engine made a function of an as-of date via month-to-date Truflation inputs."""

    def __init__(self, con):
        self.E, self.eng = _load_engine(con)
        self.daily = self._load_daily(con)
        self.actual = {
            "headline_cpi_yoy_nsa": self.E.yoy_dated(self.eng.nsa["headline"]),
            "core_cpi_yoy_nsa": self.E.yoy_dated(self.eng.nsa["core"]),
            "headline_cpi_mom_sa": self.E._mom(self.eng.sa["headline"]),
            "core_cpi_mom_sa": self.E._mom(self.eng.sa["core"]),
        }

    def _load_daily(self, con) -> dict:
        df = pd.read_csv(TF_CSV, header=2, dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]; df = df.rename(columns={df.columns[0]: "date"})
        df = df[df["date"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)].copy(); df["date"] = pd.to_datetime(df["date"])

        def col(name):
            s = pd.to_numeric(df[name], errors="coerce"); s.index = df["date"]
            return s.dropna().groupby(level=0).last().sort_index()

        rows = con.execute("SELECT reference_date,value FROM vintage WHERE series_id=? AND source='truf_network_pit' ORDER BY 1",
                           [GAS_SID]).fetchall()
        gas = pd.Series([r[1] for r in rows], index=pd.to_datetime([r[0] for r in rows])).groupby(level=0).last().sort_index()
        return {"head": col("All Items Index"), "goods": col("Commodities less Food & energy Index"),
                "house": col("Housing services_index"), "gas": gas}

    @staticmethod
    def _asof_monthly(daily: pd.Series, as_of: pd.Timestamp) -> pd.Series:
        """Monthly mean using only daily obs <= as_of (the as-of month is a partial MTD mean)."""
        return daily[daily.index <= as_of].resample("ME").mean()

    def forecast(self, target: pd.Timestamp, as_of: pd.Timestamp) -> dict:
        """All four labels for `target` as they would read on `as_of`. Truflation truncated to as_of."""
        e = self.eng
        e.tf_head = self._asof_monthly(self.daily["head"], as_of)
        e.tf_goods = self._asof_monthly(self.daily["goods"], as_of)
        e.tf_house = self._asof_monthly(self.daily["house"], as_of)
        e.gas = self._asof_monthly(self.daily["gas"], as_of)
        return e.all_labels(target - ME)

    def m1_released(self, target: pd.Timestamp, as_of: pd.Timestamp) -> bool:
        """Leak guard: the M-1 BLS print (the structural backbone's origin) must be public by as_of."""
        return as_of >= print_day(target - ME)


def build_resid(mtd: MTDNowcaster):
    """Walk-forward residuals[label][offset] = list of (forecast - actual), leak-guarded."""
    resid = {L: {o: [] for o in OFFSETS} for L in LABELS}
    targets = [t for t in mtd.actual["headline_cpi_yoy_nsa"].index if t >= TEST_START]
    for t in targets:
        for o in OFFSETS:
            as_of = print_day(t) - pd.Timedelta(days=o)
            if not mtd.m1_released(t, as_of):
                continue
            try:
                fc = mtd.forecast(t, as_of)
            except Exception:
                continue
            for L in LABELS:
                a = mtd.actual[L].get(t, np.nan)
                if np.isfinite(fc.get(L, np.nan)) and np.isfinite(a):
                    resid[L][o].append((t, fc[L] - a))
    return resid, targets


def main():
    con = duckdb.connect(str(ROOT / "data/vintage_store/thales.duckdb"), read_only=True)
    mtd = MTDNowcaster(con)
    resid, targets = build_resid(mtd)

    # ---- (1) tightening curve: MAE by days-to-print ----
    print("=" * 84)
    print("MTD TIGHTENING CURVE — OOS MAE by days-to-print (does the forecast improve intra-month?)")
    print("=" * 84)
    print(f"  {'label':<22}" + "".join(f"{f'd-{o}':>9}" for o in OFFSETS))
    print("-" * 84)
    for L in LABELS:
        cells = []
        for o in OFFSETS:
            errs = [abs(e) for _, e in resid[L][o]]
            cells.append(f"{np.mean(errs):>9.3f}" if errs else f"{'-':>9}")
        print(f"  {L:<22}" + "".join(cells))
    print(f"  (n prints per offset ~{max(len(resid['headline_cpi_yoy_nsa'][o]) for o in OFFSETS)};"
          f" lower = sharper as print day approaches)")

    # ---- (2) replay the last completed print: day-by-day convergence ----
    last = targets[-1]
    print("\n" + "=" * 84)
    print(f"REPLAY — {last.strftime('%B %Y')} print: how the nowcast converged to the actual")
    print("=" * 84)
    print(f"  {'days-to-print':<14}" + "".join(f"{L.split('_cpi_')[0][:4]+'_'+L.split('_')[-2]:>12}" for L in LABELS))
    for o in OFFSETS:
        as_of = print_day(last) - pd.Timedelta(days=o)
        if not mtd.m1_released(last, as_of):
            continue
        fc = mtd.forecast(last, as_of)
        print(f"  d-{o:<12}" + "".join(f"{fc.get(L, np.nan):>12.2f}" for L in LABELS))
    print(f"  {'ACTUAL':<14}" + "".join(f"{mtd.actual[L].get(last, np.nan):>12.2f}" for L in LABELS))

    # ---- (3) live trajectory for the next print ----
    live_t = mtd.E.me(targets[-1] + ME)        # first print after last completed actual
    dtp = (print_day(live_t) - TODAY).days
    near = min(OFFSETS, key=lambda o: abs(o - dtp))
    print("\n" + "=" * 84)
    print(f"LIVE — next print {live_t.strftime('%B %Y')} (≈{dtp}d to print, as-of {TODAY.date()})")
    print("=" * 84)
    fc = mtd.forecast(live_t, TODAY)
    print(f"  {'label':<22}{'point':>8}{'80% band (offset pool)':>26}{'pool n':>8}")
    for L in LABELS:
        pool = np.array([e for tt, e in resid[L][near] if tt < live_t])   # leak-free
        p = fc.get(L, np.nan)
        if np.isfinite(p) and len(pool) >= 8:
            q = predictive_quantiles(calibrated_samples(p, pool, n_samples=4000, window=60, halflife=12), [10, 90])
            band = f"[{q[0]:.2f}, {q[1]:.2f}]"
        else:
            band = "(insufficient pool)"
        print(f"  {L:<22}{p:>8.2f}{band:>26}{len(pool):>8}")
    print(f"\n  Band uses the residual pool at the matching offset (d-{near}); it tightens automatically")
    print("  as we approach print day. Re-run daily — the point and band update with new Truflation.")


if __name__ == "__main__":
    main()
