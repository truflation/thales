"""PPI Final Demand nowcast engine — complete: 4 labels, FD-ID indirect aggregation + energy bridge.

Labels (BLS contract, same as CPI): MoM -> SA, YoY -> NSA.
  ppi_fd_mom_sa     PPIFIS   headline FD, 1-mo % SA   -> indirect aggregation + energy/gas bridge
  ppi_fd_yoy_nsa    PPIFID   headline FD, 12-mo % NSA
  ppi_core_mom_sa   PPIFES   FD less food&energy, SA  -> persistence (no energy lever in core)
  ppi_core_yoy_nsa  PPICOR   FD less food&energy, NSA

YoY-NSA is NOT derived naively from SA MoM. We use the identity (the calendar-month seasonal
CANCELS in the 12-month difference):
    YoY_nsa(t) = YoY_nsa(t-1) + SA_MoM(t) - SA_MoM(t-12)
anchored on the actual NSA YoY history + actual SA MoM a year ago, with only SA_MoM(t) forecast.
This respects the different bases (MoM=SA, YoY=NSA) exactly.

FD-ID weights: Goods 0.290 (+ construction folded into services) / Services 0.710. Energy ~5% of
FD but a large share of its variance -> the gas bridge is the one transferable lever (-19% headline).
Trade services (~20% of FD) is a MARGIN/spread (can move opposite inputs) -> left to persistence,
the honest baseline; do NOT bridge it with input prices.

Run::  uv run python -W ignore scripts/ppi_nowcast_engine.py
"""
from __future__ import annotations
import duckdb, importlib.util, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_START = pd.Timestamp("2021-06-30")
W_GOODS, W_SERVICES, WIN = 0.290, 0.710, 48


def me(d): return pd.Timestamp(d) + pd.offsets.MonthEnd(0)
def _mom(x): return (x / x.shift(1) - 1) * 100
def _yoy(x): return (x / x.shift(12) - 1) * 100


class PPINowcaster:
    def __init__(self, con):
        spec = importlib.util.spec_from_file_location("eng", ROOT / "scripts/print_nowcast_engine.py")
        E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
        cpi = E.PrintNowcaster(con)
        self.gas = _mom(cpi.gas)
        df = pd.read_csv(ROOT / "data/ppi_series.csv", index_col=0, parse_dates=True)
        df.index = df.index + pd.offsets.MonthEnd(0)
        self.sa_fd, self.nsa_fd = df["headline_sa"], df["headline_nsa"]
        self.sa_core, self.nsa_core = df["core_fe_sa"], df["core_fe_nsa"]
        self.goods, self.svcs = _mom(df["goods_sa"]), _mom(df["services_sa"])
        self.sa_fd_mom, self.sa_core_mom = _mom(self.sa_fd), _mom(self.sa_core)
        self.nsa_fd_yoy, self.nsa_core_yoy = _yoy(self.nsa_fd), _yoy(self.nsa_core)

    def _persist(self, s, origin):
        h = s[s.index <= origin].dropna(); return float(h.iloc[-12:].mean()) if len(h) >= 12 else np.nan

    def _goods_bridge(self, origin):
        t = origin + pd.offsets.MonthEnd(1)
        d = pd.concat([self.goods.rename("y"), self.gas.rename("g")], axis=1).dropna(); tr = d[d.index < t].iloc[-WIN:]
        if len(tr) < 24 or t not in self.gas.index or not np.isfinite(self.gas[t]): return self._persist(self.goods, origin)
        b = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), tr["g"].values]), tr["y"].values, rcond=None)[0]
        return float(b[0] + b[1] * self.gas[t])

    # ---- the four labels ----
    def fd_mom_sa(self, origin):
        return W_GOODS * self._goods_bridge(origin) + W_SERVICES * self._persist(self.svcs, origin)

    def core_mom_sa(self, origin):
        return self._persist(self.sa_core_mom, origin)

    def _yoy_from_mom(self, yoy_act, sa_mom, mom_fc, origin):
        t = origin + pd.offsets.MonthEnd(1); t12 = me(t - pd.DateOffset(months=12))
        base, prev12 = yoy_act.get(origin, np.nan), sa_mom.get(t12, np.nan)
        if not all(np.isfinite(v) for v in (base, prev12, mom_fc)): return np.nan
        return base + mom_fc - prev12  # seasonal cancels in the 12-mo difference

    def fd_yoy_nsa(self, origin):
        return self._yoy_from_mom(self.nsa_fd_yoy, self.sa_fd_mom, self.fd_mom_sa(origin), origin)

    def core_yoy_nsa(self, origin):
        return self._yoy_from_mom(self.nsa_core_yoy, self.sa_core_mom, self.core_mom_sa(origin), origin)


def main():
    con = duckdb.connect(str(ROOT / "data/vintage_store/thales.duckdb"), read_only=True)
    eng = PPINowcaster(con)
    ACT = {"ppi_fd_mom_sa": (eng.sa_fd_mom, eng.fd_mom_sa, "persist_mom", eng.sa_fd_mom),
           "ppi_core_mom_sa": (eng.sa_core_mom, eng.core_mom_sa, "persist_mom", eng.sa_core_mom),
           "ppi_fd_yoy_nsa": (eng.nsa_fd_yoy, eng.fd_yoy_nsa, "persist_yoy", None),
           "ppi_core_yoy_nsa": (eng.nsa_core_yoy, eng.core_yoy_nsa, "persist_yoy", None)}
    print("=" * 76)
    print("PPI NOWCAST ENGINE — walk-forward MAE 2021-06+ (vs persistence)")
    print("=" * 76)
    for lab, (act, fc, ptype, smom) in ACT.items():
        e, ep = [], []
        for t in [t for t in act.index if t >= TEST_START]:
            org = t - pd.offsets.MonthEnd(1); a = act.get(t, np.nan); p = fc(org)
            if ptype == "persist_mom":
                h = smom[smom.index <= org].dropna(); pers = h.iloc[-12:].mean() if len(h) >= 12 else np.nan
            else:
                pers = act.get(org, np.nan)  # YoY persistence = carry last YoY
            if np.isfinite(a) and np.isfinite(p): e.append(abs(p - a))
            if np.isfinite(a) and np.isfinite(pers): ep.append(abs(pers - a))
        print(f"  {lab:<18} model MAE {np.mean(e):.3f}   persistence {np.mean(ep):.3f}   "
              f"({(np.mean(e)/np.mean(ep)-1)*100:+.0f}%)   n={len(e)}")
    print("\nLast 3 prints — model vs official (rounded):")
    last3 = [t for t in eng.sa_fd_mom.index if t >= TEST_START][-3:]
    for t in last3:
        org = t - pd.offsets.MonthEnd(1)
        fm, fy = eng.fd_mom_sa(org), eng.fd_yoy_nsa(org)
        am, ay = eng.sa_fd_mom.get(t, np.nan), eng.nsa_fd_yoy.get(t, np.nan)
        print(f"  {t.strftime('%b %Y')}  FD MoM model={fm:5.2f} official={am:5.2f} ({round(fm,1)}v{round(am,1)})"
              f"   FD YoY model={fy:4.1f} official={ay:4.1f} ({round(fy,1)}v{round(ay,1)})")
    print("\nConclusion: headline MoM gets the energy/gas edge (-19%); cores + YoY ride persistence")
    print("(no transferable Truflation lever — PPI edge is energy-only, commodity-driven).")


if __name__ == "__main__":
    main()
