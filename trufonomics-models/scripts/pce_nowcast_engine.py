"""PCE print nowcast engine — CPI->PCE bridge + optional bottom-up healthcare augmentation.

CONSISTENCY INVARIANT (enforced, tested in tests/test_pce_yoy_mom_consistency.py):
  MoM is the single forecast PRIMITIVE. YoY is ALWAYS derived by chaining the MoM nowcast
  onto the realized SA index level — never modelled independently. PCE is SA-only, so both
  MoM and YoY come from the SAME index (PCEPI / PCEPILFE). This makes the CPI-committee bug
  (MoM and YoY on mismatched bases) structurally impossible:
      YoY(t) = PCEPI(t-1) * (1 + MoM_nowcast(t)/100) / PCEPI(t-12) - 1
  and for the realized series  PCEPI(t)/PCEPI(t-12) ≡ Π_{k=0..11} PCEPI(t-k)/PCEPI(t-k-1).

PCE is ~70% the same CPI source data re-weighted (Fisher). Cleveland's method: rolling-24 OLS
of PCE-MoM on CPI-MoM (beta1 absorbs the wedge). CPI prints ~16 days BEFORE PCE, so the actual
CPI is available at nowcast time. Core PCE's residual vs core CPI is mostly HEALTHCARE (~22% of
PCE, PPI-sourced not CPI) — the bottom-up adds a PPI-healthcare regressor to capture it.

Run::  uv run python -W ignore scripts/pce_nowcast_engine.py
"""
from __future__ import annotations
import duckdb, importlib.util, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_START = pd.Timestamp("2021-06-30")
WIN = 24


def me(d): return pd.Timestamp(d) + pd.offsets.MonthEnd(0)
def _mom(x): return (x / x.shift(1) - 1) * 100


def _latest(con, sid, source):
    rows = con.execute(
        "SELECT reference_date,value FROM vintage WHERE series_id=? AND source=? "
        "AND (series_id,reference_date,as_of_date) IN (SELECT series_id,reference_date,MAX(as_of_date) "
        "FROM vintage WHERE series_id=? AND source=? GROUP BY 1,2) ORDER BY 1", [sid, source, sid, source]).fetchall()
    return pd.Series([r[1] for r in rows],
                     index=pd.to_datetime([r[0] for r in rows]) + pd.offsets.MonthEnd(0)).sort_index()


def load_cpi_engine(con):
    spec = importlib.util.spec_from_file_location("eng", ROOT / "scripts/print_nowcast_engine.py")
    E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
    return E, E.PrintNowcaster(con)


def load_ppi_health():
    """PPI healthcare MoM (proxy for PCE's PPI-sourced healthcare deflators). CSV first-cut;
    proper vintage ingest is task #156. Returns MoM series or None if not yet fetched."""
    p = ROOT / "data/ppi_healthcare.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    s = pd.Series(df["value"].values, index=pd.to_datetime(df.index) + pd.offsets.MonthEnd(0))
    return _mom(s)


class PCENowcaster:
    LABELS = ("headline", "core")
    SERIES = {"headline": "PCEPI", "core": "PCEPILFE"}

    def __init__(self, con, cpi_engine, ppi_health=None):
        self.lvl = {lab: _latest(con, sid, "fred_alfred_target") for lab, sid in self.SERIES.items()}
        self.pce_mom = {lab: _mom(self.lvl[lab]) for lab in self.LABELS}
        self.cpi = cpi_engine
        self.cpi_mom = {"headline": _mom(cpi_engine.sa["headline"]), "core": _mom(cpi_engine.sa["core"])}
        self.ppi_health = ppi_health

    # ---- the MoM forecast primitive ----
    def _bridge(self, label, origin, feats, feats_at_t):
        """rolling-24 OLS of PCE MoM on the feature MoMs; predict with feats_at_t for month t."""
        t = origin + pd.offsets.MonthEnd(1)
        y = self.pce_mom[label]
        d = pd.concat([y.rename("y")] + [s.rename(f"x{i}") for i, s in enumerate(feats)], axis=1).dropna()
        tr = d[d.index < t].iloc[-WIN:]
        if len(tr) < 12 or any(not np.isfinite(v) for v in feats_at_t):
            return np.nan
        X = np.column_stack([np.ones(len(tr))] + [tr[f"x{i}"].values for i in range(len(feats))])
        b = np.linalg.lstsq(X, tr["y"].values, rcond=None)[0]
        return float(b[0] + sum(b[i + 1] * feats_at_t[i] for i in range(len(feats))))

    def mom(self, label, origin, source="actual_cpi", bottomup=False):
        """MoM nowcast (the primitive). source: 'actual_cpi' (post-CPI-release) | 'our_cpi' (early).
        bottomup=True augments CORE with the PPI-healthcare regressor."""
        t = origin + pd.offsets.MonthEnd(1)
        if source == "actual_cpi":
            cpi_t = self.cpi_mom[label].get(t, np.nan)
        else:
            cpi_t = self.cpi.headline_mom_sa(origin) if label == "headline" else self.cpi.core_mom_sa(origin)
        feats, at_t = [self.cpi_mom[label]], [cpi_t]
        if bottomup and label == "core" and self.ppi_health is not None:
            feats.append(self.ppi_health); at_t.append(self.ppi_health.get(t, np.nan))
        return self._bridge(label, origin, feats, at_t)

    def persist_mom(self, label, origin):
        h = self.pce_mom[label][self.pce_mom[label].index <= origin].dropna()
        return float(h.iloc[-12:].mean()) if len(h) >= 12 else np.nan

    # ---- YoY is ALWAYS derived from a MoM nowcast (the invariant) ----
    def yoy_from_mom(self, label, mom_nowcast, origin):
        t = origin + pd.offsets.MonthEnd(1); t12 = me(t - pd.DateOffset(months=12))
        lvl = self.lvl[label]
        if origin not in lvl.index or t12 not in lvl.index or not np.isfinite(mom_nowcast):
            return np.nan
        return (lvl[origin] * (1 + mom_nowcast / 100) / lvl[t12] - 1) * 100

    def actual_mom(self, label, t): return self.pce_mom[label].get(t, np.nan)

    def actual_yoy(self, label, t):
        t12 = me(t - pd.DateOffset(months=12)); lvl = self.lvl[label]
        if t not in lvl.index or t12 not in lvl.index: return np.nan
        return (lvl[t] / lvl[t12] - 1) * 100


def main():
    con = duckdb.connect(str(ROOT / "data/vintage_store/thales.duckdb"), read_only=True)
    E, cpi = load_cpi_engine(con)
    ph = load_ppi_health()
    eng = PCENowcaster(con, cpi, ppi_health=ph)
    print("=" * 80)
    print(f"PCE NOWCAST ENGINE — walk-forward MAE 2021-06+  (PPI healthcare: {'loaded' if ph is not None else 'NOT yet fetched'})")
    print("=" * 80)

    def walk(label, forecaster):
        em, ey = [], []
        for t in [t for t in eng.pce_mom[label].index if t >= TEST_START]:
            org = t - pd.offsets.MonthEnd(1)
            am, ay = eng.actual_mom(label, t), eng.actual_yoy(label, t)
            p = forecaster(label, org)
            if np.isfinite(p) and np.isfinite(am):
                em.append(abs(p - am))
                py = eng.yoy_from_mom(label, p, org)
                if np.isfinite(py) and np.isfinite(ay): ey.append(abs(py - ay))
        return (np.mean(em) if em else np.nan, np.mean(ey) if ey else np.nan, len(em))

    fcs = [
        ("persist", lambda l, o: eng.persist_mom(l, o)),
        ("bridge_actual", lambda l, o: eng.mom(l, o, "actual_cpi")),
        ("bridge_ours", lambda l, o: eng.mom(l, o, "our_cpi")),
        ("bottomup_actual", lambda l, o: eng.mom(l, o, "actual_cpi", bottomup=True)),
    ]
    for label in eng.LABELS:
        print(f"\n{label}:")
        print(f"  {'forecaster':<18}{'MoM MAE':>9}{'YoY MAE':>9}{'n':>5}")
        for name, fc in fcs:
            if name == "bottomup_actual" and (label != "core" or ph is None):
                continue
            m, y, n = walk(label, fc)
            tag = "  <-baseline" if name == "persist" else ("  <-+healthcare" if name.startswith("bottomup") else "")
            print(f"  {name:<18}{m:>9.3f}{y:>9.3f}{n:>5}{tag}")


if __name__ == "__main__":
    main()
