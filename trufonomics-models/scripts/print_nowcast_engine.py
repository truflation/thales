"""Thales print nowcast engine — BLS structure + Cleveland energy + Truflation.

Unifies everything validated in the experiment series into one model that
nowcasts the four public BLS CPI prints, by indirect aggregation:

  headline = core(ex food&energy) + food + energy        (3 buckets)
  core     = core goods + core services                  (2 buckets)

Per-bucket forecasters (each validated walk-forward earlier):
  * sticky buckets (core, food)  -> 12-month moving-average persistence
       (Cleveland's choice; Truflation adds no per-component signal here)
  * energy bucket                -> gas bridge: regress energy MoM on
       Truflation's gasoline-stream MoM (corr ~0.99, real-time; -81% vs AR)
  * recombine                    -> rolling 24-month regression of the
       aggregate MoM on the bucket MoMs (Cleveland's learned-weight combine)
  * acceleration                 -> blend the headline with the Truflation
       headline bridge (the signal-mining winner) to remove the rising-regime
       persistence low-bias

Bases (BLS contract): YoY -> NSA (CUUR), MoM -> SA (CUSR). Gap-robust
date-based YoY so the Oct-2025 blackout cannot corrupt the target.

Run::  uv run python scripts/print_nowcast_engine.py
"""
from __future__ import annotations
import os
import duckdb, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/vintage_store/thales.duckdb"
TEST_START = pd.Timestamp("2021-06-30")
ACCEL_W = 0.50            # Truflation acceleration weight (headline)
ACCEL_CAP = 0.50          # cap on the bridge's upside pull; downside capped at 0.3x
                          # (upside-asymmetric: trust Truflation's lead on the way up,
                          #  discount its spurious component-divergence drops)
CORE_ACCEL_W = 0.30       # core acceleration weight — uses Truflation core GOODS only
                          # (core services is shelter-dominated and diverges; goods leads)
W_SHELTER_IN_CORE = 0.45  # shelter RI share of core (CUUR0000SAH1 / core); distributed-lag upgrade
W_USEDCARS_IN_CORE = 0.025  # used-cars RI share of core; Manheim-driven upgrade

# BLS published relative importances (~Dec-2023: food+energy+core=100; cg+cs=core).
# Structural recombine weights — exactly how BLS aggregates (convex, sum-to-1, no intercept),
# more stable than a learned rolling regression. Accuracy-neutral vs the regression in test.
RI_HEADLINE = {"core": 0.797, "food": 0.134, "energy": 0.069}
RI_CORE = {"cg": 0.2321, "cs": 0.7679}
_RI_BY_KEYS = {frozenset(RI_HEADLINE): RI_HEADLINE, frozenset(RI_CORE): RI_CORE}


def me(d): return pd.Timestamp(d) + pd.offsets.MonthEnd(0)


def yoy_dated(x: pd.Series) -> pd.Series:
    out = {}
    for t in x.index:
        ya = me(pd.Timestamp(t.year - 1, t.month, 1))
        if ya in x.index: out[t] = (x[t] / x[ya] - 1) * 100
    return pd.Series(out).sort_index()


def _series(con, sid, source):
    rows = con.execute("SELECT reference_date,value FROM vintage WHERE series_id=? AND source=? AND (series_id,reference_date,as_of_date) IN (SELECT series_id,reference_date,MAX(as_of_date) FROM vintage WHERE series_id=? AND source=? GROUP BY 1,2) ORDER BY 1", [sid, source, sid, source]).fetchall()
    s = pd.Series([r[1] for r in rows], index=pd.to_datetime([r[0] for r in rows]))
    if source == "truf_network_pit":
        return s.groupby(s.index).last().resample("ME").mean()
    return pd.Series(s.values, index=pd.to_datetime([r[0] for r in rows]) + pd.offsets.MonthEnd(0)).sort_index()


def _mom(x): return (x / x.shift(1) - 1) * 100


def _at(s: pd.Series, d):
    """Value at date d, or the latest available at/before d (blackout-robust)."""
    sub = s.loc[:d]
    return float(sub.iloc[-1]) if len(sub) else np.nan


class PrintNowcaster:
    def __init__(self, con):
        self.con = con
        b = lambda sid: _series(con, sid, "bls_direct")
        # NSA component levels (for YoY)
        self.nsa = {"core": b("CUUR0000SA0L1E"), "food": b("CUUR0000SAF1"),
                    "energy": b("CUUR0000SA0E"), "headline": b("CUUR0000SA0"),
                    "cg": b("CUUR0000SACL1E"), "cs": b("CUUR0000SASLE")}
        # SA component levels (for MoM)
        self.sa = {"core": b("CUSR0000SA0L1E"), "food": b("CUSR0000SAF1"),
                   "energy": b("CUSR0000SA0E"), "headline": b("CUSR0000SA0"),
                   "cg": b("CUSR0000SACL1E"), "cs": b("CUSR0000SASLE")}
        self.gas = _series(con, "transport_gasoline_other_fuels_and_motor_oil", "truf_network_pit")
        # Truflation headline (BLS-definition export) for the acceleration bridge
        F = os.environ.get("THALES_TF_CSV", "")
        df = pd.read_csv(F, header=2, dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]; df = df.rename(columns={df.columns[0]: "date"})
        df = df[df["date"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)].copy(); df["date"] = pd.to_datetime(df["date"])
        ai = pd.to_numeric(df["All Items Index"], errors="coerce"); ai.index = df["date"]
        self.tf_head = ai.dropna().groupby(level=0).last().sort_index().resample("ME").mean()
        cg = pd.to_numeric(df["Commodities less Food & energy Index"], errors="coerce"); cg.index = df["date"]
        self.tf_goods = cg.dropna().groupby(level=0).last().sort_index().resample("ME").mean()
        # granular-core drivers: Truflation housing (shelter dist-lag) + Manheim (used cars)
        hs = pd.to_numeric(df["Housing services_index"], errors="coerce"); hs.index = df["date"]
        self.tf_house = hs.dropna().groupby(level=0).last().sort_index().resample("ME").mean()
        self.shelter = b("CUUR0000SAH1"); self.usedcars = b("CUUR0000SETA02")
        try:
            _m = pd.read_csv(ROOT / "data" / "manheim_monthly.csv", index_col=0, parse_dates=True)
            self.manheim = _m.iloc[:, 0]
        except Exception:
            self.manheim = pd.Series(dtype=float)

    # ---- bucket forecasters (MoM of next month after origin) ----
    def _persist(self, lvl, origin):
        m = _mom(lvl); h = m[m.index <= origin].dropna()
        return h.iloc[-12:].mean() if len(h) >= 12 else (h.iloc[-1] if len(h) else 0.0)

    def _energy_bridge(self, lvl, origin):
        em = _mom(lvl); gm = _mom(self.gas)
        nxt = origin + pd.offsets.MonthEnd(1)
        d = pd.concat([em.rename("y"), gm.rename("g")], axis=1, sort=True).dropna()
        tr = d[d.index <= origin].iloc[-60:]
        if len(tr) < 24 or nxt not in gm.index:
            return self._persist(lvl, origin)
        b = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), tr["g"].values]), tr["y"].values, rcond=None)[0]
        return float(b[0] + b[1] * gm[nxt])

    def _recombine(self, agg_lvl, bucket_moms, bucket_lvls, origin):
        """Combine bucket MoM forecasts into the aggregate. Uses BLS published
        relative-importance weights (convex, sum-to-1, no intercept) when the bucket set is
        known — exactly how BLS aggregates; falls back to a rolling 24-mo regression otherwise."""
        ri = _RI_BY_KEYS.get(frozenset(bucket_moms))
        if ri is not None:
            return float(sum(ri[k] * bucket_moms[k] for k in bucket_moms))
        am = _mom(agg_lvl)
        cols = {k: _mom(v) for k, v in bucket_lvls.items()}
        d = pd.concat([am.rename("y")] + [c.rename(k) for k, c in cols.items()], axis=1, sort=True).dropna()
        tr = d[d.index <= origin].iloc[-24:]
        if len(tr) < 18:
            return float(sum(bucket_moms.values()) / len(bucket_moms))
        X = np.column_stack([np.ones(len(tr))] + [tr[k].values for k in cols])
        b = np.linalg.lstsq(X, tr["y"].values, rcond=None)[0]
        return float(b[0] + sum(b[i + 1] * bucket_moms[k] for i, k in enumerate(cols)))

    def _bridge(self, origin):
        t = origin + pd.offsets.MonthEnd(1); H = self.nsa["headline"]
        t12 = me(t - pd.DateOffset(months=12))
        H_org, H_12 = _at(H, origin), _at(H, t12)
        tf_t = self.tf_head.get(t); tf_org = _at(self.tf_head, origin)
        if np.isfinite(H_org) and np.isfinite(H_12) and tf_t is not None and np.isfinite(tf_org):
            return (H_org * (tf_t / tf_org) / H_12 - 1) * 100
        return np.nan

    # ---- the four labels ----
    def headline_yoy_nsa(self, origin, accel=ACCEL_W):
        n = self.nsa; t = origin + pd.offsets.MonthEnd(1); t12 = me(t - pd.DateOffset(months=12))
        moms = {"core": self._persist(n["core"], origin), "food": self._persist(n["food"], origin),
                "energy": self._energy_bridge(n["energy"], origin)}
        hmom = self._recombine(n["headline"], moms, {"core": n["core"], "food": n["food"], "energy": n["energy"]}, origin)
        struct = (_at(n["headline"], origin) * (1 + hmom / 100) / _at(n["headline"], t12) - 1) * 100
        br = self._bridge(origin)
        if np.isfinite(br):
            dev = np.clip(br - struct, -ACCEL_CAP * 0.3, ACCEL_CAP)   # upside-asymmetric
            return struct + accel * dev
        return struct

    def _core_bridge(self, origin):
        """Core acceleration signal: Truflation core-GOODS bridge (the only core
        decomposition with alpha; core services is shelter-divergent)."""
        t = origin + pd.offsets.MonthEnd(1); C = self.nsa["core"]; t12 = me(t - pd.DateOffset(months=12))
        C_org, C_12 = _at(C, origin), _at(C, t12)
        g_t = self.tf_goods.get(t); g_org = _at(self.tf_goods, origin)
        if np.isfinite(C_org) and np.isfinite(C_12) and g_t is not None and np.isfinite(g_org):
            return (C_org * (g_t / g_org) / C_12 - 1) * 100
        return np.nan

    def _distlag(self, target_lvl, driver_lvl, lags, origin, window=60, min_train=30):
        """Distributed-lag YoY forecast of target on its own AR + the driver at
        several lags (lease-rollover for shelter; auction-lead for used cars).
        Returns (forecast, persistence_baseline) — both as next-month YoY."""
        ty = yoy_dated(target_lvl); dy = yoy_dated(driver_lvl)
        nxt = origin + pd.offsets.MonthEnd(1)
        d = pd.DataFrame({"y": ty, "ar": ty.shift(1)})
        for k in lags: d[f"l{k}"] = dy.shift(k)
        d = d.dropna(); tr = d[d.index <= origin].iloc[-window:]
        persist = _at(ty, origin)
        if len(tr) < min_train: return np.nan, persist
        xr = [1.0, persist] + [_at(dy, me(nxt - pd.DateOffset(months=k))) for k in lags]
        if any(not np.isfinite(v) for v in xr): return np.nan, persist
        X = np.column_stack([np.ones(len(tr)), tr["ar"].values] + [tr[f"l{k}"].values for k in lags])
        b = np.linalg.lstsq(X, tr["y"].values, rcond=None)[0]
        return float(np.array(xr) @ b), persist

    def core_yoy_nsa(self, origin, accel=CORE_ACCEL_W):
        n = self.nsa; t = origin + pd.offsets.MonthEnd(1); t12 = me(t - pd.DateOffset(months=12))
        moms = {"cg": self._persist(n["cg"], origin), "cs": self._persist(n["cs"], origin)}
        cmom = self._recombine(n["core"], moms, {"cg": n["cg"], "cs": n["cs"]}, origin)
        struct = (_at(n["core"], origin) * (1 + cmom / 100) / _at(n["core"], t12) - 1) * 100
        br = self._core_bridge(origin)
        if np.isfinite(br):
            struct = struct + accel * np.clip(br - struct, -ACCEL_CAP * 0.3, ACCEL_CAP)
        # granular-core upgrades: shelter (Truflation housing dist-lag) + used cars (Manheim)
        sh_f, sh_p = self._distlag(self.shelter, self.tf_house, [6, 9, 12, 15], origin)
        if np.isfinite(sh_f) and np.isfinite(sh_p):
            struct += W_SHELTER_IN_CORE * np.clip(sh_f - sh_p, -0.4, 0.4)
        if len(self.manheim):
            uc_f, uc_p = self._distlag(self.usedcars, self.manheim, [1, 2], origin, window=48)
            if np.isfinite(uc_f) and np.isfinite(uc_p):
                struct += W_USEDCARS_IN_CORE * np.clip(uc_f - uc_p, -3.0, 3.0)
        return struct

    def headline_mom_sa(self, origin):
        sa = self.sa
        moms = {"core": self._persist(sa["core"], origin), "food": self._persist(sa["food"], origin),
                "energy": self._energy_bridge(sa["energy"], origin)}
        return self._recombine(sa["headline"], moms, {"core": sa["core"], "food": sa["food"], "energy": sa["energy"]}, origin)

    def core_mom_sa(self, origin):
        sa = self.sa
        moms = {"cg": self._persist(sa["cg"], origin), "cs": self._persist(sa["cs"], origin)}
        return self._recombine(sa["core"], moms, {"cg": sa["cg"], "cs": sa["cs"]}, origin)

    def all_labels(self, origin):
        return {"headline_cpi_yoy_nsa": self.headline_yoy_nsa(origin),
                "core_cpi_yoy_nsa": self.core_yoy_nsa(origin),
                "headline_cpi_mom_sa": self.headline_mom_sa(origin),
                "core_cpi_mom_sa": self.core_mom_sa(origin)}


def main():
    con = duckdb.connect(str(DB), read_only=True)
    eng = PrintNowcaster(con)
    actuals = {
        "headline_cpi_yoy_nsa": yoy_dated(eng.nsa["headline"]),
        "core_cpi_yoy_nsa": yoy_dated(eng.nsa["core"]),
        "headline_cpi_mom_sa": _mom(eng.sa["headline"]),
        "core_cpi_mom_sa": _mom(eng.sa["core"]),
    }
    labels = list(actuals)
    print("=" * 72)
    print("THALES PRINT NOWCAST ENGINE — forecast vs official  (origin = month-1)")
    print("=" * 72)
    for tgt in [me("2026-03-31"), me("2026-04-30"), me("2026-05-31")]:
        org = tgt - pd.offsets.MonthEnd(1)
        fc = eng.all_labels(org)
        print(f"\n{tgt.strftime('%B %Y')}:")
        for L in labels:
            a = actuals[L].get(tgt, np.nan)
            print(f"  {L:<22} model={fc[L]:6.3f}  official={a:6.3f}  miss={fc[L]-a:+.3f}  "
                  f"(rounded {round(fc[L],1)} vs {round(a,1)})")
    # walk-forward MAE per label
    print("\n" + "=" * 72)
    print("Walk-forward MAE 2021-06+ (pp):")
    months = [t for t in actuals["headline_cpi_yoy_nsa"].index if t >= TEST_START]
    for L in labels:
        es = []
        for t in months:
            org = t - pd.offsets.MonthEnd(1)
            try: p = eng.all_labels(org)[L]
            except Exception: p = np.nan
            a = actuals[L].get(t, np.nan)
            if np.isfinite(p) and np.isfinite(a): es.append(abs(p - a))
        print(f"  {L:<22} MAE={np.mean(es):.3f}  (n={len(es)})")


if __name__ == "__main__":
    main()
