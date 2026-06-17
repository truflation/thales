"""Calibrated density bands for PCE + PPI — same layer validated on CPI, generalized.

The density primitives (thales.evaluation.calibrated_density + .density.score_density) are
engine-agnostic: each label only needs three callables — point(origin), actual(target),
persist(origin). We drive the PCE and PPI engines through that interface, then for every label
report the walk-forward calibration block (EWMA-h12 calibrated density vs a persistence-density
baseline) and the live next-print bands.

Run::  uv run python -W ignore scripts/density_pce_ppi.py
"""
from __future__ import annotations
import importlib.util
import duckdb, numpy as np, pandas as pd
from pathlib import Path

from thales.evaluation.calibrated_density import calibrated_samples, predictive_quantiles
from thales.evaluation.density import samples_from_residuals, score_density

ROOT = Path(__file__).resolve().parents[1]
TEST_START = pd.Timestamp("2021-06-30")
ME = pd.offsets.MonthEnd(1)
WARMUP, S, NLIVE = 18, 2000, 4000
WINDOW, HALFLIFE = 60, 12


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def me(d): return pd.Timestamp(d) + pd.offsets.MonthEnd(0)
def _trail12(s, o):
    h = s[s.index <= o].dropna(); return float(h.iloc[-12:].mean()) if len(h) >= 12 else np.nan


def run_label(spec) -> dict:
    """spec: name, idx (target index), point(o), actual(t), persist(o). Returns calibration + live band."""
    pts, pp, acts, ts = [], [], [], []
    for t in [t for t in spec["idx"] if t >= TEST_START]:
        o = t - ME
        try: p = spec["point"](o)
        except Exception: p = np.nan
        try: q = spec["persist"](o)
        except Exception: q = np.nan
        a = spec["actual"](t)
        if np.all(np.isfinite([a, p, q])):
            pts.append(p); pp.append(q); acts.append(a); ts.append(t)
    pts, pp, acts = map(np.array, (pts, pp, acts))
    if len(acts) <= WARMUP + 3:
        return {"name": spec["name"], "ok": False}
    res_e, res_p = acts - pts, acts - pp

    rows_e, rows_p, yv = [], [], []
    for i in range(WARMUP, len(acts)):
        rows_e.append(calibrated_samples(pts[i], res_e[:i], n_samples=S, window=WINDOW, halflife=HALFLIFE, seed=i))
        rows_p.append(samples_from_residuals(pp[i], res_p[:i], n_samples=S, seed=i))
        yv.append(acts[i])
    yv = np.array(yv)
    be, bp = score_density(np.vstack(rows_e), yv), score_density(np.vstack(rows_p), yv)
    ptmae = float(np.mean(np.abs(pts[WARMUP:] - acts[WARMUP:])))

    # live next print: target after last completed
    live_t = me(ts[-1] + ME); live_o = live_t - ME
    try: lp = spec["point"](live_o)
    except Exception: lp = np.nan
    q = predictive_quantiles(calibrated_samples(lp, res_e, n_samples=NLIVE, window=WINDOW, halflife=HALFLIFE),
                             [10, 25, 50, 75, 90]) if np.isfinite(lp) else np.full(5, np.nan)
    return {"name": spec["name"], "ok": True, "be": be, "bp": bp, "ptmae": ptmae,
            "live_t": live_t, "lp": lp, "q": q, "n": be.n}


def report(title, specs):
    res = [run_label(s) for s in specs]
    print("=" * 94)
    print(f"{title} — calibrated density (EWMA h{HALFLIFE}) vs persistence-density  [walk-forward]")
    print("=" * 94)
    print(f"  {'label':<20}{'CRPS':>7}{'CRPSpers':>10}{'Δ':>6}{'cov80':>7}{'cov95':>7}{'PITp':>7}{'sharp80':>9}{'ptMAE':>8}{'n':>4}")
    print("-" * 94)
    for r in res:
        if not r["ok"]:
            print(f"  {r['name']:<20}  (insufficient history)"); continue
        be, bp = r["be"], r["bp"]
        d = (be.crps / bp.crps - 1) * 100 if bp.crps else np.nan
        print(f"  {r['name']:<20}{be.crps:>7.3f}{bp.crps:>10.3f}{d:>5.0f}%{be.cov80*100:>6.0f}%"
              f"{be.cov95*100:>6.0f}%{be.pit_ks_pvalue:>7.2f}{be.sharpness80:>9.3f}{r['ptmae']:>8.3f}{r['n']:>4}")
    print("-" * 94)
    nm = next((r for r in res if r["ok"]), None)
    if nm:
        print(f"  LIVE next print — {nm['live_t'].strftime('%B %Y')}:")
        for r in res:
            if not r["ok"] or not np.isfinite(r["lp"]):
                print(f"    {r['name']:<20}  n/a"); continue
            q = r["q"]
            print(f"    {r['name']:<20} point {r['lp']:6.2f}   50% [{q[1]:.2f}, {q[3]:.2f}]   80% [{q[0]:.2f}, {q[4]:.2f}]")
    print()


def main():
    con = duckdb.connect(str(ROOT / "data/vintage_store/thales.duckdb"), read_only=True)

    # ---- PCE ----
    P = _imp("pce_eng", "scripts/pce_nowcast_engine.py")
    E, cpi = P.load_cpi_engine(con); ph = P.load_ppi_health()
    pce = P.PCENowcaster(con, cpi, ppi_health=ph)
    bu = ph is not None
    def pce_mom(lab): return lambda o: pce.mom(lab, o, "actual_cpi", bottomup=(bu and lab == "core"))
    specs_pce = [
        {"name": "Headline MoM", "idx": pce.pce_mom["headline"].index, "point": pce_mom("headline"),
         "actual": lambda t: pce.actual_mom("headline", t), "persist": lambda o: pce.persist_mom("headline", o)},
        {"name": "Core MoM", "idx": pce.pce_mom["core"].index, "point": pce_mom("core"),
         "actual": lambda t: pce.actual_mom("core", t), "persist": lambda o: pce.persist_mom("core", o)},
        {"name": "Headline YoY", "idx": pce.pce_mom["headline"].index,
         "point": lambda o: pce.yoy_from_mom("headline", pce_mom("headline")(o), o),
         "actual": lambda t: pce.actual_yoy("headline", t),
         "persist": lambda o: pce.yoy_from_mom("headline", pce.persist_mom("headline", o), o)},
        {"name": "Core YoY", "idx": pce.pce_mom["core"].index,
         "point": lambda o: pce.yoy_from_mom("core", pce_mom("core")(o), o),
         "actual": lambda t: pce.actual_yoy("core", t),
         "persist": lambda o: pce.yoy_from_mom("core", pce.persist_mom("core", o), o)},
    ]
    report("PCE NOWCAST", specs_pce)

    # ---- PPI ----
    Q = _imp("ppi_eng", "scripts/ppi_nowcast_engine.py")
    ppi = Q.PPINowcaster(con)
    specs_ppi = [
        {"name": "FD MoM", "idx": ppi.sa_fd_mom.index, "point": ppi.fd_mom_sa,
         "actual": lambda t: ppi.sa_fd_mom.get(t, np.nan), "persist": lambda o: _trail12(ppi.sa_fd_mom, o)},
        {"name": "Core MoM", "idx": ppi.sa_core_mom.index, "point": ppi.core_mom_sa,
         "actual": lambda t: ppi.sa_core_mom.get(t, np.nan), "persist": lambda o: _trail12(ppi.sa_core_mom, o)},
        {"name": "FD YoY", "idx": ppi.nsa_fd_yoy.index, "point": ppi.fd_yoy_nsa,
         "actual": lambda t: ppi.nsa_fd_yoy.get(t, np.nan), "persist": lambda o: ppi.nsa_fd_yoy.get(o, np.nan)},
        {"name": "Core YoY", "idx": ppi.nsa_core_yoy.index, "point": ppi.core_yoy_nsa,
         "actual": lambda t: ppi.nsa_core_yoy.get(t, np.nan), "persist": lambda o: ppi.nsa_core_yoy.get(o, np.nan)},
    ]
    report("PPI NOWCAST", specs_ppi)
    print("Read: Δ = density-CRPS vs persistence (negative = better). cov80→80 / cov95→95 = calibrated.")


if __name__ == "__main__":
    main()
