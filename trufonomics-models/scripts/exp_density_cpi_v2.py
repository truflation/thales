"""CPI density v2 — does the calibrated (rolling/EWMA) scale fix the over-wide bands?

The v1 prototype (scripts/exp_density_cpi.py) bootstrapped the FULL expanding pool of prior
OOS residuals → CRPS beat persistence but cov80 ran hot (bands too wide, old-regime volatility
never leaves the pool). Here we compare four predictive distributions per label, walk-forward and
leak-free (residual pool only from origins < t), scored with the repo's proper-scoring block:

  * expanding  — v1 prototype: point + bootstrap(all prior residuals)
  * rolling36  — calibrated: recent 36-month window scale
  * ewma h12   — calibrated: EWMA scale (halflife 12m), 60-month window
  * persist    — persistence point + bootstrap(its own prior residuals)   [baseline]

Calibration target: cov80 ≈ 80%, cov95 ≈ 95%, high PIT-KS p. Sharpness (band width) lower is
better GIVEN calibration. Run::  uv run python -W ignore scripts/exp_density_cpi_v2.py
"""
from __future__ import annotations
import duckdb, importlib.util, numpy as np, pandas as pd
from pathlib import Path

from thales.evaluation.calibrated_density import calibrated_samples
from thales.evaluation.density import samples_from_residuals, score_density

ROOT = Path(__file__).resolve().parents[1]
TEST_START = pd.Timestamp("2021-06-30")
ME = pd.offsets.MonthEnd(1)
WARMUP = 18          # residuals before we start scoring
S = 2000             # samples per predictive distribution


def _engine(con):
    spec = importlib.util.spec_from_file_location("eng", ROOT / "scripts/print_nowcast_engine.py")
    E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
    return E, E.PrintNowcaster(con)


def main():
    con = duckdb.connect(str(ROOT / "data/vintage_store/thales.duckdb"), read_only=True)
    E, eng = _engine(con)
    yoy, mom, at = E.yoy_dated, E._mom, E._at

    actuals = {"headline_cpi_yoy_nsa": yoy(eng.nsa["headline"]), "core_cpi_yoy_nsa": yoy(eng.nsa["core"]),
               "headline_cpi_mom_sa": mom(eng.sa["headline"]), "core_cpi_mom_sa": mom(eng.sa["core"])}
    is_yoy = {"headline_cpi_yoy_nsa": True, "core_cpi_yoy_nsa": True,
              "headline_cpi_mom_sa": False, "core_cpi_mom_sa": False}

    def persist_point(label, org):
        a = actuals[label]
        if is_yoy[label]:
            return at(a, org)
        h = a[a.index <= org].dropna(); return h.iloc[-12:].mean() if len(h) >= 12 else np.nan

    months = [t for t in actuals["headline_cpi_yoy_nsa"].index if t >= TEST_START]
    methods = ["expanding", "rolling36", "ewma_h12", "persist"]
    print("=" * 96)
    print("CPI DENSITY v2 — calibrated scale vs expanding prototype (walk-forward, leak-free)")
    print("=" * 96)
    print(f"  {'label':<20}{'method':<11}{'CRPS':>7}{'cov80':>7}{'cov95':>8}{'PIT-KS p':>10}{'sharp80':>9}{'ptMAE':>8}{'n':>4}")
    print("-" * 96)

    for label in actuals:
        pts, pp, acts = [], [], []
        for t in months:
            org = t - ME; a = actuals[label].get(t, np.nan)
            try: p = eng.all_labels(org)[label]
            except Exception: p = np.nan
            q = persist_point(label, org)
            if np.all(np.isfinite([a, p, q])):
                pts.append(p); pp.append(q); acts.append(a)
        pts, pp, acts = map(np.array, (pts, pp, acts))
        res_e, res_p = acts - pts, acts - pp

        rows = {m: [] for m in methods}; yv = []
        for i in range(len(acts)):
            if i < WARMUP: continue
            pool_e, pool_p = res_e[:i], res_p[:i]
            rows["expanding"].append(samples_from_residuals(pts[i], pool_e, n_samples=S, seed=i))
            rows["rolling36"].append(calibrated_samples(pts[i], pool_e, n_samples=S, window=36, seed=i))
            rows["ewma_h12"].append(calibrated_samples(pts[i], pool_e, n_samples=S, window=60, halflife=12, seed=i))
            rows["persist"].append(samples_from_residuals(pp[i], pool_p, n_samples=S, seed=i))
            yv.append(acts[i])
        yv = np.array(yv); ptmae = np.mean(np.abs(pts[WARMUP:] - acts[WARMUP:]))

        for m in methods:
            blk = score_density(np.vstack(rows[m]), yv)
            pmae = ptmae if m != "persist" else np.mean(np.abs(pp[WARMUP:] - acts[WARMUP:]))
            print(f"  {label:<20}{m:<11}{blk.crps:>7.3f}{blk.cov80*100:>6.0f}%{blk.cov95*100:>7.0f}%"
                  f"{blk.pit_ks_pvalue:>10.3f}{blk.sharpness80:>9.3f}{pmae:>8.3f}{blk.n:>4}")
        print("-" * 96)

    print("\nRead: rolling36 / ewma_h12 should pull cov80 toward 80% and cov95 toward 95% (vs expanding")
    print("over-covering), at lower sharpness80 (tighter bands), without giving up CRPS. That is the")
    print("calibration fix — same point forecast, honestly-sized uncertainty.")


if __name__ == "__main__":
    main()
