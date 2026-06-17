"""Thales print nowcast — DENSITY engine. Point + calibrated bands for the next BLS CPI print.

Wraps the point engine (scripts/print_nowcast_engine.py) and adds the calibrated density layer
validated in scripts/exp_density_cpi_v2.py: an EWMA-scaled standardized-residual bootstrap
(halflife 12m) that turns each point forecast into a proper predictive distribution with
honestly-sized, regime-aware uncertainty bands.

Leak-free: the residual pool for a target uses only origins whose actual is already known
(targets strictly earlier than the one being forecast). For the live next print that is the full
completed history; EWMA weights the recent regime most.

Run::  uv run python scripts/print_density_engine.py            # live next-print bands
"""
from __future__ import annotations
import importlib.util
import duckdb, numpy as np, pandas as pd
from pathlib import Path

from thales.evaluation.calibrated_density import calibrated_samples, predictive_quantiles, MIN_RESID

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/vintage_store/thales.duckdb"
TEST_START = pd.Timestamp("2021-06-30")
ME = pd.offsets.MonthEnd(1)

WINDOW, HALFLIFE, N_SAMPLES = 60, 12, 4000     # locked from the v2 calibration eval
LABELS = ["headline_cpi_yoy_nsa", "core_cpi_yoy_nsa", "headline_cpi_mom_sa", "core_cpi_mom_sa"]
_PRETTY = {"headline_cpi_yoy_nsa": "Headline YoY (NSA)", "core_cpi_yoy_nsa": "Core YoY (NSA)",
           "headline_cpi_mom_sa": "Headline MoM (SA)", "core_cpi_mom_sa": "Core MoM (SA)"}


def _load_engine(con):
    spec = importlib.util.spec_from_file_location("eng", ROOT / "scripts/print_nowcast_engine.py")
    E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
    return E, E.PrintNowcaster(con)


class DensityPrintNowcaster:
    """Point engine + calibrated predictive bands for the four BLS CPI labels."""

    def __init__(self, con, window: int = WINDOW, halflife: float = HALFLIFE,
                 n_samples: int = N_SAMPLES):
        self.E, self.eng = _load_engine(con)
        self.window, self.halflife, self.n_samples = window, halflife, n_samples
        yoy, mom = self.E.yoy_dated, self.E._mom
        self._actual = {
            "headline_cpi_yoy_nsa": yoy(self.eng.nsa["headline"]), "core_cpi_yoy_nsa": yoy(self.eng.nsa["core"]),
            "headline_cpi_mom_sa": mom(self.eng.sa["headline"]), "core_cpi_mom_sa": mom(self.eng.sa["core"]),
        }
        self._resid: dict[str, pd.Series] = {}

    def _point(self, label: str, origin: pd.Timestamp) -> float:
        try:
            return float(self.eng.all_labels(origin)[label])
        except Exception:
            return float("nan")

    def residual_history(self, label: str) -> pd.Series:
        """Signed OOS errors (actual - forecast) indexed by target month, completed targets only."""
        if label not in self._resid:
            a = self._actual[label]
            out = {}
            for t in [t for t in a.index if t >= TEST_START and np.isfinite(a.get(t, np.nan))]:
                p = self._point(label, t - ME)
                if np.isfinite(p):
                    out[t] = a[t] - p
            self._resid[label] = pd.Series(out).sort_index()
        return self._resid[label]

    def band(self, label: str, target: pd.Timestamp, levels=(10, 25, 50, 75, 90)) -> dict:
        """Point + predictive quantiles for `target` (origin = target - 1 month)."""
        target = self.E.me(target)
        point = self._point(label, target - ME)
        hist = self.residual_history(label)
        pool = hist[hist.index < target].values            # leak-free
        samples = calibrated_samples(point, pool, n_samples=self.n_samples,
                                     window=self.window, halflife=self.halflife)
        q = predictive_quantiles(samples, list(levels))
        return {"point": point, "levels": list(levels), "quantiles": q,
                "n_resid": int(np.sum(np.isfinite(pool)))}

    def next_target(self) -> pd.Timestamp:
        """First print after the latest completed headline-YoY actual."""
        a = self._actual["headline_cpi_yoy_nsa"].dropna()
        return self.E.me(a.index[-1] + ME)

    def all_bands(self, target: pd.Timestamp) -> dict:
        return {L: self.band(L, target) for L in LABELS}


def main():
    con = duckdb.connect(str(DB), read_only=True)
    eng = DensityPrintNowcaster(con)
    target = eng.next_target()
    print("=" * 78)
    print(f"THALES CPI DENSITY ENGINE — next print: {target.strftime('%B %Y')}  "
          f"(EWMA h{HALFLIFE}, {WINDOW}m window)")
    print("=" * 78)
    print(f"  {'label':<22}{'point':>8}{'50% band':>18}{'80% band':>18}{'resid':>7}")
    print("-" * 78)
    for L in LABELS:
        b = eng.band(L, target)
        if not np.isfinite(b["point"]) or np.any(np.isnan(b["quantiles"])):
            print(f"  {_PRETTY[L]:<22}{'n/a':>8}   (insufficient residuals: n={b['n_resid']})")
            continue
        q = dict(zip(b["levels"], b["quantiles"]))
        b50 = f"[{q[25]:.2f}, {q[75]:.2f}]"; b80 = f"[{q[10]:.2f}, {q[90]:.2f}]"
        print(f"  {_PRETTY[L]:<22}{b['point']:>8.2f}{b50:>18}{b80:>18}{b['n_resid']:>7}")
    print("-" * 78)
    print("  point = committed central forecast; bands = calibrated predictive interval")
    print("  (50% ~ q25-q75, 80% ~ q10-q90). Bands track the recent volatility regime (EWMA).")
    print("  Validated walk-forward: CRPS ≪ persistence at ~half the band width "
          "(scripts/exp_density_cpi_v2.py).")


if __name__ == "__main__":
    main()
