"""Copula-based scenario console — operator-facing.

Replaces the BVAR-based scenario_console.py. Uses the empirically-
validated Copula+AR(1) model (ties naive_ar1 on CRPS, beats it on
coverage by 3-6pp; BVAR loses to both).

What it produces:

  * Multi-shock joint landed-cost distribution at horizon h, with
    proper bands (80% / 95%) derived from the copula's joint
    sampling — not a deterministic point + IRF as in the BVAR
    version.
  * Per-input scenario response (mean + bands).
  * Exposure decomposition: at horizon h, what fraction of
    landed-cost variance is attributable to each input?

Usage::

    uv run python truflation-operate/scenarios/copula_scenario_console.py \\
        --vertical auto \\
        --shock log_fx_eurusd:-0.05 \\
        --shock log_diesel:0.20 \\
        --horizon 12

Or with t-copula tails (for stress scenarios)::

    ... --family t
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "truflation-operate" / "scenarios"))

import copula_landed_cost as clc    # noqa: E402
import cost_baskets    # noqa: E402

OUT_DIR = ROOT / "truflation-operate" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _load_vertical(label: str):
    if label == "auto":
        return _load("auto_v",
                          ROOT / "truflation-operate" / "verticals" / "import_export_auto.py")
    if label == "textile":
        return _load("text_v",
                          ROOT / "truflation-operate" / "verticals" / "import_export_textile.py")
    raise ValueError(f"unknown vertical {label!r}")


def _exposure_weights(vertical: str) -> dict[str, float]:
    """Single source of truth — exposure weights from cost_baskets."""
    return cost_baskets.get_exposure_weights(vertical)


def _parse_shock(s: str) -> tuple[str, float]:
    if ":" not in s:
        raise ValueError(f"shock must be 'var:size_log', got {s!r}")
    v, x = s.split(":", 1)
    return v.strip(), float(x)


def scenario_report(vertical: str, shocks: list[tuple[str, float]],
                          h: int = 12, n_samples: int = 1000,
                          family: str = "gaussian",
                          persistent: bool = True,
                          mode: str = "conditional",
                          seed: int = 0) -> dict:
    """Produce a copula-based scenario report.

    The shocks are imposed on the **innovation paths** for each shocked
    variable (i.e., we condition the sampled innovation for that
    variable at h=1 to be the requested magnitude). The copula then
    correctly propagates joint movement to the other variables — every
    sample is a coherent joint outcome.

    Returns dict with:
      * per_variable: dict[var] -> {mean, q10, q50, q90} trajectories
      * landed_cost: same structure for the operator basket aggregate
      * exposure_decomposition_at_h: per-input contribution to landed-cost variance
    """
    mod = _load_vertical(vertical)
    panel = mod.load_panel()
    var_cols = list(panel.columns)
    Y = panel.values

    fit = clc.fit_copula_ar1(Y, family=family)
    rng = np.random.default_rng(seed)
    # Baseline: unconditioned joint copula draws
    samples_baseline = clc.sample_copula_paths(fit, Y, h, n_samples, rng)

    # Shocked: conditional copula propagation
    # The proper operator scenario primitive — non-shocked variables
    # are sampled from the conditional joint given the shocked levels,
    # preserving the empirical cross-input dependence rather than
    # ignoring it as in the innovation-override approach.
    conditioning = {v: x for v, x in shocks if v in var_cols}
    rng2 = np.random.default_rng(seed + 1)
    samples_shocked = clc.conditional_copula_sample(
        fit, Y, conditioning=conditioning, var_names=var_cols,
        h=h, persistent=persistent, n_samples=n_samples, rng=rng2)

    weights_map = _exposure_weights(vertical)
    weights = np.array([weights_map.get(c, 0.0) for c in var_cols])
    landed_baseline = samples_baseline @ weights
    landed_shocked = samples_shocked @ weights

    def _stats(samples_landed):
        return {
            "mean": [float(samples_landed[:, t].mean()) for t in range(h)],
            "q10":  [float(np.quantile(samples_landed[:, t], 0.10)) for t in range(h)],
            "q50":  [float(np.quantile(samples_landed[:, t], 0.50)) for t in range(h)],
            "q90":  [float(np.quantile(samples_landed[:, t], 0.90)) for t in range(h)],
            "q025": [float(np.quantile(samples_landed[:, t], 0.025)) for t in range(h)],
            "q975": [float(np.quantile(samples_landed[:, t], 0.975)) for t in range(h)],
        }

    per_var_baseline = {c: _stats(samples_baseline[:, :, i])
                              for i, c in enumerate(var_cols)}
    per_var_shocked = {c: _stats(samples_shocked[:, :, i])
                             for i, c in enumerate(var_cols)}

    # Exposure decomposition at horizon h: variance share per input
    # Var(landed) at h-1 ≈ Var(sum_i w_i * sample[:, h-1, i])
    # Per-input contribution ≈ w_i * Cov(sample[:, h-1, i], landed)
    landed_at_h = landed_baseline[:, h - 1]
    exposure: dict[str, float] = {}
    var_landed = float(np.var(landed_at_h))
    for i, c in enumerate(var_cols):
        cov_iL = float(np.cov(samples_baseline[:, h - 1, i], landed_at_h)[0, 1])
        contribution = weights[i] * cov_iL
        exposure[c] = contribution
    total = sum(exposure.values())
    if total > 0:
        exposure = {k: round(v / total * 100, 2) for k, v in exposure.items()}

    return {
        "vertical":   vertical,
        "shocks":     shocks,
        "horizon":    int(h),
        "family":     family,
        "t_df":       fit.t_df,
        "n_samples":  n_samples,
        "var_cols":   var_cols,
        "weights":    {c: float(w) for c, w in zip(var_cols, weights)},
        "per_variable_baseline":  per_var_baseline,
        "per_variable_shocked":   per_var_shocked,
        "landed_cost_baseline":   _stats(landed_baseline),
        "landed_cost_shocked":    _stats(landed_shocked),
        "exposure_decomposition_pct_at_h": exposure,
        "as_of_date": str(date.today()),
    }


def _print(r: dict) -> None:
    print("=" * 78)
    print(f"COPULA SCENARIO — {r['vertical']} vertical")
    print("=" * 78)
    print(f"As of: {r['as_of_date']}   Horizon: h={r['horizon']}   "
            f"Family: {r['family']}{f' (df={r["t_df"]:.0f})' if r['t_df'] else ''}   "
            f"Samples: {r['n_samples']}")
    print("\nShocks (first-step log-deviations on the named variable):")
    for v, x in r["shocks"]:
        print(f"  {v:<22s}  Δlog={x:+.3f}  ({(np.exp(x)-1)*100:+.2f}% level)")
    print("\nLanded-cost distribution under shocks (% deviation from baseline):")
    s = r["landed_cost_shocked"]
    months = [0, 1, 3, 6, r["horizon"] - 1] if r["horizon"] >= 7 else list(range(r["horizon"]))
    months = sorted(set(m for m in months if 0 <= m < r["horizon"]))
    print(f"  {'h+':>3s}  {'mean':>9s}  {'80% band':>20s}  {'95% band':>20s}")
    for t in months:
        mean = (np.exp(s['mean'][t]) - 1) * 100
        lo80 = (np.exp(s['q10'][t]) - 1) * 100
        hi80 = (np.exp(s['q90'][t]) - 1) * 100
        lo95 = (np.exp(s['q025'][t]) - 1) * 100
        hi95 = (np.exp(s['q975'][t]) - 1) * 100
        print(f"  {t+1:>3d}  {mean:>+7.2f}%  [{lo80:>+5.2f}%, {hi80:>+5.2f}%]  "
                f"[{lo95:>+5.2f}%, {hi95:>+5.2f}%]")

    print("\nExposure decomposition at h (% of landed-cost variance):")
    for v, pct in sorted(r["exposure_decomposition_pct_at_h"].items(),
                                key=lambda kv: -abs(kv[1])):
        bar = "█" * max(1, int(abs(pct) / 2))
        print(f"  {v:<22s}  {pct:>+7.2f}%  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertical", choices=["auto", "textile"], required=True)
    parser.add_argument("--shock", action="append", default=[])
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--family", choices=["gaussian", "t"], default="gaussian")
    parser.add_argument("--persistent", action="store_true", default=True,
                        help="Hold shocked vars at level for all h steps "
                             "(default; right for tariff/contract changes)")
    parser.add_argument("--decaying", dest="persistent", action="store_false",
                        help="One-step innovation that decays via AR(1) "
                             "(right for one-time spikes)")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    shocks = [_parse_shock(s) for s in args.shock]
    if not shocks:
        shocks = ([("log_fx_eurusd", -0.05), ("log_diesel", 0.10)]
                       if args.vertical == "auto"
                       else [("log_fx_cnyusd", 0.05), ("log_freight", 0.20)])

    rep = scenario_report(args.vertical, shocks, h=args.horizon,
                                  n_samples=args.n_samples, family=args.family,
                                  persistent=args.persistent)
    _print(rep)
    out = (Path(args.save) if args.save else
              OUT_DIR / f"copula_scenario_{args.vertical}_{date.today()}.json")
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(f"\nSaved JSON: {out}")


if __name__ == "__main__":
    main()
