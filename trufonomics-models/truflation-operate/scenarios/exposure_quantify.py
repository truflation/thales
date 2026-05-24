"""Shared scenario engine for Truflation Operate verticals.

Thin reusable wrapper over ``shock_scenario`` and
``conditional_forecast`` in
``src/thales/models/archetypes/bvar_minnesota.py``. The wrapper
standardises the inputs/outputs a vertical script needs so each
vertical is just data prep + a couple of calls into this module.

Two operator-facing modes:

  1. **Shock scenario.** "If diesel jumps +X%, what happens to my
     landed cost over the next h months?" Calls ``shock_scenario`` on
     the fitted BVAR, scales the user's shock through the Cholesky
     IRF, returns the trajectory for every variable plus the
     weighted landed-cost impact under the operator's cost-share
     vector.

  2. **Conditional path forecast.** "Given the path I expect for
     diesel and freight over the next 12 months, what's the joint
     distribution of FX, vehicle cost, and landed cost?" Calls
     ``conditional_forecast`` on the BVAR with the forced paths and
     returns sample-path quantiles per variable.

Both modes are deterministic given the inputs. No transaction risk,
no execution — pure exposure quantification. The vertical script
calls these, persists the resulting CSV/JSON, and is done.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thales.models.archetypes.bvar_minnesota import (    # noqa: E402
    BVARFit,
    conditional_forecast,
    shock_scenario,
)


@dataclass(frozen=True)
class CostShare:
    """One component's share of the operator's landed cost.

    `var_name` must match the BVAR column label. `share` is the
    fraction of the operator's landed cost driven by that input.
    Shares need not sum to 1.0 — the un-modelled remainder is treated
    as fixed and reported as a residual.
    """
    var_name: str
    share: float


@dataclass(frozen=True)
class ShockSpec:
    """One named shock scenario. `var_name` is the BVAR column. Shock
    is expressed in **log-space** so +0.10 ~= +10% level move for
    multiplicative log-level variables; for log-MoM frames, +0.01
    ~= +100 bp MoM."""
    name: str
    var_name: str
    size_log: float


def shock_to_landed_cost(fit: BVARFit,
                                var_cols: list[str],
                                shock: ShockSpec,
                                cost_shares: list[CostShare],
                                h: int = 12,
                                baseline_log: np.ndarray | None = None,
                                ) -> pd.DataFrame:
    """One Cholesky shock; trajectory of variable responses + landed-cost
    weighted aggregate.

    Returns a DataFrame indexed by horizon ``0..h`` with columns
    ``resp_<var>`` per BVAR variable plus a ``landed_cost_log_dev``
    column = sum over cost shares of share × resp_<var>.

    All responses are deviations from baseline, in log-space.
    """
    if shock.var_name not in var_cols:
        raise ValueError(f"shock var '{shock.var_name}' not in {var_cols}")
    shock_idx = var_cols.index(shock.var_name)

    if baseline_log is None:
        baseline_log = np.zeros(len(var_cols))

    traj = shock_scenario(
        fit=fit,
        baseline=baseline_log,
        shock_var_idx=shock_idx,
        shock_size=shock.size_log,
        h=h,
    )
    # traj shape: (h+1, k) — rows = horizons, cols = variables
    df = pd.DataFrame(traj, columns=[f"resp_{c}" for c in var_cols])
    df.index.name = "h"

    landed = np.zeros(traj.shape[0])
    for cs in cost_shares:
        if cs.var_name not in var_cols:
            continue
        j = var_cols.index(cs.var_name)
        landed += cs.share * traj[:, j]
    df["landed_cost_log_dev"] = landed
    df["landed_cost_pct_dev"] = (np.exp(landed) - 1.0) * 100
    return df


def conditional_landed_cost(fit: BVARFit,
                                    var_cols: list[str],
                                    history_log: np.ndarray,
                                    forced_paths: dict[str, np.ndarray],
                                    cost_shares: list[CostShare],
                                    h: int = 12,
                                    n_samples: int = 1000,
                                    ) -> dict:
    """Conditional forecast given a forced path for one or more
    variables. Returns deterministic mean trajectory + MC quantiles
    for every variable AND for the weighted landed-cost aggregate.
    """
    forced_idx = {var_cols.index(v): path for v, path in forced_paths.items()
                       if v in var_cols}
    res = conditional_forecast(
        fit=fit,
        history=history_log,
        forced_paths=forced_idx,
        h=h,
        n_samples=n_samples,
    )

    # res keys: mean, q05, q25, q50, q75, q95, samples (shape (h, k, S))
    # Build a per-variable dict for readability
    per_var: dict[str, pd.DataFrame] = {}
    for j, c in enumerate(var_cols):
        per_var[c] = pd.DataFrame({
            "mean":  res["mean"][:, j],
            "q05":   res["q05"][:, j],
            "q25":   res["q25"][:, j],
            "q50":   res["q50"][:, j],
            "q75":   res["q75"][:, j],
            "q95":   res["q95"][:, j],
        })

    # Landed-cost aggregate: weighted sum of variable means
    # and a sample-path-driven landed-cost distribution
    samples = res["samples"]    # (h, k, S)
    cs_lookup = {cs.var_name: cs.share for cs in cost_shares}
    # log-deviation landed cost per sample path
    weights = np.array([cs_lookup.get(c, 0.0) for c in var_cols])    # (k,)
    landed_samples = np.einsum("hks,k->hs", samples, weights)         # (h, S)

    landed_df = pd.DataFrame({
        "mean":  landed_samples.mean(axis=1),
        "q05":   np.quantile(landed_samples, 0.05, axis=1),
        "q25":   np.quantile(landed_samples, 0.25, axis=1),
        "q50":   np.quantile(landed_samples, 0.50, axis=1),
        "q75":   np.quantile(landed_samples, 0.75, axis=1),
        "q95":   np.quantile(landed_samples, 0.95, axis=1),
    })
    landed_df["mean_pct"] = (np.exp(landed_df["mean"]) - 1.0) * 100
    landed_df["q05_pct"] = (np.exp(landed_df["q05"]) - 1.0) * 100
    landed_df["q95_pct"] = (np.exp(landed_df["q95"]) - 1.0) * 100

    return {
        "per_variable":   per_var,
        "landed_cost":    landed_df,
        "n_samples":      n_samples,
    }


def fevd_to_exposure(fevd_h: np.ndarray,
                          var_cols: list[str],
                          target_var: str) -> pd.Series:
    """Express FEVD at one horizon for one target variable as an
    exposure breakdown: "what fraction of this variable's
    forecast-error variance at h periods ahead is attributable to
    each input shock?"

    fevd_h shape: (h+1, k, k). FEVD[h_idx][target, shock] is the
    contribution of shock to target's variance at horizon h_idx.
    """
    target_idx = var_cols.index(target_var)
    last = fevd_h[-1]    # take final horizon
    contributions = last[target_idx]    # (k,) shares summing to 1.0
    return pd.Series(contributions * 100, index=var_cols,
                          name=f"{target_var}_fevd_pct")
