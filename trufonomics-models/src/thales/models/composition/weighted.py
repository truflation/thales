"""Weighted composition layer — accounting-identity-respecting headline forecast.

The Phase 2.1 MVP composition. Given forecasts from each top-level
component archetype (12 categories from Truflation taxonomy) and the
published BLS/Truflation weights ``w = (w_1, …, w_R)`` summing to 1.0,
combine them into a headline YoY forecast that respects the accounting
identity:

    headline_yoy[T+h]  =  Σ_r w_r · component_r_yoy[T+h]

This is the *additive* composition. It's the simplest version of CBDF
(O'Keeffe & Petrova 2025) — the full version layers a shared latent
factor across components to handle cross-component dependence in
forecast errors. That's Phase 2.1b; this module is the 2.1a foundation
the factor layer builds on.

Density:
  * Point: closed-form weighted sum.
  * Bands: Monte Carlo. Sample from each component's predictive
    distribution (assumed Gaussian from point + symmetric band, OR
    from explicit samples if provided), compute the weighted sum
    sample-by-sample, take quantiles. Captures the cross-component
    correlation structure if samples were drawn jointly upstream;
    assumes independence otherwise.

Attribution:
  * For each component, the contribution to a *change* in headline is
    ``w_r · Δcomponent_r``. The composer exposes this as a separate
    method so it doesn't pollute the basic compose() API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from thales.evaluation.harness import Forecast


@dataclass
class WeightedComposer:
    """Compose per-component forecasts into a headline forecast.

    ``weights`` is a dict ``{component_id: weight}`` whose values must sum
    to ~1 within the tolerance check. Components with zero weight are
    permitted (they're just dropped from the sum).
    """
    weights: dict[str, float]
    weight_sum_tol: float = 1e-3
    n_mc_samples: int = 2000
    seed: int = 0

    def __post_init__(self) -> None:
        s = sum(self.weights.values())
        if abs(s - 1.0) > self.weight_sum_tol:
            raise ValueError(
                f"weights sum to {s:.6f}, must be within "
                f"{self.weight_sum_tol} of 1.0")

    def compose(self,
                  component_forecasts: dict[str, Forecast],
                  origin: pd.Timestamp,
                  target: pd.Timestamp,
                  ) -> Forecast:
        """Compose component forecasts into a headline Forecast.

        ``component_forecasts`` maps component_id → Forecast. Every
        ``component_id`` in ``self.weights`` with non-zero weight must be
        present.

        Returns a Forecast with the headline point + 80%/95% bands +
        metadata recording the per-component contributions.
        """
        # Validate coverage
        active = {k: w for k, w in self.weights.items() if w > 0}
        missing = set(active) - set(component_forecasts)
        if missing:
            raise ValueError(
                f"composition needs forecasts for these components, "
                f"missing: {sorted(missing)}")

        # Point forecast — closed form
        point = sum(active[k] * component_forecasts[k].point for k in active)

        # Density via Monte Carlo
        rng = np.random.default_rng(self.seed)
        samples_acc = np.zeros(self.n_mc_samples)
        per_component_samples = {}
        for k, w in active.items():
            fc = component_forecasts[k]
            comp_samples = self._sample_component(fc, rng)
            per_component_samples[k] = comp_samples
            samples_acc += w * comp_samples

        lo80, hi80 = (float(np.percentile(samples_acc, 10)),
                       float(np.percentile(samples_acc, 90)))
        lo95, hi95 = (float(np.percentile(samples_acc, 2.5)),
                       float(np.percentile(samples_acc, 97.5)))

        # Per-component contribution to point
        contributions = {
            k: float(active[k] * component_forecasts[k].point)
            for k in active
        }

        return Forecast(
            origin=origin,
            target=target,
            point=float(point),
            lo80=lo80, hi80=hi80,
            lo95=lo95, hi95=hi95,
            samples=samples_acc,
            metadata={
                "composer": "weighted",
                "n_components": len(active),
                "n_mc": self.n_mc_samples,
                "contributions": contributions,
                "weight_sum": float(sum(active.values())),
            },
        )

    def attribution(self,
                      component_forecasts: dict[str, Forecast],
                      component_today: dict[str, float],
                      ) -> pd.DataFrame:
        """Per-component contribution to the *change* in headline.

        Returns a DataFrame with columns
            (component_id, weight, today, forecast, delta, contribution_pp)

        ``contribution_pp = w_r · (component_forecast_r − component_today_r)``
        is each component's contribution to the headline forecast
        movement. Sorted by absolute contribution descending.
        """
        active = {k: w for k, w in self.weights.items() if w > 0}
        rows = []
        for k, w in active.items():
            today = float(component_today.get(k, np.nan))
            fc = component_forecasts[k].point
            delta = fc - today if not np.isnan(today) else np.nan
            rows.append({
                "component_id": k,
                "weight": w,
                "today": today,
                "forecast": fc,
                "delta": delta,
                "contribution_pp": w * delta if not np.isnan(delta) else np.nan,
            })
        df = pd.DataFrame(rows)
        df = df.sort_values("contribution_pp",
                              key=lambda s: s.abs(),
                              ascending=False).reset_index(drop=True)
        return df

    # ─── Internals ────────────────────────────────────────────────────────

    def _sample_component(self, fc: Forecast,
                            rng: np.random.Generator) -> np.ndarray:
        """Draw ``n_mc_samples`` from a component's predictive distribution.

        Priority: explicit ``fc.samples`` ► Gaussian from band ► point only.
        """
        if fc.samples is not None and len(fc.samples) > 0:
            # Resample with replacement to get our target count
            return rng.choice(fc.samples, size=self.n_mc_samples,
                                 replace=True)
        if fc.has_bands:
            # Recover Gaussian SD from 80% band: hi80 − lo80 = 2 × 1.2816 σ
            sigma = max((fc.hi80 - fc.lo80) / (2.0 * 1.2816), 1e-9)
            return rng.normal(fc.point, sigma, size=self.n_mc_samples)
        # Point-only forecast: degenerate distribution
        return np.full(self.n_mc_samples, fc.point)
