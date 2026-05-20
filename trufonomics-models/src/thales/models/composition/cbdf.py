"""CBDF (Component-Based Dynamic Factor) composition — Phase 2.1b.

Extends ``WeightedComposer`` to capture **cross-component residual
correlation**. Plain weighted composition assumes independent
component forecast errors. Reality: a fuel-price shock hits Utilities
AND Food-at-home AND Transportation simultaneously, so component
errors are positively correlated; treating them as independent makes
the composed headline band too tight (under-coverage) or, conversely,
treating them as fully correlated makes it too wide.

CBDF (O'Keeffe & Petrova 2025, NY Fed SR 1152) handles this via a
shared latent factor:

    component_r_resid_t  =  β_r · F_t  +  λ_r,t  +  ε_r,t

with F_t common across components and (β_r, λ_r) recovered jointly
with the headline accounting identity. The full O'Keeffe-Petrova
estimation is a multi-step EM procedure on the residual panel; this
module ships the simpler **multivariate-Gaussian residual** version,
which:

  1. Estimates the empirical residual covariance Σ_resid from a
     historical residual panel (component_r forecast errors over
     past origins)
  2. Composes future forecasts by drawing JOINT samples from
     N(point_per_component, Σ_resid) and weighted-summing each draw

This captures cross-component dependence without explicitly modeling F_t.
The full DFM (with explicit F_t) is straightforward additive surgery —
the same hierarchical-housing JAX code from Phase 1.5 generalizes —
but for the MVP the multivariate-Gaussian residual is enough to
correct band calibration without 50× more code.

Usage:

    composer = CBDFComposer(weights={...})
    composer.fit_residual_covariance(historical_residual_panel)
    out = composer.compose(per_component_forecasts, origin, target)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from thales.evaluation.harness import Forecast
from thales.models.composition.weighted import WeightedComposer


@dataclass
class CBDFComposer(WeightedComposer):
    """Composer that draws joint Monte Carlo samples from a fitted
    multivariate-Gaussian residual covariance.

    Falls back to the parent ``WeightedComposer`` (independent
    components) if ``fit_residual_covariance`` hasn't been called.
    """
    _residual_cov: np.ndarray | None = field(default=None, init=False, repr=False)
    _residual_components: list[str] | None = field(default=None, init=False, repr=False)

    def fit_residual_covariance(self, residual_panel: pd.DataFrame) -> None:
        """Estimate cross-component residual covariance from historical errors.

        ``residual_panel`` is a DataFrame indexed by origin date with
        one column per component_id, values being component forecast
        errors (component_actual − component_forecast) over a training
        window. Component IDs must match those in ``self.weights``.

        The estimator is **shrunk toward diagonal** — a small ridge on
        the off-diagonal — to keep the covariance positive-definite
        when the residual panel is short relative to the number of
        components (n_origins < 2× n_components).
        """
        active = sorted(k for k, w in self.weights.items() if w > 0)
        missing = set(active) - set(residual_panel.columns)
        if missing:
            raise ValueError(
                f"residual_panel missing columns for active components: "
                f"{sorted(missing)}")
        panel = residual_panel[active].dropna()
        if len(panel) < 12:
            raise ValueError(
                f"need ≥12 origins of residuals; got {len(panel)}")

        # Empirical covariance with light shrinkage for PSD safety
        emp = np.cov(panel.values, rowvar=False, ddof=1)
        diag = np.diag(np.diag(emp))
        n_origins = len(panel)
        n_comp = len(active)
        # Ledoit-Wolf-ish shrinkage intensity: more shrinkage when n_origins is small
        shrinkage = max(0.05, min(0.5, n_comp / (n_comp + n_origins)))
        cov_shrunk = (1 - shrinkage) * emp + shrinkage * diag

        # Final PSD safeguard: clip negative eigenvalues
        eigvals, eigvecs = np.linalg.eigh(cov_shrunk)
        eigvals = np.clip(eigvals, 1e-9, None)
        cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        self._residual_cov = cov_psd
        self._residual_components = active

    def compose(self,
                  component_forecasts: dict[str, Forecast],
                  origin: pd.Timestamp,
                  target: pd.Timestamp,
                  ) -> Forecast:
        """Compose with joint MC sampling if covariance fitted, else fall
        back to independent draws (parent behaviour)."""
        if self._residual_cov is None:
            return super().compose(component_forecasts, origin, target)

        active = self._residual_components
        missing = set(active) - set(component_forecasts)
        if missing:
            raise ValueError(
                f"composition needs forecasts for all fitted components, "
                f"missing: {sorted(missing)}")

        # Joint sampling: N(point_vector, residual_cov)
        point_vec = np.array([component_forecasts[k].point for k in active])
        rng = np.random.default_rng(self.seed)
        joint_draws = rng.multivariate_normal(
            mean=point_vec,
            cov=self._residual_cov,
            size=self.n_mc_samples,
        )
        # Weighted sum per draw
        w_vec = np.array([self.weights[k] for k in active])
        samples_acc = joint_draws @ w_vec
        point = float((point_vec * w_vec).sum())

        lo80 = float(np.percentile(samples_acc, 10))
        hi80 = float(np.percentile(samples_acc, 90))
        lo95 = float(np.percentile(samples_acc, 2.5))
        hi95 = float(np.percentile(samples_acc, 97.5))

        contributions = {
            k: float(self.weights[k] * component_forecasts[k].point)
            for k in active
        }

        return Forecast(
            origin=origin, target=target, point=point,
            lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
            samples=samples_acc,
            metadata={
                "composer": "cbdf_multivariate_gaussian",
                "n_components": len(active),
                "n_mc": self.n_mc_samples,
                "contributions": contributions,
                "weight_sum": float(w_vec.sum()),
                "residual_cov_norm_frobenius":
                    float(np.linalg.norm(self._residual_cov, ord="fro")),
            },
        )
