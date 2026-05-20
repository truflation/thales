"""Truflation → BLS / PCE bridge — converts a Truflation-targeted
forecaster's output to an official-target prediction.

Path A v1 demonstrated that a simple linear bridge

    BLS_CPI_yoy[T+1]  ≈  α  +  β · truflation_yoy[T+1]  +  ε

closes the methodology gap between Truflation's daily-aggregated CPI
and BLS's monthly retail-price CPI. The two series are highly
correlated but not identical (Truflation runs ~13% below BLS in
Clothing, similar gaps in other categories — methodology differences,
not signal).

This bridge wraps any Forecaster whose target is a Truflation YoY
series and produces a Forecast for the corresponding official target
(BLS CPI YoY, BEA PCE YoY, etc.). Bridge coefficients fit per-origin
on a sliding window of recent paired (Truflation, BLS) observations,
respecting walk-forward discipline.

Architecture role: this is the **structural piece** that converts
Tier 1's internal Truflation prediction into the BLS / PCE prediction
the institutional product publishes. See `docs/planning/01-architecture.md`
§Composition layer.

Usage:

    composer = CBDFComposer(weights={"r1": 0.4, ...})
    composed = ComposedForecaster(components={...}, composer=composer,
                                       model_id="composed_v1")
    bridged = TruflationToBLSBridge(
        inner=composed,
        target_truf_col="trufl_headline_yoy",
        target_bls_col="bls_cpi_yoy",
        bridge_window_months=24,
        model_id="thales_bls_v1",
    )
    forecasts = walk_forward(bridged, panel, "bls_cpi_yoy", origins, 1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from thales.evaluation.harness import Forecast, Forecaster


@dataclass
class TruflationToBLSBridge:
    """Wrap an inner Truflation-target forecaster + linear bridge to BLS.

    ``inner``                 inner Forecaster (any target series)
    ``target_truf_col``       column name of the inner forecaster's target
                              (used to fit the bridge regression)
    ``target_bls_col``        column name of the BLS / PCE outer target
    ``bridge_window_months``  sliding-window length for bridge OLS estimate
    ``model_id``              free-form identifier for scoring DB
    """
    inner: Forecaster
    target_truf_col: str
    target_bls_col: str
    bridge_window_months: int = 24
    model_id: str = "bridged_v1"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        # 1) Run inner forecaster (predicts Truflation YoY at target)
        inner_fc = self.inner.fit_predict(panel, origin, target)

        # 2) Fit bridge α + β · trufl_yoy on a sliding window
        train = panel[[self.target_truf_col, self.target_bls_col]].dropna()
        train = train.loc[train.index <= origin]
        if self.bridge_window_months and len(train) > self.bridge_window_months:
            train = train.iloc[-self.bridge_window_months:]
        if len(train) < 6:
            raise ValueError(
                f"bridge: need ≥6 paired (truf, bls) obs in window; "
                f"have {len(train)}")

        x = train[self.target_truf_col].values
        y = train[self.target_bls_col].values
        X = np.column_stack([np.ones_like(x), x])
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta = float(coefs[0]), float(coefs[1])

        residuals = y - X @ coefs
        residual_sd = float(np.std(residuals, ddof=2))

        # 3) Apply bridge to inner forecast
        bridged_point = alpha + beta * inner_fc.point

        # Bridge bands: linear transformation of inner bands.
        # If beta > 0, ordering preserved. If beta < 0, lo/hi swap.
        # Add bridge-residual variance to bands (the bridge itself has noise).
        def _bridge_band(p_inner: float | None,
                            band_inflation: float) -> float | None:
            if p_inner is None:
                return None
            return alpha + beta * p_inner + band_inflation

        # 80% / 95% inflations from bridge residual SD
        inflate_80 = 1.2816 * residual_sd
        inflate_95 = 1.96 * residual_sd

        if inner_fc.has_bands:
            inner_lo80, inner_hi80 = inner_fc.lo80, inner_fc.hi80
            inner_lo95, inner_hi95 = inner_fc.lo95, inner_fc.hi95
            if beta < 0:
                inner_lo80, inner_hi80 = inner_hi80, inner_lo80
                inner_lo95, inner_hi95 = inner_hi95, inner_lo95
            lo80 = _bridge_band(inner_lo80, -inflate_80)
            hi80 = _bridge_band(inner_hi80, +inflate_80)
            lo95 = _bridge_band(inner_lo95, -inflate_95)
            hi95 = _bridge_band(inner_hi95, +inflate_95)
        else:
            # Inner gave only point — bridge bands solely from bridge residual SD
            lo80 = bridged_point - inflate_80
            hi80 = bridged_point + inflate_80
            lo95 = bridged_point - inflate_95
            hi95 = bridged_point + inflate_95

        # Bridge samples: linear transform + residual noise
        if inner_fc.samples is not None:
            rng = np.random.default_rng(0)
            bridge_noise = rng.normal(0.0, residual_sd, size=len(inner_fc.samples))
            samples = alpha + beta * inner_fc.samples + bridge_noise
        else:
            samples = None

        return Forecast(
            origin=inner_fc.origin,
            target=inner_fc.target,
            point=bridged_point,
            lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
            samples=samples,
            metadata={
                **inner_fc.metadata,
                "bridge_alpha": alpha,
                "bridge_beta": beta,
                "bridge_residual_sd": residual_sd,
                "bridge_window_n": len(train),
            },
        )
