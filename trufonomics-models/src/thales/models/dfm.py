"""Stock-Watson 2002 dynamic factor model — the standard baseline for
the O'Keeffe-Petrova CBDF head-to-head comparison.

Single latent factor extracted from a panel of monthly observable
components, then a regression of the target series on the factor.
This is the canonical "FAVAR-light" / "DFM" baseline used in the
inflation-nowcasting literature (e.g. Stock-Watson 2002 JBES,
Bańbura-Modugno 2014, and the O'Keeffe-Petrova 2025 NY Fed SR 1152
which extends to component-based DFM aka CBDF).

Estimation is two-step PCA-plus-OLS (Stock-Watson 2002), not joint
EM. The principal-components estimator is consistent under weak
factor structure and is the canonical baseline. (Bai-Ng 2002 give
the asymptotic theory.)

State-space form (informational, we don't run a Kalman filter here):

    y_t  =  Λ f_t  +  ε_t            ε_t ~ N(0, Ψ)   diag idiosyncratic
    f_t  =  φ f_{t-1}  +  η_t        η_t ~ N(0, σ_η²)

Where ``y_t`` is the k-dimensional component panel and ``f_t`` is the
1-dimensional latent factor. The target series ``z_t`` (e.g. headline
CPI YoY) loads on the factor:

    z_t  =  α_z  +  β_z f_t  +  ν_t

Forecast at h=1:

    f̂_{T+1}  =  φ f̂_T
    ẑ_{T+1}  =  α_z  +  β_z f̂_{T+1}

Density (Gaussian):

    Var(ẑ_{T+1})  ≈  β_z² · σ_η²  +  σ_ν²

This is the DFM baseline against which CBDF claims +15% RMSE / +20%
density improvement on GDP (O'Keeffe-Petrova 2025). Our test
re-validates that claim on inflation.

This module deliberately mirrors the simplicity of the canonical
DFM; richer specifications (multiple factors, dynamic loadings,
mixed-frequency MIDAS) are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from thales.evaluation.density import (
    DEFAULT_N_SAMPLES,
    samples_from_gaussian,
)
from thales.evaluation.harness import Forecast


@dataclass
class DFMFit:
    """Fitted Stock-Watson DFM."""
    factor: np.ndarray         # (T,) standardized factor scores
    loadings: np.ndarray       # (k,) factor loadings on standardized y
    mu: np.ndarray             # (k,) component means
    sigma: np.ndarray          # (k,) component SDs (used for de-standardization)
    phi_f: float               # AR(1) on factor
    sigma_eta: float           # factor innovation SD
    alpha_z: float             # target loading intercept
    beta_z: float              # target loading slope
    sigma_nu: float            # target idiosyncratic SD
    n_train: int


def _pca_first_factor(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
    """Standardize columns and extract the first principal component.

    Returns:
      factor: (T,) — scaled so SD = 1, sign normalized so first
              loading is positive
      loadings: (k,) — loadings on the standardized panel
      mu: (k,) — column means used for standardization
      sd: (k,) — column SDs used for standardization
    """
    T, k = Y.shape
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    Y_std = (Y - mu) / sd
    # SVD of the standardized panel: Y_std = U Σ V^T
    # First principal component = U[:, 0] · Σ[0]
    U, S, Vt = np.linalg.svd(Y_std, full_matrices=False)
    factor = U[:, 0] * S[0]
    loadings = Vt[0]
    # Sign normalization
    if loadings[0] < 0:
        factor = -factor
        loadings = -loadings
    # Rescale factor to unit SD (PC scaled by S[0] has SD = S[0]/sqrt(T-1))
    factor_sd = factor.std(ddof=1)
    if factor_sd > 0:
        factor = factor / factor_sd
        loadings = loadings * factor_sd
    return factor, loadings, mu, sd


def fit_dfm(Y_components: np.ndarray, z_target: np.ndarray) -> DFMFit:
    """Fit a single-factor Stock-Watson DFM.

    Parameters
    ----------
    Y_components : (T, k) — component panel (monthly observations)
    z_target     : (T,) — target series (e.g. headline CPI YoY) at the
                   same monthly frequency

    Returns
    -------
    DFMFit
    """
    Y = np.asarray(Y_components, dtype=float)
    z = np.asarray(z_target, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y_components must be 2-D (T, k)")
    if len(z) != Y.shape[0]:
        raise ValueError("z_target length must match Y_components rows")
    if Y.shape[0] < Y.shape[1] + 5:
        raise ValueError(
            f"DFM: need T > k+5; have T={Y.shape[0]}, k={Y.shape[1]}")

    factor, loadings, mu, sd = _pca_first_factor(Y)

    # AR(1) on factor
    f_lag = factor[:-1]
    f_target = factor[1:]
    X_f = np.column_stack([np.ones_like(f_lag), f_lag])
    coef_f, *_ = np.linalg.lstsq(X_f, f_target, rcond=None)
    intercept_f, phi_f = float(coef_f[0]), float(coef_f[1])
    resid_f = f_target - X_f @ coef_f
    sigma_eta = float(np.std(resid_f, ddof=2))

    # Target loading: z_t = α_z + β_z f_t + ν_t
    X_z = np.column_stack([np.ones_like(factor), factor])
    coef_z, *_ = np.linalg.lstsq(X_z, z, rcond=None)
    alpha_z, beta_z = float(coef_z[0]), float(coef_z[1])
    resid_z = z - X_z @ coef_z
    sigma_nu = float(np.std(resid_z, ddof=2))

    # Note: we drop intercept_f (typically near zero for standardized factor);
    # forecast uses pure AR phi_f * f_T projection.
    return DFMFit(
        factor=factor, loadings=loadings, mu=mu, sigma=sd,
        phi_f=phi_f, sigma_eta=sigma_eta,
        alpha_z=alpha_z, beta_z=beta_z, sigma_nu=sigma_nu,
        n_train=len(z),
    )


@dataclass
class StockWatsonDFMForecaster:
    """Single-factor DFM as a Forecaster-Protocol object.

    Operates on a panel containing ``component_cols`` (the k components)
    plus ``target_col`` (e.g. headline CPI YoY). At each origin:

      1. Fit DFM on training window via two-step PCA + OLS
      2. Project factor one step ahead: f̂_{T+1} = φ_f · f̂_T
      3. Project target: ẑ_{T+1} = α_z + β_z · f̂_{T+1}
      4. Density via Gaussian: σ²(ẑ_{T+1}) ≈ β_z²·σ_η² + σ_ν²

    No conformal upgrade here — Stock-Watson 2002 baseline form. The
    O'Keeffe-Petrova CBDF claim is precisely that their architecture
    improves over this baseline; we use it AS the baseline.
    """
    component_cols: list[str]
    target_col: str
    horizon: int = 1
    train_min: int = 36
    train_window: int | None = None
    n_samples: int = DEFAULT_N_SAMPLES
    seed: int = 0
    model_id: str = "dfm_stock_watson_v1"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        if self.target_col in self.component_cols:
            raise ValueError(
                "target_col cannot also be in component_cols")
        cols = [self.target_col] + self.component_cols
        data = panel[cols].copy().dropna()
        data = data.loc[data.index <= origin]
        if self.train_window:
            data = data.iloc[-self.train_window:]
        if len(data) < self.train_min:
            raise ValueError(
                f"DFM: need ≥{self.train_min} obs; have {len(data)}")

        Y = data[self.component_cols].values
        z = data[self.target_col].values
        fit = fit_dfm(Y, z)

        # Iterate factor forward `horizon` steps
        f_h = fit.factor[-1]
        for _ in range(self.horizon):
            f_h = fit.phi_f * f_h
        # Target forecast
        point = fit.alpha_z + fit.beta_z * f_h

        # h-step Gaussian variance (sum of geometrically-decaying factor
        # innovation variance × β_z² + idiosyncratic variance)
        var_factor_h = 0.0
        decay = 1.0
        for _ in range(self.horizon):
            var_factor_h = var_factor_h + decay * fit.sigma_eta ** 2
            decay = decay * fit.phi_f ** 2
        sigma_h = float(np.sqrt(fit.beta_z ** 2 * var_factor_h
                                          + fit.sigma_nu ** 2))

        samples = (samples_from_gaussian(
            mu=point, sigma=sigma_h,
            n_samples=self.n_samples,
            seed=self.seed + hash(origin) % 10_000)
            if self.n_samples > 0 else None)

        return Forecast(
            origin=origin, target=target, point=point,
            lo80=point - 1.2816 * sigma_h, hi80=point + 1.2816 * sigma_h,
            lo95=point - 1.96 * sigma_h, hi95=point + 1.96 * sigma_h,
            samples=samples,
            metadata={
                "model": "dfm_stock_watson",
                "n_train": fit.n_train,
                "k_components": len(self.component_cols),
                "phi_f": fit.phi_f,
                "beta_z": fit.beta_z,
                "alpha_z": fit.alpha_z,
                "sigma_h": sigma_h,
                "factor_loading_first": float(fit.loadings[0]),
            },
        )
