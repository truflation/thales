"""Same-month nowcaster — Tier 1 product use case.

At end of month T, predict BLS_yoy[T] before BLS publishes (~mid-T+1).
Knowledge available at origin = end-of-T:

  * BLS_yoy[T-1], BLS_yoy[T-2], … — last published BLS prints
  * truf_yoy[T] — Truflation aggregated through end-of-T (daily)
  * 12 Truflation per-component values through end-of-T

Structural form (Path A v1's choice):

    BLS_yoy[T]  =  α  +  β · BLS_yoy[T-1]  +  γ · truf_yoy[T]  +  ε

Estimated by OLS on a sliding training window. The β coefficient
captures BLS persistence; γ captures the Truflation lead value
(Truflation has 13-25 days of information BLS doesn't yet have).

Walk-forward semantics: at origin = T, the forecaster reads
``panel.loc[panel.index < T, "bls_yoy"]`` (excludes BLS[T] — that's
what we're predicting) and ``panel.loc[panel.index <= T, "truf_yoy"]``
(includes truf[T] — known by end-of-T).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from thales.evaluation.conformal import (
    conformal_band_offsets,
    min_n_for_alpha,
)
from thales.evaluation.density import (
    DEFAULT_N_SAMPLES,
    samples_from_gaussian,
    samples_from_residuals,
)
from thales.evaluation.harness import Forecast


CompressionMethod = Literal["pca", "pls", "grouped"]
BandMethod = Literal["gaussian", "in_sample", "rolling_conformal"]


def _samples_for_bridge(point: float,
                          residuals_or_None: np.ndarray | None,
                          residual_sd: float,
                          n_samples: int,
                          seed: int) -> np.ndarray | None:
    """Emit predictive samples consistent with the bridge band path.

    If a non-empty residual vector is provided (rolling-conformal or
    in-sample residuals when n ≥ 9), bootstrap from it. Otherwise fall
    back to Gaussian draws around ``residual_sd`` to match the band's
    Gaussian fallback. Returns None if neither source is usable.
    """
    if n_samples <= 0:
        return None
    if residuals_or_None is not None and len(residuals_or_None) >= 9:
        return samples_from_residuals(point, residuals_or_None,
                                        n_samples=n_samples, seed=seed)
    if residual_sd > 0:
        return samples_from_gaussian(point, residual_sd,
                                       n_samples=n_samples, seed=seed)
    return None


def _bands_from_residuals(point: float,
                            errors: np.ndarray
                            ) -> tuple[float, float, float, float]:
    """Two-sided 80% / 95% bands with per-α conformal-or-Gaussian fallback.

    Uses finite-sample conformal quantiles (Lei et al. 2018) when n ≥
    ``min_n_for_alpha(α)``; falls back to Gaussian z·σ for that α only
    if the calibration set is too small. This avoids the rank-clamp
    artifact (an under-calibrated 95% band silently equal to the 80%
    band when only 24 calibration months are available).
    """
    n = len(errors)
    sigma = float(np.std(errors)) if n > 1 else 0.0
    if n >= min_n_for_alpha(0.20):
        a, b = conformal_band_offsets(errors, alpha=0.20)
        lo80, hi80 = point + a, point + b
    else:
        lo80, hi80 = point - 1.2816 * sigma, point + 1.2816 * sigma
    if n >= min_n_for_alpha(0.05):
        a, b = conformal_band_offsets(errors, alpha=0.05)
        lo95, hi95 = point + a, point + b
    else:
        lo95, hi95 = point - 1.96 * sigma, point + 1.96 * sigma
    return lo80, hi80, lo95, hi95


def _gaussian_bands(point: float,
                      sigma: float) -> tuple[float, float, float, float]:
    """Gaussian bands at z=1.2816 (80%) and z=1.96 (95%)."""
    return (point - 1.2816 * sigma, point + 1.2816 * sigma,
              point - 1.96 * sigma,   point + 1.96 * sigma)


def _rolling_oos_residuals_bridge(train_full: pd.DataFrame,
                                     fit_fn,
                                     calib_months: int,
                                     train_min: int) -> np.ndarray:
    """Rolling-origin OOS residuals helper for same-month bridge family.

    For each calibration position c in the trailing ``calib_months``
    rows of ``train_full``, fit the bridge on rows ``[0:c]`` via
    ``fit_fn(train_slice) -> (predict_fn, ...)``, predict y at row c,
    record signed residual = actual - predicted.

    ``fit_fn`` must accept a DataFrame slice and return a 2-tuple of
    ``(predict_one_row, _unused)``, where ``predict_one_row(row)``
    returns the scalar prediction for a pandas row.
    """
    n_total = len(train_full)
    cal_start = max(n_total - calib_months, train_min)
    if cal_start >= n_total:
        return np.array([])
    residuals: list[float] = []
    for c in range(cal_start, n_total):
        tr = train_full.iloc[:c]
        predict_one, _ = fit_fn(tr)
        row = train_full.iloc[c]
        pred = predict_one(row)
        actual = float(row["__y_target"])
        residuals.append(actual - pred)
    return np.asarray(residuals)


@dataclass
class SameMonthBridgeNowcaster:
    """Tier 1 same-month nowcaster — α + β·BLS_lag + γ·truf at origin.

    Bands controlled by ``band_method``:

      * ``"gaussian"`` (legacy default for back-compat) — bands from
        ``z · σ_residual`` with z ∈ {1.2816, 1.96}. Fast but assumes
        Gaussian residuals; tends to undercover on heavy-tailed data.
      * ``"in_sample"`` — finite-sample conformal quantiles of
        in-sample residuals. Better than Gaussian for non-Gaussian
        residuals but biased (in-sample residuals are tighter than OOS).
      * ``"rolling_conformal"`` (recommended for production) — bands
        from rolling-origin OOS residuals over the trailing
        ``calib_months`` window. Finite-sample conformal quantiles.
        Same-month frame: target_for_calib_row[c] = BLS_yoy[c].
    """
    target_bls_col: str = "bls_yoy"
    truf_col: str = "truf_yoy"
    train_window_months: int = 36
    train_min: int = 12
    band_method: BandMethod = "gaussian"
    calib_months: int = 24
    n_samples: int = DEFAULT_N_SAMPLES
    seed: int = 0
    model_id: str = "same_month_bridge_v1"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        # Build (BLS_yoy[t], BLS_yoy[t-1], truf_yoy[t]) training rows
        # using only data BEFORE origin (since BLS[origin] is the target)
        data = panel[[self.target_bls_col, self.truf_col]].copy()
        data["bls_lag1"] = data[self.target_bls_col].shift(1)
        data["__y_target"] = data[self.target_bls_col]

        # Training: rows where origin-of-training-window has all 3 known
        train = data.dropna()
        train = train.loc[train.index < origin]
        if self.train_window_months and len(train) > self.train_window_months:
            train = train.iloc[-self.train_window_months:]
        if len(train) < self.train_min:
            raise ValueError(
                f"same-month: need ≥{self.train_min} training rows; "
                f"have {len(train)}")

        y = train[self.target_bls_col].values
        x_lag = train["bls_lag1"].values
        x_truf = train[self.truf_col].values
        X = np.column_stack([np.ones_like(y), x_lag, x_truf])
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta, gamma = (float(coefs[0]), float(coefs[1]),
                                  float(coefs[2]))
        residuals = y - X @ coefs
        residual_sd = float(np.std(residuals, ddof=3))

        # Predict BLS_yoy[origin] using BLS_yoy[origin-1] and truf_yoy[origin]
        # origin-1 = last index strictly before origin
        before_origin = data.loc[data.index < origin]
        if before_origin.empty:
            raise ValueError("no data strictly before origin")
        bls_lag1_at_origin = float(before_origin[self.target_bls_col].iloc[-1])
        truf_at_origin = panel.loc[origin, self.truf_col]
        if pd.isna(truf_at_origin):
            raise ValueError(f"truf_yoy at origin {origin} is NaN")

        point = alpha + beta * bls_lag1_at_origin + gamma * float(truf_at_origin)

        # ── Bands ─────────────────────────────────────────────────────
        meta_band: dict = {"residual_sd": residual_sd}
        # Track which residual vector to use for samples emission. We use
        # the same residuals that produced the band — guarantees the
        # density and the band agree.
        sample_residuals: np.ndarray | None = None
        if self.band_method == "rolling_conformal":
            target_col = self.target_bls_col
            truf_col = self.truf_col

            def _fit(tr: pd.DataFrame):
                yy = tr[target_col].values
                Xtr = np.column_stack([np.ones(len(yy)),
                                            tr["bls_lag1"].values,
                                            tr[truf_col].values])
                cf, *_ = np.linalg.lstsq(Xtr, yy, rcond=None)
                def _predict_one(row):
                    return float(cf[0]
                                  + cf[1] * row["bls_lag1"]
                                  + cf[2] * row[truf_col])
                return _predict_one, cf

            cal_residuals = _rolling_oos_residuals_bridge(
                train, _fit, calib_months=self.calib_months,
                train_min=self.train_min)
            if len(cal_residuals) >= 9:    # min_n_for_alpha(0.20) = 9
                lo80, hi80, lo95, hi95 = _bands_from_residuals(
                    point, cal_residuals)
                meta_band["band_source"] = "rolling_conformal"
                meta_band["n_calib"] = int(len(cal_residuals))
                sample_residuals = cal_residuals
            else:
                lo80, hi80, lo95, hi95 = _gaussian_bands(point, residual_sd)
                meta_band["band_source"] = "gaussian_fallback_n_too_small"
        elif self.band_method == "in_sample":
            if len(residuals) >= 9:
                lo80, hi80, lo95, hi95 = _bands_from_residuals(point, residuals)
                meta_band["band_source"] = "in_sample_conformal"
                sample_residuals = residuals
            else:
                lo80, hi80, lo95, hi95 = _gaussian_bands(point, residual_sd)
                meta_band["band_source"] = "gaussian_fallback_n_too_small"
        else:    # gaussian (legacy default)
            lo80, hi80, lo95, hi95 = _gaussian_bands(point, residual_sd)
            meta_band["band_source"] = "gaussian"

        samples = _samples_for_bridge(
            point, sample_residuals, residual_sd,
            n_samples=self.n_samples,
            seed=self.seed + hash(origin) % 10_000)

        return Forecast(
            origin=origin, target=target, point=point,
            lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
            samples=samples,
            metadata={
                "model": "same_month_bridge",
                "alpha": alpha, "beta_lag": beta, "gamma_truf": gamma,
                "n_train": len(train),
                "bls_lag1_at_origin": bls_lag1_at_origin,
                "truf_at_origin": float(truf_at_origin),
                **meta_band,
            },
        )


@dataclass
class MultiComponentBridgeNowcaster:
    """Tier 1 same-month nowcaster with per-component Truflation features.

    Generalizes ``SameMonthBridgeNowcaster``: instead of one Truflation
    headline feature ``truf_yoy``, uses 12 per-component Truflation YoY
    features:

        BLS_yoy[T]  =  α  +  β · BLS_yoy[T-1]  +  Σ_r γ_r · truf_r[T]  +  ε

    With Ridge regularization (small alpha) to handle multicollinearity
    among the 12 component series.
    """
    target_bls_col: str = "bls_yoy"
    truf_component_cols: list[str] = None
    train_window_months: int = 36
    train_min: int = 24
    ridge_alpha: float = 0.1
    band_method: BandMethod = "gaussian"
    calib_months: int = 24
    model_id: str = "multi_component_bridge_v1"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        if not self.truf_component_cols:
            raise ValueError("truf_component_cols must be a non-empty list")

        feature_cols = self.truf_component_cols
        data = panel[[self.target_bls_col] + feature_cols].copy()
        data["bls_lag1"] = data[self.target_bls_col].shift(1)
        data["__y_target"] = data[self.target_bls_col]

        train = data.dropna()
        train = train.loc[train.index < origin]
        if self.train_window_months and len(train) > self.train_window_months:
            train = train.iloc[-self.train_window_months:]
        if len(train) < self.train_min:
            raise ValueError(
                f"multi-comp: need ≥{self.train_min} training rows; "
                f"have {len(train)}")

        def _fit(tr: pd.DataFrame):
            yy = tr[self.target_bls_col].values
            Xtr = np.column_stack([
                np.ones(len(yy)),
                tr["bls_lag1"].values,
                tr[feature_cols].values,
            ])
            n_p = Xtr.shape[1]
            ridge_diag = np.full(n_p, self.ridge_alpha)
            ridge_diag[0] = 0.0
            cf = np.linalg.solve(Xtr.T @ Xtr + np.diag(ridge_diag),
                                       Xtr.T @ yy)

            def _predict_one(row):
                vec = np.concatenate([
                    [1.0, row["bls_lag1"]],
                    np.asarray(row[feature_cols], dtype=float),
                ])
                return float(vec @ cf)
            return _predict_one, cf

        predict_one, coefs = _fit(train)
        n_features = len(coefs)
        y = train[self.target_bls_col].values
        X = np.column_stack([np.ones(len(y)), train["bls_lag1"].values,
                                 train[feature_cols].values])
        residuals = y - X @ coefs
        residual_sd = float(np.std(residuals, ddof=n_features))

        before_origin = data.loc[data.index < origin]
        if before_origin.empty:
            raise ValueError("no data strictly before origin")
        bls_lag1 = float(before_origin[self.target_bls_col].iloc[-1])
        truf_components = panel.loc[origin, feature_cols].values
        if np.any(pd.isna(truf_components)):
            raise ValueError(f"missing truf component(s) at origin {origin}")

        x_origin = np.concatenate([[1.0, bls_lag1], truf_components.astype(float)])
        point = float(x_origin @ coefs)

        meta_band: dict = {"residual_sd": residual_sd}
        if self.band_method == "rolling_conformal":
            cal_residuals = _rolling_oos_residuals_bridge(
                train, _fit, calib_months=self.calib_months,
                train_min=self.train_min)
            if len(cal_residuals) >= 9:
                lo80, hi80, lo95, hi95 = _bands_from_residuals(
                    point, cal_residuals)
                meta_band["band_source"] = "rolling_conformal"
                meta_band["n_calib"] = int(len(cal_residuals))
            else:
                lo80, hi80, lo95, hi95 = _gaussian_bands(point, residual_sd)
                meta_band["band_source"] = "gaussian_fallback_n_too_small"
        elif self.band_method == "in_sample":
            if len(residuals) >= 9:
                lo80, hi80, lo95, hi95 = _bands_from_residuals(
                    point, residuals)
                meta_band["band_source"] = "in_sample_conformal"
            else:
                lo80, hi80, lo95, hi95 = _gaussian_bands(point, residual_sd)
                meta_band["band_source"] = "gaussian_fallback_n_too_small"
        else:
            lo80, hi80, lo95, hi95 = _gaussian_bands(point, residual_sd)
            meta_band["band_source"] = "gaussian"

        return Forecast(
            origin=origin, target=target, point=point,
            lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
            metadata={
                "model": "multi_component_bridge",
                "alpha": float(coefs[0]),
                "beta_lag": float(coefs[1]),
                "gamma_components": {col: float(g) for col, g in zip(
                    feature_cols, coefs[2:])},
                "n_train": len(train),
                "n_features": n_features,
                "ridge_alpha": self.ridge_alpha,
                **meta_band,
            },
        )


@dataclass
class CompressedMultiComponentBridge:
    """Multi-component bridge with feature compression — fixes Fix #2.

    The naive ``MultiComponentBridgeNowcaster`` overfits because 12
    component features on a 36-month training window leaves only 24
    degrees of freedom. Solution: compress 12 features → ``n_components``
    (typically 3-5) before regressing.

    Three compression strategies:

      * ``"pca"`` — unsupervised. SVD of standardized component matrix at
        each origin (no leakage). Top ``n_components`` principal
        components.
      * ``"pls"`` — supervised. Partial Least Squares directions chosen
        to maximize covariance with BLS_yoy. Tends to do better than
        PCA when target is correlated with low-variance features
        (Wold 1975, Helland 1990).
      * ``"grouped"`` — economic taxonomy. ``component_groups`` is a
        dict mapping group_name → list of truf component cols. Optional
        ``component_weights`` (col → weight) gives weighted-mean
        aggregation; otherwise simple mean.

    Compressor is **refit at every origin** to avoid look-ahead leakage.
    Standardization (PCA / PLS) uses training-window mean/SD only.

    The structural form is otherwise identical to the parent class:

        BLS_yoy[T]  =  α + β·BLS_yoy[T-1] + Σ_k γ_k · z_k[T] + ε

    where ``z_k`` is the k-th compressed feature. With ``n_components=3``
    we have 5 parameters fit on 36 obs — well-conditioned even with
    ridge_alpha=0.
    """
    target_bls_col: str = "bls_yoy"
    truf_component_cols: list[str] = field(default_factory=list)
    train_window_months: int = 36
    train_min: int = 24
    feature_compression: CompressionMethod = "pca"
    n_components: int = 3
    component_groups: dict[str, list[str]] | None = None
    component_weights: dict[str, float] | None = None
    ridge_alpha: float = 0.0
    band_method: BandMethod = "gaussian"
    calib_months: int = 24
    model_id: str = "compressed_bridge_v1"

    def _fit_and_predict_compressor(self, train: pd.DataFrame
                                          ) -> tuple[callable, np.ndarray, float, int]:
        """Fit compressor + OLS on a training slice; return a predict-one
        callable plus (residuals, residual_sd, n_features). Factored so
        the rolling-conformal loop can reuse it."""
        feature_cols = self.truf_component_cols
        y = train[self.target_bls_col].values
        bls_lag = train["bls_lag1"].values
        comp_train = train[feature_cols].values

        if self.feature_compression == "pca":
            mu = comp_train.mean(axis=0)
            sd = comp_train.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0
            comp_std = (comp_train - mu) / sd
            _, S, Vt = np.linalg.svd(comp_std, full_matrices=False)
            k = min(self.n_components, len(S))
            loadings = Vt[:k]
            z_train = comp_std @ loadings.T

            def _proj(comp_raw: np.ndarray) -> np.ndarray:
                return loadings @ ((comp_raw - mu) / sd)
        elif self.feature_compression == "pls":
            from sklearn.cross_decomposition import PLSRegression
            mu = comp_train.mean(axis=0)
            sd = comp_train.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0
            comp_std = (comp_train - mu) / sd
            k = min(self.n_components, comp_std.shape[1])
            pls = PLSRegression(n_components=k, scale=False)
            pls.fit(comp_std, y)
            z_train = pls.transform(comp_std)

            def _proj(comp_raw: np.ndarray) -> np.ndarray:
                std = ((comp_raw - mu) / sd).reshape(1, -1)
                return pls.transform(std).ravel()
        else:    # grouped
            group_keys = sorted(self.component_groups.keys())
            k = len(group_keys)
            z_train = np.zeros((len(train), k))
            group_idx_ws: list[tuple[np.ndarray, np.ndarray]] = []
            for j, g in enumerate(group_keys):
                cols = self.component_groups[g]
                if self.component_weights:
                    ws = np.array([self.component_weights.get(c, 0.0)
                                       for c in cols], dtype=float)
                    if ws.sum() <= 0:
                        ws = np.ones(len(cols))
                    ws = ws / ws.sum()
                else:
                    ws = np.ones(len(cols)) / len(cols)
                idx = np.asarray([feature_cols.index(c) for c in cols])
                z_train[:, j] = comp_train[:, idx] @ ws
                group_idx_ws.append((idx, ws))

            def _proj(comp_raw: np.ndarray) -> np.ndarray:
                out = np.zeros(k)
                for j, (idx, ws) in enumerate(group_idx_ws):
                    out[j] = comp_raw[idx] @ ws
                return out

        X = np.column_stack([np.ones(len(y)), bls_lag, z_train])
        n_features = X.shape[1]
        if self.ridge_alpha > 0:
            ridge_diag = np.full(n_features, self.ridge_alpha)
            ridge_diag[0] = 0.0
            coefs = np.linalg.solve(X.T @ X + np.diag(ridge_diag), X.T @ y)
        else:
            coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefs
        residual_sd = float(np.std(residuals, ddof=n_features))

        def _predict_one(row) -> float:
            comp_raw = np.asarray(row[feature_cols], dtype=float)
            z = _proj(comp_raw)
            vec = np.concatenate([[1.0, row["bls_lag1"]], z])
            return float(vec @ coefs)

        return _predict_one, residuals, residual_sd, n_features

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        if not self.truf_component_cols:
            raise ValueError("truf_component_cols must be a non-empty list")
        if self.feature_compression == "grouped" and not self.component_groups:
            raise ValueError("component_groups required for grouped method")

        feature_cols = self.truf_component_cols
        data = panel[[self.target_bls_col] + feature_cols].copy()
        data["bls_lag1"] = data[self.target_bls_col].shift(1)
        data["__y_target"] = data[self.target_bls_col]

        train = data.dropna()
        train = train.loc[train.index < origin]
        if self.train_window_months and len(train) > self.train_window_months:
            train = train.iloc[-self.train_window_months:]
        if len(train) < self.train_min:
            raise ValueError(
                f"compressed-bridge: need ≥{self.train_min} training "
                f"rows; have {len(train)}")

        # Fit compressor + OLS on the full training window — this gives
        # the production point predictor.
        predict_one, residuals, residual_sd, n_features = (
            self._fit_and_predict_compressor(train))

        # Predict at origin
        before = data.loc[data.index < origin]
        if before.empty:
            raise ValueError("no data strictly before origin")
        bls_lag1_origin = float(before[self.target_bls_col].iloc[-1])
        comp_origin_raw = panel.loc[origin, feature_cols].values.astype(float)
        if np.any(pd.isna(comp_origin_raw)):
            raise ValueError(f"missing truf component(s) at origin {origin}")

        # Build a stub row with the panel's origin values so predict_one
        # (which expects pandas-style ``row[col]`` access) works.
        origin_row = pd.Series({**{c: v for c, v in zip(feature_cols,
                                                              comp_origin_raw)},
                                     "bls_lag1": bls_lag1_origin})
        point = predict_one(origin_row)

        # Compression metadata for the meta dict
        compress_meta: dict = {}
        if self.feature_compression == "pca":
            comp_train = train[feature_cols].values
            mu = comp_train.mean(axis=0)
            sd = comp_train.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0
            _, S, _ = np.linalg.svd((comp_train - mu) / sd,
                                          full_matrices=False)
            k = min(self.n_components, len(S))
            compress_meta = {"explained_var_ratio":
                                 (S[:k] ** 2 / (S ** 2).sum()).tolist()}
        elif self.feature_compression == "pls":
            compress_meta = {"pls_n_components":
                                 int(min(self.n_components,
                                            len(feature_cols)))}
        else:
            compress_meta = {"groups": sorted(self.component_groups.keys())}

        # ── Bands ─────────────────────────────────────────────────────
        meta_band: dict = {"residual_sd": residual_sd}
        if self.band_method == "rolling_conformal":
            n_total = len(train)
            cal_start = max(n_total - self.calib_months, self.train_min)
            cal_residuals: list[float] = []
            for c in range(cal_start, n_total):
                tr = train.iloc[:c]
                pred_one_c, *_ = self._fit_and_predict_compressor(tr)
                row = train.iloc[c]
                cal_residuals.append(
                    float(row["__y_target"]) - pred_one_c(row))
            cal_residuals = np.asarray(cal_residuals)
            if len(cal_residuals) >= 9:
                lo80, hi80, lo95, hi95 = _bands_from_residuals(
                    point, cal_residuals)
                meta_band["band_source"] = "rolling_conformal"
                meta_band["n_calib"] = int(len(cal_residuals))
            else:
                lo80, hi80, lo95, hi95 = _gaussian_bands(point, residual_sd)
                meta_band["band_source"] = "gaussian_fallback_n_too_small"
        elif self.band_method == "in_sample":
            if len(residuals) >= 9:
                lo80, hi80, lo95, hi95 = _bands_from_residuals(
                    point, residuals)
                meta_band["band_source"] = "in_sample_conformal"
            else:
                lo80, hi80, lo95, hi95 = _gaussian_bands(point, residual_sd)
                meta_band["band_source"] = "gaussian_fallback_n_too_small"
        else:    # gaussian (legacy default)
            lo80, hi80, lo95, hi95 = _gaussian_bands(point, residual_sd)
            meta_band["band_source"] = "gaussian"

        return Forecast(
            origin=origin, target=target, point=point,
            lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
            metadata={
                "model": "compressed_bridge",
                "compression": self.feature_compression,
                "n_components": int(n_features - 2),    # subtract intercept + lag
                "n_train": len(train),
                "n_features": int(n_features),
                **meta_band,
                **compress_meta,
            },
        )


RegimeBuffer = Literal["filtered", "transition", "transition_max"]


def _markov_one_step_p_high(p_high: float,
                                 p_stay_low: float, p_stay_high: float,
                                 ) -> float:
    """One-step-ahead high-regime probability from the Markov transition.

        p_h[T+1 | T]  =  (1 − p_h[T]) · P(low → high)
                          +  p_h[T]   · P(high → high)
                       =  (1 − p_h)·(1 − p_stay_low) + p_h · p_stay_high
    """
    return ((1.0 - p_high) * (1.0 - p_stay_low)
              + p_high * p_stay_high)


def _regime_sigma(p_high: float, sigma_low: float, sigma_high: float,
                     buffer_method: RegimeBuffer,
                     p_stay_low: float, p_stay_high: float,
                     transition_threshold: float = 0.20) -> tuple[float, float]:
    """Combine regime σ's into a forecast-time σ̂. Returns (σ̂, p̃_high).

    Three buffer methods:

      * ``"filtered"`` — original behavior. σ̂ = (1−p_h) σ_low + p_h σ_high.
        Reactive: bands widen *after* the filter recognizes a regime change.

      * ``"transition"`` — one-step-ahead Markov forecast (default).
        Uses the transition matrix to compute p̃_h = P(S_{T+1}=high | data_T)
        instead of P(S_T=high | data_T). Forward-looking; widens bands
        slightly *before* a flip happens because p̃_h drifts toward the
        stationary distribution faster than the smoothed filter does.

      * ``"transition_max"`` — most conservative. Uses ``transition``
        blending unless the system is uncertain (min(p̃_h, 1−p̃_h) ≥
        ``transition_threshold``), in which case σ̂ = max(σ_low, σ_high).
        Aggressive widening at regime boundaries; protects against the
        worst-case where the next observation could come from either
        regime.
    """
    p_h_eff = p_high
    if buffer_method in ("transition", "transition_max"):
        p_h_eff = _markov_one_step_p_high(p_high, p_stay_low, p_stay_high)
    if buffer_method == "transition_max":
        if min(p_h_eff, 1.0 - p_h_eff) >= transition_threshold:
            return max(sigma_low, sigma_high), p_h_eff
    sigma_t = (1.0 - p_h_eff) * sigma_low + p_h_eff * sigma_high
    return sigma_t, p_h_eff


@dataclass
class RegimeConditionalBridgeNowcaster:
    """Same-month bridge nowcaster with REGIME-CONDITIONAL bands.

    Same point forecast as ``SameMonthBridgeNowcaster``:
        BLS_yoy[T]  =  α + β · BLS_yoy[T-1] + γ · truf_yoy[T] + ε

    But bands widen when the model is in (or near) a high-volatility
    regime, estimated by per-origin pure-MS Hamilton fit on training
    residuals. Constant-width bands (Cleveland Fed, persistence
    baselines) ignore regime; this is a Tier 1 differentiator.

    Implementation:
      1. Fit OLS bridge on training window
      2. Compute training residuals
      3. Fit Hamilton 2-state regime model on residuals → σ_low,
         σ_high, transition matrix
      4. Compute σ̂_t at origin via the chosen ``buffer_method`` (see
         ``_regime_sigma``)
      5. Bands = point ± z · σ̂_t

    Fix #6 — ``buffer_method`` defaults to ``"transition"`` (one-step-
    ahead Markov forecast) so bands widen *before* a regime flip
    rather than reactively after.
    """
    target_bls_col: str = "bls_yoy"
    truf_col: str = "truf_yoy"
    train_window_months: int = 60   # need more data for MS to fit
    train_min: int = 24
    buffer_method: RegimeBuffer = "transition"
    transition_threshold: float = 0.20
    model_id: str = "regime_conditional_bridge_v1"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        # Step 1+2: OLS bridge + residuals
        data = panel[[self.target_bls_col, self.truf_col]].copy()
        data["bls_lag1"] = data[self.target_bls_col].shift(1)
        train = data.dropna().loc[lambda d: d.index < origin]
        if self.train_window_months and len(train) > self.train_window_months:
            train = train.iloc[-self.train_window_months:]
        if len(train) < self.train_min:
            raise ValueError(
                f"regime-cond: need ≥{self.train_min} training rows; "
                f"have {len(train)}")

        y = train[self.target_bls_col].values
        x_lag = train["bls_lag1"].values
        x_truf = train[self.truf_col].values
        X = np.column_stack([np.ones_like(y), x_lag, x_truf])
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta, gamma = (float(coefs[0]), float(coefs[1]),
                                  float(coefs[2]))
        residuals = y - X @ coefs

        # Step 3: fit Hamilton 2-state on residuals
        from thales.models.archetypes.regime_switching import fit_hamilton_2state
        p_high = float("nan")
        p_high_eff = float("nan")
        if len(residuals) < 50:
            sigma_t = float(np.std(residuals, ddof=3))
            band_source = "constant"
        else:
            try:
                ms_fit = fit_hamilton_2state(residuals)
                p_high = float(ms_fit.smoothed_prob_high[-1])
                sigma_t, p_high_eff = _regime_sigma(
                    p_high=p_high,
                    sigma_low=ms_fit.sigma_low,
                    sigma_high=ms_fit.sigma_high,
                    buffer_method=self.buffer_method,
                    p_stay_low=ms_fit.p_stay_low,
                    p_stay_high=ms_fit.p_stay_high,
                    transition_threshold=self.transition_threshold,
                )
                band_source = f"regime_conditional_{self.buffer_method}"
            except Exception:
                sigma_t = float(np.std(residuals, ddof=3))
                band_source = "constant_fallback"

        # Step 4: predict at origin
        before = data.loc[data.index < origin]
        bls_lag1 = float(before[self.target_bls_col].iloc[-1])
        truf_at_origin = float(panel.loc[origin, self.truf_col])
        point = alpha + beta * bls_lag1 + gamma * truf_at_origin

        lo80 = point - 1.2816 * sigma_t
        hi80 = point + 1.2816 * sigma_t
        lo95 = point - 1.96 * sigma_t
        hi95 = point + 1.96 * sigma_t

        return Forecast(
            origin=origin, target=target, point=point,
            lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
            metadata={
                "model": "regime_conditional_bridge",
                "alpha": alpha, "beta_lag": beta, "gamma_truf": gamma,
                "sigma_conditional": sigma_t,
                "p_high": p_high,
                "p_high_eff": p_high_eff,
                "buffer_method": self.buffer_method,
                "band_source": band_source,
                "n_train": len(train),
            },
        )


@dataclass
class LastReleaseBaseline:
    """Same-month baseline: predict BLS_yoy[T] = BLS_yoy[T-1].

    The natural baseline at h=0 because BLS_yoy[T] hasn't been published
    yet at origin = T; the most recent published value is BLS_yoy[T-1].
    """
    target_col: str = "bls_yoy"
    model_id: str = "last_release_v1"

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        before_origin = panel.loc[panel.index < origin, self.target_col].dropna()
        if len(before_origin) < 1:
            raise ValueError("no observations before origin")
        point = float(before_origin.iloc[-1])
        # Bands from empirical first-difference SD on training window
        diffs = before_origin.diff().dropna().values
        if len(diffs) < 6:
            return Forecast(origin=origin, target=target, point=point,
                              metadata={"baseline": "last_release"})
        sd = float(np.std(diffs, ddof=1))
        return Forecast(
            origin=origin, target=target, point=point,
            lo80=point - 1.2816 * sd, hi80=point + 1.2816 * sd,
            lo95=point - 1.96 * sd, hi95=point + 1.96 * sd,
            metadata={"baseline": "last_release", "diff_sd": sd},
        )
