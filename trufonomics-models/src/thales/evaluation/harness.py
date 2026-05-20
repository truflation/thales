"""Walk-forward evaluation harness.

The spine of every Thales evaluation: synthetic-DGP recovery, historical
walk-forward, and live tracking. Wraps the metrics and tests modules in a
re-usable simulator that takes any object implementing the ``Forecaster``
protocol and produces a standard ``ScoreBlock``.

Three abstractions:

  * ``Forecast`` — a single prediction with optional bands and metadata.
  * ``Forecaster`` — a callable protocol producing one ``Forecast`` per
    (panel, origin, target) call. Stateless from the harness's view; the
    forecaster handles its own fitting.
  * ``walk_forward`` — drives the loop over origins, materializing a list
    of forecasts; ``attach_actuals`` joins to the realized series, then
    ``score`` computes the metric block.

Idempotent and DataFrame-native; no global state. Density support via
``Forecast.samples`` (shape ``(S,)``) feeds CRPS / PIT scoring downstream.

Usage:

    forecaster = MyForecaster()  # implements fit_predict
    forecasts = walk_forward(forecaster, panel, target_col="y",
                              origins=panel.index[-90:], horizon=1)
    df = attach_actuals(forecasts, panel["y"])
    block = score(df, today_col="y_today")
    print(block.summary())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Protocol

import numpy as np
import pandas as pd

from thales.evaluation import metrics as M


# ─── Forecast object ──────────────────────────────────────────────────────


@dataclass
class Forecast:
    """One prediction at one origin for one target.

    Bands and samples are optional; only the point is required. Metadata
    is a free-form dict the forecaster can use to record alphas, residual
    SDs, the today-baseline value, etc.
    """
    origin: pd.Timestamp
    target: pd.Timestamp
    point: float
    lo80: float | None = None
    hi80: float | None = None
    lo95: float | None = None
    hi95: float | None = None
    samples: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_bands(self) -> bool:
        return self.lo80 is not None and self.hi80 is not None

    @property
    def has_density(self) -> bool:
        return self.samples is not None and len(self.samples) > 0


class Forecaster(Protocol):
    """Stateless protocol for any forecasting model.

    Implementations may be classes (carrying configuration) or simple
    closures over ``fit_predict``. The harness only requires:

      * ``model_id`` — string label used as the primary key in scoring DB
      * ``fit_predict(panel, origin, target) -> Forecast`` — given the
        panel TRUNCATED at origin (no peeking) plus an explicit target
        date, return one Forecast.
    """
    model_id: str

    def fit_predict(self, panel: pd.DataFrame,
                     origin: pd.Timestamp,
                     target: pd.Timestamp) -> Forecast:
        ...


# ─── Walk-forward driver ──────────────────────────────────────────────────


def walk_forward(forecaster: Forecaster,
                  panel: pd.DataFrame,
                  target_col: str,
                  origins: Iterable[pd.Timestamp],
                  horizon: int = 1,
                  verbose: bool = False,
                  ) -> list[Forecast]:
    """Drive the forecaster across origins. Slices the panel to ``[:origin]``
    so the forecaster physically can't see future rows.

    ``target_col`` is informational; the harness only uses it to validate
    that the target row exists in the panel index. The forecaster is
    responsible for choosing its features inside the slice.

    ``horizon`` is in *index steps*, not days. For a daily panel that's
    days; for a monthly panel that's months. Leave at 1 for day-ahead.
    """
    forecasts: list[Forecast] = []
    origins_list = list(origins)
    for i, origin in enumerate(origins_list, 1):
        if origin not in panel.index:
            continue
        origin_pos = panel.index.get_loc(origin)
        target_pos = origin_pos + horizon
        if target_pos >= len(panel.index):
            continue
        target = panel.index[target_pos]
        # Slice up to and including origin — the forecaster decides its
        # own training window from here.
        slice_panel = panel.iloc[: origin_pos + 1]
        try:
            fc = forecaster.fit_predict(slice_panel, origin, target)
        except Exception as e:
            if verbose:
                print(f"  [{i}/{len(origins_list)}] origin={origin.date()} "
                      f"FAILED: {type(e).__name__}: {e}")
            continue
        forecasts.append(fc)
        if verbose and (i % 15 == 0 or i == len(origins_list)):
            print(f"  [{i}/{len(origins_list)}] origin={origin.date()} "
                  f"point={fc.point:.4f}")
    return forecasts


def attach_actuals(forecasts: list[Forecast],
                    actuals: pd.Series,
                    today_baseline: pd.Series | None = None,
                    ) -> pd.DataFrame:
    """Join a list of forecasts to a realized series (and optionally a
    'today' baseline used for direction tests). Returns a flat DataFrame
    with one row per scored forecast.

    Forecasts whose ``target`` is missing from ``actuals`` are dropped
    (the realized value isn't in yet). ``today_baseline`` defaults to
    ``actuals[origin]`` if not provided.

    Predictive samples (``Forecast.samples``) ride along as a column of
    1-D ``np.ndarray`` objects when present, so :func:`score` can pull
    them out and compute CRPS / PIT / interval coverage automatically.
    """
    rows = []
    for f in forecasts:
        if f.target not in actuals.index:
            continue
        actual = float(actuals.loc[f.target])
        if pd.isna(actual):
            continue

        if today_baseline is not None and f.origin in today_baseline.index:
            today = float(today_baseline.loc[f.origin])
        elif f.origin in actuals.index:
            today = float(actuals.loc[f.origin])
        else:
            today = float("nan")

        row = {
            "origin": f.origin,
            "target": f.target,
            "point": f.point,
            "lo80": f.lo80,
            "hi80": f.hi80,
            "lo95": f.lo95,
            "hi95": f.hi95,
            "actual": actual,
            "today": today,
            "error": f.point - actual,
            "abs_error": abs(f.point - actual),
            "naive_error": today - actual if not np.isnan(today) else float("nan"),
        }
        if f.has_bands:
            row["hit_80"] = bool(f.lo80 <= actual <= f.hi80)
            row["hit_95"] = bool(f.lo95 <= actual <= f.hi95)
            row["width_80"] = f.hi80 - f.lo80
            row["width_95"] = f.hi95 - f.lo95
        if not np.isnan(today):
            row["pred_up"] = bool(f.point > today)
            row["actual_up"] = bool(actual > today)
            row["direction_hit"] = bool(row["pred_up"] == row["actual_up"])
        if f.has_density:
            row["samples"] = np.asarray(f.samples, dtype=float)
        if f.metadata:
            row["metadata_json"] = json.dumps(f.metadata, default=str)
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Scoring ──────────────────────────────────────────────────────────────


@dataclass
class ScoreBlock:
    """Standard metric block over a window of scored forecasts.

    Reports BOTH RMSE-reduction and MSE-reduction relative to the naive
    baseline, since they're easy to confuse:

        RMSE_red  =  1 - RMSE_method / RMSE_naive
        MSE_red   =  1 - MSE_method  / MSE_naive
                  =  1 - (1 - RMSE_red)²

    A 24% RMSE reduction is a 42% MSE reduction (Path A v1 reports
    MSE-reduction; many recent papers report RMSE-reduction). Showing
    both prevents the "are we beating Path A or not" confusion that
    cost us a couple of hours on 2026-04-25.

    Density metrics (``crps``, ``pit_ks_pvalue``, ``cov80_density``,
    ``cov95_density``, ``sharp80_density``, ``sharp95_density``) are
    populated when forecasts carry ``samples`` arrays (or ``score`` is
    given a ``density_samples`` matrix). Otherwise None.
    """
    n: int
    window_start: str
    window_end: str
    rmse: float
    mae: float
    rmse_naive: float | None
    rmse_reduction_pct: float | None
    mse_reduction_pct: float | None     # NEW — derived from rmse_reduction_pct
    cov80: float | None
    cov95: float | None
    width80: float | None
    width95: float | None
    dir_hit: float | None
    base_rate_up: float | None
    crps: float | None
    pit_ks_pvalue: float | None = None
    cov80_density: float | None = None
    cov95_density: float | None = None
    sharp80_density: float | None = None
    sharp95_density: float | None = None
    n_density: int | None = None
    ship_verdict: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def summary(self) -> str:
        lines = [
            f"n={self.n}  window {self.window_start} → {self.window_end}",
            f"RMSE: {self.rmse:.4f}",
        ]
        if self.rmse_naive is not None:
            lines[-1] += (f"  vs naive {self.rmse_naive:.4f}  "
                           f"({self.rmse_reduction_pct:+.2f}% RMSE / "
                           f"{self.mse_reduction_pct:+.2f}% MSE)")
        lines.append(f"MAE:  {self.mae:.4f}")
        if self.cov80 is not None:
            lines.append(f"80% coverage: {self.cov80:.1%}  "
                          f"(width {self.width80:.4f} pp)")
            lines.append(f"95% coverage: {self.cov95:.1%}  "
                          f"(width {self.width95:.4f} pp)")
        if self.dir_hit is not None:
            lines.append(f"Directional acc: {self.dir_hit:.1%}  "
                          f"(base-rate up: {self.base_rate_up:.1%})")
        if self.crps is not None:
            crps_line = f"CRPS: {self.crps:.4f}"
            if self.pit_ks_pvalue is not None:
                crps_line += f"  PIT-KS p={self.pit_ks_pvalue:.3f}"
            lines.append(crps_line)
        if self.cov80_density is not None:
            lines.append(
                f"Density 80%/95% cov: {self.cov80_density:.1%} / "
                f"{self.cov95_density:.1%}  "
                f"(sharp {self.sharp80_density:.3f} / "
                f"{self.sharp95_density:.3f})")
        lines.append(f"Verdict: {self.ship_verdict}")
        return "\n".join(lines)


def _ship_gate(cov80: float | None, cov95: float | None,
                rmse_red: float | None) -> str:
    """SHIP iff: 80% coverage within ±7pp of nominal AND 95% within ±4pp
    AND model not catastrophically worse than naive (RMSE reduction > -10%).
    """
    if cov80 is None or cov95 is None:
        return "INSUFFICIENT-METRICS"
    calibrated = abs(cov80 - 0.80) < 0.07 and abs(cov95 - 0.95) < 0.04
    competitive = rmse_red is None or rmse_red > -10
    return "SHIP" if (calibrated and competitive) else "HOLD"


def score(df: pd.DataFrame,
           density_samples: np.ndarray | None = None,
           ) -> ScoreBlock:
    """Compute the metric block from a frame produced by ``attach_actuals``.

    ``density_samples`` is an optional ``(n, S)`` array aligned to the
    rows of ``df``. When omitted, the function looks for a ``samples``
    column on ``df`` (carried from ``Forecast.samples``) and stacks
    them automatically; if neither is present, density metrics are None.
    """
    if df.empty:
        return ScoreBlock(
            n=0, window_start="", window_end="",
            rmse=float("nan"), mae=float("nan"),
            rmse_naive=None, rmse_reduction_pct=None,
            mse_reduction_pct=None,
            cov80=None, cov95=None, width80=None, width95=None,
            dir_hit=None, base_rate_up=None, crps=None,
            ship_verdict="INSUFFICIENT-DATA",
        )

    actual = df["actual"].values
    point = df["point"].values

    rmse_v = M.rmse(point, actual)
    mae_v = M.mae(point, actual)

    rmse_naive = None
    rmse_red = None
    mse_red = None
    if "naive_error" in df.columns and df["naive_error"].notna().any():
        rmse_naive = float(np.sqrt(np.mean(df["naive_error"].dropna() ** 2)))
        if rmse_naive > 0:
            rmse_red = (1 - rmse_v / rmse_naive) * 100
            # MSE_red = 1 - (1 - RMSE_red)²  — exact algebraic identity.
            mse_red = (1 - (1 - rmse_red / 100) ** 2) * 100

    cov80 = float(df["hit_80"].mean()) if "hit_80" in df.columns else None
    cov95 = float(df["hit_95"].mean()) if "hit_95" in df.columns else None
    width80 = float(df["width_80"].mean()) if "width_80" in df.columns else None
    width95 = float(df["width_95"].mean()) if "width_95" in df.columns else None
    dir_hit = float(df["direction_hit"].mean()) if "direction_hit" in df.columns else None
    base_up = float(df["actual_up"].mean()) if "actual_up" in df.columns else None

    # ── Density metrics (CRPS / PIT / coverage / sharpness) ───────────
    density_block = None
    samples_matrix = None
    if density_samples is not None and len(density_samples) == len(df):
        samples_matrix = np.asarray(density_samples, dtype=float)
    elif "samples" in df.columns and df["samples"].notna().any():
        # Stack from per-row arrays; rows with NaN/None get NaN row.
        n_samples = max(
            (len(s) for s in df["samples"]
             if isinstance(s, np.ndarray) and len(s)),
            default=0,
        )
        if n_samples > 0:
            samples_matrix = np.full((len(df), n_samples), np.nan)
            for i, s in enumerate(df["samples"].values):
                if isinstance(s, np.ndarray) and len(s) >= n_samples:
                    samples_matrix[i, :] = s[:n_samples]

    if samples_matrix is not None:
        from thales.evaluation.density import score_density
        density_block = score_density(samples_matrix, actual)

    crps = density_block.crps if density_block else None
    pit_ks = density_block.pit_ks_pvalue if density_block else None
    cov80_d = density_block.cov80 if density_block else None
    cov95_d = density_block.cov95 if density_block else None
    sharp80_d = density_block.sharpness80 if density_block else None
    sharp95_d = density_block.sharpness95 if density_block else None
    n_d = density_block.n if density_block else None

    return ScoreBlock(
        n=len(df),
        window_start=str(df["origin"].min()),
        window_end=str(df["origin"].max()),
        rmse=rmse_v, mae=mae_v,
        rmse_naive=rmse_naive, rmse_reduction_pct=rmse_red,
        mse_reduction_pct=mse_red,
        cov80=cov80, cov95=cov95, width80=width80, width95=width95,
        dir_hit=dir_hit, base_rate_up=base_up,
        crps=crps,
        pit_ks_pvalue=pit_ks,
        cov80_density=cov80_d, cov95_density=cov95_d,
        sharp80_density=sharp80_d, sharp95_density=sharp95_d,
        n_density=n_d,
        ship_verdict=_ship_gate(cov80, cov95, rmse_red),
    )


# ─── End-to-end convenience ───────────────────────────────────────────────


def evaluate(forecaster: Forecaster,
              panel: pd.DataFrame,
              target_col: str,
              origins: Iterable[pd.Timestamp],
              horizon: int = 1,
              today_baseline: pd.Series | None = None,
              verbose: bool = False,
              ) -> tuple[pd.DataFrame, ScoreBlock]:
    """Convenience: walk-forward + attach + score in one call.

    Returns the materialized prediction frame (one row per scored origin)
    and the aggregate ScoreBlock.
    """
    forecasts = walk_forward(forecaster, panel, target_col, origins,
                              horizon=horizon, verbose=verbose)
    df = attach_actuals(forecasts, panel[target_col],
                         today_baseline=today_baseline)
    block = score(df)
    return df, block
