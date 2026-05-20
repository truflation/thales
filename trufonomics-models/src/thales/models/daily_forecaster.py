"""One-day-ahead forecaster for Truflation's daily US CPI YoY index.

Target: ``y[T+1]`` where ``y[T]`` is Truflation's reported daily US CPI YoY
reading at reference date T.

Model:
    y[T+1] = β0 + β1·y[T] + β2·Δy[T] + β3·Δy5[T]
              + β4·gas_yoy[T] + β5·Δgas[T]
              + β6·T5YIE[T] + β7·DTWEXBGS[T] + ε

Density:
    point + empirical quantiles of (point + bootstrap-resampled training
    residuals). Produces 80% and 95% bands without Bayesian machinery.

Walk-forward:
    Every origin refits on all training data strictly prior to T.
    Used both for the live daily prediction (single origin = today) and for
    the rolling-origin backtest (many origins).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from thales.vintage import VintageStore

# Feature names in the fitted panel — kept as constants so ingest and model
# agree on ordering.
FEATURES = [
    "y_t",          # today's Truflation value
    "delta_y_1",    # 1-day change
    "delta_y_5",    # 5-day momentum
    "gas_yoy",      # today's gasoline YoY
    "delta_gas",    # 1-day change in gasoline YoY
    "t5yie",        # 5Y breakeven, today
    "dxy",          # broad trade-weighted USD, today
]

TARGET = "y_tp1"
TRAIN_START = pd.Timestamp("2021-01-01")


@dataclass
class Forecast:
    """Output of a single-origin prediction."""
    origin_date: pd.Timestamp          # the "today" feature date
    target_date: pd.Timestamp          # the predicted reference date (T+1)
    point: float                        # point forecast
    band_80: tuple[float, float]       # (lo, hi)
    band_95: tuple[float, float]
    p_up: float                         # P(target > today) from samples
    samples: np.ndarray = field(repr=False)  # length n_boot
    coefficients: dict[str, float] = field(default_factory=dict)
    today_value: float = float("nan")   # y[T], for "direction vs today"
    n_train: int = 0


@dataclass
class BacktestResult:
    predictions: pd.DataFrame   # columns: origin, target, pred, lo80, hi80, lo95, hi95, actual
    residuals: np.ndarray


# ─── Panel assembly ──────────────────────────────────────────────────────────

def build_panel(
    truflation: pd.Series,
    gasoline: pd.Series,
    t5yie: pd.Series,
    dxy: pd.Series,
) -> pd.DataFrame:
    """Assemble the feature panel from raw daily series.

    All inputs indexed by date, values clean (no NaN). Missing days on
    daily covariates are forward-filled to the nearest prior business-day
    observation — standard for weekday-only FRED series bleeding into
    weekends.
    """
    y = truflation.copy()
    y.name = "y_t"
    y.index = pd.to_datetime(y.index)

    # Align to daily calendar (Truflation is daily including weekends; FRED
    # series are business-day only). Forward-fill weekday FRED on weekends.
    idx = pd.date_range(y.index.min(), y.index.max(), freq="D")
    y = y.reindex(idx)

    gas = gasoline.reindex(idx)
    t5 = t5yie.reindex(idx).ffill()
    dx = dxy.reindex(idx).ffill()

    panel = pd.DataFrame({
        "y_t": y,
        "delta_y_1": y - y.shift(1),
        "delta_y_5": y - y.shift(5),
        "gas_yoy": gas,
        "delta_gas": gas - gas.shift(1),
        "t5yie": t5,
        "dxy": dx,
        "y_tp1": y.shift(-1),   # target
    })
    return panel


# ─── Fit + predict ───────────────────────────────────────────────────────────

def _fit(panel: pd.DataFrame) -> tuple[LinearRegression, np.ndarray]:
    """Fit OLS on the training slice, return (model, in-sample residuals)."""
    valid = panel.dropna(subset=FEATURES + [TARGET])
    X = valid[FEATURES].values
    y = valid[TARGET].values
    model = LinearRegression().fit(X, y)
    residuals = y - model.predict(X)
    return model, residuals


def predict_next_day(
    panel: pd.DataFrame,
    origin: pd.Timestamp | str | date | None = None,
    train_start: pd.Timestamp = TRAIN_START,
    n_boot: int = 1000,
    seed: int = 0,
) -> Forecast:
    """Fit on all data through `origin` and predict y[T+1].

    `origin` defaults to the last row with all features present (i.e., the
    most recent day we could form features on).
    """
    if origin is None:
        feature_ok = panel[FEATURES].dropna()
        origin = feature_ok.index.max()
    origin = pd.Timestamp(origin)

    # Training: rows with target known AND origin < T (strictly prior)
    train = panel.loc[(panel.index >= train_start) & (panel.index < origin)]
    train = train.dropna(subset=FEATURES + [TARGET])
    if len(train) < 60:
        raise ValueError(
            f"Not enough training rows ({len(train)}) at origin {origin:%Y-%m-%d}")

    model = LinearRegression().fit(train[FEATURES].values,
                                    train[TARGET].values)
    residuals = train[TARGET].values - model.predict(train[FEATURES].values)

    # Feature row at origin
    row = panel.loc[origin, FEATURES]
    if row.isna().any():
        missing = row[row.isna()].index.tolist()
        raise ValueError(f"Missing features at origin {origin:%Y-%m-%d}: {missing}")

    point = float(model.predict(row.values.reshape(1, -1))[0])

    # Bootstrap residual draws
    rng = np.random.default_rng(seed)
    draws = rng.choice(residuals, size=n_boot, replace=True)
    samples = point + draws

    today_value = float(panel.loc[origin, "y_t"])

    return Forecast(
        origin_date=origin,
        target_date=origin + pd.Timedelta(days=1),
        point=point,
        band_80=(float(np.percentile(samples, 10)),
                  float(np.percentile(samples, 90))),
        band_95=(float(np.percentile(samples, 2.5)),
                  float(np.percentile(samples, 97.5))),
        p_up=float((samples > today_value).mean()),
        samples=samples,
        coefficients=dict(zip(FEATURES, model.coef_)) | {"intercept": float(model.intercept_)},
        today_value=today_value,
        n_train=len(train),
    )


def walk_forward_backtest(
    panel: pd.DataFrame,
    start: pd.Timestamp | str = "2023-01-01",
    end: pd.Timestamp | str | None = None,
    train_start: pd.Timestamp = TRAIN_START,
    n_boot: int = 500,
    seed: int = 0,
) -> BacktestResult:
    """Roll through origins in [start, end], fit on strict prior, predict T+1.

    Returns a DataFrame of predictions + actuals.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end else panel.index.max()
    origins = panel.loc[start:end].dropna(subset=FEATURES).index

    rows = []
    for origin in origins:
        target_date = origin + pd.Timedelta(days=1)
        if target_date not in panel.index or pd.isna(panel.loc[target_date, "y_t"]):
            continue  # truth not available → skip
        try:
            fc = predict_next_day(panel, origin=origin,
                                    train_start=train_start,
                                    n_boot=n_boot, seed=seed)
        except ValueError:
            continue
        rows.append({
            "origin": origin,
            "target": target_date,
            "pred": fc.point,
            "lo80": fc.band_80[0],
            "hi80": fc.band_80[1],
            "lo95": fc.band_95[0],
            "hi95": fc.band_95[1],
            "actual": float(panel.loc[target_date, "y_t"]),
            "today_value": fc.today_value,
        })
    df = pd.DataFrame(rows)
    residuals = (df["pred"] - df["actual"]).values if len(df) else np.array([])
    return BacktestResult(predictions=df, residuals=residuals)


# ─── Convenience: load from existing sources ────────────────────────────────

def load_panel_from_existing_sources(
    truflation_parquet: Path,
    vintage_store: VintageStore,
    as_of_date: date | str | None = None,
) -> pd.DataFrame:
    """Build the feature panel from the kairos Truflation parquet + FRED store.

    This is the Phase 0 data path — uses the kairos parquet for historical
    daily Truflation + gasoline, and the new FRED vintage store for T5YIE
    and DTWEXBGS. Transitional: a live Truflation ingest will replace the
    parquet read when enterprise API is wired up.
    """
    tf = pd.read_parquet(truflation_parquet)
    tf["date"] = pd.to_datetime(tf["date"])
    tf = tf.set_index("date").sort_index()

    truf = tf["truflation_us_cpi_yoy/truflation_us_cpi_yoy"].dropna()
    gas = tf["us_gasoline_yoy/us_gasoline_yoy"].dropna()

    # FRED daily covariates from the vintage store, snapshotted at as_of
    as_of = as_of_date or date.today()
    t5yie = vintage_store.get_vintage("T5YIE", as_of)
    dxy = vintage_store.get_vintage("DTWEXBGS", as_of)

    return build_panel(truf, gas, t5yie, dxy)
