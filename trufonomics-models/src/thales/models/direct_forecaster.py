"""Direct-target day-ahead forecaster.

Forecasts Truflation's published frozen YoY directly, using the 12 top-level
component index values as features. Replaces the "composite → composition
math → YoY" path of ``component_forecaster.py`` with a single Ridge
regression whose residuals reflect TRUE prediction uncertainty for the
target series — including any composition drift between our reconstructed
composite and Truflation's published aggregate.

Motivation: the composite-based forecaster's 80% band covered only 2.4% of
realizations in the 90-day backtest because the composite drifts 0.2–0.3pp
from published YoY in recent weeks, and per-component residual bootstraps
don't capture that drift.

Model:

    published_yoy[T+1]
        ~ α  +  Σᵢ βᵢ · component_iᵢ[T]
             +  φ · published_yoy[T]
             +  ε

    ε_t are the bootstrap draws for the 80% / 95% bands.

Attribution: βᵢ coefficients tell us sensitivity of tomorrow's published YoY
to each component's current value. Combined with today's recent moves,
produces a component-level contribution table for Stefan's post.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from thales.vintage import VintageStore
from thales.weights import build_crosswalk, top_level_category_ids

KAIROS_PARQUET = Path("/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
PUBLISHED_YOY_COL = "truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy"
DEFAULT_TRAIN_START = pd.Timestamp("2022-01-01")
DEFAULT_N_BOOT = 1000
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass
class ComponentContribution:
    """Per-category sensitivity + estimated contribution to tomorrow's YoY."""
    category_id: int
    category_name: str
    raw_name: str
    today_index: float
    coef: float                # β_i — sensitivity of tomorrow's published YoY to this component
    recent_move_pct: float     # 7-day % change in the component index
    contribution_pp: float     # ≈ β_i * (component_i[T] - component_i[T-7])


@dataclass
class DirectForecast:
    origin_date: date
    target_date: date
    point_yoy_pct: float
    band_80: tuple[float, float]
    band_95: tuple[float, float]
    today_published_yoy: float        # the true "today" baseline for direction
    residual_sd_pct: float            # in pp
    intercept: float
    lag_coef: float                   # φ — coefficient on published_yoy[T]
    ridge_alpha: float
    n_train: int
    contributions: list[ComponentContribution] = field(default_factory=list)


def _load_published_yoy() -> pd.Series:
    pq = pd.read_parquet(KAIROS_PARQUET)
    pq["date"] = pd.to_datetime(pq["date"])
    pq = pq.set_index("date").sort_index()
    return pq[PUBLISHED_YOY_COL].dropna()


def _load_components(store: VintageStore, raw_names: Iterable[str],
                      as_of: date) -> pd.DataFrame:
    cols = {}
    for sid in raw_names:
        s = store.get_vintage(sid, as_of)
        if not s.empty:
            cols[sid] = s
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols)
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    return df.reindex(idx).ffill()


def build_training_panel(store: VintageStore,
                          as_of: date | None = None
                          ) -> tuple[pd.DataFrame, list[tuple[int, str, str]]]:
    """Wide panel with 12 component columns + `__published_yoy` column."""
    as_of = as_of or date.today()

    streams_csv = Path(__file__).resolve().parents[3] / \
        "data" / "truflation" / "streams_catalog.csv"
    streams_df = pd.read_csv(streams_csv)
    crosswalk = build_crosswalk(streams_df["raw_name"])
    tops = top_level_category_ids()
    top_streams = crosswalk[crosswalk["category_id"].isin(tops)].copy()
    top_streams["category_id"] = top_streams["category_id"].astype(int)

    top_info = [
        (int(r.category_id), r.raw_name, r.category)
        for _, r in top_streams.iterrows()
    ]
    raw_names = [t[1] for t in top_info]

    components = _load_components(store, raw_names, as_of=as_of)
    if components.empty:
        raise RuntimeError("vintage store has no component data")

    published = _load_published_yoy()
    panel = components.copy()
    panel["__published_yoy"] = published

    # Inner-join on the dates where BOTH components and published are present
    panel = panel.dropna(subset=["__published_yoy"])
    panel = panel.dropna(subset=raw_names)   # all 12 must be present
    return panel, top_info


def _fit_ridge(panel: pd.DataFrame, raw_names: list[str],
                train_start: pd.Timestamp, origin: pd.Timestamp
                ) -> tuple[RidgeCV, np.ndarray, float, float] | None:
    """Fit RidgeCV on training slice. Returns (model, residuals, intercept, alpha)."""
    feature_cols = raw_names + ["__published_yoy"]
    feat = panel[feature_cols].copy()
    feat["__target_tp1"] = panel["__published_yoy"].shift(-1)

    train = feat.loc[(feat.index >= train_start) &
                      (feat.index < origin)].dropna()
    if len(train) < 90:
        return None

    X = train[feature_cols].values
    y = train["__target_tp1"].values
    model = RidgeCV(alphas=list(RIDGE_ALPHAS)).fit(X, y)
    residuals = y - model.predict(X)
    return model, residuals, float(model.intercept_), float(model.alpha_)


def direct_target_forecast(store: VintageStore,
                             as_of: date | None = None,
                             train_start: pd.Timestamp = DEFAULT_TRAIN_START,
                             n_boot: int = DEFAULT_N_BOOT,
                             seed: int = 0,
                             ) -> DirectForecast:
    """Fit Ridge on historical (components, published_yoy_lag) → published_yoy,
    predict T+1, return point + bootstrap bands + attribution.
    """
    as_of = as_of or date.today()
    panel, top_info = build_training_panel(store, as_of=as_of)
    raw_names = [t[1] for t in top_info]
    feature_cols = raw_names + ["__published_yoy"]

    origin = panel.index.max()
    fit = _fit_ridge(panel, raw_names, train_start, origin + pd.Timedelta(days=1))
    # origin+1 here because _fit_ridge uses feat.index < origin_param internally
    # and we want to include origin itself in the training set (we KNOW today).
    if fit is None:
        raise RuntimeError("not enough training data for direct forecast")
    model, residuals, intercept, alpha = fit

    # Feature vector at origin = today
    x_origin = panel.loc[origin, feature_cols].values.reshape(1, -1)
    point = float(model.predict(x_origin)[0])

    # Bootstrap bands
    rng = np.random.default_rng(seed)
    draws = point + rng.choice(residuals, size=n_boot)
    band_80 = (float(np.percentile(draws, 10)),
                float(np.percentile(draws, 90)))
    band_95 = (float(np.percentile(draws, 2.5)),
                float(np.percentile(draws, 97.5)))

    today_published = float(panel.loc[origin, "__published_yoy"])
    coef_vec = model.coef_

    # Attribution: β_i × (component_i[T] - component_i[T-7]) for top categories
    target_week_ago = origin - pd.Timedelta(days=7)
    if target_week_ago not in panel.index:
        idx = panel.index.get_indexer([target_week_ago], method="nearest")[0]
        target_week_ago = panel.index[idx]

    contributions: list[ComponentContribution] = []
    for i, (cid, raw, cat_name) in enumerate(top_info):
        today_val = float(panel.loc[origin, raw])
        week_ago_val = float(panel.loc[target_week_ago, raw])
        recent_move_pct = (today_val - week_ago_val) / week_ago_val * 100.0 \
            if week_ago_val else 0.0
        beta = float(coef_vec[i])
        # Contribution in pp of yoy: β_i × the CHANGE the component made this week
        contribution_pp = beta * (today_val - week_ago_val)
        contributions.append(ComponentContribution(
            category_id=cid, category_name=cat_name, raw_name=raw,
            today_index=today_val, coef=beta,
            recent_move_pct=recent_move_pct,
            contribution_pp=contribution_pp,
        ))

    contributions.sort(key=lambda c: abs(c.contribution_pp), reverse=True)
    # Lag coef is the last entry — the published-yoy persistence coefficient
    lag_coef = float(coef_vec[-1])

    return DirectForecast(
        origin_date=origin.date(),
        target_date=(origin + timedelta(days=1)).date(),
        point_yoy_pct=point,
        band_80=band_80,
        band_95=band_95,
        today_published_yoy=today_published,
        residual_sd_pct=float(residuals.std()),
        intercept=intercept,
        lag_coef=lag_coef,
        ridge_alpha=alpha,
        n_train=len(residuals),
        contributions=contributions,
    )
