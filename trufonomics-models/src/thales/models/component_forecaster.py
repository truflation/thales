"""Per-component day-ahead forecaster for Truflation's US CPI YoY.

For each of the 12 top-level category streams, fits a tiny walk-forward OLS
with (1) persistence of the component itself and (2) one or two category-
specific exogenous daily covariates from the vintage store. Weight-composes
the 12 component forecasts into a headline index forecast using the
2026 v2 top-level weights (verified via the composition sanity check —
median residual 0.000 pp when reconstructing published frozen headline).

Method signposted in the Stefan day-ahead design review; MVP version of the
full archetype-specific SSM approach specified in the Trufonomics planning
architecture.

Outputs:
    ComponentForecast dataclass with:
        - point (aggregate YoY for T+1)
        - 80% / 95% bands via independent per-component bootstrap residuals
        - per-component point forecasts + contributions (for attribution)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from thales.vintage import VintageStore
from thales.weights import (
    build_crosswalk,
    get_top_level_weights,
    top_level_category_ids,
)

# ─── Category-specific exogenous feature plan ──────────────────────────────
# Maps category_id → list of FRED/EIA series ids that move daily and make
# sense as predictors for that component. Sticky categories (Health,
# Education, Communications, Alcohol & Tobacco) get persistence-only.
# Transport + Utilities pick up commodity proxies; Housing picks up rate
# proxies; Clothing/Durables pick up DXY.
CATEGORY_EXOG: dict[int, list[str]] = {
    78: ["DCOILWTICO"],                       # Food — crude as macro-inflation proxy
    79: ["MORTGAGE30US", "DGS10"],           # Housing — rate-sensitive
    80: ["DCOILWTICO", "DCOILBRENTEU"],      # Transport — commodity pass-through
    81: ["DHHNGSP"],                          # Utilities — nat gas spot
    82: [],                                   # Health — persistence only
    83: ["DTWEXBGS"],                         # Household durables — USD / import
    84: [],                                   # Alcohol & Tobacco — persistence
    85: ["DTWEXBGS"],                         # Clothing — USD / import
    86: [],                                   # Communications — persistence
    87: [],                                   # Education — persistence
    88: [],                                   # Recreation & Culture — persistence
    89: [],                                   # Other — persistence
}

DEFAULT_TRAIN_START = pd.Timestamp("2022-01-01")
DEFAULT_N_BOOT = 1000


@dataclass
class ComponentFit:
    """Per-component training output."""
    raw_name: str
    category_id: int
    category_name: str
    feature_names: list[str]
    coef: dict[str, float]
    intercept: float
    residuals: np.ndarray
    today_value: float
    today_features: dict[str, float]
    point_forecast: float


@dataclass
class HeadlineForecast:
    """Composed headline forecast output."""
    origin_date: date
    target_date: date
    point_yoy_pct: float
    band_80: tuple[float, float]
    band_95: tuple[float, float]
    today_value: float                              # current published yoy today
    component_fits: list[ComponentFit]
    attribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    weights_used: pd.DataFrame = field(default_factory=pd.DataFrame)
    composite_index_today: float = float("nan")
    composite_index_tomorrow: float = float("nan")


# ─── Panel assembly ─────────────────────────────────────────────────────────

def load_panel_from_store(store: VintageStore,
                           series_ids: Iterable[str],
                           as_of: date | None = None
                           ) -> pd.DataFrame:
    """Wide panel: row = date, columns = series_ids, values forward-filled
    to daily frequency where the underlying series is weekly/monthly.
    """
    as_of = as_of or date.today()
    cols = {}
    for sid in series_ids:
        s = store.get_vintage(sid, as_of)
        if s.empty:
            continue
        cols[sid] = s
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols)
    # Reindex to continuous daily range, forward-fill weekly/monthly features
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    return df.reindex(idx).ffill()


def fit_component(panel: pd.DataFrame, target_col: str,
                   feature_cols: list[str],
                   train_start: pd.Timestamp = DEFAULT_TRAIN_START,
                   origin: pd.Timestamp | None = None,
                   ) -> ComponentFit | None:
    """Walk-forward OLS fit for a single component; predicts T+1."""
    if target_col not in panel.columns:
        return None

    origin = origin or panel[target_col].dropna().index.max()
    origin = pd.Timestamp(origin)

    # Feature set: persistence (target at T) + exogenous features at T,
    # predicting target at T+1
    y = panel[target_col]
    X_cols = [f"{target_col}__lag0"] + [f"{f}__lag0" for f in feature_cols]

    # Build lagged panel
    feat_panel = pd.DataFrame(index=panel.index)
    feat_panel[f"{target_col}__lag0"] = panel[target_col]
    for f in feature_cols:
        if f in panel.columns:
            feat_panel[f"{f}__lag0"] = panel[f]
        else:
            return None  # missing covariate — caller decides what to do
    feat_panel["__target_tp1"] = panel[target_col].shift(-1)

    train = feat_panel.loc[(feat_panel.index >= train_start) &
                            (feat_panel.index < origin)].dropna()
    if len(train) < 60:
        return None

    X = train[X_cols].values
    y_train = train["__target_tp1"].values
    model = LinearRegression().fit(X, y_train)
    residuals = y_train - model.predict(X)

    # Forecast row at origin
    x_origin = feat_panel.loc[origin, X_cols]
    if x_origin.isna().any():
        return None
    point = float(model.predict(x_origin.values.reshape(1, -1))[0])

    today_value = float(panel.loc[origin, target_col])
    today_features = {f: float(panel.loc[origin, f]) for f in feature_cols
                       if f in panel.columns}

    category_meta = None
    return ComponentFit(
        raw_name=target_col,
        category_id=-1,   # filled in by caller
        category_name="",
        feature_names=X_cols,
        coef=dict(zip(X_cols, model.coef_.tolist())),
        intercept=float(model.intercept_),
        residuals=residuals,
        today_value=today_value,
        today_features=today_features,
        point_forecast=point,
    )


# ─── Composition ────────────────────────────────────────────────────────────

def compose_headline_forecast(store: VintageStore,
                                as_of: date | None = None,
                                train_start: pd.Timestamp = DEFAULT_TRAIN_START,
                                n_boot: int = DEFAULT_N_BOOT,
                                seed: int = 0,
                                ) -> HeadlineForecast:
    """End-to-end: pull component + covariate data, fit 12 OLS models,
    weight-compose into headline, produce bands + attribution.
    """
    as_of = as_of or date.today()

    # Resolve the 12 top-level component streams
    streams_csv = Path(__file__).resolve().parents[3] / \
        "data" / "truflation" / "streams_catalog.csv"
    streams_df = pd.read_csv(streams_csv)
    crosswalk = build_crosswalk(streams_df["raw_name"])
    tops = top_level_category_ids()
    top_streams = crosswalk[crosswalk["category_id"].isin(tops)].copy()
    top_streams["category_id"] = top_streams["category_id"].astype(int)

    # Determine all needed series (components + exogenous covariates)
    component_ids = top_streams["raw_name"].tolist()
    exog_ids = {eid for ids in CATEGORY_EXOG.values() for eid in ids}
    needed = set(component_ids) | exog_ids

    # Pull panel
    panel = load_panel_from_store(store, sorted(needed), as_of=as_of)
    if panel.empty:
        raise RuntimeError("vintage store has no data for the needed streams")

    # Origin = latest date where every component has a value
    origin = panel[component_ids].dropna().index.max()
    if origin is pd.NaT:
        raise RuntimeError("no date where all 12 component streams are populated")

    target_date = origin + timedelta(days=1)
    weights_df = get_top_level_weights(origin.date())

    # Fit each component
    fits: list[ComponentFit] = []
    for _, row in top_streams.iterrows():
        cid = row["category_id"]
        raw = row["raw_name"]
        cat_name = row["category"]
        exog = [e for e in CATEGORY_EXOG.get(cid, []) if e in panel.columns]
        fit = fit_component(panel, target_col=raw, feature_cols=exog,
                             train_start=train_start, origin=origin)
        if fit is None:
            continue
        fit.category_id = cid
        fit.category_name = cat_name
        fits.append(fit)

    if len(fits) != len(tops):
        got = {f.category_id for f in fits}
        missing = [c for c in tops if c not in got]
        raise RuntimeError(f"Component forecaster missing categories: {missing}")

    # Build weights lookup
    w_lookup = dict(zip(weights_df["category_id"].astype(int),
                         weights_df["weight"].astype(float)))

    # ── Composite today vs composite tomorrow ──
    def composite_index(today: bool) -> float:
        total_w, total_wv = 0.0, 0.0
        for fit in fits:
            w = w_lookup[fit.category_id]
            v = fit.today_value if today else fit.point_forecast
            total_wv += w * v
            total_w += w
        return total_wv / total_w if total_w > 0 else float("nan")

    composite_today = composite_index(today=True)
    composite_tomorrow = composite_index(today=False)

    # ── Bootstrap density ──
    # For each draw: resample one residual per component, add to point,
    # recompose. Take percentiles of the draws.
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        total_w, total_wv = 0.0, 0.0
        for fit in fits:
            w = w_lookup[fit.category_id]
            res = rng.choice(fit.residuals) if len(fit.residuals) else 0.0
            v = fit.point_forecast + res
            total_wv += w * v
            total_w += w
        draws[b] = total_wv / total_w

    # ── Convert composite index to YoY ──
    # Find composite index at target_date - 365 days by recomposing from panel
    # history (same method that validated at 0.000 pp median residual)
    t_minus_365 = origin - pd.DateOffset(days=365)
    if t_minus_365 not in panel.index:
        # Nearest match
        nearest_idx = panel.index.get_indexer([t_minus_365], method="nearest")[0]
        t_minus_365 = panel.index[nearest_idx]
    prior_composite = 0.0
    wsum = 0.0
    for fit in fits:
        val = panel.loc[t_minus_365, fit.raw_name]
        if pd.isna(val):
            continue
        w = w_lookup[fit.category_id]
        prior_composite += w * float(val)
        wsum += w
    prior_composite /= wsum if wsum > 0 else 1.0

    yoy_today_pct = (composite_today / prior_composite - 1.0) * 100.0
    yoy_tomorrow_pct = (composite_tomorrow / prior_composite - 1.0) * 100.0
    yoy_draws_pct = (draws / prior_composite - 1.0) * 100.0

    # ── Attribution ──
    attribution_rows = []
    for fit in fits:
        w = w_lookup[fit.category_id] / 100.0
        move = fit.point_forecast - fit.today_value
        contrib = w * move / prior_composite * 100.0  # in pp of YoY
        attribution_rows.append({
            "category_id": fit.category_id,
            "category": fit.category_name,
            "weight_pct": w_lookup[fit.category_id],
            "today_index": fit.today_value,
            "tomorrow_index": fit.point_forecast,
            "index_move": move,
            "yoy_contribution_pp": contrib,
        })
    attribution = pd.DataFrame(attribution_rows).sort_values(
        "yoy_contribution_pp", key=lambda s: s.abs(), ascending=False)

    return HeadlineForecast(
        origin_date=origin.date(),
        target_date=target_date.date(),
        point_yoy_pct=yoy_tomorrow_pct,
        band_80=(float(np.percentile(yoy_draws_pct, 10)),
                  float(np.percentile(yoy_draws_pct, 90))),
        band_95=(float(np.percentile(yoy_draws_pct, 2.5)),
                  float(np.percentile(yoy_draws_pct, 97.5))),
        today_value=yoy_today_pct,
        component_fits=fits,
        attribution=attribution,
        weights_used=weights_df,
        composite_index_today=composite_today,
        composite_index_tomorrow=composite_tomorrow,
    )
