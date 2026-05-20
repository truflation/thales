"""DuckDB-backed scoring database for forecast tracking across models.

Two tables share the same schema across every Thales model — the day-ahead
LIVE forecaster, Path A nowcast, future archetype SSMs, the CBDF
composition. One queryable scoreboard is the point: 'across all models on
the last 30 days, which had the best 80% coverage in the disinflation
regime?'

Schema:

    forecasts(model_id, target_series, origin_date, target_date,
              point, lo80, hi80, lo95, hi95, metadata_json,
              created_at)
        PK (model_id, target_series, origin_date, target_date)

    scoring(model_id, target_series, target_date, actual,
            today_baseline, error, abs_error, naive_error,
            hit_80, hit_95, pred_up, actual_up, direction_hit,
            scored_at)
        PK (model_id, target_series, target_date)

Append-mostly. Re-inserting on the same PK overwrites (handy for
idempotent backtests). Density samples deliberately *not* stored here —
that grows fast and belongs in a side parquet keyed off
``(model_id, origin_date)`` if ever needed.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator

import duckdb
import numpy as np
import pandas as pd

from thales.evaluation.harness import Forecast, ScoreBlock, score

DDL_FORECASTS = """
CREATE TABLE IF NOT EXISTS forecasts (
    model_id      VARCHAR NOT NULL,
    target_series VARCHAR NOT NULL,
    origin_date   DATE NOT NULL,
    target_date   DATE NOT NULL,
    point         DOUBLE,
    lo80          DOUBLE,
    hi80          DOUBLE,
    lo95          DOUBLE,
    hi95          DOUBLE,
    metadata_json VARCHAR,
    created_at    TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (model_id, target_series, origin_date, target_date)
);
"""

DDL_SCORING = """
CREATE TABLE IF NOT EXISTS scoring (
    model_id        VARCHAR NOT NULL,
    target_series   VARCHAR NOT NULL,
    target_date     DATE NOT NULL,
    actual          DOUBLE,
    today_baseline  DOUBLE,
    error           DOUBLE,
    abs_error       DOUBLE,
    naive_error     DOUBLE,
    hit_80          BOOLEAN,
    hit_95          BOOLEAN,
    pred_up         BOOLEAN,
    actual_up       BOOLEAN,
    direction_hit   BOOLEAN,
    scored_at       TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (model_id, target_series, target_date)
);
"""

INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS forecasts_model ON forecasts (model_id, target_series);",
    "CREATE INDEX IF NOT EXISTS forecasts_origin ON forecasts (origin_date);",
    "CREATE INDEX IF NOT EXISTS scoring_model ON scoring (model_id, target_series);",
    "CREATE INDEX IF NOT EXISTS scoring_target ON scoring (target_date);",
]


@dataclass
class ScoringDB:
    """DuckDB connection wrapper with the forecast/scoring schema."""
    db_path: Path
    read_only: bool = False
    _conn: duckdb.DuckDBPyConnection | None = None

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        if not self.read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "ScoringDB":
        self._conn = duckdb.connect(str(self.db_path), read_only=self.read_only)
        if not self.read_only:
            self._conn.execute(DDL_FORECASTS)
            self._conn.execute(DDL_SCORING)
            for ddl in INDEX_DDL:
                self._conn.execute(ddl)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("ScoringDB not opened — use as context manager")
        return self._conn

    # ── Inserts ──────────────────────────────────────────────────────────

    def insert_forecast(self, model_id: str, target_series: str,
                          forecast: Forecast) -> None:
        meta = json.dumps(forecast.metadata, default=str) if forecast.metadata else None
        self.conn.execute(
            """
            INSERT OR REPLACE INTO forecasts
              (model_id, target_series, origin_date, target_date,
               point, lo80, hi80, lo95, hi95, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [model_id, target_series,
             pd.Timestamp(forecast.origin).date(),
             pd.Timestamp(forecast.target).date(),
             forecast.point,
             forecast.lo80, forecast.hi80,
             forecast.lo95, forecast.hi95,
             meta],
        )

    def insert_forecasts(self, model_id: str, target_series: str,
                           forecasts: Iterable[Forecast]) -> int:
        n = 0
        for f in forecasts:
            self.insert_forecast(model_id, target_series, f)
            n += 1
        return n

    def attach_actual(self, model_id: str, target_series: str,
                        target_date: date, actual: float,
                        today_baseline: float | None = None) -> bool:
        """Look up the corresponding forecast row, compute scoring fields,
        upsert into scoring table. Returns True on success, False if the
        forecast row is missing (nothing to score).
        """
        target_date = pd.Timestamp(target_date).date()
        row = self.conn.execute(
            """
            SELECT origin_date, point, lo80, hi80, lo95, hi95
            FROM forecasts
            WHERE model_id = ? AND target_series = ? AND target_date = ?
            """,
            [model_id, target_series, target_date],
        ).fetchone()
        if row is None:
            return False
        origin_date, point, lo80, hi80, lo95, hi95 = row

        if today_baseline is None or np.isnan(today_baseline):
            today_baseline = float("nan")

        error = point - actual if point is not None else None
        abs_err = abs(error) if error is not None else None
        naive_err = (today_baseline - actual
                       if not np.isnan(today_baseline) else None)
        hit_80 = (lo80 <= actual <= hi80) if (lo80 is not None and hi80 is not None) else None
        hit_95 = (lo95 <= actual <= hi95) if (lo95 is not None and hi95 is not None) else None
        if not np.isnan(today_baseline) and point is not None:
            pred_up = bool(point > today_baseline)
            actual_up = bool(actual > today_baseline)
            dir_hit = bool(pred_up == actual_up)
        else:
            pred_up = actual_up = dir_hit = None

        self.conn.execute(
            """
            INSERT OR REPLACE INTO scoring
              (model_id, target_series, target_date, actual, today_baseline,
               error, abs_error, naive_error,
               hit_80, hit_95, pred_up, actual_up, direction_hit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [model_id, target_series, target_date,
             float(actual),
             float(today_baseline) if not np.isnan(today_baseline) else None,
             error, abs_err, naive_err,
             hit_80, hit_95, pred_up, actual_up, dir_hit],
        )
        return True

    # ── Reads ────────────────────────────────────────────────────────────

    def read_predictions(self, model_id: str, target_series: str,
                           since: date | None = None,
                           until: date | None = None) -> pd.DataFrame:
        sql = ("SELECT * FROM forecasts WHERE model_id = ? "
                "AND target_series = ?")
        params: list[object] = [model_id, target_series]
        if since is not None:
            sql += " AND origin_date >= ?"
            params.append(pd.Timestamp(since).date())
        if until is not None:
            sql += " AND origin_date <= ?"
            params.append(pd.Timestamp(until).date())
        sql += " ORDER BY origin_date"
        return self.conn.execute(sql, params).fetchdf()

    def read_scoring(self, model_id: str, target_series: str,
                       since: date | None = None,
                       until: date | None = None) -> pd.DataFrame:
        """Return joined forecasts × scoring rows for a model.

        Adds derived columns ``today``, ``width_80``, ``width_95`` so the
        result is identical in schema to ``harness.attach_actuals`` output —
        any code that operates on one works on the other.
        """
        sql = """
            SELECT f.origin_date AS origin, s.target_date AS target,
                   f.point, f.lo80, f.hi80, f.lo95, f.hi95,
                   s.actual, s.today_baseline AS today,
                   s.error, s.abs_error, s.naive_error,
                   s.hit_80, s.hit_95,
                   s.pred_up, s.actual_up, s.direction_hit,
                   (f.hi80 - f.lo80) AS width_80,
                   (f.hi95 - f.lo95) AS width_95,
                   f.metadata_json
            FROM forecasts f
            JOIN scoring s
              ON f.model_id = s.model_id
              AND f.target_series = s.target_series
              AND f.target_date = s.target_date
            WHERE f.model_id = ? AND f.target_series = ?
        """
        params: list[object] = [model_id, target_series]
        if since is not None:
            sql += " AND s.target_date >= ?"
            params.append(pd.Timestamp(since).date())
        if until is not None:
            sql += " AND s.target_date <= ?"
            params.append(pd.Timestamp(until).date())
        sql += " ORDER BY s.target_date"
        return self.conn.execute(sql, params).fetchdf()

    def summarize(self, model_id: str, target_series: str,
                    since: date | None = None,
                    until: date | None = None) -> ScoreBlock:
        df = self.read_scoring(model_id, target_series,
                                  since=since, until=until)
        return score(df)

    def list_models(self) -> pd.DataFrame:
        """Inventory: one row per (model_id, target_series) with counts."""
        return self.conn.execute(
            """
            SELECT f.model_id, f.target_series,
                   COUNT(*) AS n_forecasts,
                   COUNT(s.actual) AS n_scored,
                   MIN(f.origin_date) AS first_origin,
                   MAX(f.origin_date) AS last_origin
            FROM forecasts f
            LEFT JOIN scoring s
              ON f.model_id = s.model_id
              AND f.target_series = s.target_series
              AND f.target_date = s.target_date
            GROUP BY f.model_id, f.target_series
            ORDER BY f.model_id, f.target_series
            """
        ).fetchdf()


@contextmanager
def open_scoring_db(path: Path | str,
                      read_only: bool = False) -> Iterator[ScoringDB]:
    """Convenience context manager for callers who don't want to instantiate."""
    db = ScoringDB(Path(path), read_only=read_only)
    with db:
        yield db
