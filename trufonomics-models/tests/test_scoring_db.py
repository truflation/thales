"""Tests for thales.evaluation.scoring_db — the cross-model scoreboard."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from thales.evaluation.harness import Forecast
from thales.evaluation.scoring_db import ScoringDB, open_scoring_db


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "scoring_test.duckdb"


def make_forecast(origin_str: str, target_str: str,
                    point: float, band: float = 0.1) -> Forecast:
    return Forecast(
        origin=pd.Timestamp(origin_str),
        target=pd.Timestamp(target_str),
        point=point,
        lo80=point - band, hi80=point + band,
        lo95=point - 2 * band, hi95=point + 2 * band,
        metadata={"alpha": 0.01},
    )


def test_init_creates_tables(db_path):
    with open_scoring_db(db_path) as db:
        tables = db.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        names = {t[0] for t in tables}
        assert "forecasts" in names
        assert "scoring" in names


def test_insert_and_read_forecast(db_path):
    with open_scoring_db(db_path) as db:
        f = make_forecast("2026-04-24", "2026-04-25", point=1.75)
        db.insert_forecast("test_model", "test_target", f)
        df = db.read_predictions("test_model", "test_target")
        assert len(df) == 1
        assert df.iloc[0]["point"] == pytest.approx(1.75)
        assert df.iloc[0]["lo80"] == pytest.approx(1.65)
        assert pd.Timestamp(df.iloc[0]["origin_date"]).date() == date(2026, 4, 24)


def test_insert_idempotent_on_pk(db_path):
    """Re-inserting the same (model, target_series, origin, target) row
    overwrites — no duplicate-key crash."""
    with open_scoring_db(db_path) as db:
        f = make_forecast("2026-04-24", "2026-04-25", point=1.75)
        db.insert_forecast("test_model", "test_target", f)

        f2 = make_forecast("2026-04-24", "2026-04-25", point=1.80)
        db.insert_forecast("test_model", "test_target", f2)

        df = db.read_predictions("test_model", "test_target")
        assert len(df) == 1
        assert df.iloc[0]["point"] == pytest.approx(1.80)


def test_attach_actual_full_round_trip(db_path):
    """Insert a forecast, attach the realized value, read back the joined
    scoring frame and confirm derived columns match expectation."""
    with open_scoring_db(db_path) as db:
        f = make_forecast("2026-04-24", "2026-04-25", point=1.75)
        db.insert_forecast("live_v1", "trufl_yoy", f)

        ok = db.attach_actual("live_v1", "trufl_yoy",
                                 date(2026, 4, 25),
                                 actual=1.78, today_baseline=1.76)
        assert ok is True

        df = db.read_scoring("live_v1", "trufl_yoy")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["actual"] == pytest.approx(1.78)
        assert row["today"] == pytest.approx(1.76)
        assert row["error"] == pytest.approx(1.75 - 1.78)
        assert row["abs_error"] == pytest.approx(0.03)
        assert row["naive_error"] == pytest.approx(1.76 - 1.78)
        # 1.78 ∈ [1.65, 1.85] ⇒ hit_80 True
        assert bool(row["hit_80"]) is True
        # Direction: pred 1.75 < today 1.76 ⇒ pred down; actual 1.78 > today ⇒ up ⇒ miss
        assert bool(row["pred_up"]) is False
        assert bool(row["actual_up"]) is True
        assert bool(row["direction_hit"]) is False


def test_attach_actual_returns_false_on_missing_forecast(db_path):
    with open_scoring_db(db_path) as db:
        ok = db.attach_actual("nonexistent", "missing",
                                 date(2026, 4, 25), actual=1.0)
        assert ok is False


def test_summarize_returns_score_block(db_path):
    """Insert several forecasts + actuals, confirm summarize produces a
    ScoreBlock with sensible coverage / direction."""
    rng = np.random.default_rng(0)
    with open_scoring_db(db_path) as db:
        for i in range(20):
            origin = pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
            target = origin + pd.Timedelta(days=1)
            today = 2.0 + 0.01 * i
            point = today + rng.normal(0, 0.02)
            f = Forecast(
                origin=origin, target=target, point=point,
                lo80=point - 0.05, hi80=point + 0.05,
                lo95=point - 0.10, hi95=point + 0.10,
            )
            db.insert_forecast("test", "y", f)
            actual = today + rng.normal(0, 0.03)
            db.attach_actual("test", "y", target.date(),
                                actual=actual, today_baseline=today)

        block = db.summarize("test", "y")
        assert block.n == 20
        assert block.cov80 is not None
        assert block.dir_hit is not None
        assert 0 <= block.cov80 <= 1


def test_list_models_inventory(db_path):
    with open_scoring_db(db_path) as db:
        for model_id, target_series in [("a", "y"), ("a", "z"), ("b", "y")]:
            f = make_forecast("2026-04-24", "2026-04-25", point=1.0)
            db.insert_forecast(model_id, target_series, f)
            db.attach_actual(model_id, target_series,
                                date(2026, 4, 25), actual=1.05,
                                today_baseline=1.0)

        inv = db.list_models()
        assert len(inv) == 3
        assert set(inv["model_id"]) == {"a", "b"}
        assert (inv["n_scored"] == 1).all()


def test_read_with_date_window(db_path):
    with open_scoring_db(db_path) as db:
        for i in range(10):
            origin = pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
            target = origin + pd.Timedelta(days=1)
            f = make_forecast(str(origin.date()), str(target.date()),
                                 point=1.0)
            db.insert_forecast("m", "y", f)

        windowed = db.read_predictions("m", "y",
                                          since=date(2026, 1, 5),
                                          until=date(2026, 1, 7))
        assert len(windowed) == 3
        assert pd.Timestamp(windowed["origin_date"].min()).date() == date(2026, 1, 5)
        assert pd.Timestamp(windowed["origin_date"].max()).date() == date(2026, 1, 7)


def test_metadata_round_trip(db_path):
    with open_scoring_db(db_path) as db:
        f = Forecast(
            origin=pd.Timestamp("2026-04-24"),
            target=pd.Timestamp("2026-04-25"),
            point=1.75,
            metadata={"alpha": 0.01, "n_train": 1544, "lag_coef": 0.995},
        )
        db.insert_forecast("m", "y", f)
        df = db.read_predictions("m", "y")
        import json
        meta = json.loads(df.iloc[0]["metadata_json"])
        assert meta["alpha"] == 0.01
        assert meta["n_train"] == 1544
