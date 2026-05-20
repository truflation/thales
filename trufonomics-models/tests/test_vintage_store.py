"""Tests for the vintage store — the piece every backtest depends on.

The critical invariants:

  1. `get_vintage(series_id, as_of_date)` returns only observations whose
     `as_of_date` is ≤ the requested date (point-in-time correctness).
  2. Revisions are handled — re-ingesting the same reference_date at a later
     as_of_date preserves both vintages, and `get_vintage(as_of_of_first)`
     still returns the original value.
  3. Append-only: re-ingesting an identical row is a no-op; re-ingesting the
     same PK with a different value raises ``ValueError``.
  4. `snapshot(as_of)` respects vintage for every series simultaneously.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from thales.vintage import VintageStore


@pytest.fixture
def store(tmp_path: Path) -> VintageStore:
    return VintageStore(tmp_path / "test.duckdb")


def _obs(*pairs):
    return [(date.fromisoformat(d), v) for d, v in pairs]


# ─── Point-in-time correctness ───────────────────────────────────────────────

def test_get_vintage_returns_only_known_rows(store: VintageStore) -> None:
    # Ingest Jan and Feb values on 2024-03-01
    store.ingest("CPI", _obs(("2024-01-31", 3.1), ("2024-02-29", 3.2)),
                  as_of_date="2024-03-01", source="test")
    # Ingest Mar value on 2024-04-01
    store.ingest("CPI", _obs(("2024-03-31", 3.3)),
                  as_of_date="2024-04-01", source="test")

    # On 2024-03-15 we knew Jan and Feb, not March
    v_mar = store.get_vintage("CPI", "2024-03-15")
    assert len(v_mar) == 2
    assert v_mar.loc[pd.Timestamp("2024-01-31")] == 3.1
    assert v_mar.loc[pd.Timestamp("2024-02-29")] == 3.2

    # On 2024-04-15 we knew all three
    v_apr = store.get_vintage("CPI", "2024-04-15")
    assert len(v_apr) == 3
    assert v_apr.loc[pd.Timestamp("2024-03-31")] == 3.3


def test_get_vintage_empty_before_first_asof(store: VintageStore) -> None:
    store.ingest("X", _obs(("2024-01-31", 1.0)), "2024-03-01", "test")
    assert store.get_vintage("X", "2024-02-01").empty


# ─── Revisions ──────────────────────────────────────────────────────────────

def test_revisions_preserve_original_vintage(store: VintageStore) -> None:
    # First-release March value on 2024-04-01
    store.ingest("GDP", _obs(("2024-03-31", 2.5)),
                  as_of_date="2024-04-01", source="bea")
    # Revised value on 2024-05-01
    store.ingest("GDP", _obs(("2024-03-31", 2.7)),
                  as_of_date="2024-05-01", source="bea")

    # As of 2024-04-15 we only knew the first release
    v_apr = store.get_vintage("GDP", "2024-04-15")
    assert len(v_apr) == 1
    assert v_apr.iloc[0] == 2.5

    # As of 2024-05-15 we know the revision
    v_may = store.get_vintage("GDP", "2024-05-15")
    assert v_may.iloc[0] == 2.7

    # Full revision history available via revisions()
    rev = store.revisions("GDP", "2024-03-31")
    assert len(rev) == 2
    assert list(rev["value"]) == [2.5, 2.7]


# ─── Append-only / idempotency ───────────────────────────────────────────────

def test_reingest_identical_is_noop(store: VintageStore) -> None:
    r1 = store.ingest("X", _obs(("2024-01-31", 5.0)), "2024-02-01", "test")
    r2 = store.ingest("X", _obs(("2024-01-31", 5.0)), "2024-02-01", "test")
    assert r1.rows_inserted == 1
    assert r2.rows_inserted == 0
    assert r2.rows_duplicate == 1
    assert store.row_count() == 1


def test_reingest_conflicting_value_raises(store: VintageStore) -> None:
    store.ingest("X", _obs(("2024-01-31", 5.0)), "2024-02-01", "test")
    with pytest.raises(ValueError, match="Vintage conflict"):
        store.ingest("X", _obs(("2024-01-31", 7.0)), "2024-02-01", "test")
    # First value unchanged
    assert store.get_vintage("X", "2024-02-15").iloc[0] == 5.0


def test_different_sources_coexist(store: VintageStore) -> None:
    # Same series, same reference, same as_of, but two different sources
    store.ingest("CPI", _obs(("2024-01-31", 3.1)), "2024-02-15", "bls_direct")
    store.ingest("CPI", _obs(("2024-01-31", 3.11)), "2024-02-15", "fred_mirror")
    assert store.row_count() == 2
    # get_vintage with source filter picks the requested one
    v_bls = store.get_vintage("CPI", "2024-03-01", source="bls_direct")
    v_fred = store.get_vintage("CPI", "2024-03-01", source="fred_mirror")
    assert v_bls.iloc[0] == 3.1
    assert v_fred.iloc[0] == 3.11


# ─── Snapshot ────────────────────────────────────────────────────────────────

def test_snapshot_respects_vintage_across_series(store: VintageStore) -> None:
    store.ingest("CPI", _obs(("2024-01-31", 3.1)), "2024-02-15", "bls")
    store.ingest("GDP", _obs(("2024-03-31", 2.5)), "2024-04-01", "bea")

    snap_mar = store.snapshot("2024-03-01")
    assert set(snap_mar.keys()) == {"CPI", "GDP"}
    assert len(snap_mar["CPI"]) == 1
    assert snap_mar["GDP"].empty  # GDP not yet known

    snap_may = store.snapshot("2024-05-01")
    assert len(snap_may["GDP"]) == 1


def test_list_series(store: VintageStore) -> None:
    store.ingest("A", _obs(("2024-01-31", 1.0)), "2024-02-01", "src")
    store.ingest("B", _obs(("2024-01-31", 2.0)), "2024-02-01", "src")
    assert store.list_series() == ["A", "B"]


# ─── Series ingest convenience ───────────────────────────────────────────────

def test_ingest_accepts_pandas_series(store: VintageStore) -> None:
    s = pd.Series(
        {pd.Timestamp("2024-01-31"): 3.1, pd.Timestamp("2024-02-29"): 3.2},
        name="CPI",
    )
    res = store.ingest("CPI", s, as_of_date="2024-03-15", source="test")
    assert res.rows_inserted == 2
    v = store.get_vintage("CPI", "2024-04-01")
    assert len(v) == 2


def test_ingest_drops_nans(store: VintageStore) -> None:
    import numpy as np
    s = pd.Series(
        {
            pd.Timestamp("2024-01-31"): 3.1,
            pd.Timestamp("2024-02-29"): np.nan,
            pd.Timestamp("2024-03-31"): 3.3,
        }
    )
    res = store.ingest("CPI", s, "2024-04-01", "test")
    assert res.rows_submitted == 2  # NaN dropped
    assert res.rows_inserted == 2
