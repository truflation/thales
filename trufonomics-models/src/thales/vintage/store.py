"""Point-in-time vintage data store backed by DuckDB.

Schema (single append-only table):

    CREATE TABLE vintage (
        series_id       VARCHAR NOT NULL,
        reference_date  DATE    NOT NULL,  -- the date the value refers to
        as_of_date      DATE    NOT NULL,  -- when we learned the value
        value           DOUBLE  NOT NULL,
        source          VARCHAR NOT NULL,  -- 'fred', 'bls', 'clevfed_scrape', ...
        source_hash     VARCHAR,           -- optional response hash for repro
        ingestion_ts    TIMESTAMP NOT NULL DEFAULT current_timestamp,
        PRIMARY KEY (series_id, reference_date, as_of_date, source)
    )

Invariants:
  - Append-only. UPDATE and DELETE are not exposed on the public API.
  - Unique on (series_id, reference_date, as_of_date, source): re-ingesting
    an identical vintage is a no-op; re-ingesting the same key with a
    different value raises.
  - `get_vintage(series_id, as_of_date)` returns, for each reference_date, the
    most recent value that was available at or before `as_of_date`. This is
    the standard real-time-vintage query.

This module intentionally has no knowledge of FRED/BLS/etc. — it's a generic
bitemporal store. Source-specific ingest lives in thales.ingest.*.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS vintage (
    series_id       VARCHAR NOT NULL,
    reference_date  DATE    NOT NULL,
    as_of_date      DATE    NOT NULL,
    value           DOUBLE  NOT NULL,
    source          VARCHAR NOT NULL,
    source_hash     VARCHAR,
    ingestion_ts    TIMESTAMP NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (series_id, reference_date, as_of_date, source)
);
CREATE INDEX IF NOT EXISTS ix_vintage_series_asof
    ON vintage (series_id, as_of_date);
CREATE INDEX IF NOT EXISTS ix_vintage_series_ref
    ON vintage (series_id, reference_date);
"""


def _as_date(d: date | datetime | str | pd.Timestamp) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return pd.Timestamp(d).date()


def _hash_payload(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class IngestResult:
    """Summary of an `ingest` call."""
    series_id: str
    source: str
    as_of_date: date
    rows_submitted: int
    rows_inserted: int
    rows_duplicate: int


class VintageStore:
    """Bitemporal append-only data store.

    Parameters
    ----------
    db_path : str | Path
        Path to the DuckDB file. Parent directories are created.
    read_only : bool, default False
        Open in read-only mode (useful for backtest workers).
    """

    def __init__(self, db_path: str | Path, read_only: bool = False):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path), read_only=read_only)
        if not read_only:
            self._conn.execute(SCHEMA)

    # ─── Public API ──────────────────────────────────────────────────────

    def ingest(
        self,
        series_id: str,
        observations: Iterable[tuple[date | str, float]] | pd.Series,
        as_of_date: date | str,
        source: str,
        source_hash: str | None = None,
    ) -> IngestResult:
        """Insert a batch of (reference_date, value) observations at a given as_of.

        If `observations` is a pandas Series, its index is taken as the
        reference_date.  Duplicate (series_id, reference_date, as_of_date,
        source) keys with *identical* values are silently skipped; with
        *different* values they raise ``ValueError``.
        """
        asof = _as_date(as_of_date)
        rows = list(_normalize_observations(observations))
        if not rows:
            return IngestResult(series_id, source, asof, 0, 0, 0)

        payload = "\n".join(f"{r.isoformat()}\t{v:.12g}" for r, v in rows)
        resolved_hash = source_hash or _hash_payload(payload)

        # Stage in a temp table, then MERGE-style insert.
        self._conn.execute("BEGIN")
        try:
            self._conn.execute("""
                CREATE OR REPLACE TEMP TABLE _staging (
                    reference_date DATE,
                    value DOUBLE
                )
            """)
            self._conn.executemany(
                "INSERT INTO _staging VALUES (?, ?)",
                [(r, float(v)) for r, v in rows],
            )

            # Detect conflicts: same PK, different value
            conflicts = self._conn.execute("""
                SELECT v.reference_date, v.value AS existing, s.value AS incoming
                FROM vintage v
                JOIN _staging s USING (reference_date)
                WHERE v.series_id = ?
                  AND v.as_of_date = ?
                  AND v.source    = ?
                  AND v.value    <> s.value
            """, [series_id, asof, source]).fetchall()
            if conflicts:
                sample = conflicts[:3]
                raise ValueError(
                    f"Vintage conflict for series_id={series_id!r} "
                    f"as_of={asof} source={source!r}: "
                    f"{len(conflicts)} row(s) would overwrite existing values. "
                    f"Sample: {sample}"
                )

            # Insert non-duplicates
            inserted = self._conn.execute("""
                INSERT INTO vintage
                    (series_id, reference_date, as_of_date, value, source, source_hash)
                SELECT ?, s.reference_date, ?, s.value, ?, ?
                FROM _staging s
                LEFT JOIN vintage v
                       ON v.series_id = ?
                      AND v.reference_date = s.reference_date
                      AND v.as_of_date = ?
                      AND v.source = ?
                WHERE v.series_id IS NULL
                RETURNING series_id
            """, [series_id, asof, source, resolved_hash,
                   series_id, asof, source]).fetchall()
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

        rows_inserted = len(inserted)
        return IngestResult(
            series_id=series_id,
            source=source,
            as_of_date=asof,
            rows_submitted=len(rows),
            rows_inserted=rows_inserted,
            rows_duplicate=len(rows) - rows_inserted,
        )

    def get_vintage(
        self,
        series_id: str,
        as_of_date: date | str,
        source: str | None = None,
    ) -> pd.Series:
        """Return the series as known at `as_of_date`.

        For each reference_date, picks the row with the largest as_of_date
        that is ≤ `as_of_date`.  Returns a pandas Series indexed by
        reference_date (DatetimeIndex), sorted ascending.
        """
        asof = _as_date(as_of_date)
        params = [series_id, asof]
        extra = ""
        if source is not None:
            extra = " AND source = ?"
            params.append(source)
        df = self._conn.execute(f"""
            WITH ranked AS (
                SELECT reference_date, value,
                       ROW_NUMBER() OVER (
                           PARTITION BY reference_date
                           ORDER BY as_of_date DESC, ingestion_ts DESC
                       ) AS rn
                FROM vintage
                WHERE series_id = ?
                  AND as_of_date <= ?
                  {extra}
            )
            SELECT reference_date, value
            FROM ranked
            WHERE rn = 1
            ORDER BY reference_date
        """, params).fetchdf()
        if df.empty:
            return pd.Series(dtype=float, name=series_id)
        df["reference_date"] = pd.to_datetime(df["reference_date"])
        s = df.set_index("reference_date")["value"]
        s.name = series_id
        return s

    def snapshot(
        self,
        as_of_date: date | str,
        series_ids: Iterable[str] | None = None,
    ) -> dict[str, pd.Series]:
        """Return {series_id: Series} of all known series at `as_of_date`."""
        if series_ids is None:
            series_ids = self.list_series()
        return {sid: self.get_vintage(sid, as_of_date) for sid in series_ids}

    def revisions(
        self,
        series_id: str,
        reference_date: date | str,
    ) -> pd.DataFrame:
        """Full revision history for a single (series_id, reference_date)."""
        ref = _as_date(reference_date)
        return self._conn.execute("""
            SELECT as_of_date, value, source, source_hash, ingestion_ts
            FROM vintage
            WHERE series_id = ? AND reference_date = ?
            ORDER BY as_of_date, ingestion_ts
        """, [series_id, ref]).fetchdf()

    def list_series(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT series_id FROM vintage ORDER BY series_id"
        ).fetchall()
        return [r[0] for r in rows]

    def list_sources(self, series_id: str | None = None) -> list[str]:
        if series_id is None:
            q = "SELECT DISTINCT source FROM vintage ORDER BY source"
            rows = self._conn.execute(q).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT DISTINCT source FROM vintage WHERE series_id = ? ORDER BY source",
                [series_id],
            ).fetchall()
        return [r[0] for r in rows]

    def row_count(self) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM vintage").fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "VintageStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _normalize_observations(
    observations: Iterable[tuple[date | str, float]] | pd.Series,
) -> Iterable[tuple[date, float]]:
    """Normalize an iterable or Series into [(date, float), ...] tuples."""
    if isinstance(observations, pd.Series):
        for idx, v in observations.dropna().items():
            yield _as_date(idx), float(v)
        return
    for ref, value in observations:
        if value is None or pd.isna(value):
            continue
        yield _as_date(ref), float(value)
