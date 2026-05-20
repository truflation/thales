"""ALFRED (Archival FRED) ingest — true point-in-time vintages.

Replaces the single-as_of-tag approach in ``fred.py`` with proper historical
vintages: every release and every revision of every series, each tagged with
the ``realtime_start`` date (i.e., the as_of_date when that value was first
or last known to the market).

Uses the same FRED_API_KEY. The vintage behaviour is unlocked by passing
``realtime_start=1776-07-04`` and ``realtime_end=9999-12-31`` to the
standard `/fred/series/observations` endpoint; FRED treats those as
sentinels meaning "give me every vintage of every observation."

Data shape per series:
  Each (reference_date) can have multiple rows, one per vintage. We write
  one vintage-row to the store per FRED observation:
      source = 'fred_alfred'
      reference_date = observation.date
      as_of_date     = observation.realtime_start   <-- the real as-of
      value          = observation.value

Existing ``source='fred'`` rows (as_of=today fallback) stay in the store
for backward compatibility; new work should query with
``source='fred_alfred'`` for vintage-correct results.

Usage:
    python -m thales.ingest.fred_alfred                    # full panel
    python -m thales.ingest.fred_alfred --series UNRATE    # one series
    python -m thales.ingest.fred_alfred --limit 5          # first 5 (smoke)
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from thales.ingest.fred import FRED_PANEL, FredSeriesSpec
from thales.vintage import IngestResult, VintageStore

ROOT = Path(__file__).resolve().parents[3]
FRED_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
SOURCE = "fred_alfred"
SOURCE_TARGET = "fred_alfred_target"
START_DATE = "2010-01-01"
SLEEP_BETWEEN = 0.15  # courtesy throttle; free-tier is 120 req/min, we need ~47 at most


# Official inflation targets — distinct source tag so leakage prevention is
# queryable: feature builders MUST exclude `source = 'fred_alfred_target'`.
# These four are the headline / core CPI and PCE price indexes from BEA/BLS
# mirrored on FRED. Vintage-correct revisions matter for PCE in particular
# (annual NIPA revisions revise the entire history).
FRED_TARGET_PANEL: tuple[FredSeriesSpec, ...] = (
    FredSeriesSpec("CPIAUCSL", "target", "BLS Consumer Price Index, all items, SA", 13),
    FredSeriesSpec("CPILFESL", "target", "BLS Core CPI (less food + energy), SA", 13),
    FredSeriesSpec("PCEPI",    "target", "BEA PCE price index, all items, SA",     30),
    FredSeriesSpec("PCEPILFE", "target", "BEA Core PCE (less food + energy), SA",  30),
)


def _fred_api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("FRED_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise RuntimeError("FRED_API_KEY not set")
    return key


@dataclass
class VintageObs:
    reference_date: date
    as_of_date: date
    value: float


class FREDVintageCapError(RuntimeError):
    """Raised when FRED rejects a request for exceeding the 2000 vintage-date cap."""


def _fetch_one_window(series_id: str, key: str,
                       obs_start: str, obs_end: str,
                       rt_start: str, rt_end: str,
                       max_retries: int = 3) -> list[VintageObs]:
    """Single API call for one realtime-window slice. Retries 500s; raises on 400."""
    params = {
        "series_id": series_id, "api_key": key, "file_type": "json",
        "observation_start": obs_start, "observation_end": obs_end,
        "realtime_start": rt_start, "realtime_end": rt_end,
    }
    last_err = None
    for attempt in range(max_retries):
        r = requests.get(FRED_ENDPOINT, params=params, timeout=60)
        if r.status_code == 400:
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            msg = body.get("error_message", r.text[:200])
            if "vintage dates" in msg and "maximum" in msg:
                raise FREDVintageCapError(msg)
            raise RuntimeError(f"FRED 400: {msg}")
        if 500 <= r.status_code < 600:
            last_err = f"status {r.status_code}"
            time.sleep(0.5 * (attempt + 1))
            continue
        r.raise_for_status()
        body = r.json()
        out: list[VintageObs] = []
        for obs in body.get("observations", []):
            raw_val = obs.get("value", ".")
            if raw_val in (".", "", None):
                continue
            try:
                v = float(raw_val)
                ref = date.fromisoformat(obs["date"])
                asof = date.fromisoformat(obs["realtime_start"])
            except (TypeError, ValueError, KeyError):
                continue
            out.append(VintageObs(ref, asof, v))
        return out
    raise RuntimeError(f"FRED failed after {max_retries} retries: {last_err}")


def fetch_vintages(series_id: str, key: str,
                    start: str = START_DATE) -> list[VintageObs]:
    """Pull every (reference_date × as_of_date) vintage pair for a series.

    Strategy: try a single full-range call first (best results — gets the
    true first-publication date for every observation). Fall back to
    year-chunking only if FRED trips the 2000-vintage-dates cap (which
    happens for daily series). Year-chunked results lose some precision
    on the exact first-publication date at chunk boundaries, but are
    still vintage-correct for downstream use.
    """
    obs_end = date.today().isoformat()
    # Cap realtime_end at today (FRED rejects future dates except the 9999 sentinel)
    # so we use the sentinel to ensure we capture any values released between
    # the date FRED considers "today" and the query time.
    try:
        return _fetch_one_window(
            series_id, key,
            obs_start=start, obs_end=obs_end,
            rt_start="1776-07-04", rt_end="9999-12-31",
        )
    except FREDVintageCapError:
        pass

    # Fallback: year-chunked realtime window
    all_obs: dict[tuple[date, date], float] = {}
    start_year = pd.Timestamp(start).year
    end_year = date.today().year
    today_iso = date.today().isoformat()
    for yr in range(start_year, end_year + 1):
        rt_start = f"{yr}-01-01"
        rt_end = today_iso if yr == end_year else f"{yr}-12-31"
        chunk = _fetch_one_window(
            series_id, key,
            obs_start=start, obs_end=obs_end,
            rt_start=rt_start, rt_end=rt_end,
        )
        for v in chunk:
            all_obs[(v.reference_date, v.as_of_date)] = v.value
    return [VintageObs(ref, asof, val)
            for (ref, asof), val in all_obs.items()]


def ingest_alfred_panel(
    store: VintageStore,
    specs: Iterable[FredSeriesSpec] = FRED_PANEL,
    start: str = START_DATE,
    source: str = SOURCE,
) -> dict[str, IngestResult | str]:
    """Pull every series' full vintage history and write to the store.

    ``source`` defaults to ``'fred_alfred'`` for the macro covariate panel.
    Pass ``SOURCE_TARGET`` for the official-target panel so the rows are
    distinguishable at query time (leakage prevention).

    Returns {series_id: IngestResult or error-string}.
    """
    key = _fred_api_key()
    results: dict[str, IngestResult | str] = {}

    for spec in specs:
        sid = spec.series_id
        t0 = time.monotonic()
        try:
            vintages = fetch_vintages(sid, key, start=start)
        except Exception as exc:   # noqa: BLE001
            results[sid] = f"{type(exc).__name__}: {exc}"
            print(f"  [err]  {sid:<18s} {results[sid]}")
            time.sleep(SLEEP_BETWEEN)
            continue

        # Group by as_of_date so each ingest call is one batch per vintage
        per_asof: dict[date, list[tuple[date, float]]] = defaultdict(list)
        for v in vintages:
            per_asof[v.as_of_date].append((v.reference_date, v.value))

        total_inserted = 0
        total_dup = 0
        for asof, obs in per_asof.items():
            try:
                res = store.ingest(
                    series_id=sid, observations=obs,
                    as_of_date=asof, source=source,
                )
            except ValueError as e:
                results[sid] = f"conflict on as_of={asof}: {e}"
                break
            total_inserted += res.rows_inserted
            total_dup += res.rows_duplicate
        else:
            dt = time.monotonic() - t0
            results[sid] = IngestResult(
                series_id=sid, source=source, as_of_date=date.today(),
                rows_submitted=len(vintages),
                rows_inserted=total_inserted,
                rows_duplicate=total_dup,
            )
            print(f"  {sid:<18s} {len(vintages):>5d} vintages  "
                  f"{len(per_asof):>4d} as_of dates  "
                  f"{total_inserted:>5d} inserted  ({dt:.1f}s)  "
                  f"({spec.category})")
        time.sleep(SLEEP_BETWEEN)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", nargs="*",
                        help="restrict to these FRED series ids")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N series only (smoke test)")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--targets", action="store_true",
                        help="ingest the official-target panel (CPIAUCSL, "
                              "CPILFESL, PCEPI, PCEPILFE) under "
                              f"source='{SOURCE_TARGET}' instead of the "
                              "macro covariate panel")
    args = parser.parse_args()

    if args.targets:
        specs_pool = list(FRED_TARGET_PANEL)
        source = SOURCE_TARGET
        panel_label = "TARGET"
    else:
        specs_pool = list(FRED_PANEL)
        source = SOURCE
        panel_label = "covariate"

    specs = specs_pool
    if args.series:
        wanted = set(args.series)
        specs = [s for s in specs if s.series_id in wanted]
    if args.limit:
        specs = specs[: args.limit]

    print(f"ALFRED {panel_label} ingest: {len(specs)} series, "
          f"from {args.start}, source={source}")
    db = ROOT / "data" / "vintage_store" / "thales.duckdb"
    with VintageStore(db) as store:
        results = ingest_alfred_panel(store, specs, start=args.start,
                                          source=source)

    ok = [k for k, v in results.items() if isinstance(v, IngestResult)]
    err = [k for k, v in results.items() if isinstance(v, str)]
    rows = sum(v.rows_inserted for v in results.values() if isinstance(v, IngestResult))
    print(f"\n=== {len(ok)} ok, {len(err)} failed, {rows:,} new rows ===")


if __name__ == "__main__":
    main()
