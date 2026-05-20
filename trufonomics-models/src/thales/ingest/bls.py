"""BLS direct ingest — headline CPI + major subindexes via BLS public API v2.

Reads BLS_API_KEY from environment (or .env). Writes to vintage store with
source='bls_direct'. Separate from FRED because:

  * FRED mirrors BLS with its own release lag and may differ in final digits.
  * The headline target (CPIAUCSL on FRED = CUSR0000SA0 on BLS) should come
    from BLS for the nowcast comparison to be clean.

BLS API v2 limits (registered key):
  * 500 queries/day
  * 25 series per request
  * 20 years per request

For our 16-series subindex pull over 2010–2026, we batch into 2 requests (one
for 2010–2019, one for 2020–2026) to stay under the 20-year window.

Usage:
    python -m thales.ingest.bls
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from thales.vintage import IngestResult, VintageStore

ROOT = Path(__file__).resolve().parents[3]
SOURCE = "bls_direct"
BLS_ENDPOINT = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# Each request: up to 25 series, up to 20 years.
MAX_YEARS_PER_REQ = 20

# CPI subindex panel — BLS-native series IDs. CUSR = seasonally-adjusted CPI.
# These are the TARGET family for the nowcast; kept out of the FRED covariate
# panel to prevent leakage into Ridge baselines.


@dataclass(frozen=True)
class BLSSeriesSpec:
    series_id: str
    category: str
    description: str


BLS_PANEL: list[BLSSeriesSpec] = [
    # Headline + core
    BLSSeriesSpec("CUSR0000SA0",      "headline",   "All items, SA — headline CPI"),
    BLSSeriesSpec("CUSR0000SA0L1E",   "core",       "All items less food and energy, SA — core CPI"),
    BLSSeriesSpec("CUSR0000SA0L5",    "sticky",     "All items less food, shelter, energy, SA"),

    # Major groups
    BLSSeriesSpec("CUSR0000SAF1",     "food",       "Food, SA"),
    BLSSeriesSpec("CUSR0000SAF11",    "food",       "Food at home, SA"),
    BLSSeriesSpec("CUSR0000SEFV",     "food",       "Food away from home, SA"),
    BLSSeriesSpec("CUSR0000SA0E",     "energy",     "Energy, SA"),
    BLSSeriesSpec("CUSR0000SETB01",   "energy",     "Gasoline (all types), SA"),
    BLSSeriesSpec("CUSR0000SEHF01",   "energy",     "Electricity, SA"),
    BLSSeriesSpec("CUSR0000SEHF02",   "energy",     "Utility (piped) gas service, SA"),

    # Housing / shelter
    BLSSeriesSpec("CUSR0000SAH",      "housing",    "Housing, SA"),
    BLSSeriesSpec("CUSR0000SAH1",     "shelter",    "Shelter, SA"),
    BLSSeriesSpec("CUSR0000SAH2",     "housing",    "Fuels and utilities, SA"),
    BLSSeriesSpec("CUSR0000SAH3",     "housing",    "Household furnishings and operations, SA"),
    BLSSeriesSpec("CUSR0000SEHA",     "shelter",    "Rent of primary residence, SA"),
    BLSSeriesSpec("CUSR0000SEHC01",   "shelter",    "Owners' equivalent rent of residences, SA"),

    # Transport
    BLSSeriesSpec("CUSR0000SAT",      "transport",  "Transportation, SA"),
    BLSSeriesSpec("CUSR0000SETA01",   "transport",  "New vehicles, SA"),
    BLSSeriesSpec("CUSR0000SETA02",   "transport",  "Used cars and trucks, SA"),

    # Medical
    BLSSeriesSpec("CUSR0000SAM",      "medical",    "Medical care, SA"),
    BLSSeriesSpec("CUSR0000SAM1",     "medical",    "Medical care commodities, SA"),
    BLSSeriesSpec("CUSR0000SAM2",     "medical",    "Medical care services, SA"),

    # Other major categories
    BLSSeriesSpec("CUSR0000SAA",      "apparel",    "Apparel, SA"),
    BLSSeriesSpec("CUSR0000SAR",      "recreation", "Recreation, SA"),
    BLSSeriesSpec("CUSR0000SAE",      "edu_comm",   "Education and communication, SA"),
    BLSSeriesSpec("CUSR0000SEFW",     "alcohol",    "Alcoholic beverages, SA"),
    BLSSeriesSpec("CUSR0000SAG",      "other",      "Other goods and services, SA"),

    # Core-CPI clean decomposition (added 2026-05-17)
    # Core CPI = Goods-less-food-energy + Services-less-energy-services
    BLSSeriesSpec("CUSR0000SACL1E",   "core_goods",    "Commodities less food and energy commodities, SA"),
    BLSSeriesSpec("CUSR0000SASLE",    "core_services", "Services less energy services, SA"),
    # Headline goods/services aggregates (cross-check)
    BLSSeriesSpec("CUSR0000SAC",      "all_goods",     "All commodities, SA"),
    BLSSeriesSpec("CUSR0000SAS",      "all_services",  "All services, SA"),
    # Transport detail (for finer core ablations later)
    BLSSeriesSpec("CUSR0000SETC",     "transport",  "Motor vehicle parts and equipment, SA"),
    BLSSeriesSpec("CUSR0000SETD",     "transport",  "Motor vehicle maintenance and repair, SA"),
    BLSSeriesSpec("CUSR0000SETE",     "transport",  "Motor vehicle insurance, SA"),
    BLSSeriesSpec("CUSR0000SETG",     "transport",  "Public transportation, SA"),

    # NSA (Not Seasonally Adjusted) equivalents — input for X-13ARIMA-SEATS
    # replication exercise. CUUR prefix = Consumer-price Urban URban (NSA).
    BLSSeriesSpec("CUUR0000SA0",      "headline_nsa", "All items, NSA — headline CPI"),
    BLSSeriesSpec("CUUR0000SA0L1E",   "core_nsa",     "All items less food and energy, NSA — core CPI"),
]


def _bls_key() -> str:
    key = os.environ.get("BLS_API_KEY")
    if not key:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("BLS_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise RuntimeError(
            "BLS_API_KEY not found. Set it in the environment or in "
            "trufonomics-models/.env"
        )
    return key


def _request_batch(series_ids: list[str], start_year: int, end_year: int,
                    key: str) -> dict:
    payload = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
        "registrationkey": key,
    }
    r = requests.post(
        BLS_ENDPOINT,
        data=json.dumps(payload),
        headers={"Content-type": "application/json"},
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("status") != "REQUEST_SUCCEEDED":
        msg = body.get("message") or body.get("status")
        raise RuntimeError(f"BLS API error: {msg}")
    return body


def _month_end(year: int, period: str) -> pd.Timestamp:
    """BLS periods: M01..M12 (monthly), M13 (annual). We skip M13."""
    if period == "M13":
        return pd.NaT
    month = int(period[1:])
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def _to_series(results_item: dict) -> pd.Series:
    rows = []
    for d in results_item.get("data", []):
        ts = _month_end(int(d["year"]), d["period"])
        if pd.isna(ts):
            continue
        try:
            v = float(d["value"])
        except ValueError:
            continue
        rows.append((ts, v))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({ts: v for ts, v in rows}).sort_index()
    return s


def fetch_bls_panel(specs: Iterable[BLSSeriesSpec],
                      start_year: int = 2010,
                      end_year: int | None = None,
                      ) -> dict[str, pd.Series]:
    """Pull all specs from BLS. Batches by year-range to stay within 20y limit."""
    key = _bls_key()
    end_year = end_year or date.today().year
    series_ids = [s.series_id for s in specs]

    # Batch in chunks of up to MAX_YEARS_PER_REQ and 25 series at a time
    out: dict[str, list[pd.Series]] = {sid: [] for sid in series_ids}
    for chunk_start in range(start_year, end_year + 1, MAX_YEARS_PER_REQ):
        chunk_end = min(chunk_start + MAX_YEARS_PER_REQ - 1, end_year)
        for i in range(0, len(series_ids), 25):
            batch = series_ids[i:i + 25]
            print(f"  requesting {len(batch)} series "
                  f"[{chunk_start}-{chunk_end}]...")
            body = _request_batch(batch, chunk_start, chunk_end, key)
            for item in body["Results"]["series"]:
                sid = item["seriesID"]
                s = _to_series(item)
                if not s.empty:
                    out[sid].append(s)

    merged = {sid: pd.concat(parts).sort_index()
              for sid, parts in out.items() if parts}
    # Deduplicate any overlap across chunks
    for sid, s in merged.items():
        merged[sid] = s[~s.index.duplicated(keep="last")]
    return merged


def ingest_bls_panel(
    store: VintageStore,
    specs: Iterable[BLSSeriesSpec] = BLS_PANEL,
    as_of_date: date | str | None = None,
    start_year: int = 2010,
    end_year: int | None = None,
) -> dict[str, IngestResult]:
    if as_of_date is None:
        as_of_date = date.today()
    specs = list(specs)
    print(f"  {len(specs)} series from BLS, {start_year}–{end_year or date.today().year}")
    data = fetch_bls_panel(specs, start_year=start_year, end_year=end_year)
    results: dict[str, IngestResult] = {}
    for spec in specs:
        s = data.get(spec.series_id)
        if s is None or s.empty:
            print(f"  [skip] {spec.series_id}: empty")
            continue
        res = store.ingest(
            series_id=spec.series_id,
            observations=s,
            as_of_date=as_of_date,
            source=SOURCE,
        )
        print(f"  {spec.series_id:<18s} {res.rows_inserted:>4d} inserted  "
              f"({spec.category}: {spec.description})")
        results[spec.series_id] = res
    return results


def main() -> None:
    db_path = ROOT / "data" / "vintage_store" / "thales.duckdb"
    print(f"Opening vintage store: {db_path}")
    with VintageStore(db_path) as store:
        results = ingest_bls_panel(store)
    total = sum(r.rows_inserted for r in results.values())
    print(f"\nDone. {len(results)}/{len(BLS_PANEL)} series; {total:,} new rows.")


if __name__ == "__main__":
    main()
