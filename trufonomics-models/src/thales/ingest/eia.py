"""EIA ingest — energy spot prices and regional retail gasoline.

Fills the granularity gap FRED can't: PADD-regional retail gasoline needed
for the Commodity pass-through archetype (docs/planning/01-architecture.md
§Archetype 1). Also gives us EIA-direct WTI / Brent / Henry Hub which are
the canonical upstream commodity drivers for Utilities and fuel portions of
Transportation.

Usage:
    python -m thales.ingest.eia

API reference: https://www.eia.gov/opendata/documentation.php
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from thales.vintage import IngestResult, VintageStore

ROOT = Path(__file__).resolve().parents[3]
SOURCE = "eia"
BASE_URL = "https://api.eia.gov/v2"


@dataclass(frozen=True)
class EIASeriesSpec:
    route: str               # e.g. "/petroleum/pri/spt/data/"
    series_id: str            # facets[series][] value
    frequency: str            # "daily" | "weekly" | "monthly"
    category: str
    description: str


# Priority energy panel
EIA_PANEL: list[EIASeriesSpec] = [
    # Upstream crude spots — feeds Commodity pass-through archetype
    EIASeriesSpec("/petroleum/pri/spt/data/",  "RWTC",  "daily",
                   "crude",    "WTI Cushing spot, $/bbl"),
    EIASeriesSpec("/petroleum/pri/spt/data/",  "RBRTE", "daily",
                   "crude",    "Brent FOB spot, $/bbl"),
    # Natural gas
    EIASeriesSpec("/natural-gas/pri/fut/data/", "RNGWHHD", "daily",
                   "natgas",   "Henry Hub natural gas spot, $/MMBtu"),
    # Retail gasoline — US total (overlap with FRED's GASREGW, for cross-check)
    EIASeriesSpec("/petroleum/pri/gnd/data/", "EMM_EPMR_PTE_NUS_DPG", "weekly",
                   "gasoline", "US regular retail gasoline, $/gal"),
    # PADD regional — not in FRED, genuinely new information
    EIASeriesSpec("/petroleum/pri/gnd/data/", "EMM_EPMR_PTE_R10_DPG", "weekly",
                   "gasoline", "PADD 1 (East Coast) regular retail, $/gal"),
    EIASeriesSpec("/petroleum/pri/gnd/data/", "EMM_EPMR_PTE_R50_DPG", "weekly",
                   "gasoline", "PADD 5 (West Coast) regular retail, $/gal"),
]


def _eia_key() -> str:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("EIA_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise RuntimeError(
            "EIA_API_KEY not found. Set it in env or in trufonomics-models/.env"
        )
    return key


def fetch_series(spec: EIASeriesSpec, key: str,
                  start: str = "2010-01-01") -> pd.Series:
    """Fetch one EIA series via v2 API, return a Series indexed by date."""
    url = BASE_URL + spec.route
    params = {
        "api_key": key,
        "frequency": spec.frequency,
        "data[0]": "value",
        "facets[series][]": spec.series_id,
        "start": start,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    rows: list[tuple[pd.Timestamp, float]] = []
    offset = 0
    while True:
        params["offset"] = offset
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        body = r.json()
        resp = body.get("response", {})
        data = resp.get("data", [])
        if not data:
            break
        for d in data:
            try:
                ts = pd.Timestamp(d["period"])
                v = float(d["value"])
            except (ValueError, TypeError, KeyError):
                continue
            rows.append((ts, v))
        if len(data) < int(params["length"]):
            break  # last page
        offset += int(params["length"])

    if not rows:
        return pd.Series(dtype=float, name=spec.series_id)
    # EIA occasionally returns duplicate (period, value) rows across pages
    s = pd.Series({ts: v for ts, v in rows}).sort_index()
    s.name = spec.series_id
    return s


def ingest_eia_panel(
    store: VintageStore,
    specs: Iterable[EIASeriesSpec] = EIA_PANEL,
    as_of_date: date | str | None = None,
    start: str = "2010-01-01",
) -> dict[str, IngestResult]:
    key = _eia_key()
    if as_of_date is None:
        as_of_date = date.today()
    results: dict[str, IngestResult] = {}
    for spec in specs:
        try:
            s = fetch_series(spec, key, start=start)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {spec.series_id}: {type(e).__name__}: {e}")
            continue
        if s.empty:
            print(f"  [skip] {spec.series_id}: empty")
            continue
        res = store.ingest(
            series_id=spec.series_id,
            observations=s,
            as_of_date=as_of_date,
            source=SOURCE,
        )
        print(f"  {spec.series_id:<22s} {res.rows_inserted:>5d} inserted  "
              f"({spec.frequency}, {spec.category}: {spec.description})")
        results[spec.series_id] = res
    return results


def main() -> None:
    db_path = ROOT / "data" / "vintage_store" / "thales.duckdb"
    print(f"Opening vintage store: {db_path}")
    with VintageStore(db_path) as store:
        print(f"Pulling {len(EIA_PANEL)} EIA series...")
        results = ingest_eia_panel(store)
    total = sum(r.rows_inserted for r in results.values())
    print(f"\nDone. {len(results)}/{len(EIA_PANEL)} series; {total:,} new rows.")


if __name__ == "__main__":
    main()
