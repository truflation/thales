"""Phase 3.1b — ingest logistics FRED panel into the vintage store.

Adds the labor / maintenance / volume / freight series needed for the
6-variable logistics transmission VAR. The fuel chain (DCOILWTICO,
GASREGW, DHHNGSP) is already in the store from earlier ingest.

Series added by this script:

  * CES4300000008  — Avg hourly earnings, Transportation & Warehousing
                       (the closest BLS NAICS aggregate covering
                       trucking; NAICS 484-only earnings aren't in FRED)
  * CES4348400001  — Trucking sector employment (NAICS 484, all)
  * GASDESW         — US retail diesel, weekly (EIA via FRED)
  * WPU057303       — PPI: Diesel fuel
  * CUSR0000SETD    — CPI: Vehicle maintenance & repair
  * PCU48414841     — PPI: Specialized freight (truck) trucking
  * TRUCKD11        — ATA Truck Tonnage Index (volume)
  * TRFVOLUSM227NFWA— TSA total freight volume (volume backup)
  * VMT             — Vehicle miles traveled (activity)

Each is tagged with ``source='fred'`` and ``as_of_date=today``.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.ingest.fred import (    # noqa: E402
    FredSeriesSpec, _fred_client, ingest_fred_panel,
)
from thales.vintage import VintageStore    # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"


LOGISTICS_PANEL: list[FredSeriesSpec] = [
    # Labor (no NAICS-484-only wage in FRED; T&W is the closest aggregate)
    FredSeriesSpec("CES4300000008", "logistics_labor",
                       "Avg hourly earnings, Transportation & Warehousing", 30),
    FredSeriesSpec("CES4348400001", "logistics_labor",
                       "Trucking sector employment (NAICS 484)", 30),

    # Fuel costs (diesel — primary cost variable for trucking)
    FredSeriesSpec("GASDESW", "logistics_fuel",
                       "US retail diesel, weekly (EIA)", 2),
    FredSeriesSpec("WPU057303", "logistics_fuel",
                       "PPI: Diesel fuel", 30),

    # Maintenance
    FredSeriesSpec("CUSR0000SETD", "logistics_maintenance",
                       "CPI: Vehicle maintenance & repair", 30),

    # Freight rates / pricing pass-through
    FredSeriesSpec("PCU48414841", "logistics_freight_rate",
                       "PPI: Specialized freight (truck) trucking", 30),

    # Activity / volume
    FredSeriesSpec("TRUCKD11", "logistics_volume",
                       "ATA Truck Tonnage Index", 30),
    FredSeriesSpec("TRFVOLUSM227NFWA", "logistics_volume",
                       "TSA total freight volume", 30),
    FredSeriesSpec("VMT", "logistics_volume",
                       "Vehicle miles traveled, all", 30),
]


def main() -> None:
    print("=" * 78)
    print("Phase 3.1b — logistics FRED ingest")
    print("=" * 78)

    # Load FRED API key from .env
    env = (ROOT / ".env").read_text()
    for line in env.splitlines():
        if line.startswith("FRED_API_KEY="):
            os.environ["FRED_API_KEY"] = line.split("=", 1)[1].strip()
            break

    print(f"\nIngesting {len(LOGISTICS_PANEL)} series → {VINTAGE_DB}")
    print()

    with VintageStore(VINTAGE_DB) as store:
        results = ingest_fred_panel(
            store, specs=LOGISTICS_PANEL,
            as_of_date=date.today(),
            start="2010-01-01")

    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"  {'series_id':<22s}  {'status':>10s}  {'n_inserted':>11s}  desc")
    print("  " + "-" * 100)
    for spec in LOGISTICS_PANEL:
        r = results.get(spec.series_id)
        status = "OK" if r and r.n_inserted > 0 else (
            "DUPE" if r else "MISSING")
        n = r.n_inserted if r else 0
        print(f"  {spec.series_id:<22s}  {status:>10s}  {n:>11d}  "
                f"{spec.description}")


if __name__ == "__main__":
    main()
