"""Phase 3.2 — ingest restaurants FRED panel into the vintage store.

Adds the food-cogs / labor / rent / utilities / menu / traffic series
needed for the 6-variable restaurants transmission VAR.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.ingest.fred import (    # noqa: E402
    FredSeriesSpec, ingest_fred_panel,
)
from thales.vintage import VintageStore    # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"


RESTAURANT_PANEL: list[FredSeriesSpec] = [
    FredSeriesSpec("WPSFD49207", "restaurant_food_cogs",
                       "PPI: Final demand foods (upstream food cost)", 30),
    FredSeriesSpec("CES7000000008", "restaurant_labor",
                       "L&H avg hourly earnings (restaurant wages)", 30),
    FredSeriesSpec("PCU531120531120", "restaurant_rent",
                       "PPI: Lessors of nonresidential buildings", 30),
    FredSeriesSpec("CUSR0000SEHF", "restaurant_utilities",
                       "CPI: Energy services (utilities)", 30),
    FredSeriesSpec("CUSR0000SEFV", "restaurant_menu",
                       "CPI: Food away from home (menu / output side)", 30),
    FredSeriesSpec("RSFSDP", "restaurant_traffic",
                       "Retail sales: Food services & drinking places (volume)", 30),
]


def main() -> None:
    print("=" * 78)
    print("Phase 3.2 — restaurants FRED ingest")
    print("=" * 78)
    env = (ROOT / ".env").read_text()
    for line in env.splitlines():
        if line.startswith("FRED_API_KEY="):
            os.environ["FRED_API_KEY"] = line.split("=", 1)[1].strip()
            break
    print(f"\nIngesting {len(RESTAURANT_PANEL)} series → {VINTAGE_DB}\n")
    with VintageStore(VINTAGE_DB) as store:
        ingest_fred_panel(store, specs=RESTAURANT_PANEL,
                              as_of_date=date.today(),
                              start="2010-01-01")


if __name__ == "__main__":
    main()
