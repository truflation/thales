"""Phase 3.3 — ingest FRED panel for the 4 additional verticals.

12 series across retail, healthcare, real estate, manufacturing. Some
overlap with existing ingests (utilities, rent, T&W wages from Phase
3.1/3.2 are reused). Only NEW series added here.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.ingest.fred import FredSeriesSpec, ingest_fred_panel  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"


PANEL: list[FredSeriesSpec] = [
    # Retail mid-market
    FredSeriesSpec("CES4200000008", "retail",
                       "Avg hourly earnings, Retail Trade", 30),
    FredSeriesSpec("RSGMS", "retail",
                       "Retail sales: General merchandise stores", 30),
    FredSeriesSpec("PCU423423", "retail",
                       "PPI: Wholesale trade — durable goods wholesalers", 30),

    # Healthcare operators
    FredSeriesSpec("CES6500000008", "healthcare",
                       "Avg hourly earnings, Health Care & Social Assistance", 30),
    FredSeriesSpec("CUSR0000SAM2", "healthcare",
                       "CPI: Medical care services", 30),
    FredSeriesSpec("PCU325412325412", "healthcare",
                       "PPI: Pharmaceutical preparation manufacturing", 30),

    # Real estate operators
    FredSeriesSpec("CES5500000008", "real_estate",
                       "Avg hourly earnings, Financial Activities (incl R/E)", 30),
    FredSeriesSpec("WPUSI012011", "real_estate",
                       "PPI: Construction materials", 30),
    FredSeriesSpec("CUUR0000SAH1", "real_estate",
                       "CPI: Shelter (output / pricing-side)", 30),
    FredSeriesSpec("USCONS", "real_estate",
                       "Construction employment (activity proxy)", 30),

    # Manufacturing
    FredSeriesSpec("CES3000000008", "manufacturing",
                       "Avg hourly earnings, Manufacturing", 30),
    FredSeriesSpec("PPIIDC", "manufacturing",
                       "PPI: Industrial commodities", 30),
]


def main() -> None:
    print("=" * 78)
    print("Phase 3.3 — additional vertical FRED ingest")
    print("=" * 78)
    env = (ROOT / ".env").read_text()
    for line in env.splitlines():
        if line.startswith("FRED_API_KEY="):
            os.environ["FRED_API_KEY"] = line.split("=", 1)[1].strip()
            break
    print(f"\nIngesting {len(PANEL)} series → {VINTAGE_DB}\n")
    with VintageStore(VINTAGE_DB) as store:
        ingest_fred_panel(store, specs=PANEL,
                              as_of_date=date.today(),
                              start="2010-01-01")


if __name__ == "__main__":
    main()
