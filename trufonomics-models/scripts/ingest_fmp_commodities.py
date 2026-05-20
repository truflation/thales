"""One-shot ingest of FMP commodity panel into the vintage store."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.ingest.fmp import FMP_COMMODITY_PANEL, ingest_fmp_panel  # noqa: E402
from thales.vintage import VintageStore  # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"


def main() -> None:
    print("=" * 78)
    print("FMP commodity ingest")
    print("=" * 78)
    print(f"\nIngesting {len(FMP_COMMODITY_PANEL)} symbols → {VINTAGE_DB}\n")
    with VintageStore(VINTAGE_DB) as store:
        ingest_fmp_panel(store, as_of_date=date.today(), start="2010-01-01")


if __name__ == "__main__":
    main()
