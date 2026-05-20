"""One-off ingest: PCE sub-component price indexes from FRED ALFRED.

Pulls vintage-correct PCE component chain-type price indexes — Durable
Goods, Nondurable Goods, Services — into the vintage store. These are
the components needed for a BLS-native-CBDF-style PCE forecaster.

These are PCE *covariates* (decomposition inputs), not targets, so they
go in under source='fred_alfred'.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.ingest.fred import FredSeriesSpec    # noqa: E402
from thales.ingest.fred_alfred import (    # noqa: E402
    SOURCE,
    fetch_vintages,
    _fred_api_key,
)
from thales.vintage import VintageStore    # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"

# BEA PCE chain-type price indexes (monthly, 2012=100 base):
PCE_COMPONENTS = [
    FredSeriesSpec("DDURRG3M086SBEA", "pce_component",
                       "PCE: Durable Goods, chain-type price index", 30),
    FredSeriesSpec("DNDGRG3M086SBEA", "pce_component",
                       "PCE: Nondurable Goods, chain-type price index", 30),
    FredSeriesSpec("DSERRG3M086SBEA", "pce_component",
                       "PCE: Services, chain-type price index", 30),
]


def main() -> None:
    key = _fred_api_key()
    print(f"Ingesting {len(PCE_COMPONENTS)} PCE sub-component price indexes "
            f"to vintage store (source={SOURCE!r})")
    with VintageStore(VINTAGE_DB) as store:
        for spec in PCE_COMPONENTS:
            sid = spec.series_id
            print(f"  fetching {sid} …")
            t0 = time.monotonic()
            try:
                vintages = fetch_vintages(sid, key, start="2010-01-01")
            except Exception as e:    # noqa: BLE001
                print(f"    error: {type(e).__name__}: {e}")
                continue
            per_asof = defaultdict(list)
            for v in vintages:
                per_asof[v.as_of_date].append((v.reference_date, v.value))
            total = 0
            for asof, obs in per_asof.items():
                res = store.ingest(
                    series_id=sid, observations=obs,
                    as_of_date=asof, source=SOURCE,
                )
                total += res.rows_inserted
            dt = time.monotonic() - t0
            print(f"    {sid}: {len(vintages)} vintages, "
                    f"{len(per_asof)} as_of dates, {total} inserted "
                    f"({dt:.1f}s)")
    print("\nDone.")


if __name__ == "__main__":
    main()
