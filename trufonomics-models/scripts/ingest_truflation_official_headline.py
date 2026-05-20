"""Ingest Truflation's official published US CPI series into the vintage
store.

Source: ``/Users/kluless/kairos/data/truflation/api/all_streams.parquet``
(kairos's existing Truflation API snapshot). We pick four series:

  * ``truflation_us_cpi_index``        — live US CPI level (revises)
  * ``truflation_us_cpi_yoy``           — live US CPI YoY
  * ``truflation_us_cpi_frozen_index`` — frozen vintage level (no revisions)
  * ``truflation_us_cpi_frozen_yoy``   — frozen vintage YoY

Writes to the vintage store under ``source='truf_network_published'``
so it's separate from the component streams ingested under
``source='truf_network'``.

The frozen series is the apples-to-apples comparator for backtests
because it never revises. The live series is the right input for
nowcasts that want to use the latest published reading.

Run::

    uv run python scripts/ingest_truflation_official_headline.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.vintage import VintageStore    # noqa: E402

KAIROS_PARQUET = Path("/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
SOURCE = "truf_network_published"

OFFICIAL_SERIES = {
    "truflation_us_cpi_index":         "truflation_us_cpi_index/truflation_us_cpi_index",
    "truflation_us_cpi_yoy":           "truflation_us_cpi_yoy/truflation_us_cpi_yoy",
    "truflation_us_cpi_frozen_index":  "truflation_us_cpi_frozen_index/truflation_us_cpi_frozen_index",
    "truflation_us_cpi_frozen_yoy":    "truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy",
}


def main() -> None:
    print("=" * 78)
    print("Ingest Truflation official published US CPI series")
    print("=" * 78)

    print(f"\nReading {KAIROS_PARQUET}…")
    pq = pd.read_parquet(KAIROS_PARQUET)
    pq["date"] = pd.to_datetime(pq["date"])
    pq = pq.set_index("date").sort_index()
    print(f"  parquet: {pq.shape[0]} rows × {pq.shape[1]} cols")

    today = date.today()
    print(f"\nIngesting into {VINTAGE_DB} (source='{SOURCE}', as_of_date={today}):")
    with VintageStore(VINTAGE_DB) as store:
        for new_sid, pq_col in OFFICIAL_SERIES.items():
            if pq_col not in pq.columns:
                print(f"  [skip] {pq_col} not in parquet")
                continue
            s = pq[pq_col].dropna()
            if s.empty:
                print(f"  [skip] {new_sid}: empty after dropna")
                continue
            res = store.ingest(
                series_id=new_sid,
                observations=s,
                as_of_date=today,
                source=SOURCE,
            )
            print(f"  {new_sid:<35s} {res.rows_inserted:>5d} rows  "
                    f"({s.index.min().date()} → {s.index.max().date()})")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
