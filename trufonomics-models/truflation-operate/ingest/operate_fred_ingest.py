"""Truflation Operate — FRED ingest for import/export transmission VAR.

Pulls four families of series that the existing FRED panel does not
already cover (or covers only partially):

  * **Retail diesel.** ``GASDESW`` — US On-Highway Diesel weekly retail
    price, EIA-sourced via FRED. Replaces the PPI diesel proxy
    (correlation 0.74 monthly with retail) used in the earlier logistics
    BVAR. This is the right operator-facing input.

  * **Wholesale / PPI fuel context.** ``WPU057303`` (PPI for No. 2
    Diesel) — kept for cross-check, not as the primary input.

  * **Freight cost — PPI long-distance trucking.** ``PCU484121484121``
    (Producer Price Index by Industry, Long-Distance Truckload Trucking).
    First-cut freight signal; route-level spot indices (Drewry WCI,
    Freightos FBX) are paid-tier upgrades documented in the README.

  * **FX rates.** Five major exchange rates relevant to US-facing
    importers — EUR, CNY, MXN, CAD, INR. Each is a US-Dollars-vs-Local
    or Local-vs-US-Dollar spot exchange rate; tagged with a `category`
    that flags directionality for downstream code.

All series are tagged ``source='operate_fred'`` in the vintage store
to separate them from the general macro `fred` panel; this also means
they can be reingested independently without touching the larger panel.

Run::

    uv run python -m truflation_operate.ingest.operate_fred_ingest
        # or directly:
    uv run python truflation-operate/ingest/operate_fred_ingest.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thales.ingest.fred import _fred_client, fetch_series    # noqa: E402
from thales.vintage import IngestResult, VintageStore       # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
SOURCE = "operate_fred"


@dataclass(frozen=True)
class OperateSeriesSpec:
    series_id: str
    category: str
    description: str


# ── Series panel for Truflation Operate (import/export vertical, v1) ──
OPERATE_PANEL: list[OperateSeriesSpec] = [
    # Diesel (retail + wholesale)
    OperateSeriesSpec("GASDESW",         "diesel_retail",
                          "US On-Highway Diesel weekly retail $/gal (EIA via FRED)"),
    OperateSeriesSpec("WPU057303",       "diesel_wholesale",
                          "PPI: No. 2 Diesel Fuel (wholesale, monthly)"),

    # Freight cost
    OperateSeriesSpec("PCU484121484121", "freight_trucking",
                          "PPI: Long-Distance Truckload Trucking (monthly)"),

    # FX rates relevant to US-facing importers. DEXUSEU is USD-per-EUR
    # (rises ⇒ EUR strengthens). DEXCHUS, DEXMXUS, DEXCAUS, DEXINUS are
    # local-per-USD (rises ⇒ local currency weakens vs USD).
    OperateSeriesSpec("DEXUSEU", "fx_usd_per_local",
                          "EUR/USD spot — US Dollars to Euro (daily)"),
    OperateSeriesSpec("DEXCHUS", "fx_local_per_usd",
                          "USD/CNY spot — Chinese Yuan per US Dollar (daily)"),
    OperateSeriesSpec("DEXMXUS", "fx_local_per_usd",
                          "USD/MXN spot — Mexican Pesos per US Dollar (daily)"),
    OperateSeriesSpec("DEXCAUS", "fx_local_per_usd",
                          "USD/CAD spot — Canadian Dollars per US Dollar (daily)"),
    OperateSeriesSpec("DEXINUS", "fx_local_per_usd",
                          "USD/INR spot — Indian Rupees per US Dollar (daily)"),
]


def main() -> None:
    print("=" * 78)
    print("Truflation Operate — FRED ingest (diesel, freight, FX)")
    print("=" * 78)
    print(f"\nVintage store: {VINTAGE_DB}")
    print(f"Source tag:    {SOURCE}")
    print(f"Panel:         {len(OPERATE_PANEL)} series\n")

    fred = _fred_client()
    today = date.today()
    results: dict[str, IngestResult] = {}
    with VintageStore(VINTAGE_DB) as store:
        for spec in OPERATE_PANEL:
            try:
                s = fetch_series(fred, spec.series_id, start="2010-01-01")
            except Exception as e:    # noqa: BLE001
                print(f"  [skip] {spec.series_id}: {type(e).__name__}: {e}")
                continue
            if s.empty:
                print(f"  [skip] {spec.series_id}: no observations")
                continue
            res = store.ingest(
                series_id=spec.series_id,
                observations=s,
                as_of_date=today,
                source=SOURCE,
            )
            results[spec.series_id] = res
            print(f"  {spec.series_id:<18s} {res.rows_inserted:>5d} inserted "
                    f"{res.rows_duplicate:>4d} dup  "
                    f"({spec.category}: {spec.description})")

    total = sum(r.rows_inserted for r in results.values())
    print(f"\nDone. {len(results)}/{len(OPERATE_PANEL)} series; "
            f"{total:,} new rows.")


if __name__ == "__main__":
    main()
