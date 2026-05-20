"""Cleveland Fed inflation nowcast scraper.

Parses the three tables on
https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting
and writes each cell into the vintage store.

Table structure:
    Monthly MoM:  month × {CPI, Core CPI, PCE, Core PCE}
    Monthly YoY:  month × {CPI, Core CPI, PCE, Core PCE}
    Quarterly SAAR: quarter × {CPI, Core CPI, PCE, Core PCE}

Series IDs emitted:
    clevfed_{cpi|corecpi|pce|corepce}_{mom|yoy|saar}

Reference dates:
  * monthly cells: reference_date = month-end
  * quarterly cells: reference_date = quarter-end

Design: a daily cron runs this script. Each run is a fresh scrape tagged
with today's as_of_date, and append-only conflict detection in the vintage
store prevents duplicates on reruns. Over time this builds a full daily
archive of the Cleveland Fed nowcast vintages — which is what the Thales
planning doc §04-data-sources calls for as the primary comparator.

Historical backfill (pre-today archive) requires the DevTools JSON endpoint
option from §04 — not implemented in this module; add later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

from thales.vintage import IngestResult, VintageStore

ROOT = Path(__file__).resolve().parents[3]
SOURCE = "clevfed_scrape"
URL = "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"

MEASURE_MAP = {
    "CPI": "cpi",
    "Core CPI": "corecpi",
    "PCE": "pce",
    "Core PCE": "corepce",
}


@dataclass
class ParsedCell:
    series_id: str
    reference_date: date
    value: float


def _month_end(text: str) -> date:
    """'April 2026' → 2026-04-30."""
    ts = pd.Timestamp(text.strip()) + pd.offsets.MonthEnd(0)
    return ts.date()


def _quarter_end(text: str) -> date:
    """'2026:Q1' → 2026-03-31."""
    m = re.match(r"(\d{4}):?\s*Q?(\d)", text.strip())
    if not m:
        raise ValueError(f"can't parse quarter {text!r}")
    year = int(m.group(1))
    q = int(m.group(2))
    month_end = {1: 3, 2: 6, 3: 9, 4: 12}[q]
    ts = pd.Timestamp(year=year, month=month_end, day=1) + pd.offsets.MonthEnd(0)
    return ts.date()


def _parse_table(table, freq_suffix: str, period_parser) -> list[ParsedCell]:
    """Parse one HTML table into ParsedCells.

    freq_suffix: 'mom', 'yoy', or 'saar'
    period_parser: function str → date (month-end or quarter-end)
    """
    rows = table.find_all("tr")
    if not rows:
        return []
    header = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
    # Expected header: [Period, CPI, Core CPI, PCE, Core PCE, Updated]
    try:
        period_col = 0
        measure_cols = {MEASURE_MAP[h]: i for i, h in enumerate(header) if h in MEASURE_MAP}
    except KeyError:
        return []
    if not measure_cols:
        return []

    cells: list[ParsedCell] = []
    for tr in rows[1:]:
        tds = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
        if len(tds) < len(header) or tds[0].startswith("Note"):
            continue
        try:
            ref = period_parser(tds[period_col])
        except (ValueError, IndexError):
            continue
        for measure_key, col_idx in measure_cols.items():
            if col_idx >= len(tds):
                continue
            raw = tds[col_idx]
            if not raw or raw in {"-", "—", "N/A"}:
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            cells.append(ParsedCell(
                series_id=f"clevfed_{measure_key}_{freq_suffix}",
                reference_date=ref,
                value=val,
            ))
    return cells


def scrape_nowcasts(url: str = URL) -> list[ParsedCell]:
    """Fetch and parse the Cleveland Fed page. Returns all non-blank cells."""
    r = requests.get(url, timeout=30, headers={
        "User-Agent": "Mozilla/5.0 (compatible; Thales/0.1)"
    })
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    tables = soup.find_all("table")

    # Heuristic: first table with CPI/Core/PCE in header is MoM; second is YoY;
    # third (headed by 'Quarter') is quarterly SAAR.
    monthly_tables = []
    quarterly_tables = []
    for t in tables:
        headers = [th.get_text(strip=True) for th in t.find_all("th")]
        if "CPI" not in headers:
            continue
        if "Quarter" in headers:
            quarterly_tables.append(t)
        elif "Month" in headers:
            monthly_tables.append(t)

    cells: list[ParsedCell] = []
    if len(monthly_tables) >= 1:
        cells += _parse_table(monthly_tables[0], "mom", _month_end)
    if len(monthly_tables) >= 2:
        cells += _parse_table(monthly_tables[1], "yoy", _month_end)
    for t in quarterly_tables:
        cells += _parse_table(t, "saar", _quarter_end)
    return cells


def ingest_cleveland_fed(
    store: VintageStore,
    as_of_date: date | str | None = None,
) -> dict[str, IngestResult]:
    as_of = as_of_date or date.today()
    cells = scrape_nowcasts()
    if not cells:
        print("  [warn] no cells parsed — page layout may have changed")
        return {}

    # Group by series_id and ingest per-series
    by_series: dict[str, list[tuple[date, float]]] = {}
    for c in cells:
        by_series.setdefault(c.series_id, []).append((c.reference_date, c.value))

    results: dict[str, IngestResult] = {}
    for sid, obs in sorted(by_series.items()):
        res = store.ingest(series_id=sid, observations=obs,
                             as_of_date=as_of, source=SOURCE)
        print(f"  {sid:<26s} {len(obs):>2d} cells, {res.rows_inserted:>2d} inserted")
        results[sid] = res
    return results


def main() -> None:
    db_path = ROOT / "data" / "vintage_store" / "thales.duckdb"
    print(f"Scraping {URL}")
    with VintageStore(db_path) as store:
        print(f"  vintage store: {db_path}")
        results = ingest_cleveland_fed(store)
    total = sum(r.rows_inserted for r in results.values())
    print(f"\nDone. {len(results)} series; {total} new rows.")


if __name__ == "__main__":
    main()
