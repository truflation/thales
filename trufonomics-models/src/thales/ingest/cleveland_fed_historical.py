"""Cleveland Fed historical backfill — parses the three JSON files that drive
the inflation-nowcasting page's interactive chart and writes the complete
daily vintage archive into the Thales vintage store.

URLs (public, no auth):
    https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json
    https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_year.json
    https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_quarter.json

Each file is a list of target-period entries. Every entry has:
  * chart.subcaption: the target period ("2014-1" or "2026:Q2")
  * categories[0].category: labels array (mostly "MM/DD" date labels, plus
    milestone markers like "CPI Mar" that mark BLS release dates)
  * dataset: list of series — first 4 are the nowcast evolution for CPI,
    Core CPI, PCE, Core PCE; next 4 are the "Actual" values once BLS
    releases them

Output: rows ingested to the Thales vintage store with source='clevfed'
keyed on:
    (series_id='clevfed_{measure}_{freq}', reference_date=target period end,
     as_of_date=date on the label, source='clevfed')
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from thales.vintage import VintageStore

ROOT = Path(__file__).resolve().parents[3]
VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
BASE = "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting"
SOURCE = "clevfed"

FILES = {
    "nowcast_month":   "mom",
    "nowcast_year":    "yoy",
    "nowcast_quarter": "saar",
}

MEASURE_MAP = {
    "CPI Inflation":      "cpi",
    "Core CPI Inflation": "corecpi",
    "PCE Inflation":      "pce",
    "Core PCE Inflation": "corepce",
}

DATE_LABEL_RE = re.compile(r"^(\d{1,2})/(\d{1,2})$")
QUARTER_RE = re.compile(r"^(\d{4}):?Q?(\d)$")
MONTH_RE = re.compile(r"^(\d{4})-(\d{1,2})$")


def parse_target_period(subcaption: str) -> tuple[date, int]:
    """Return (reference_date = period-end, target_first_month_for_year_inference)."""
    m = QUARTER_RE.match(subcaption)
    if m:
        year, q = int(m.group(1)), int(m.group(2))
        end_month = q * 3
        ref = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
        first_month = (q - 1) * 3 + 1
        return ref.date(), first_month
    m = MONTH_RE.match(subcaption)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        ref = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        return ref.date(), month
    raise ValueError(f"Unrecognized target period: {subcaption!r}")


def decode_label_dates(labels: list[str], target_year: int,
                        target_month: int) -> list[date | None]:
    """Given a labels array for one target, return same-length list of dates
    (None for non-date labels like 'CPI Mar').

    Year-rollover rule: start year = target_year, except if the first date
    label's month exceeds target_month + 6 (indicating labels start in the
    previous calendar year), in which case start year = target_year − 1.
    Then walk forward; increment year whenever label month DECREASES.
    """
    first_m = None
    for lbl in labels:
        match = DATE_LABEL_RE.match(lbl)
        if match:
            first_m = int(match.group(1))
            break
    if first_m is None:
        return [None] * len(labels)

    start_year = target_year - 1 if first_m > target_month + 6 else target_year

    out: list[date | None] = []
    year = start_year
    prev_m: int | None = None
    for lbl in labels:
        match = DATE_LABEL_RE.match(lbl)
        if not match:
            out.append(None)
            continue
        m, d = int(match.group(1)), int(match.group(2))
        if prev_m is not None and m < prev_m:
            year += 1
        try:
            out.append(pd.Timestamp(year=year, month=m, day=d).date())
        except ValueError:
            out.append(None)
        prev_m = m
    return out


@dataclass
class NowcastRow:
    series_id: str
    target_date: date
    as_of_date: date
    value: float


def parse_nowcast_file(url: str, freq_suffix: str) -> list[NowcastRow]:
    resp = requests.get(url, timeout=120, headers={
        "User-Agent": "Mozilla/5.0 (compatible; Thales/0.1)"
    })
    resp.raise_for_status()
    data = resp.json()

    rows: list[NowcastRow] = []
    for entry in data:
        chart = entry.get("chart", {})
        subcaption = chart.get("subcaption", "")
        try:
            target_date, target_month = parse_target_period(subcaption)
        except ValueError:
            continue
        target_year = pd.Timestamp(target_date).year

        labels = [c["label"] for c in entry["categories"][0]["category"]]
        as_of_dates = decode_label_dates(labels, target_year, target_month)

        for series in entry.get("dataset", []):
            name = series.get("seriesname", "")
            if name not in MEASURE_MAP:
                continue  # skip "Actual ..." series
            series_id = f"clevfed_{MEASURE_MAP[name]}_{freq_suffix}"
            for idx, point in enumerate(series.get("data", [])):
                if idx >= len(as_of_dates):
                    break
                if as_of_dates[idx] is None:
                    continue
                raw = point.get("value", "")
                if raw == "" or raw is None:
                    continue
                try:
                    val = float(raw)
                except (ValueError, TypeError):
                    continue
                rows.append(NowcastRow(
                    series_id=series_id,
                    target_date=target_date,
                    as_of_date=as_of_dates[idx],
                    value=val,
                ))
    return rows


def main() -> None:
    print(f"Opening vintage store: {VINTAGE_DB}")
    store = VintageStore(VINTAGE_DB)

    for filename, freq in FILES.items():
        url = f"{BASE}/{filename}.json"
        print(f"\n--- {filename}.json ({freq}) ---")
        rows = parse_nowcast_file(url, freq)
        print(f"  parsed {len(rows):,} rows")

        batches: dict[tuple[str, date], list[tuple[date, float]]] = defaultdict(list)
        for r in rows:
            batches[(r.series_id, r.as_of_date)].append((r.target_date, r.value))

        total_inserted = 0
        total_duplicate = 0
        conflicts = 0
        for (series_id, asof), obs in batches.items():
            try:
                res = store.ingest(
                    series_id=series_id,
                    observations=obs,
                    as_of_date=asof,
                    source=SOURCE,
                )
            except ValueError:
                conflicts += 1
                continue
            total_inserted += res.rows_inserted
            total_duplicate += res.rows_duplicate
        n_series = len({sid for sid, _ in batches})
        n_asof = len({asof for _, asof in batches})
        print(f"  {n_series} series × {n_asof} as_of dates")
        print(f"  inserted: {total_inserted:,}  duplicates: {total_duplicate:,}  conflicts: {conflicts}")

    print(f"\nFinal vintage store row count: {store.row_count():,}")
    store.close()


if __name__ == "__main__":
    main()
