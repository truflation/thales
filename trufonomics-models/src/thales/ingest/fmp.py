"""FMP (Financial Modeling Prep) ingest — commodities and futures.

Provides daily historical prices for the commodity/futures symbols we
need for the Phase 3 transmission products:

  * HOUSD — Heating Oil (NY Harbor ULSD futures, the standard diesel
    hedge instrument; correlates ~0.95 with retail diesel)
  * CLUSD — WTI Crude Oil futures (alternative hedge / oil-shock proxy)
  * NGUSD — Natural Gas (Henry Hub futures)
  * GCUSD — Gold (regime / risk-off indicator, optional)
  * SIUSD — Silver (industrial demand proxy, optional)

Uses the **stable** FMP endpoint (the `v3/...` legacy endpoints return
403 for new accounts as of 2025-Q3).

API contract (stable):
    GET /stable/historical-price-eod/light
        ?symbol=<symbol>&from=YYYY-MM-DD&to=YYYY-MM-DD&apikey=<key>
    →  list of {symbol, date, price, volume}, newest first.
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
SOURCE = "fmp"
BASE_URL = "https://financialmodelingprep.com/stable"


@dataclass(frozen=True)
class FmpCommoditySpec:
    symbol: str
    category: str
    description: str


# Default panel for Phase 3 transmission products
FMP_COMMODITY_PANEL: list[FmpCommoditySpec] = [
    FmpCommoditySpec("HOUSD", "fuel_hedge",
                          "NY Harbor heating oil futures (diesel hedge)"),
    FmpCommoditySpec("CLUSD", "oil",
                          "WTI crude oil futures"),
    FmpCommoditySpec("BZUSD", "oil",
                          "Brent crude oil futures"),
    FmpCommoditySpec("RBUSD", "fuel_hedge",
                          "RBOB gasoline futures"),
    FmpCommoditySpec("NGUSD", "gas",
                          "Henry Hub natural gas futures"),
    # Liquid energy ETFs — investable instruments for shipper hedging
    FmpCommoditySpec("DBO", "fuel_hedge",
                          "Invesco DB Oil ETF (top diesel correlator)"),
    FmpCommoditySpec("USO", "oil",
                          "United States Oil Fund ETF"),
    FmpCommoditySpec("UGA", "fuel_hedge",
                          "United States Gasoline Fund ETF"),
    FmpCommoditySpec("XLE", "energy_equity",
                          "Energy Select Sector SPDR"),
]


def _fmp_key() -> str:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("FMP_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise RuntimeError(
            "FMP_API_KEY not found. Set in environment or in .env at repo root.")
    return key


def fetch_historical_eod(symbol: str,
                            start: str = "2010-01-01",
                            end: str | None = None) -> pd.Series:
    """Fetch daily EOD price history for a single FMP symbol.

    Returns a pd.Series indexed by reference_date (ascending), name=symbol.
    Drops the volume column — for the BVAR use case price is what we
    need.
    """
    if end is None:
        end = date.today().isoformat()
    key = _fmp_key()
    url = (f"{BASE_URL}/historical-price-eod/light"
            f"?symbol={symbol}&from={start}&to={end}&apikey={key}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        raise ValueError(f"FMP returned empty/unexpected response for {symbol}")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    s = df["price"].astype(float)
    s.name = symbol
    return s


def ingest_fmp_panel(
    store: VintageStore,
    specs: Iterable[FmpCommoditySpec] = FMP_COMMODITY_PANEL,
    as_of_date: date | str | None = None,
    start: str = "2010-01-01",
) -> dict[str, IngestResult]:
    """Pull the FMP commodity panel into the vintage store.

    Each row tagged ``source='fmp'`` and ``as_of_date=today`` (or as
    specified). Idempotent: re-running on the same as_of_date dedupes.
    """
    if as_of_date is None:
        as_of_date = date.today()
    elif isinstance(as_of_date, str):
        as_of_date = date.fromisoformat(as_of_date)

    results: dict[str, IngestResult] = {}
    for spec in specs:
        try:
            s = fetch_historical_eod(spec.symbol, start=start)
            r = store.ingest(
                series_id=spec.symbol,
                observations=s,
                as_of_date=as_of_date,
                source=SOURCE,
            )
            results[spec.symbol] = r
            print(f"  {spec.symbol:<10s}  {r.rows_inserted:>6d} inserted  "
                    f"{r.rows_duplicate:>4d} dup  ({spec.category})")
        except Exception as e:    # noqa: BLE001
            print(f"  {spec.symbol:<10s}  ERROR: {type(e).__name__}: {e}")
    return results
