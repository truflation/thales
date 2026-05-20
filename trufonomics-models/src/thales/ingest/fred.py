"""FRED ingest — pulls the covariate panel into the vintage store.

Reads FRED_API_KEY from environment (or .env via python-dotenv if present).
Each pull tags every observation with as_of_date = today. FRED does not
expose release-date metadata through the public series endpoint — for true
point-in-time vintages you need ALFRED, which is TODO (see note below).

Usage:
    # Pull the full priority set
    python -m thales.ingest.fred

    # Or from Python
    from thales.ingest.fred import ingest_fred_panel
    from thales.vintage import VintageStore
    store = VintageStore("data/vintage_store/thales.duckdb")
    ingest_fred_panel(store)

TODO: Migrate to ALFRED (FRED Archival) for true release-date vintages. For
Phase 0 the "pulled today, tagged today" approximation is acceptable — we
document the limitation, and forward scrapes from today on will have clean
as_of_date tagging by construction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
from fredapi import Fred

from thales.vintage import IngestResult, VintageStore

ROOT = Path(__file__).resolve().parents[3]
SOURCE = "fred"


@dataclass(frozen=True)
class FredSeriesSpec:
    series_id: str
    category: str
    description: str
    release_lag_days: int  # publication lag vs reference period


# Priority panel — locked in pre-reg §7 of the Thales forecast track
# and in docs/planning/04-data-sources.md of the Trufonomics plan.
FRED_PANEL: list[FredSeriesSpec] = [
    # Rates and curve
    FredSeriesSpec("DGS2",          "rates",          "2Y Treasury constant maturity",             1),
    FredSeriesSpec("DGS10",         "rates",          "10Y Treasury constant maturity",            1),
    FredSeriesSpec("DGS30",         "rates",          "30Y Treasury constant maturity",            1),
    FredSeriesSpec("T10Y2Y",        "yield_curve",    "10Y minus 2Y Treasury spread",              1),
    FredSeriesSpec("T10Y3M",        "yield_curve",    "10Y minus 3M Treasury spread",              1),
    FredSeriesSpec("MORTGAGE30US",  "mortgage",       "Freddie Mac 30Y fixed mortgage rate",       7),
    FredSeriesSpec("FEDFUNDS",      "policy_rate",    "Effective federal funds rate, monthly avg", 20),
    FredSeriesSpec("DFEDTARU",      "policy_rate",    "Fed funds target range, upper",             1),
    FredSeriesSpec("DFF",           "policy_rate",    "Daily effective federal funds rate",        1),

    # Credit spreads and financial conditions
    FredSeriesSpec("BAMLC0A0CM",    "credit_spread",  "ICE BofA US Corporate Index OAS (IG)",      1),
    FredSeriesSpec("BAMLH0A0HYM2",  "credit_spread",  "ICE BofA US HY Index OAS",                  1),
    FredSeriesSpec("NFCI",          "fin_conditions", "Chicago Fed NFCI (weekly)",                 7),
    FredSeriesSpec("ANFCI",         "fin_conditions", "Adjusted NFCI (business cycle removed)",    7),

    # Inflation expectations and breakevens
    FredSeriesSpec("T5YIE",         "breakeven",      "5Y breakeven inflation rate",               1),
    FredSeriesSpec("T10YIE",        "breakeven",      "10Y breakeven inflation rate",              1),
    FredSeriesSpec("T5YIFR",        "breakeven",      "5Y 5Y forward inflation expectation",       1),
    FredSeriesSpec("EXPINF1YR",     "inflation_exp",  "Cleveland Fed 1Y inflation expectation",    30),
    FredSeriesSpec("EXPINF10YR",    "inflation_exp",  "Cleveland Fed 10Y inflation expectation",   30),
    FredSeriesSpec("MICH",          "inflation_exp",  "UMich 1Y inflation expectation",            30),

    # Commodities
    FredSeriesSpec("DCOILWTICO",    "commodity",      "WTI crude oil spot, daily",                 2),
    FredSeriesSpec("DCOILBRENTEU",  "commodity",      "Brent crude oil spot, daily",               2),
    FredSeriesSpec("DHHNGSP",       "commodity",      "Henry Hub natural gas spot, daily",         2),
    FredSeriesSpec("GASREGW",       "commodity",      "US retail regular gasoline, weekly (EIA)",  2),

    # FX
    FredSeriesSpec("DTWEXBGS",      "fx",             "Broad trade-weighted USD index",            1),

    # Labor
    FredSeriesSpec("UNRATE",        "labor",          "Unemployment rate",                         30),
    FredSeriesSpec("PAYEMS",        "labor",          "Nonfarm payrolls",                          30),
    FredSeriesSpec("ICSA",          "labor",          "Initial jobless claims (weekly)",           5),
    FredSeriesSpec("CCSA",          "labor",          "Continued jobless claims (weekly)",         12),
    FredSeriesSpec("JTSJOL",        "labor",          "JOLTS job openings",                        60),
    FredSeriesSpec("CIVPART",       "labor",          "Labor force participation rate",            30),
    FredSeriesSpec("CES0500000003", "labor",          "Avg hourly earnings, total private",        30),

    # Activity
    FredSeriesSpec("INDPRO",        "output",         "Industrial production index",               30),
    FredSeriesSpec("RSAFS",         "retail",         "Retail sales ex-food services",             30),
    FredSeriesSpec("UMCSENT",       "sentiment",      "UMich consumer sentiment",                  30),
    FredSeriesSpec("DSPIC96",       "income",         "Real disposable personal income",           30),
    FredSeriesSpec("PCE",           "spending",       "Personal consumption expenditures (level)", 30),

    # Money and reserves
    FredSeriesSpec("M2SL",          "money_supply",   "M2 money stock (monthly)",                  30),
    FredSeriesSpec("WALCL",         "fed_balance",    "Fed total assets (Wednesday-level)",        1),
    FredSeriesSpec("TOTRESNS",      "fed_reserves",   "Reserves of depository institutions",       30),
    FredSeriesSpec("RRPONTSYD",     "fed_reserves",   "Overnight reverse repo operations, daily",  1),

    # Housing
    FredSeriesSpec("CSUSHPISA",     "housing_price",  "Case-Shiller US national home price index", 60),
    FredSeriesSpec("HOUST",         "housing_activity","Housing starts, new privately owned",       30),
    FredSeriesSpec("PERMIT",        "housing_activity","New private housing permits",               30),
    FredSeriesSpec("EXHOSLUSM495S", "housing_activity","Existing home sales",                       30),

    # External
    FredSeriesSpec("BOPGSTB",       "trade",          "US trade balance, goods and services",      60),

    # GDP (quarterly)
    FredSeriesSpec("GDP",           "gdp",            "Gross domestic product, nominal",           90),
    FredSeriesSpec("GDPC1",         "gdp",            "Real GDP",                                  90),

    # NOTE: CPI-family (CPIAUCSL, CPILFESL, PCEPI, PCEPILFE, PPIFIS) is
    # deliberately excluded from this panel. Those are TARGET series and
    # come in via thales.ingest.bls with source='bls_direct' to avoid
    # leakage in Ridge-style baselines. See pre-reg §7.
]


def _fred_client() -> Fred:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        # Fall back to .env file at repo root
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("FRED_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise RuntimeError(
            "FRED_API_KEY not found. Set it in the environment or in "
            "trufonomics-models/.env"
        )
    return Fred(api_key=key)


def fetch_series(fred: Fred, series_id: str,
                  start: str = "2010-01-01") -> pd.Series:
    """Fetch a single FRED series, clean-NaN'd, indexed by reference_date."""
    s = fred.get_series(series_id, observation_start=start)
    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    s.name = series_id
    return s


def ingest_fred_panel(
    store: VintageStore,
    specs: Iterable[FredSeriesSpec] = FRED_PANEL,
    as_of_date: date | str | None = None,
    start: str = "2010-01-01",
) -> dict[str, IngestResult]:
    """Pull the full FRED panel into the vintage store.

    Every observation is tagged with `as_of_date` (defaults to today).
    Returns a {series_id: IngestResult} dict.
    """
    fred = _fred_client()
    if as_of_date is None:
        as_of_date = date.today()
    results: dict[str, IngestResult] = {}
    for spec in specs:
        try:
            s = fetch_series(fred, spec.series_id, start=start)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {spec.series_id}: {type(e).__name__}: {e}")
            continue
        if s.empty:
            print(f"  [skip] {spec.series_id}: no observations")
            continue
        res = store.ingest(
            series_id=spec.series_id,
            observations=s,
            as_of_date=as_of_date,
            source=SOURCE,
        )
        print(f"  {spec.series_id:<16s} {res.rows_inserted:>5d} inserted "
              f"{res.rows_duplicate:>4d} dup  ({spec.category})")
        results[spec.series_id] = res
    return results


def main() -> None:
    db_path = ROOT / "data" / "vintage_store" / "thales.duckdb"
    print(f"Opening vintage store: {db_path}")
    with VintageStore(db_path) as store:
        print(f"Pulling {len(FRED_PANEL)} FRED series (as_of = today)...")
        results = ingest_fred_panel(store)
        total_inserted = sum(r.rows_inserted for r in results.values())
        print(f"\nDone. {len(results)}/{len(FRED_PANEL)} series; "
              f"{total_inserted:,} new rows.")
        print(f"Row count in store: {store.row_count():,}")


if __name__ == "__main__":
    main()
