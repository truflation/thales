"""Official inflation targets — single read-side entry point.

Wraps vintage-store access for the four target series:

  * ``CPIAUCSL``  — BLS Headline CPI
  * ``CPILFESL``  — BLS Core CPI
  * ``PCEPI``     — BEA Headline PCE price index
  * ``PCEPILFE``  — BEA Core PCE price index

Both BLS pairs are ingested via ALFRED (``source='fred_alfred_target'``)
so revisions are point-in-time correct. The Cleveland Fed nowcast
(``clevfed_*``) is the canonical comparator series.

All loaders return monthly ``pd.Series`` indexed at month-end. YoY
transformations are computed on demand.

Example usage:

    with VintageStore(DB, read_only=True) as store:
        target = load_target_yoy(store, "cpi", as_of=date.today())
        nowcast = load_nowcast_comparator(store, "cpi", as_of=date.today())
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from thales.vintage import VintageStore

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"


# Canonical short-name → (level series_id, comparator nowcast series_id)
TARGETS: dict[str, tuple[str, str]] = {
    "cpi":      ("CPIAUCSL", "clevfed_cpi_yoy"),
    "core_cpi": ("CPILFESL", "clevfed_corecpi_yoy"),
    "pce":      ("PCEPI",    "clevfed_pce_yoy"),
    "core_pce": ("PCEPILFE", "clevfed_corepce_yoy"),
}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    level_series_id: str
    nowcast_series_id: str


def get_spec(name: str) -> TargetSpec:
    if name not in TARGETS:
        raise ValueError(
            f"Unknown target {name!r}. Available: {sorted(TARGETS)}"
        )
    level_id, nowcast_id = TARGETS[name]
    return TargetSpec(name=name, level_series_id=level_id,
                       nowcast_series_id=nowcast_id)


def _normalize_monthly(s: pd.Series) -> pd.Series:
    """Snap to month-end (PeriodIndex 'M' converted to Timestamp)."""
    if s.empty:
        return s
    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    return s.sort_index()


def load_target_level(store: VintageStore, name: str,
                        as_of: date | None = None) -> pd.Series:
    """Load monthly LEVEL series (e.g. CPI index value) at the as-of vintage."""
    spec = get_spec(name)
    as_of = as_of or date.today()
    s = store.get_vintage(spec.level_series_id, as_of)
    return _normalize_monthly(s)


def load_target_yoy(store: VintageStore, name: str,
                      as_of: date | None = None) -> pd.Series:
    """Year-over-year inflation rate (date-based 12-month change, in %).

    Uses date-based lookup (level[t] / level[t-1y]) rather than positional
    `shift(12)` so the YoY is robust to missing months in the vintage
    store. Without this, a single gap (e.g. missing Oct 2025 in the BLS
    panel) silently shifts the denominator by one month and produces
    wrong YoY.
    """
    level = load_target_level(store, name, as_of=as_of)
    if len(level) < 13:
        return pd.Series(dtype=float, name=f"{name}_yoy")
    out: dict = {}
    for t in level.index:
        denom = pd.Timestamp(year=t.year - 1, month=t.month, day=1) \
                    + pd.offsets.MonthEnd(0)
        if denom in level.index:
            out[t] = (level.loc[t] / level.loc[denom] - 1.0) * 100.0
    yoy = pd.Series(out).sort_index()
    yoy.name = f"{name}_yoy"
    return yoy.dropna()


def load_nowcast_comparator(store: VintageStore, name: str,
                              as_of: date | None = None) -> pd.Series:
    """Cleveland Fed nowcast YoY for the given target. Comparator baseline."""
    spec = get_spec(name)
    as_of = as_of or date.today()
    s = store.get_vintage(spec.nowcast_series_id, as_of)
    s = _normalize_monthly(s)
    s.name = f"{name}_clevfed_yoy"
    return s


def load_panel(store: VintageStore, name: str,
                 as_of: date | None = None) -> pd.DataFrame:
    """Convenience: monthly panel with columns

      * ``y``      — target YoY (the prediction target)
      * ``level``  — the underlying index level
      * ``clevfed``— Cleveland Fed nowcast YoY (comparator)

    Indexed at month-end. Missing rows where any series is unavailable are
    kept (caller's choice to dropna).
    """
    spec = get_spec(name)
    level = load_target_level(store, name, as_of=as_of)
    yoy = load_target_yoy(store, name, as_of=as_of)
    clev = load_nowcast_comparator(store, name, as_of=as_of)
    return pd.DataFrame({
        "y": yoy,
        "level": level,
        "clevfed": clev,
    }).sort_index()


def list_available(store: VintageStore,
                     as_of: date | None = None) -> pd.DataFrame:
    """Inventory: one row per target with first/last obs of level + nowcast."""
    as_of = as_of or date.today()
    rows = []
    for name, (level_id, nowcast_id) in TARGETS.items():
        level = store.get_vintage(level_id, as_of)
        clev = store.get_vintage(nowcast_id, as_of)
        rows.append({
            "name": name,
            "level_series": level_id,
            "level_n": len(level),
            "level_min": level.index.min() if not level.empty else None,
            "level_max": level.index.max() if not level.empty else None,
            "nowcast_series": nowcast_id,
            "nowcast_n": len(clev),
            "nowcast_min": clev.index.min() if not clev.empty else None,
            "nowcast_max": clev.index.max() if not clev.empty else None,
        })
    return pd.DataFrame(rows)
