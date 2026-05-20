"""Truflation category weights — loader + hierarchy utilities.

The weights files live in data/truflation/weights/ and come in two versions:

  * ``categories-tables-v1.csv`` — weights effective 2010-01-01 through 2025-12-31
  * ``categories-tables-v2.csv`` — weights effective 2026-01-01 onward

Each row is a (category_id, subcategory_id, source_id) tuple with:

  * ``relative_importance``         — Truflation weight (% of headline, 0-100)
  * ``bls_relative_importance``     — BLS-equivalent weight
  * ``pce_relative_importance``     — PCE-equivalent weight
  * plus ``Goods or Services`` and ``Core or NonCore`` tags on leaf rows

The structure nests properly: for a category with children, the SUM of its
children's weights equals the category's own weight. The 12 top-level
weights sum to ~100.

Taxonomy tree (name ↔ id) lives in ``categories-metadata.csv``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = ROOT / "data" / "truflation" / "weights"
V2_EFFECTIVE_FROM = date(2026, 1, 1)


# ─── Category metadata ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_category_tree() -> pd.DataFrame:
    """Load `(category_id, category, parent_id)` for every taxonomy node."""
    df = pd.read_csv(WEIGHTS_DIR / "categories-metadata.csv")
    df["parent_id"] = df["parent_id"].astype("Int64")
    return df


def top_level_category_ids() -> list[int]:
    tree = load_category_tree()
    return tree[tree["parent_id"].isna()]["category_id"].astype(int).tolist()


def category_name(category_id: int) -> str | None:
    tree = load_category_tree()
    hit = tree[tree["category_id"] == category_id]
    return str(hit.iloc[0]["category"]) if len(hit) else None


def path_to_category(category_id: int) -> list[int]:
    """Return [root, ..., category_id] — the full ancestor chain."""
    tree = load_category_tree()
    lookup = dict(zip(tree["category_id"], tree["parent_id"]))
    path: list[int] = [category_id]
    while True:
        parent = lookup.get(path[0])
        if pd.isna(parent) or parent is pd.NA:
            break
        path.insert(0, int(parent))
    return path


# ─── Weights ────────────────────────────────────────────────────────────────

def _load_weights_table(as_of: date) -> pd.DataFrame:
    csv = "categories-tables-v2.csv" if as_of >= V2_EFFECTIVE_FROM else "categories-tables-v1.csv"
    df = pd.read_csv(WEIGHTS_DIR / csv)
    # Normalize types
    for col in ["category_id", "subcategory_id", "source_id"]:
        df[col] = df[col].astype("Int64")
    return df


def get_top_level_weights(as_of: date | str | pd.Timestamp) -> pd.DataFrame:
    """Return the 12 top-level category weights effective at `as_of`.

    Columns: ``category_id``, ``category``, ``weight``, ``bls_weight``,
    ``pce_weight``.
    """
    as_of_d = _as_date(as_of)
    wts = _load_weights_table(as_of_d)
    top = wts[(wts["subcategory_id"] == 0) & (wts["source_id"] == 0)].copy()
    tree = load_category_tree()
    name_map = dict(zip(tree["category_id"], tree["category"]))
    top["category"] = top["category_id"].map(name_map)
    out = top[["category_id", "category",
                "relative_importance",
                "bls_relative_importance",
                "pce_relative_importance"]].rename(columns={
                    "relative_importance": "weight",
                    "bls_relative_importance": "bls_weight",
                    "pce_relative_importance": "pce_weight",
                })
    return out.sort_values("category_id").reset_index(drop=True)


def get_subcategory_weights(as_of: date | str | pd.Timestamp,
                              parent_id: int | None = None
                              ) -> pd.DataFrame:
    """Return subcategory-level weights (subcategory_id != 0, source_id = 0).

    If ``parent_id`` is given, limit to children of that category. Otherwise
    returns every (parent, child) weight pair in the panel.
    """
    as_of_d = _as_date(as_of)
    wts = _load_weights_table(as_of_d)
    sub = wts[(wts["subcategory_id"] != 0) & (wts["source_id"] == 0)].copy()
    if parent_id is not None:
        sub = sub[sub["category_id"] == parent_id]
    tree = load_category_tree()
    name_map = dict(zip(tree["category_id"], tree["category"]))
    sub["parent_name"] = sub["category_id"].map(name_map)
    sub["category"] = sub["subcategory_id"].map(name_map)
    out = sub[["category_id", "parent_name", "subcategory_id", "category",
                "relative_importance",
                "bls_relative_importance",
                "pce_relative_importance"]].rename(columns={
                    "relative_importance": "weight",
                    "bls_relative_importance": "bls_weight",
                    "pce_relative_importance": "pce_weight",
                })
    return out.sort_values(["category_id", "subcategory_id"]).reset_index(drop=True)


# ─── Raw-name → category_id cross-walk ──────────────────────────────────────

def _normalize_name(s: str) -> str:
    """'Food & Non-alcoholic Beverages' → 'food_and_non_alcoholic_beverages'"""
    s = s.lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^\w\s]", " ", s)   # strip punctuation
    s = re.sub(r"\s+", "_", s.strip())
    return s


@dataclass
class StreamMapping:
    raw_name: str
    category_id: int | None
    matched_path: list[int]     # ancestor chain ending at category_id
    normalized_candidates: list[str]  # what we tried to match against


def build_crosswalk(raw_names: Iterable[str]) -> pd.DataFrame:
    """For each stream raw_name, find the matching category_id.

    Strategy: for every node in the tree, generate candidate raw_name forms
    (just-name, parent_child, full-path) and index them. Then look up each
    stream raw_name in that index.

    Returns DataFrame with columns:
        raw_name, category_id, category, matched_on, ancestor_path
    """
    tree = load_category_tree()
    name_by_id = dict(zip(tree["category_id"], tree["category"]))
    parent_by_id = {int(cid): (int(pid) if not pd.isna(pid) else None)
                     for cid, pid in zip(tree["category_id"], tree["parent_id"])}

    def path_names(cid: int) -> list[str]:
        """Return normalized name chain [root_name, ..., cid_name]."""
        out: list[str] = []
        cur = cid
        while cur is not None:
            out.insert(0, _normalize_name(name_by_id[cur]))
            cur = parent_by_id.get(cur)
        return out

    # Build candidate index: canonical_name → category_id
    index: dict[str, int] = {}
    for cid in tree["category_id"].astype(int):
        names = path_names(cid)
        candidates = {
            names[-1],                                         # just leaf
            "_".join(names[-2:]) if len(names) >= 2 else names[-1],
            "_".join(names),                                   # full path
        }
        for cand in candidates:
            # If collision: prefer deeper nodes (more specific match)
            if cand not in index or len(path_names(cid)) > len(path_names(index[cand])):
                index[cand] = cid

    rows = []
    for raw in raw_names:
        r = raw.strip()
        cid = index.get(r)
        rows.append({
            "raw_name": r,
            "category_id": cid,
            "category": name_by_id.get(cid) if cid else None,
            "matched_on": r if cid else None,
            "ancestor_path": path_to_category(cid) if cid else [],
        })
    return pd.DataFrame(rows)


# ─── helpers ────────────────────────────────────────────────────────────────

def _as_date(d: date | str | pd.Timestamp) -> date:
    if isinstance(d, date) and not isinstance(d, pd.Timestamp):
        return d
    return pd.Timestamp(d).date()
