"""Industry cost-structure registry — Phase 3.1c.

Maps each industry vertical to the cost-share weights used by the
transmission VAR's interpretation layer. Weights are *operating-cost*
shares (not revenue), which is what hedging and pricing decisions are
conditioned on.

Source for logistics weights: Phase 3.1 planning doc (matches American
Transportation Research Institute's *Operational Costs of Trucking*
2023 annual analysis, ~75% of total marginal cost). Future verticals
should cite their source explicitly.

Design note: kept as a plain Python registry (not DuckDB) because:
  * The data is tiny (single-digit rows per industry)
  * It's effectively a configuration constant — versioned with code
  * No need for as-of vintaging at this granularity (cost structures
    drift slowly; if/when they shift, bump a version)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CostStructure:
    """Cost-share weights for an industry vertical."""
    industry: str
    weights: dict[str, float]
    source: str
    as_of: str    # YYYY-MM-DD

    def __post_init__(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"{self.industry}: weights sum to {total:.4f}, not 1.0")


COST_STRUCTURES: dict[str, CostStructure] = {
    "logistics": CostStructure(
        industry="logistics",
        weights={
            "fuel":        0.35,    # diesel + gasoline
            "labor":       0.25,    # driver wages
            "maintenance": 0.10,    # vehicle upkeep
            "insurance":   0.05,    # premiums
            "other":       0.25,    # tires, tolls, depreciation, admin
        },
        source="ATRI Operational Costs of Trucking 2023; planning doc §3.1",
        as_of="2026-04-25",
    ),

    "restaurants": CostStructure(
        industry="restaurants",
        weights={
            "food_cogs": 0.30,
            "labor":     0.30,
            "rent":      0.08,
            "utilities": 0.04,
            "other":     0.28,
        },
        source="planning doc §3.2",
        as_of="2026-04-25",
    ),

    # Phase 3.3 — additional verticals
    "retail_midmarket": CostStructure(
        industry="retail_midmarket",
        weights={
            "cogs":      0.65,    # wholesale goods purchased
            "labor":     0.15,    # store + admin staff
            "rent":      0.08,    # commercial real estate
            "utilities": 0.02,
            "other":     0.10,    # marketing, insurance, depreciation
        },
        source="NRF ROI report 2024 (mid-market dept-store / general merch); planning doc §3.3",
        as_of="2026-04-26",
    ),

    "healthcare_operators": CostStructure(
        industry="healthcare_operators",
        weights={
            "labor":          0.50,    # medical + admin staff
            "pharma_supplies": 0.20,   # drugs + medical supplies
            "utilities":      0.08,    # facilities energy
            "insurance":      0.05,    # malpractice + liability
            "other":          0.17,    # capex/depreciation, IT, admin
        },
        source="AHA Hospital Statistics 2024; planning doc §3.3",
        as_of="2026-04-26",
    ),

    "real_estate_operators": CostStructure(
        industry="real_estate_operators",
        weights={
            "maintenance":  0.25,    # property upkeep + construction
            "property_tax": 0.20,
            "utilities":    0.15,    # often passed through, but treated as opex line
            "labor":        0.10,    # property management + admin
            "insurance":    0.08,
            "other":        0.22,    # interest reserves, leasing fees, depreciation
        },
        source="NAREIT operating-cost benchmarks 2024; planning doc §3.3",
        as_of="2026-04-26",
    ),

    "manufacturing_durables": CostStructure(
        industry="manufacturing_durables",
        weights={
            "raw_materials": 0.50,   # commodity inputs
            "labor":         0.20,
            "energy":        0.05,
            "logistics":     0.05,
            "other":         0.20,   # capex/depreciation, R&D, SG&A
        },
        source="NAM cost-structure averages durable goods 2024; planning doc §3.3",
        as_of="2026-04-26",
    ),
}


def get_cost_structure(industry: str) -> CostStructure:
    """Return the registered cost structure for an industry, or raise."""
    if industry not in COST_STRUCTURES:
        raise KeyError(
            f"no cost structure for {industry!r}; "
            f"known: {sorted(COST_STRUCTURES)}")
    return COST_STRUCTURES[industry]
