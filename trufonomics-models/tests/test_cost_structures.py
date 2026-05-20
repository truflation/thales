"""Tests for cost-structure registry — Phase 3.1c."""
from __future__ import annotations

import pytest

from thales.cost_structures import (
    COST_STRUCTURES,
    CostStructure,
    get_cost_structure,
)


def test_logistics_present_and_sums_to_one():
    cs = get_cost_structure("logistics")
    assert cs.industry == "logistics"
    assert "fuel" in cs.weights
    assert abs(sum(cs.weights.values()) - 1.0) < 0.01


def test_restaurants_present_and_sums_to_one():
    cs = get_cost_structure("restaurants")
    assert abs(sum(cs.weights.values()) - 1.0) < 0.01


def test_unknown_industry_raises():
    with pytest.raises(KeyError):
        get_cost_structure("unknown_vertical")


def test_invalid_weights_rejected_at_construction():
    with pytest.raises(ValueError, match="sum"):
        CostStructure(industry="bad", weights={"a": 0.5, "b": 0.4},
                          source="test", as_of="2026-04-26")


def test_logistics_weights_match_planning_doc():
    """Pin the logistics weights so silent edits get caught in CI."""
    cs = get_cost_structure("logistics")
    assert cs.weights == {
        "fuel": 0.35, "labor": 0.25, "maintenance": 0.10,
        "insurance": 0.05, "other": 0.25,
    }


def test_phase_3_3_verticals_all_present_and_valid():
    """All four Phase 3.3 verticals registered with weights summing to 1.0."""
    for industry in ("retail_midmarket", "healthcare_operators",
                          "real_estate_operators", "manufacturing_durables"):
        cs = get_cost_structure(industry)
        assert abs(sum(cs.weights.values()) - 1.0) < 0.01, (
            f"{industry}: weights sum to {sum(cs.weights.values())}")
        assert "labor" in cs.weights, f"{industry} missing labor weight"
