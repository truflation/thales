"""Single source of truth for operator cost baskets.

Resolves the inconsistency between ``docs/cost_structures.md`` (which
quoted operating-cost shares like "vehicle 60%, freight 12%, diesel
6%, duty 10%, FX implicit") and the variance-attribution weights used
in earlier code ("vehicle 45%, FX 30%, freight 12%, diesel 6%,
transport 7%"). Both are real concepts, they shouldn't share a name.

Two distinct quantities per operator basket, both versioned here:

1. **cost_share** — operating-cost share (per the docs). Sums to 1.0
   across all components including fixed lines (duty, overhead). What
   the operator's accountant would tell you.

2. **landed_cost_exposure_weight** — the vector that multiplies the
   modelled variables in the landed-cost aggregation:
   ``landed_log_dev = Σ w_i × Δlog x_i``.
   This is *not* a probability distribution. For variables that
   transform the same cost line (e.g., vehicle wholesale USD price
   and EUR/USD FX both affect the EUR-denominated vehicle cost), each
   gets its own weight equal to the cost share of that portion — so
   the vector can sum above 1.0. That's correct.

Worked example (auto importer):
  Vehicle cost is 60% of landed cost, paid in USD.
    Δlog(landed_EUR) contribution = 0.60 × (Δlog vehicle_USD + Δlog EUR/USD)
  So weights are:
    log_truf_vehicle       0.60   (USD vehicle wholesale price moves)
    log_fx_eurusd          0.60   (same 60% cost portion, FX moves)
    log_freight            0.12   (ocean ro-ro share)
    log_diesel             0.06   (inland trucking share)
    log_truf_transport     0.07   (small extra transport, mostly EU-side)
  Sum = 1.45. Fixed lines (duty 10%, insurance/overhead 12%) absent
  from this vector by design — they don't get modelled as time-series
  variables, they enter via scenario knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CostShareItem:
    label: str                       # human-readable
    share: float                     # fraction of landed cost (sum across all items = 1.0)
    modelled_var: str | None = None  # BVAR/copula variable name (if modelled)
    fx_pair: str | None = None       # if cost is in foreign currency,
                                       # the FX variable that drives EUR/USD conversion


@dataclass(frozen=True)
class OperatorBasket:
    version: str                       # bump when shares change
    label: str
    description: str
    cost_share: list[CostShareItem]    # canonical operating-cost decomposition
    notes: str = ""


def landed_cost_exposure_weights(basket: OperatorBasket) -> dict[str, float]:
    """Derive the exposure-weight vector (the thing that multiplies
    modelled-variable log-deviations to produce landed-cost log-dev).

    Rule: for each cost line that is modelled (modelled_var != None),
    the modelled variable gets weight = cost_share. If the line is in
    a foreign currency, the FX variable ALSO gets weight = cost_share
    (since both the foreign-currency price and the conversion rate
    move landed cost).
    """
    weights: dict[str, float] = {}
    for item in basket.cost_share:
        if item.modelled_var:
            weights[item.modelled_var] = weights.get(item.modelled_var, 0.0) + item.share
        if item.fx_pair:
            weights[item.fx_pair] = weights.get(item.fx_pair, 0.0) + item.share
    return weights


# ─── Auto importer (Paris) — operating-cost basket v1 ────────────────────
# Sources: ATRI 2023 Trucking Cost averages; EU light-vehicle MFN duty
# (HS 8703) = 10%; ro-ro shipping share from industry reports.

AUTO_IMPORTER_V1 = OperatorBasket(
    version="v1-2026-05-24",
    label="paris_auto_importer",
    description="EU-side importer of US-made light vehicles, EUR-denominated",
    cost_share=[
        CostShareItem("Vehicle wholesale (USD invoice)", 0.60,
                          modelled_var="log_truf_vehicle",
                          fx_pair="log_fx_eurusd"),
        CostShareItem("Ocean ro-ro shipping", 0.12,
                          modelled_var="log_freight"),
        CostShareItem("Inland trucking (EU side)", 0.06,
                          modelled_var="log_diesel"),
        CostShareItem("EU inland transport overhead", 0.07,
                          modelled_var="log_truf_transport"),
        # Fixed / non-modelled lines (absent from the modelled vector)
        CostShareItem("Import duty (EU 10% MFN, HS 8703)", 0.10,
                          modelled_var=None),
        CostShareItem("Insurance + dealer overhead", 0.05,
                          modelled_var=None),
    ],
    notes=(
        "Vehicle cost is USD-denominated, so it carries TWO variance "
        "drivers — the USD wholesale price (Truflation `vehicle_purchases_"
        "net_outlay_cars_and_trucks_new`) AND the EUR/USD spot rate. "
        "Both get weight 0.60 in the exposure vector. Cost shares sum to "
        "1.00. Exposure weights sum > 1.0 by design (foreign-currency "
        "double-pass)."),
)

# ─── Textile importer (US) — operating-cost basket v1 ────────────────────

TEXTILE_IMPORTER_V1 = OperatorBasket(
    version="v1-2026-05-24",
    label="us_textile_importer",
    description="US-side importer of finished textiles from Asia (CNY-source)",
    cost_share=[
        CostShareItem("Factory FOB goods (CNY invoice)", 0.50,
                          modelled_var="log_truf_clothing",
                          fx_pair="log_fx_cnyusd"),
        CostShareItem("Container ocean freight", 0.20,
                          modelled_var="log_freight"),
        CostShareItem("US inland trucking", 0.10,
                          modelled_var="log_diesel"),
        CostShareItem("US distribution overhead", 0.07,
                          modelled_var="log_truf_transport"),
        # Fixed / non-modelled
        CostShareItem("Import duty (US MFN textiles, HS 50-63 avg)", 0.10,
                          modelled_var=None),
        CostShareItem("Insurance + handling", 0.03,
                          modelled_var=None),
    ],
    notes=(
        "Goods cost is CNY-denominated. `log_truf_clothing` captures the "
        "US-retail price level proxy; FX `log_fx_cnyusd` captures the "
        "conversion. Both contribute. Duty is a scenario knob, not in "
        "the BVAR/copula state vector."),
)


REGISTRY: dict[str, OperatorBasket] = {
    "auto":    AUTO_IMPORTER_V1,
    "textile": TEXTILE_IMPORTER_V1,
}


def get_basket(label: str) -> OperatorBasket:
    if label not in REGISTRY:
        raise ValueError(f"unknown basket {label!r}; known: {list(REGISTRY)}")
    return REGISTRY[label]


def get_exposure_weights(label: str) -> dict[str, float]:
    return landed_cost_exposure_weights(get_basket(label))
