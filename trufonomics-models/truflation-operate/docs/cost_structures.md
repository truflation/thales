# Truflation Operate — Cost structures per client (v1)

Operating-cost share weights for each import/export client. These weights drive the BVAR's interpretation layer: when the model produces a +X% shock to component Y, the cost-structure weight tells the client what fraction of their total landed cost moves.

**Important.** These are first-cut estimates from public industry-average sources. Per-client weights should be refreshed with the client's actual invoice / P&L data when available. The accuracy of the exposure quantification is bounded above by the accuracy of these weights.

## Sources

| Source | Used for |
|---|---|
| ATRI 2023 *Operational Costs of Trucking* | US trucking baseline (already in `src/thales/cost_structures.py`) |
| US Census *International Trade in Goods and Services* | Import structure baseline |
| WITS World Bank Tariff Data | Duty estimates (vehicle / textile HS codes) |
| EU TARIC / USITC | Cross-checked duty rates |
| ATRI plus EU ro-ro carrier average industry reports | Ro-ro shipping cost share for autos |

## Client 1 — Paris auto importer (importing US-made vehicles to the EU)

| Component | Share of landed cost | Driver variable |
|---|---|---|
| Vehicle wholesale cost | 60% | `truf:vehicle_purchases_*` + USD-denominated invoice (FX) |
| Ocean ro-ro shipping | 12% | Freight indices (today: PPI `PCU484121484121` as proxy; upgrade to Drewry ro-ro index later) |
| Inland trucking (EU side, port → dealership) | 6% | EU diesel + trucking labor (today: extrapolated from US `GASDESW` + EU adjustment; upgrade to Eurostat diesel) |
| Import duty | 10% | EU 10% MFN duty on light vehicles (HS 8703) — fixed unless trade-deal change |
| FX (USD invoice exposure) | 100% of vehicle cost | `DEXUSEU` (USD/EUR spot) |
| Insurance + handling | 4% | small, treat as fixed |
| Dealer operating cost (overhead) | 8% | local, not modeled |

**Primary BVAR variables:** EUR/USD, ocean freight, EU inland diesel, vehicle wholesale cost (Truflation `vehicle_purchases_net_outlay_cars_and_trucks_new`).

**Exposure quantification:** the model expresses the client's landed-cost variance as: how much of the variance comes from FX vs freight vs vehicle cost vs duty changes? At a horizon of N months, what is the distribution of landed cost under shock scenarios?

## Client 2 — US textile importer (importing finished textiles from Asia)

| Component | Share of landed cost | Driver variable |
|---|---|---|
| Goods cost (factory FOB) | 50% | Local currency cost (FX exposure to source-country FX) |
| Container ocean freight | 20% | Freight indices (today: `PCU484121484121` proxy; upgrade to Freightos FBX Asia-US later) |
| Inland trucking (US side, port → DC → store) | 10% | `GASDESW` (US retail diesel) + PPI trucking |
| Import duty | 15% | US MFN duty on textiles (HS 50-63 range, varies 0-32%; conservative average) |
| FX (foreign-currency invoice exposure) | 100% of goods cost | Primary: `DEXCHUS` (CNY/USD); secondary: `DEXINUS` (INR/USD), other Asia FX |
| Insurance + handling | 5% | small |

**Primary BVAR variables:** CNY/USD (or chosen Asia source-country FX), container freight, US inland diesel, Truflation `clothing_and_footwear` (anchors to US retail tradables inflation), tariff regime dummy.

**Exposure quantification:** how much of landed-cost variance comes from each input? Under tariff scenarios (current MFN vs Section 301 escalation), what is the distribution shift?

## Cross-cutting notes

- **FX is treated as exposure, not forecast.** No model promises a future EUR/USD or CNY/USD value. The scenario engine takes user shocks (e.g., "assume EUR/USD moves to 1.05") and propagates through the transmission to landed cost.
- **Duty is treated as scenario, not forecast.** Same reason — duty changes are policy events, not statistically predictable. Scenario console lets the user toggle duty rates.
- **Cost shares are versioned.** When refreshed with client invoice data, bump a version tag in the cost-structure dict.

## Out of scope (v1)

- Carrier-specific freight rates (would need direct carrier ingest).
- Per-route ocean freight (Drewry WCI / Freightos FBX paid tier).
- EU diesel direct (Eurostat ingest pending; using US `GASDESW` as proxy for v1).
- Per-shipment HS-code-level duty resolution (using basket-average duties).

These upgrades close the gap between "good first cut" and "production-grade per-client analysis." Each is independently scopable.
