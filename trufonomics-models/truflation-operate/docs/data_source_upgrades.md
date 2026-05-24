# Data-source upgrade backlog

v1 ingest uses public-FRED proxies that are reasonable for a prototype but insufficient for production credibility. This doc catalogues the upgrade targets in priority order, with the gap each one closes.

## Status of v1 proxies

| Cost component | v1 proxy in `operate_fred_ingest.py` | What it actually measures | Gap |
|---|---|---|---|
| US inland diesel | `GASDESW` (EIA via FRED) | US on-highway retail diesel, weekly | None — this is the right series |
| US wholesale diesel | `WPU057303` | PPI No. 2 diesel | Reasonable cross-check |
| Ocean / inland freight | `PCU484121484121` | PPI: US long-distance truckload | **Not ocean freight** — no route-specific signal |
| EU inland diesel | (currently using `GASDESW` US as proxy) | US, not EU | **Region wrong** — auto importer needs EU |
| Major FX | `DEXUSEU`, `DEXCHUS`, `DEXMXUS`, `DEXCAUS`, `DEXINUS` | FRED daily spot | Good for these pairs |

## Priority 1 — Ocean freight (route-specific)

Both clients are import/export; ocean freight is a primary exposure. PPI trucking is the wrong instrument.

| Source | Coverage | Cadence | Cost | Notes |
|---|---|---|---|---|
| [Freightos FBX](https://www.freightos.com/data/) | Daily container-spot indexes across 13 trade lanes (China-US East, China-US West, Asia-N. Europe, etc.) | Daily | Free press-tier; paid API for raw | Most useful for textile importer (Asia → US lanes) |
| [Drewry WCI](https://www.drewry.co.uk/logistics-executive-briefing/logistics-executive-briefing-articles/world-container-index-methodology) | Weekly container-spot, 8 routes (Shanghai-LA, Shanghai-Rotterdam, etc.) | Weekly | Paid via Drewry | Established industry benchmark |
| Baltic Exchange BDI / BDTI | Dry bulk + tanker, multi-route | Daily | Paid (Baltic) | Not container — different segment |
| EU [TARIC](https://taxation-customs.ec.europa.eu/customs-4/calculation-customs-duties/customs-tariff/eu-customs-tariff-taric_en) — Eurostat external trade | Trade volumes / values by HS code, monthly | Monthly | Free | Not freight rates but useful for export-volume context |

**Recommendation:** start with Freightos FBX (their free reporting tier publishes the headline FBX01 daily); upgrade to Drewry WCI for the textile vertical once paid budget allows. Add as new vars `log_freight_asia_us` and `log_freight_asia_eu`.

## Priority 2 — EU diesel (region-correct)

Auto importer's inland trucking is EU-side; US diesel is a proxy correlation, not the price the operator pays.

| Source | Coverage | Cadence | Cost | Notes |
|---|---|---|---|---|
| [European Commission Weekly Oil Bulletin](https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en) | Country-level retail and wholesale petroleum prices for all EU member states | Weekly | Free | The right series for EU operator. Aggregates to euro-area weighted mean. |
| Eurostat energy prices | Country-level monthly | Monthly | Free | Cross-check |

**Recommendation:** ingest the EU Weekly Oil Bulletin via scraping the published Excel. Add as `log_diesel_eu` to the auto importer panel.

## Priority 3 — Tariff / duty resolution

Both v1 baskets quote a single representative MFN duty rate (10% for EU light vehicles, 15% blended for US textiles). Real operators have specific HTS / HS codes; duty rates vary inside each category.

| Source | Coverage | Cadence | Cost | Notes |
|---|---|---|---|---|
| [USITC HTS](https://www.usitc.gov/harmonized_tariff_information) | US HTS schedule, statutory + Section 301 + Section 232 modifications | Updated continuously | Free | Right for US-importer (textile vertical) |
| [TARIC](https://taxation-customs.ec.europa.eu/customs-4/calculation-customs-duties/customs-tariff/eu-customs-tariff-taric_en) | EU customs tariff with all measures | Continuous | Free | Right for EU-importer (auto vertical) |
| WITS World Bank | Multi-country bilateral MFN duties by HS | Annual | Free | Cross-check |

**Recommendation:** v1.5 should accept an HS code per client; pull the corresponding duty from USITC/TARIC at scenario time rather than the basket-average rate. Adds duty as a true scenario knob (with current rate + announced changes) rather than a fixed share.

## Priority 4 — Wages and labour cost (per vertical)

Inland trucking labor and warehouse labor are missing from the v1 panel. Truflation has `transport` but it's an aggregate; the operator's actual labor exposure is more specific.

| Source | Coverage | Cadence | Cost | Notes |
|---|---|---|---|---|
| FRED `CES4300000008` | US transportation & warehousing avg hourly earnings | Monthly | Free | Already used in earlier logistics BVAR |
| FRED `ECIWAG` | US employment cost index | Quarterly | Free | Headline wage growth |
| Eurostat LCI (labour cost index) | EU by activity | Quarterly | Free | EU-side labor for auto importer |

**Recommendation:** add per-vertical labor series in v1.5 panel expansions. Auto importer should include EU transport labor; textile importer should include US distribution labor.

## Priority 5 — Targeted commodity inputs

For specific verticals, targeted commodity series add signal:

- **Auto:** steel (HRC futures), aluminum (LME), copper. Auto wholesale costs respond to these with 3-6m lag.
- **Textile:** cotton (ICE), polyester (synthetic, paid feeds), wool (ASX). Goods-cost prediction inputs.
- **Restaurants (future vertical):** beef + chicken + dairy + grain spot.

Most are free via FMP commodity endpoints already wired in `src/thales/ingest/fmp.py`.

## Implementation order

1. **Freightos FBX free tier** → ingest `log_freight_asia_us`, `log_freight_asia_eu`. Update textile panel.
2. **EC Weekly Oil Bulletin** → ingest `log_diesel_eu`. Update auto panel.
3. **HS-code-aware duty layer** → modify scenario console to accept `--hs-code` and look up current rate.
4. **Per-vertical labor series** → adds 1-2 more vars to each panel.
5. **Targeted commodity inputs** → vertical-specific expansions.

Each of these is independently scopable. Priorities 1 and 2 close the biggest credibility gaps for the existing two clients.

## Honest caveat

Production-grade ingest of paid feeds (Drewry, Freightos paid tier, S&P Global) requires real budget. The free tier upgrades (EU Oil Bulletin, USITC HTS, TARIC) close most of the credibility gap at zero marginal cost.
