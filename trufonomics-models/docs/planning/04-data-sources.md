# 04 — Data Sources

Every data source used in the foundational model stack, with URLs, access method, and notes on vintage handling.

## Contents

- [Truflation (primary data)](#truflation-primary-data)
- [Official comparators](#official-comparators)
- [Cleveland Fed Nowcast (benchmark)](#cleveland-fed-nowcast-benchmark)
- [FRED covariates](#fred-covariates)
- [Commodity and FX data](#commodity-and-fx-data)
- [Housing data](#housing-data)
- [Shipping / trade](#shipping--trade)
- [Weather](#weather)
- [Surveys and sentiment](#surveys-and-sentiment)
- [Industry data for transmission VARs](#industry-data-for-transmission-vars)

---

## Truflation (primary data)

**API:** Subscription-based, details with enterprise team.
**Site:** https://truflation.com/
**Methodology:** https://truflation.com/blog/everything-you-need-to-know-about-truflations-index-methodology
**PCE Index launch:** https://truflation.com/blog/introducing-the-truflation-pce-index
**2026 weights update:** https://blog.truflation.com/truflation-2026-cpi-weights/
**UK CPI vs ONS:** https://truflation.com/blog/truflation-ons-cpi-vs-cpih
**Leading indicator analysis:** https://truflation.com/blog/truflation-leading-indicator-of-us-cpi
**Why below BLS:** https://truflation.com/blog/why-truflations-cpi-number-is-lower-than-the-bls
**Monthly trends blog:** https://blog.truflation.com/trends-driving-us-inflation-april-2026/

**Data ingest:**
- Daily aggregate CPI + 12 categories — available now
- Subcategory and component-level data — coming soon per Angad
- 24-hour delay in Truflation publication (they pull at 10:30 PM UTC, publish 24hr later after QC)
- Base date: January 1, 2010
- Weights updated annually in February using prior-year expenditure data

**Vintage handling:** Truflation's reading for date T is published on date T+1. Store `reference_date = T`, `as_of_date = T+1` (= publication date).

---

## Official comparators

### BLS CPI
- **Headline CPI series on FRED:** `CPIAUCSL` → https://fred.stlouisfed.org/series/CPIAUCSL
- **Core CPI:** `CPILFESL` → https://fred.stlouisfed.org/series/CPILFESL
- **BLS CPI overview:** https://www.bls.gov/cpi/overview.htm
- **BLS CPI FAQ:** https://www.bls.gov/cpi/questions-and-answers.htm
- **Vintage-aware access:** ALFRED (FRED's archival system) — https://alfred.stlouisfed.org/
- **Release calendar:** https://www.bls.gov/schedule/news_release/cpi.htm

**Subindex series IDs (partial list — expand for all Truflation-equivalent categories):**
- Food: `CPIUFDSL`
- Food at home: `CUSR0000SAF11`
- Food away from home: `CUSR0000SEFV`
- Energy: `CPIENGSL`
- Gasoline: `CUSR0000SETB01`
- Housing: `CUSR0000SAH`
- Shelter: `CUSR0000SAH1`
- OER: `CUSR0000SEHC01`
- Rent: `CUSR0000SEHA`
- Medical care: `CPIMEDSL`
- Apparel: `CPIAPPSL`
- Transportation: `CPITRNSL`
- New vehicles: `CUSR0000SETA01`
- Used vehicles: `CUSR0000SETA02`
- Recreation: `CUSR0000SAR`
- Education and communication: `CUSR0000SAE`

### BEA PCE
- **Headline PCE:** `PCEPI` → https://fred.stlouisfed.org/series/PCEPI
- **Core PCE:** `PCEPILFE` → https://fred.stlouisfed.org/series/PCEPILFE
- **BEA releases:** https://www.bea.gov/data/personal-consumption-expenditures-price-index

### Trimmed mean measures (regime proxies)
- **Dallas Fed trimmed mean PCE:** `PCETRIM12M680SFRBDAL` → https://fred.stlouisfed.org/series/PCETRIM12M680SFRBDAL
- **Atlanta Fed sticky-price CPI:** `STICKCPIM159SFRBATL` → https://fred.stlouisfed.org/series/STICKCPIM159SFRBATL
- **Atlanta Fed flexible-price CPI:** `CORESTICKM159SFRBATL` (core sticky)
- **Cleveland Fed median CPI:** `MEDCPIM159SFRBCLE` → https://fred.stlouisfed.org/series/MEDCPIM159SFRBCLE
- **Cleveland Fed median PCE:** https://www.clevelandfed.org/indicators-and-data/median-pce-inflation

---

## Cleveland Fed Nowcast (benchmark)

**Site:** https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting
**User's guide:** https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/indicators-and-data/inflation-nowcasting/nowcasting_users_guide.pdf
**FAQ:** https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/indicators-and-data/inflation-nowcasting/nowcasting_faqs.pdf
**Underlying working paper (Knotek & Zaman 2014 / revised):** https://www.clevelandfed.org/publications/working-paper/2014/wp-1403-nowcasting-us-headline-and-core-inflation
**2024 update (WP 2406):** https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/publications/working-papers/2024/wp2406.pdf
**Real-time assessment paper:** https://www.clevelandfed.org/publications/economic-commentary/2023/ec-202306-real-time-assessment-inflation-nowcasting-cleveland-fed
**Public information contact:** public.information@clev.frb.org

**Access strategy (see `03-checklist.md` section 0.4):**
1. **Email** `public.information@clev.frb.org` requesting historical nowcast archive — direct, clean, request as Head of AI at Truflation
2. **DevTools scrape** — inspect Network tab on the nowcasting page, click through charting tool history for various target periods, find JSON endpoint pattern, write scraper for full archive back to 2014
3. **MacroMicro** has the series hosted — paid tier allows CSV: https://en.macromicro.me/charts/27688/us-cleveland-inflation-cpi
4. **Forward scrape** — daily at 10:30 AM Eastern, store in vintage DB. Build archive from today forward regardless of other options.
5. **Replication files** — check Knotek faculty page and journal appendices for frozen historical slice

**What they publish:** Point nowcasts only for CPI, Core CPI, PCE, Core PCE at monthly MoM, monthly YoY, and quarterly annualized. No density bands on the website.

---

## FRED covariates

All accessed via `fredapi` Python package. API key: https://fred.stlouisfed.org/docs/api/api_key.html

### Rates
- `DGS2` — 2-year Treasury constant maturity
- `DGS10` — 10-year Treasury constant maturity
- `DGS30` — 30-year Treasury
- `MORTGAGE30US` — Freddie Mac 30-year fixed mortgage rate
- `FEDFUNDS` — Effective federal funds rate
- `DFEDTARU` / `DFEDTARL` — target range

### Commodities
- `DCOILWTICO` — WTI spot, daily
- `DCOILBRENTEU` — Brent spot, daily
- `DHHNGSP` — Henry Hub natural gas spot, daily
- `GASREGW` — US retail regular gasoline, weekly (EIA)
- Grain futures — sourced externally or via CME

### FX
- `DTWEXBGS` — Broad trade-weighted dollar index
- `DEXCHUS` — USD/CNY
- `DEXUSEU` — USD/EUR
- `DEXMXUS` — USD/MXN
- `DEXJPUS` — USD/JPY

### Labor
- `UNRATE` — unemployment rate
- `JTSQUR` — quits rate
- `ECIWAG` — Employment Cost Index, wages and salaries
- `FRBATLWGT3MMAUMHWGO` — Atlanta Fed Wage Growth Tracker
- `CIVPART` — labor force participation

### Prices / inflation context
- `T5YIE` — 5-year breakeven inflation
- `T10YIE` — 10-year breakeven inflation
- `EXPINF1YR`, `EXPINF5YR`, `EXPINF10YR` — Cleveland Fed inflation expectations

### Activity
- `INDPRO` — industrial production
- `RSAFS` — retail sales
- `PAYEMS` — nonfarm payrolls

---

## Commodity and FX data

Beyond FRED, for finer granularity:

- **EIA:** https://www.eia.gov/opendata/ — API for all US energy prices, retail gasoline by region
- **CME futures:** for forward curves (WTI, Brent, natural gas, grains)
- **ICE futures:** Brent and soft commodities
- **Oanda / FX** — intraday FX if needed

---

## Housing data

### Rent and price feeds
- **Zillow Research:** https://www.zillow.com/research/data/ — Zillow Rent Index, ZHVI, free CSV download
- **Apartment List:** https://www.apartmentlist.com/research/category/data-rent-estimates
- **RealPage / CoStar** — paid, premium
- **Trulia** — now part of Zillow
- **CoreLogic HPI** — paid
- **Case-Shiller HPI:** `CSUSHPISA` on FRED → https://fred.stlouisfed.org/series/CSUSHPISA

### Mortgage
- **Freddie Mac PMMS:** https://www.freddiemac.com/pmms — weekly primary mortgage market survey
- **ICE Mortgage Technology / Optimal Blue** — rate lock data (paid)
- **HMDA** — Home Mortgage Disclosure Act data, annual

### Inventory / tightness
- **NAR:** existing home sales, months supply
- **Census:** new home sales, housing starts, building permits (`HOUST`, `PERMIT`)
- **Realtor.com** — active listings, days on market

---

## Shipping / trade

### Global shipping
- **Baltic Exchange:** https://www.balticexchange.com/ — Baltic Dry Index (BDI), BDTI, BCTI. Paid for real-time.
- **Drewry WCI:** https://www.drewry.co.uk/supply-chain-advisors/supply-chain-expertise/world-container-index-assessed-by-drewry
- **Shanghai Containerized Freight Index (SCFI):** https://en.sse.net.cn/indices/scfinew.jsp
- **Freightos Baltic Index (FBX):** https://fbx.freightos.com/
- **FRED has some:** `BDIY` (Baltic Dry)

### Trade flows
- **US Census trade data:** https://www.census.gov/foreign-trade/
- **USITC tariff schedule:** https://hts.usitc.gov/ — for tariff regime dummies

---

## Weather

- **NOAA:** https://www.ncei.noaa.gov/access/search/data-search/global-summary-of-the-day
- **Daily HDD/CDD by region:** compute from station-level data
- **NOAA API:** https://www.ncdc.noaa.gov/cdo-web/webservices/v2
- **Drought Monitor:** https://droughtmonitor.unl.edu/ — for food/agriculture

---

## Surveys and sentiment

- **UMich Consumer Sentiment:** `UMCSENT` on FRED
- **Conference Board Consumer Confidence:** paid, or proxy via UMCSENT
- **SPF (Philadelphia Fed):** https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters — quarterly, free CSV archive
- **Blue Chip Economic Indicators:** https://www.wolterskluwer.com/en/solutions/enrs/economic-forecasting — paid subscription
- **Cleveland Fed Survey of Firms' Inflation Expectations (SoFIE):** https://www.clevelandfed.org/indicators-and-data/survey-of-firms-inflation-expectations
- **Atlanta Fed Business Inflation Expectations:** https://www.atlantafed.org/research/surveys/business-inflation-expectations
- **NY Fed Survey of Consumer Expectations:** https://www.newyorkfed.org/microeconomics/sce

---

## Industry data for transmission VARs

### BLS Producer Price Index (industry level)
- **PPI main page:** https://www.bls.gov/ppi/
- **Industry-level PPI series IDs:** see BLS reference or query API
- Key series for transmission verticals:
  - Trucking: `PCU484---484---` family
  - Restaurants: PPI is limited, use PCE components for food + labor costs
  - Retail trade: `PCU44-45--44-45--`
  - Healthcare: extensive PPI coverage

### Census economic indicators
- **Monthly Retail Trade:** https://www.census.gov/retail/
- **Transportation Services Index:** https://www.bts.gov/tsi
- **Advance Economic Indicators Report**

### Industry outputs for VAR endogenous variables
- **FRED industry-level:** `IPMAN` (manufacturing), `IPFINAL`, etc.
- **ISM PMI:** https://www.ismworld.org/ — manufacturing and services PMIs
- **Markit PMI:** S&P Global

### Wage data by sector
- **BLS Current Employment Statistics (CES):** average hourly earnings by industry (`CES` family series)
- **Quarterly Census of Employment and Wages (QCEW)**

---

## Data governance principles

Applied consistently across all ingest:

1. **Every series tagged with `as_of_date`** = date of our pull, not the reference date
2. **Historical backfill clearly marked** — if pulling today, tag vintage as "today" not the original release date (unless original dates are reliably retrievable)
3. **Source hash logged per ingestion** for reproducibility
4. **Append-only** — no updates, no deletes, only new rows
5. **Source metadata** — URL, endpoint, parameters captured with each pull
6. **Schema versioning** — changes to ingest format trigger a schema version bump

This is boring infrastructure work. Skipping it produces fake backtests. Don't skip.
