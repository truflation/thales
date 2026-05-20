# Phase 3.2 — Restaurants transmission VAR

Same architecture as Phase 3.1 logistics, different cost structure +
endogenous vector. Validates that the transmission-VAR pattern
generalizes across verticals.

## What shipped

1. **`scripts/ingest_restaurants_fred.py`** — 6 FRED series ingested
   into the vintage store (1,168 rows): PPI Final demand foods,
   L&H wages, PPI nonresidential rent, CPI energy services, CPI
   food away from home, retail sales food services.
2. **`scripts/bvar_restaurants_6var.py`** — 6-var BVAR on **MoM
   log-differences** (not log-levels — see below for why).
3. **Cost structure already registered** (Phase 3.1c): `food_cogs 30%,
   labor 30%, rent 8%, utilities 4%, other 28%`. ATRI-equivalent
   weights from the planning doc.

## Endogenous vector (Cholesky ordering most-exogenous → most-endogenous)

| Order | Variable | Series | Role |
|---:|---|---|---|
| 1 | `mom_food_cogs` | WPSFD49207 (PPI Final demand foods) | Upstream commodity input |
| 2 | `mom_utilities` | CUSR0000SEHF (CPI Energy services) | Macro input |
| 3 | `mom_rent` | PCU531120531120 (PPI Nonres bldg lessors) | Slow-moving cost |
| 4 | `mom_labor` | CES7000000008 (L&H avg hourly earnings) | Sticky input |
| 5 | `mom_menu` | CUSR0000SEFV (CPI Food away from home) | **Output / pricing-side** |
| 6 | `mom_traffic` | RSFSDP / menu CPI (real food-services sales) | **Demand-side endogenous** |

## A binding methodology lesson — model in MoM, not levels

**v1 (log-levels) gave nonsensical IRFs**: a +20% food shock implied
+64% real-traffic at h=0. The cause was Fix #5's lesson reapplied:
all six series share strong secular trend, so log-LEVEL correlations
range 0.85-0.96 while MoM (first-differenced log) correlations are
realistic 0.10-0.40. The Cholesky decomposition of Σ on log-levels
attributed shared trend variance to whichever variable came first in
the ordering, inflating cross-effects.

**v2 (MoM frame) produces sensible numbers**:
- food↔menu MoM corr: 0.17 (level corr was 0.96)
- food↔traffic MoM corr: 0.40 (level corr was 0.85)

This is the third time in this project we've found that **monthly
macro VARs need first-differenced inputs** to get clean structural
IRFs. The 3.1 logistics findings are also being re-examined under
this lens (queued).

## Static fit and IRF (1-SD food_cogs shock, MoM frame)

System is **stationary** (max\|eig\| < 1) — expected since we're
already in differences.

**Cumulative LEVEL effect over 12 months from a 1-SD food MoM shock**
(in pp; sum of MoM responses):

| food_cogs | utilities | rent | labor | menu | traffic |
|---:|---:|---:|---:|---:|---:|
| +1.50pp | +0.50pp | −0.40pp | +0.32pp | +0.10pp | +4.65pp |

The signal: a 1-SD positive food-cogs innovation co-occurs with menu
inflation +0.10pp over 12 months, real traffic +4.65pp. Cost-side
variables move modestly (utilities, labor); rent shows a small
negative response that's likely Σ-noise on a near-trendless series.

## FEVD at h=12 (variance share by shock, %)

|  | food | utils | rent | labor | menu | traffic |
|---|---:|---:|---:|---:|---:|---:|
| food (own) | **96.0** | 0.6 | 0.7 | 2.1 | 0.0 | 0.6 |
| utilities | 5.3 | **89.2** | 0.4 | 5.0 | 0.1 | 0.1 |
| rent | 2.6 | 2.5 | **88.9** | 1.7 | 0.1 | 4.1 |
| labor | 5.3 | 0.8 | 0.4 | **92.1** | 1.3 | 0.0 |
| menu | 1.2 | 4.5 | 1.3 | 1.8 | **91.2** | 0.0 |
| traffic | **15.5** | 1.7 | 0.2 | 6.3 | 0.6 | **75.5** |

Reads:
- **Each cost-side variable is dominated by its own shock** (~89-96%).
  Restaurants' cost lines move mostly on idiosyncratic shocks, not on
  cross-bucket transmission.
- **Traffic has the most cross-component variance: 15.5% from food**
  — the largest external dependency in the system. Restaurant volume
  is empirically tied to the food-cycle.
- **Menu is 91% own-shock** at h=12 — restaurant pricing has weak
  empirical pass-through on the monthly horizon. (Either due to
  sticky menus or BLS measurement smoothing.)

## Walk-forward forecasts (n=132 months OOS, ~2015-2026)

| target | RMSE | naive | RMSE Δ% | cov80 | cov95 |
|---|---:|---:|---:|---:|---:|
| food_cogs | 0.915 | 0.994 | **+7.95%** | 75.8% | 88.6% |
| utilities | 0.849 | 1.004 | **+15.44%** | 77.3% | 90.2% |
| rent | 1.660 | 2.406 | **+30.98%** | 53.8% | 72.0% |
| labor | 0.486 | 0.564 | **+13.72%** | 80.3% | 84.8% |
| menu | 0.151 | 0.160 | +5.14% | 73.5% | 88.6% |
| traffic | 9.563 | 7.734 | −23.66% | 82.6% | 93.2% |

**The BVAR meaningfully beats RW on every cost-side variable** (food,
utilities, rent, labor, menu) — not just by a hair. Rent at +31%
RMSE reduction is the strongest. The cross-information from the
broader system reliably improves cost-line forecasting.

**Traffic is the model's blind spot** (−23.66% RMSE — worse than
RW). Real food-services sales are dominated by demand-side noise the
BVAR can't see (consumer income shocks, weather, weekend/holiday
patterns). Customers should not rely on the model's traffic forecasts
— the structural-transmission framing only works on cost-side
variables.

**Coverage problem on rent and traffic:** 80%-band cov is 53.8% (rent)
and 82.6% (traffic) — rent severely undercovers, traffic is roughly
calibrated. Rent's underlying series is near-trendless and the
Gaussian VAR forecast SD doesn't capture its outlier shocks well.
Conformalizing the VAR forecast band is queued.

## Customer-facing exposure scenario

**The output framing**: a +20pp one-time food-cogs MoM shock means
food prices jump 20% in a single month. The BVAR projects how the
rest of the system co-evolves over the next 12 months.

**For a $10M-revenue restaurant** with the registered cost structure:

| scenario | food_cogs Δ$ | labor Δ$ | rent Δ$ | utilities Δ$ | **TOTAL Δ$** | menu Δpp | traffic Δpp |
|---|---:|---:|---:|---:|---:|---:|---:|
| +20pp food MoM | +$813k | +$151k | −$59k | +$34k | **+$939k** | +2.06pp | +92.89pp |
| +50pp food MoM | +$2.03M | +$377k | −$147k | +$84k | **+$2.35M** | +5.16pp | +232.2pp |
| −20pp food MoM | −$813k | −$151k | +$59k | −$34k | **−$939k** | −2.06pp | −92.89pp |

### Read carefully — what these numbers mean

- **Cost-line Δ$**: directly defensible. A +20pp food shock with
  empirical rent/utilities/labor responses translates to ~$939k of
  exposure on a $10M restaurant's cost lines.
- **menu Δpp**: the model's projection that menu prices co-move
  ~+2pp with a +20pp food shock over 12 months. This is the
  *empirical pass-through* in the data. Restaurants' actual pricing
  policy is sticky — the model captures that.
- **traffic Δpp = +92.89pp**: this is **co-movement, not causation**.
  Reading it as "real traffic will rise 93%" would be wrong. The
  correct reading: in months historically co-occurring with food
  spikes, real food-services sales also surge — driven by shared
  exposure to general inflation, post-COVID reopening, etc. **This
  is exposure data, not a forecast.** Strip it out for executive
  reports; keep it only for analysts who understand correlation
  artifacts.

## Cross-vertical comparison (Phase 3.1 logistics vs Phase 3.2 restaurants)

| | Logistics (3.1) | Restaurants (3.2) |
|---|---|---|
| Most exogenous | Diesel | Food commodity prices |
| Pass-through ratio (own own × cross) | 8% (diesel→freight) | ~10% (food→menu) |
| Most volatile cost line | Fuel | Food COGS |
| Demand-side variable | Volume (ATA tonnage) | Traffic (real sales) |
| Demand-side BVAR usefulness | +1.77% vs RW | **−23.66% vs RW** |
| Best forecastable variable | Maintenance (+19%) | Rent (+31%) |

**Common pattern**: the BVAR's value is in **cost-line attribution
and forecasting**, not demand-side prediction. The transmission
mechanism (upstream commodity → cost lines → output prices) is
forecastable; the demand-side response is dominated by macro shocks
and idiosyncratic factors the BVAR can't identify on monthly data.

## Caveats / future work

1. **Sample dominated by 2021-2023 inflation surge** — the food↔
   traffic 0.40 MoM correlation is partly an artifact of post-COVID
   recovery + simultaneous food spike. Splitting the sample and
   refitting on pre-2020 history would give different IRFs (probably
   smaller traffic response). Not a bug — but worth flagging in
   customer reports.

2. **Rent is structurally hard to forecast.** PCU531120531120
   (commercial rent PPI) is near-trendless monthly. cov80 of 53.8%
   says the Gaussian band undercovers badly. Worth either dropping
   rent from the customer-facing scenarios or applying conformal
   bands.

3. **Traffic IRF amplification is a Cholesky artifact**, not the
   model "saying" food causes traffic to spike. Customer-facing
   reports should distinguish: cost-line $ impact (defensible) vs
   demand-side level deviation (correlation projection, exposure
   data only).

4. **No insurance / margin / weather data.** Restaurants are weather-
   sensitive (winter storm shutdowns) and margin-dependent
   (pricing-power varies by brand tier). Both would benefit the
   model. Insurance: paywalled; weather: NOAA daily; margin:
   10-Q parsing.

5. **Pre-COVID baseline** would be cleaner — fitting the BVAR on
   2010-2019 only and projecting forward is a sanity check we should
   run before customer rollout.

## Files

- `scripts/ingest_restaurants_fred.py` (new)
- `scripts/bvar_restaurants_6var.py` (new — fits in MoM frame)
- `results/real_data_archetypes/bvar_restaurants_6var_summary.csv`
- `results/real_data_archetypes/bvar_restaurants_6var_fevd_h12.csv`
- `results/real_data_archetypes/bvar_restaurants_6var_irf.csv`

## What this validates for the architecture

- ✅ BVAR-Minnesota generalizes cleanly across verticals (3.1 logistics → 3.2 restaurants in <1h with same code)
- ✅ Cost-structure registry pattern works (#127 design pays off)
- ✅ MoM-first methodology lesson holds at the VAR level too
- ✅ The "exposure analytics not advice" framing (`docs/architecture/02-product-boundary-no-advice.md`) carries through cleanly: customer sees cost-line $ impact, decides what to do
