# 06 — References

All papers, blog posts, methodological references, and key URLs mentioned in the planning docs.

## Contents

- [Truflation methodology and research](#truflation-methodology-and-research)
- [Nowcasting methodology — frontier research](#nowcasting-methodology--frontier-research)
- [Cleveland Fed (primary benchmark)](#cleveland-fed-primary-benchmark)
- [State-space inflation modeling](#state-space-inflation-modeling)
- [Sticky/flexible and disaggregated inflation](#stickyflexible-and-disaggregated-inflation)
- [GDP nowcasting (adjacent literature)](#gdp-nowcasting-adjacent-literature)
- [Tools and libraries](#tools-and-libraries)
- [Data portals](#data-portals)

---

## Truflation methodology and research

- **Methodology (comprehensive):** https://truflation.com/blog/everything-you-need-to-know-about-truflations-index-methodology
- **Methodology PDF:** https://truflation.com/Methodology.pdf
- **Introducing Truflation PCE Index:** https://truflation.com/blog/introducing-the-truflation-pce-index
- **Truflation vs BLS CPI analysis:** https://truflation.com/blog/why-truflations-cpi-number-is-lower-than-the-bls
- **Truflation as leading indicator of US inflation:** https://truflation.com/blog/truflation-leading-indicator-of-us-cpi (quantified lead analysis, regime-dependent)
- **UK CPI vs ONS comparison:** https://truflation.com/blog/truflation-ons-cpi-vs-cpih
- **2026 CPI weights update:** https://blog.truflation.com/truflation-2026-cpi-weights/
- **Monthly inflation driver blog:** https://blog.truflation.com/trends-driving-us-inflation-april-2026/

Third-party coverage:
- **Gryphon FP on Truflation:** https://gryphonfp.com/blog/truflation-a-high-frequency-inflation-measurement/
- **InflationData.com comparison:** https://inflationdata.com/articles/2026/04/13/truflation-compared-to-cpi/
- **Gryphon on inflationary signals from Truflation:** https://gryphonfp.com/blog/what-the-truflation-measure-is-telling-us-about-inflationary-signals/

---

## Nowcasting methodology — frontier research

### CBDF (composition layer — our primary reference)
- **O'Keeffe, H. & Petrova, K. (2025), "Component-Based Dynamic Factor Nowcast Model"** — NY Fed Staff Report 1152.
  - NY Fed page: https://www.newyorkfed.org/research/staff_reports/sr1152
  - SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5230835
  - 15% RMSE improvement and 20% log-score improvement over standard DFM. Beats NY Fed's current Almuzara-Baker-O'Keeffe-Sbordone (2023) model and Atlanta Fed's Higgins (2014) GDPNow.

### Time-varying mixed-frequency DFM
- **TVP-MF-DFM-SV (Bayesian dynamic model averaging):** https://www.sciencedirect.com/science/article/pii/S0169207022001078 — fast Kalman-filter-based algorithm, handles mixed frequencies and time-varying parameters with SV. Reference for our archetype model design.

### ML vs econometric comparison
- **IMF WP 2025/252 — Traditional econometric vs ML for GDP nowcasting:** https://www.imf.org/en/publications/wp/issues/2025/12/05/gdp-nowcasting-performance-of-traditional-econometric-models-vs-machine-learning-572360
  - Finding: traditional econometric models (bridge, DFM) tend to outperform ML; linear ML (LASSO, Elastic Net) is the only ML class that sometimes competes.

### Neural + DFM hybrid
- **NCDENow (Neural Controlled Differential Equations + DFM):** https://arxiv.org/abs/2409.08732
  - Hybrid model combining DFM interpretability with neural net flexibility for irregular time series. Useful reference but econometric base is what we prioritize.

### Bayesian DFM with unknown factor number
- **Horseshoe-prior DFM:** https://www.mdpi.com/2227-7390/9/22/2865
- **Bayesian SV DFM:** https://www.sciencedirect.com/science/article/abs/pii/S2452306221001039

### ML for DFM-nowcasting benchmark
- **Kant, Pick & Winter (2025) "Nowcasting GDP using ML":** https://link.springer.com/article/10.1007/s10182-024-00515-0

---

## Cleveland Fed (primary benchmark)

- **Nowcasting page:** https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting
- **User's guide:** https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/indicators-and-data/inflation-nowcasting/nowcasting_users_guide.pdf
- **FAQ / model overview:** https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/indicators-and-data/inflation-nowcasting/nowcasting_faqs.pdf
- **Knotek & Zaman (2014/rev) working paper:** https://www.clevelandfed.org/publications/working-paper/2014/wp-1403-nowcasting-us-headline-and-core-inflation
- **Knotek & Zaman 2024 — real-time point and density nowcasting:** https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/publications/working-papers/2024/wp2406.pdf
- **Real-time assessment (2023 Economic Commentary):** https://www.clevelandfed.org/publications/economic-commentary/2023/ec-202306-real-time-assessment-inflation-nowcasting-cleveland-fed
- **Public information contact:** public.information@clev.frb.org

Also worth following:
- **Cleveland Fed Center for Inflation Research:** https://www.clevelandfed.org/center-for-inflation-research
- **Cleveland Fed all indicators:** https://www.clevelandfed.org/indicators-and-data

---

## State-space inflation modeling

### Primary references for UC-SV-MS and nowcasting
- **BSP WP 2022-02 — State-space approach to nowcasting inflation with high-frequency data (Philippines):** https://www.bsp.gov.ph/Media_And_Research/WPS/WPS202202.pdf
  - Excellent reference for applying state-space models to mixed-frequency inflation nowcasting.
- **BIS WP 713 — Mertens, "Inflation and professional forecast dynamics":** https://www.bis.org/publ/work713.pdf
  - UC-SV-TVP with sticky information. Methodological foundation for our regime model.
- **RBA Bulletin 2019 — Disaggregated component inflation modeling:** https://www.rba.gov.au/publications/bulletin/2019/jun/explaining-low-inflation-using-models.html
  - Component-level inflation modeling at a central bank.

### Robust inflation forecasting
- **Improving inflation forecasts using robust measures (SkewKLT + trimmed mean):** https://www.sciencedirect.com/science/article/abs/pii/S016920702300047X
  - Cross-sectional skewness and trimmed-mean as inputs to density forecasts. Stochastic volatility in mean model.

### Aptech tutorial and teaching materials
- **Aptech blog: Understanding state-space models (inflation example):** https://www.aptech.com/blog/understanding-state-space-models-an-inflation-example/
- **Donsker class — teaching notes on state space models:** https://donskerclass.github.io/Forecasting/StateSpace.html

---

## Sticky/flexible and disaggregated inflation

- **Boston Fed 2025 — "Transitory or Persistent? Frequency of Price Changes":** https://www.bostonfed.org/publications/current-policy-perspectives/2025/frequency-of-price-changes-and-the-nature-of-inflation.aspx
  - Bils-Klenow sticky/flex decomposition, updated with post-pandemic data. Foundation for our regime model's component classification.
- **Richmond Fed — Forecasting Inflation overview:** https://www.richmondfed.org/publications/research/econ_focus/2021/q4_federal_reserve
- **Cleveland Fed median CPI:** https://www.clevelandfed.org/indicators-and-data/median-cpi
- **Cleveland Fed median PCE:** https://www.clevelandfed.org/indicators-and-data/median-pce-inflation
- **Atlanta Fed sticky-price CPI:** https://www.atlantafed.org/research/inflationproject/stickyprice
- **Dallas Fed trimmed mean PCE:** https://www.dallasfed.org/research/pce

---

## GDP nowcasting (adjacent literature, useful when we extend beyond inflation)

- **NY Fed Staff Nowcast:** https://www.newyorkfed.org/research/policy/nowcast — weekly GDP nowcast, archive publicly available
- **Atlanta Fed GDPNow:** https://www.atlantafed.org/cqer/research/gdpnow — GDPNow model, downloadable archive

---

## Tools and libraries

### State-space modeling
- **dynamax (JAX-backed SSMs):** https://github.com/probml/dynamax — Kalman, particle, switching, nonlinear in one API
- **statsmodels:** https://www.statsmodels.org/ — VECM, UCM, MarkovSwitching reference implementations
- **tsfoos (time series Forecasting with Bayesian):** for reference

### Bayesian inference
- **NumPyro:** https://num.pyro.ai/ — JAX-backed, NUTS, primary Bayesian tool
- **PyMC 5:** https://www.pymc.io/ — alternative, secondary
- **CmdStanPy:** https://mc-stan.org/cmdstanpy/ — Stan interface, fallback for complex models
- **Blackjax:** https://blackjax-devs.github.io/blackjax/ — JAX sampling primitives

### Scoring
- **properscoring (CRPS, log score):** https://github.com/properscoring/properscoring

### Data
- **fredapi:** https://github.com/mortada/fredapi — FRED Python client
- **DuckDB:** https://duckdb.org/ — vintage store backend
- **polars:** https://pola.rs/ — fast DataFrame

### Storage and deployment
- **Vast.ai:** https://vast.ai/ — GPU rental
- **Streamlit:** https://streamlit.io/ — dashboard
- **Prefect:** https://www.prefect.io/ — orchestration when needed

---

## Data portals

- **FRED:** https://fred.stlouisfed.org/
- **ALFRED (vintage FRED):** https://alfred.stlouisfed.org/
- **BLS:** https://www.bls.gov/
- **BEA:** https://www.bea.gov/
- **Census:** https://www.census.gov/
- **EIA:** https://www.eia.gov/opendata/
- **NOAA:** https://www.ncei.noaa.gov/
- **Philadelphia Fed SPF:** https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters
- **Zillow Research:** https://www.zillow.com/research/data/
- **USITC tariff schedule:** https://hts.usitc.gov/

---

## Related context references (for our positioning)

- **MacroMicro Truflation chart:** https://en.macromicro.me/charts/116496/us-truflation-index-yoy
- **MacroMicro Cleveland Fed nowcast (CPI):** https://en.macromicro.me/charts/27688/us-cleveland-inflation-cpi
- **MacroMicro Cleveland Fed nowcast (PCE):** https://en.macromicro.me/charts/27686/us-cleveland-inflation-pce
- **Grokipedia on Truflation US Inflation Index:** https://grokipedia.com/page/Truflation_US_Inflation_Index
- **Babypips Forexpedia Truflation definition:** https://www.babypips.com/forexpedia/truflation
