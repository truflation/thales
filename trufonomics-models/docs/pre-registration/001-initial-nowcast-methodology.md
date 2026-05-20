# 001 — Initial Nowcast Methodology Pre-Registration

**Status:** DRAFT (unlocked)
**Draft date:** 2026-04-24
**Lock date:** pending — cannot lock until `src/thales/evaluation/harness.py` exists and one baseline has been run end-to-end against its spine. Locking without the spine would commit us to an evaluation we cannot yet actually run.
**Version:** v0.1 (first skeleton)
**Scope:** the **initial three-baseline nowcast evaluation** for the Thales-v2 Bayesian/state-space stack. This is the Phase 0 §0.10 pre-reg. Later phases get their own pre-reg docs (002+).

This document is the authoritative specification for the first official evaluation in the Thales-v2 repo. Amendments append to the bottom with a timestamp; text above the lock line is never silently edited.

---

## 1. Task definition

- **Primary target:** BLS CPI YoY — headline — published monthly by the U.S. Bureau of Labor Statistics around day 14 of month `M+1` for the period ending month `M`.
- **Secondary target:** BEA PCE YoY — headline — published monthly by the U.S. Bureau of Economic Analysis around day 25-30 of month `M+1` for the period ending month `M`.
- **Forecast origin:** daily, at 18:00 UTC, using only information publicly available at that timestamp (respects vintage and release calendars).
- **Primary horizons:** current-period nowcast (h=0 months, ≈ 0 to 30 days before the release depending on day of month) and short-forecast (h=+1 month). Wider horizon evaluation (h=3, 6, 12) is pre-registered in a separate forecast-track doc (not this one).
- **Distinct from the Stefan daily-post product.** The Stefan product predicts tomorrow's Truflation daily index value. That is a different target, a different pre-reg (003 when authored), and not what this document commits to.

---

## 2. Data sources (frozen at lock time)

### 2.1 Target series — official releases
- BLS CPI: `CUSR0000SA0` (headline), `CUSR0000SA0L1E` (core) — ingested via `thales.ingest.bls`, source='bls_direct'
- BEA PCE: not yet ingested as a direct BEA feed; for now sourced via FRED series `PCEPI` / `PCEPILFE` which mirror BEA releases with a short lag — must be swapped for direct BEA ingest before first real evaluation

### 2.2 Truflation panel
- 80 TN Network component streams (daily, 2020-01-01 → present), frozen-index variants
- 12 top-level categories + 36 subcategories + 32 components = 80 of 81 taxonomy nodes; one gap (id 158 Misc products & services under Other) flagged to backend
- Weights from `data/truflation/weights/categories-tables-v{1,2}.csv` (v1 ≤ 2025-12-31; v2 ≥ 2026-01-01); composition math validated 2026-04-24 (median residual 0.000 pp)

### 2.3 Public covariates
- FRED: 47 series (rates, yield curve, credit spreads, inflation expectations, commodities, FX, labor, housing, GDP, money). CPI-family deliberately excluded from the covariate set to prevent leakage.
- EIA: 6 energy series (WTI, Brent, Henry Hub, US + PADD-1 + PADD-5 retail gasoline)

### 2.4 Comparator nowcasts
- Cleveland Fed inflation nowcast (CPI, Core CPI, PCE, Core PCE; MoM, YoY, quarterly SAAR) — full historical vintage archive 2013-2026 from the public JSON endpoints, plus daily forward scrape
- **Not-yet-ingested (required before first formal run):** SPF (Philadelphia Fed), Atlanta Fed GDPNow, NY Fed Staff Nowcast. Blue Chip is skipped (paid).

### 2.5 Vintage discipline
- Target series: BLS + BEA prints carry true release-calendar `as_of_date`
- Cleveland Fed: full historical daily vintages (~3,178 as_of dates per series)
- FRED covariates: **current single-as_of tag** (historical pulls tagged with today's date; ALFRED migration is task #72). **Caveat:** pseudo-real-time claims against pre-2026 origins are weaker than they will be post-ALFRED integration. This pre-reg acknowledges the limitation and reports results both ways (as-of-today and as-of-release where available) once the ALFRED migration is complete.
- Truflation: component streams are frozen-index (revision-pinned by design); `as_of_date` = today of ingest. Forward ingests tag `as_of = pull_day`.

---

## 3. Baseline ladder (§0.10 — the three baselines + comparators)

The goal of this evaluation is to establish that the Thales-v2 pipeline works end-to-end by beating the simplest possible prediction. The full archetype SSM + CBDF stack will be evaluated in subsequent pre-regs (002+).

| # | Method | Description |
|---|---|---|
| B0 | Persistence | `CPI[M] = CPI[M−1]` (last publicly-released value); zero-parameter |
| B1 | Truflation-carry | Today's Truflation YoY = next BLS CPI print (single-number heuristic) |
| B2 | OLS optimal-lag | Walk-forward OLS with `(persistence, Truflation_day25, gasoline_YoY_day25)` as features; same spec as kairos Path A for direct cross-repo comparability |
| B3 | Simple UCM-with-Truflation | Unobserved-components state-space with Truflation as observed driver and BLS as target; first real SSM of the new stack. Built in `statsmodels.UnobservedComponents` first; NumPyro rewrite is a later ablation |
| C1 | Cleveland Fed published nowcast | Their CPI / Core CPI monthly-MoM and monthly-YoY values at the matching `as_of_date` (from our historical archive). Primary external comparator |
| C2 | SPF consensus | When ingested. Secondary external comparator |
| N1 | Naive random walk | Trivial null hypothesis |

---

## 4. Evaluation specification

### 4.1 Walk-forward protocol
- **Origins:** daily from 2014-01-01 through latest release (Cleveland Fed comparator starts 2013-08; use 2014-01-01 as the effective window start)
- **Training history lower bound:** 2010-01-01 for covariates; 2021-01-01 for the Truflation live period
- **Minimum training window for fitted baselines (B2, B3):** 24 months
- **Target known:** a target is in the eval set only if the official BLS release has happened by the time the prediction would be scored

### 4.2 Metrics
- **Primary point:** RMSE reduction vs persistence (B0): `1 − RMSE(method) / RMSE(B0)`
- **Primary density:** CRPS (sample-based; bootstrap residuals for the linear baselines, posterior samples for the SSM)
- **Calibration:** PIT histogram + KS p-value; 80% and 95% interval coverage
- **Point secondaries:** MAE, MASE, directional accuracy
- **Density secondary:** log predictive score (Gaussian approximation)
- **Statistical significance:** Diebold-Mariano (non-nested), Clark-West (nested), Newey-West HAC lag-3. p-values are supporting, not decisive.
- **Uncertainty:** bootstrap 95% CI on every headline statistic, 1,000 paired-row resamples

### 4.3 Regime stratification (mandatory — every headline table stratified)
- **Stable low inflation:** 2015-01 → 2019-12, 2024-10 → present-baseline
- **COVID shock:** 2020-03 → 2020-12
- **Inflation surge:** 2021-01 → 2023-07
- **Disinflation:** 2023-08 → 2024-09
- **Tariff regime:** 2025-04 → present (overlaps with baseline regimes)
- **Government shutdown (BLS disruption):** 2025-10, 2026-01 (special flag)

### 4.4 Horizon stratification
- Nowcast at release minus 15 days (h = −15)
- Nowcast at release (h = 0)
- Short forecast h = +1 month

---

## 5. Success criteria

A method is considered to have **meaningful nowcast signal** when ALL of the following hold, on the full sample:

1. RMSE reduction vs persistence ≥ **20%** (realistic bar per Medeiros 2021 ~30% peer-reviewed benchmark and kairos Path A's +42% prior)
2. Bootstrap 95% CI on RMSE reduction excludes zero
3. Method does not materially underperform persistence on MAE
4. 80% density band has empirical coverage in `[0.75, 0.85]` (± 5 pp of nominal)

**Robust** = condition (1) holds in ≥ 2 of the 5 regime sub-periods listed in §4.3.

**Stopping rule:** if after all three baselines (B1, B2, B3) are run, none achieves **meaningful** signal, the pre-reg freezes the conclusion as: *"Thales-v2 Phase 0 baselines do not clear the 20% RMSE-reduction bar on daily/monthly US CPI nowcasting. Phase 1 archetype work proceeds only after the aggregate-level pipeline is fixed."* (This is a sanity gate — if the simplest methods can't beat persistence meaningfully, something in the pipeline is broken.)

---

## 6. Methods explicitly considered and decided

Outcome of the 2026-04-24 methodology review (see `docs/planning/01-architecture.md` §Methodology review 2026-04-24).

### 6.1 Added — HRNN as an alternative archetype class
Hierarchical Recurrent Neural Networks (Benchimol et al. 2022, International Journal of Forecasting) added as a **6th archetype class** available to any category where the structural SSM fit is weak. Does not replace any planned SSM. Benchmarked alongside archetype SSMs in every per-category ablation in Phase 1. This Phase 0 pre-reg does NOT evaluate HRNN at the aggregate level; that work lands in pre-reg 002 alongside the first archetype SSMs.

### 6.2 Rejected — zero-shot pretrained TSFMs as primary forecaster
Chronos, TimesFM, Moirai, TimeMoE, Toto, Sundial evaluated. Literature evidence (Chen & Kelly et al., arXiv 2511.18578, Nov 2025) shows zero-shot TSFMs underperform CatBoost/LightGBM on financial returns; the transferable lesson is that zero-shot pretrained TSFMs on macro data are unlikely to beat domain-tuned econometric methods. **Not included in the baseline ladder.** If ever revisited, only via fine-tuning on Truflation component data as a separate research program.

### 6.3 Not in scope for this pre-reg
- CBDF composition layer (Phase 2 — pre-reg 002)
- UC-SV-MS regime model (Phase 2 — pre-reg 003)
- Transmission VARs (Phase 3 — pre-reg 004+)
- Stefan daily-post product (separate product + pre-reg when it goes live)

---

## 7. Blockers before lock

Pre-reg cannot move from DRAFT to LOCKED until:

1. `src/thales/evaluation/harness.py` + `src/thales/evaluation/scoring_db.py` exist and self-test passes on a synthetic DGP (task #70)
2. SPF ingest exists (at least historical archive) OR this pre-reg is amended to remove SPF as a required comparator
3. BEA PCE direct ingest exists OR this pre-reg is amended to accept FRED-mirror PCE as the target
4. One baseline (B0 persistence is the easiest — it has no model to fit) has been run end-to-end through the harness and scored against Cleveland Fed historical vintages, producing a usable scoring-DB row and a dashboard-ready metric table

When (1)–(4) are satisfied, change `Status: DRAFT (unlocked)` above to `Status: LOCKED — YYYY-MM-DD` and lock this file. All further changes append to the Amendment log.

---

## 8. What this pre-reg does NOT claim

- Does not claim the full Thales-v2 architecture (archetype SSMs + CBDF + regime + transmission) works. That's Phase 1/2/3.
- Does not claim anything about the Stefan daily-post product.
- Does not claim Thales-v2 beats the kairos Path A / Gated v2 result — those are cross-pre-reg comparisons that land later.
- Does not commit to a specific NumPyro vs statsmodels implementation for B3; whichever ships first is fine, ablation lands in pre-reg 002.

---

## Amendment log

*(empty — no amendments yet)*
