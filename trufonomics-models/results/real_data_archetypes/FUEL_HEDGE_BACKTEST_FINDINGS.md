# Phase 3.1e — fuel-hedging economic-value backtest

Tests whether the BVAR's structural information actually improves
**hedge sizing in dollars** for a $100M-revenue logistics shipper
bearing ~$30M annual fuel exposure. v1 produced misleading results
(static 50% hedge with the wrong instrument); this v2 re-does it
with optimal hedge ratios and the best tradeable instrument we could
find on FMP.

## What shipped

1. **`thales.ingest.fmp`** — FMP commodities ingest module using the
   stable `/historical-price-eod/light` endpoint. v3 endpoints are
   legacy.
2. **9 commodity series** ingested into the vintage store: HOUSD, CLUSD,
   BZUSD, RBUSD, NGUSD futures + DBO, USO, UGA, XLE ETFs (24,880
   daily rows total, 2010-2026).
3. **`scripts/fuel_hedge_backtest_v2.py`** — 5-strategy comparison on
   130 months of OOS data (2015-01 → 2025-12).

## The headline finding — there's a low ceiling

Even the **PPI Diesel Fuel index itself** (WPU057303 — a wholesale
diesel price measure, FRED-published) only correlates **0.74 monthly**
with retail diesel (GASDESW). That's the **theoretical hedging
ceiling** — and it's set by retail-diesel's irreducible idiosyncratic
variance from margins, taxes, and regional distribution costs.

Best tradeable hedge correlations vs retail diesel:

| Instrument | Monthly corr |
|---|---:|
| DBO (Invesco DB Oil ETF) | **0.60** |
| BNO (Brent Oil Fund ETF) | 0.59 |
| HOUSD (heating oil futures) | 0.60 |
| BZUSD (Brent futures) | 0.56 |
| USO (US Oil Fund) | 0.56 |
| CLUSD (WTI futures) | 0.51 |
| UGA (Gasoline Fund) | 0.49 |
| RBUSD (RBOB gasoline futures) | 0.46 |
| XLE (Energy Select SPDR) | 0.42 |
| NGUSD (Natural Gas) | 0.11 |

**No tradeable instrument correlates above 0.60** with retail diesel.
Multi-instrument basket OLS (DBO + HOUSD + RBUSD) maxes out at **R² =
0.34**, implying a theoretical maximum variance reduction of **18.5%**.
That's the wall.

## Strategy comparison (n=130 months OOS, 2015-01 → 2025-12)

| Strategy | annual_sd | **σ-reduction** | annual_mean | total P&L |
|---|---:|---:|---:|---:|
| 1. Unhedged | $412,750 | (baseline) | −$82k | −$890k |
| 2. **Static DBO** (β-optimal, full-sample) | $348,728 | **+15.5%** | −$46k | −$502k |
| 3. **Rolling DBO** (β-optimal, 36-month) | $357,555 | **+13.4%** | −$54k | −$589k |
| 4. Rolling basket OLS (DBO+HOUSD+RBUSD) | $380,918 | +7.7% | −$16k | −$175k |
| 5. BVAR-modulated basket | $382,982 | +7.2% | −$26k | −$281k |

### Reads

1. **All strategies actually reduce variance** (vs v1 where the wrong
   hedge ratio added noise). The fix was the **β-optimal hedge ratio**:
   `h* = corr × σ_diesel / σ_hedge`, which gave DBO a hedge ratio of
   0.30 (not 0.50 or 1.0).

2. **Static DBO with optimal ratio is the winner**: 15.5% σ-reduction
   — within 3pp of the theoretical 18.5% ceiling. The rolling version
   gives up only 2pp because the optimal ratio is reasonably stable
   across the sample.

3. **Multi-instrument basket UNDERPERFORMS single DBO** (7.7% vs
   13.4%). DBO and HOUSD are correlated > 0.85, so the basket weights
   are unstable in rolling OLS — classic multicollinearity problem.
   The basket isn't worth the complexity.

4. **BVAR-modulated basket adds no value** (7.2% vs 7.7% unmodulated).
   The volatility-regime multiplier (1.5× when σ above median, 0.5×
   below) doesn't help. The BVAR's diesel-vol signal at the monthly
   horizon isn't predictive enough to justify dynamic sizing — by the
   time vol is "high" in the BVAR posterior, the spike has already
   happened.

5. **Mean P&L slightly improves with hedging** (−$82k → −$46k for
   static DBO) because diesel had a small upward drift over the
   sample, and being long the hedge instrument captured part of that.
   But the *Sharpe is still negative* for everything — the hedge
   doesn't add a profit edge, it just cuts variance.

## What this is — and what this is NOT

**This backtest is an internal validation.** It tests whether the
BVAR's monthly volatility signal is sharp enough to drive
*action* (specifically: dynamic hedge sizing) better than naive
methods. **Answer: no, not at the monthly frequency.** The five
strategies cluster within 8pp on σ-reduction; the BVAR's modulation
adds nothing over a static β-optimal hedge.

**This is NOT a product feature.** Thales does not give hedging
advice. Thales does not recommend instruments. Thales does not size
positions. Those are RIA-regulated activities (suitability,
fiduciary duty, E&O insurance) and they're a different business.

### Thales' actual product surface

What Thales **does** deliver, validated by the work in
`BVAR_LOGISTICS_5VAR_FINDINGS.md` and `BVAR_CONDITIONAL_FINDINGS.md`:

- **Outcome-exposure scenarios.** "If diesel +20%, your fuel cost
  line moves +$5.3M / 12 months. 8% of your freight revenue line
  partially offsets via pass-through. Maintenance and labor are
  uncorrelated with this shock at the monthly horizon."
- **Cost-bucket attribution.** What fraction of your operating-cost
  variance comes from each macro variable (FEVD).
- **Pass-through tracking.** How much of an upstream price shock
  shows up in your revenue line at each horizon.

What customers do with that information — hedge, raise prices,
revise budgets, raise capital, do nothing — is **their decision**,
not Thales'. Thales is a **data and analytics product**, not an
advisor.

### Why the distinction matters operationally

- **No RIA registration** required → faster go-to-market
- **No suitability obligation** → uniform product across customers
- **No fiduciary duty** → simpler T&Cs, lower E&O premiums
- **Customers retain agency** → no "we told you to hedge and you
  lost money" liability
- **Cleaner pricing** → SaaS subscription per dashboard, not AUM-based

### Why this backtest is still useful

It's a **methodology validation**: it confirms that the BVAR's
predictive power on monthly diesel volatility is below the threshold
needed to make actionable hedge-sizing claims. That's what tells us
**not** to add a "recommended hedge ratio" feature to the product —
even if a customer asked for it. The honest answer is "the underlying
signal isn't sharp enough at this cadence for us to put a number on
that with confidence." Saying so explicitly is more credible than
shipping a feature that adds no measurable value.

## Caveats

1. **Monthly frequency is the binding constraint.** Daily diesel-vs-
   DBO correlation is likely higher (~0.7+) — at daily, futures and
   spot move together more tightly. Monthly aggregation washes out
   correlated short-term noise. A weekly backtest might show better
   numbers; queued for v3.

2. **Hedge sizing here ignores margin requirements.** Real-world,
   futures positions need posted margin (~10-15% of notional). The
   strategy P&L doesn't account for capital tied up in margin or for
   funding costs. Material for ROI calculations but doesn't change
   the σ-reduction conclusion.

3. **No rollover costs or basis risk.** Continuous front-month
   futures backtest assumes free roll. Real shippers using HOUSD pay
   roll yield (currently ~2-4%/year in contango). The basket
   strategies are particularly sensitive — three contracts to roll.

4. **The BVAR-modulation logic was simple.** A more sophisticated
   regime-detection signal (e.g. the Markov-switching diesel-vol
   from Phase 1.3) might add real value. v2 used a crude threshold
   on the BVAR's Σ diagonal.

5. **Out-of-sample window is 2015-2025**, which includes COVID
   crashes (2020) and the 2022 fuel spike. Different regime mixes
   would give different numbers. The 18.5% ceiling is regime-
   dependent — could be higher in stable periods, lower in crises.

## Files

- `src/thales/ingest/fmp.py` (new — FMP stable-endpoint ingest)
- `scripts/ingest_fmp_commodities.py` (new)
- `scripts/fuel_hedge_backtest_v2.py` (new — proper β-optimal backtest)
- `results/real_data_archetypes/fuel_hedge_backtest_v2.csv`
- `results/real_data_archetypes/fuel_hedge_backtest_v2_summary.csv`

## Glossary (stats terms)

- **β-optimal hedge ratio:** `h* = cov(target, hedge) / var(hedge)
  = corr × σ_target / σ_hedge`. The minimum-variance hedge size for
  one unit of underlying exposure. Standard textbook (e.g. Hull,
  *Options, Futures and Other Derivatives*, Ch. 3).
- **Hedge effectiveness / R²:** the fraction of target variance
  explained by the hedge instrument. Equals corr² for a single hedge,
  or the OLS R² for a basket. Caps the achievable σ-reduction.
- **Multicollinearity in basket hedging:** when two hedge instruments
  are highly correlated (DBO and HOUSD ≈ 0.85), the OLS coefficients
  on each become unstable across rolling windows. A single instrument
  is often more robust than a basket.
- **Sharpe of a hedge program:** annualized mean P&L / annualized SD
  of P&L. A pure hedge has Sharpe ≈ 0 (no expected return, just
  variance reduction). Anything > 0 is luck or directional bet.
