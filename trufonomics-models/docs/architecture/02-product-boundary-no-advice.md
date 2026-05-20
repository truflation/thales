# Product boundary: outcome exposure, not investment advice

**Status:** firm decision, 2026-04-26.

## The rule

**Thales delivers outcome-exposure analytics. Thales does not give
investment, hedging, or capital-allocation advice.**

This is a hard boundary. It applies to every output of every product
tier — Tier 1 nowcasts, Tier 2 forecasts, Tier 3 transmission VARs,
regime indicators, all of it.

## What "outcome exposure" means in practice

Acceptable outputs (data and analytics):

- "If diesel +20%, your fuel cost line moves +$5.3M / 12 months."
- "Your freight revenue line will partially offset by 8% pass-through
  at the empirical historical rate."
- "78% of your fuel-cost variance is hedge-able via DBO; 22% is
  irreducible idiosyncratic."
- "Your maintenance and labor lines are uncorrelated with this shock
  at the monthly horizon."
- "Inflation is in the 'high-vol' regime with smoothed P=0.85."
- "Cleveland Fed nowcast says 1.85% YoY for next month, 95% band
  [1.62, 2.08]."

NOT acceptable outputs (investment advice / recommendations):

- ❌ "You should hedge 50% of your fuel exposure."
- ❌ "Buy DBO for fuel hedging."
- ❌ "Increase your cash reserves by $X."
- ❌ "Reduce your variable-rate debt exposure."
- ❌ "Take profits on your inflation-linked bond holdings."
- ❌ "Rotate from growth to value."

The line: **Thales describes the world and the customer's exposure
to it. Thales never tells the customer what to do about it.**

## Why this matters

### Regulatory

In the US, recommending specific securities or position sizes
triggers SEC / state RIA registration with fiduciary duty,
suitability obligations, and ongoing compliance burden. Even
"educational" framing can trigger registration if the recommendations
are specific enough.

The exposure-analytics framing avoids this entirely. Bloomberg sells
the same kinds of analytics to the same kinds of customers without
RIA registration because they describe markets and exposures rather
than recommend trades.

### Liability

An advisor who said "hedge X" and the customer lost money has clear
fault attribution. An analytics provider who said "your exposure is
$Y" has none — the customer made the trading decision themselves.

### Product simplicity

Advice products require **suitability assessments** per customer
(risk tolerance, sophistication, jurisdiction). Analytics products
ship the same numbers to everyone who pays. The former is a high-
touch sales motion; the latter is SaaS.

### Customer agency

Treasury and finance teams have their own hedging policies, broker
relationships, and execution preferences. Telling them what
instrument to use is presumptuous and often wrong. Telling them
*how exposed they are* is genuinely useful regardless of how they
choose to manage that exposure.

## Edge cases

**"What's the empirical pass-through ratio of diesel to freight?"**
✅ Allowed — this is a measurement, not advice.

**"What's the β-optimal hedge ratio for diesel via DBO?"**
🟡 Borderline — this is computable from the data and could be
framed as analytics ("the historical correlation gives a minimum-
variance ratio of 0.30"), but it's leaning toward "telling the
customer what to do." Safe framing: present as a *measurement of
the data* ("DBO has a 0.30 minimum-variance ratio against retail
diesel monthly returns over 2015-2025") rather than a prescription
for them.

**"Should we widen our credit spread band?"**
❌ Not allowed — this is a portfolio-management decision.

**"Is inflation likely to surprise to the upside next month?"**
✅ Allowed — this is a forecast (with calibrated bands), and stating
a forecast probabilistically is description, not advice.

**"Given today's regime indicator at 0.85, your P&L expected
variance over the next quarter is $X."**
✅ Allowed — exposure measurement.

## Implementation guardrails

1. **Never use action verbs in customer-facing output.** "Hedge,"
   "buy," "sell," "increase," "reduce," "rotate," "rebalance" — none
   of these go in API responses or report templates.

2. **Use exposure verbs instead.** "Exposed to," "correlated with,"
   "partially offset by," "expected impact of," "variance attributable
   to," "regime probability of."

3. **Always present customer-facing numbers as measurements, never
   as recommendations.** The customer infers the action; Thales
   describes the data.

4. **In findings docs and internal analysis**, it's fine to discuss
   what a strategy *could* do (e.g. fuel hedge backtest). This is
   methodology validation, not customer-facing output. The boundary
   applies to what gets exposed *through the product*, not to
   internal R&D analysis.

5. **If a customer specifically asks "what should we do?"**, the
   correct response is to surface the relevant exposure data and let
   their treasury / advisory team decide. Not "we recommend X" — even
   if X is obvious.

## Connection to existing decisions

- The Phase 3.1e fuel-hedge backtest (`FUEL_HEDGE_BACKTEST_FINDINGS.md`)
  is **internal validation only**. The σ-reduction numbers won't ship
  as a customer feature. They tell us "the model's monthly vol signal
  isn't sharp enough to be useful for action" — useful internal
  finding.
- The Phase 3.1d shock-scenario API (`BVAR_CONDITIONAL_FINDINGS.md`)
  IS customer-facing — but as exposure mapping ("if X then your P&L
  moves Y"), not as advice.
- The "VIX for inflation" Tier 3a regime indicator (already in the
  product spec) is exposure data — the customer's portfolio
  exposure to high-vol regimes is theirs to manage.
