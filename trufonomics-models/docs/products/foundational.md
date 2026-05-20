# What "foundational" means for Thales

Direct response to `obsidian:15 - trufonomics/thales/my questions.md`.

## Q1 — what makes a model foundational? how is our model foundational?

**Your answer was right and I'll sharpen it:**

> "Our models are foundational in the sense you can do a number of
> downstream tasks with them — downstream systems consume the model's
> outputs as inputs."

That's the key. By analogy with foundation models in ML (BERT, GPT,
etc.), a foundational econometric model is one whose **outputs are
themselves inputs** to many distinct downstream systems and decisions.
Three properties make it foundational rather than narrow:

1. **One pre-trained representation, many output surfaces.**
   In ML: a single transformer is fine-tuned for translation,
   classification, summarization. In Thales: one fitted state-space
   stack (per-category archetypes + CBDF composition + regime layer)
   exposes nowcast density, multi-horizon forecasts, regime indicator,
   and transmission VARs — four separate products from one model.

2. **Outputs go into other people's models.**
   A point-forecast service is consumed by humans reading a chart.
   A foundational model's outputs are consumed *programmatically*: by
   a trading desk's risk model, by a CFO's budget Monte Carlo, by an
   industry-VAR's exogenous-driver vector. This is the clearer test —
   **does anything else in the world build on top of your output?**

3. **The latent state has economic meaning that survives composition.**
   Tier 1 publishes a number. The thing *behind* that number is a
   joint posterior over (level, log-vol, regime, per-component β,
   loading on national factor). Each piece is independently useful
   and consumed by different products:
   - μ_t  (level)        → headline forecast
   - h_t  (log-volatility) → density bands
   - S_t  (regime)        → VIX-for-inflation
   - β_r  (loadings)      → transmission VARs
   - λ_r  (idiosyncratic) → per-component attribution

This is the structure that makes the same internal state useful to a
trading desk AND a CFO AND a logistics operator AND an "is inflation
sticky?" macroeconomist.

## Why it's not foundational if you skip the architecture

Path A v1 was Tier 1 only — a single-output Ridge stacker. It's a great
nowcast (+42% MSE reduction), but it's not foundational because:

- No latent state to interpret (Ridge produces one number, no
  intermediates)
- No density (point forecast only)
- No regime indicator
- No path for adding scenarios
- No way to compose into industry VARs

Thales rebuilds on a state-space backbone *specifically so the latent
state can be exposed for downstream consumption*.

## Q2 — what are the downstream tasks it can work on? where do we input its outputs?

Full answer in `README.md` "What downstream systems consume Thales"
table. Summary by Thales output:

### Tier 1 density nowcast goes into:
- Trading risk models (TIPS-vs-nominal trade sizing, options pricing)
- Asset-allocation rebalancing rules
- Bond-portfolio duration management
- Economic research note (cited alongside Cleveland Fed)
- News explanatory journalism (per-component attribution)
- Internal Truflation product analytics

### Tier 2 multi-horizon forecast goes into:
- CFO budget scenarios (annual budgets, multi-year capex)
- Lender pricing models (floating-rate loans, credit spreads)
- Hedge-fund medium-term thematic positioning
- Treasury cash management
- Counterfactual policy research

### Tier 3a regime indicator (VIX for inflation) goes into:
- Macro hedge fund regime-conditional triggers
- VaR multipliers (regime-conditional risk sizing)
- Asset-allocation regime rules
- News "is inflation back to normal?" framing
- Lightweight subscriber product for non-quant users

### Tier 3b transmission VARs go into:
- Industry-operator P&L forecasting (logistics, restaurants, retail)
- Equity-analyst sector cost-pressure models
- Industrial-firm hedge-ratio sizing
- Credit analyst industry-level spread forecasts

## How this differs from "just publishing forecasts"

A vendor publishing forecasts → subscribers read the forecast and
decide what to do.

A foundational model → subscribers' systems consume the forecast
*alongside the latent state and per-component decomposition*, and
their own systems make decisions algorithmically.

The former is a research subscription. The latter is infrastructure.
The price-point and customer behavior differ accordingly.

## How foundational survives the per-target architecture lessons

Today's Phase 2.2 finding (UC layer wrong for monthly YoY) is exactly
the kind of result foundational models *must* surface clearly — because
the same internal state powers four products, getting it wrong
miscalibrates all four. The "three variants tested, pure MS wins" run
in `results/regime/PHASE_2_2_RESOLUTION.md` is foundational
methodology in action: validate the spec carefully, document the
mismatch, ship the right variant.

That's also why we're building each archetype + composition layer with
synthetic recovery FIRST and real-data fits SECOND. Both gates have to
pass for foundational use. Synthetic alone isn't enough; real-data
alone doesn't tell you whether your architecture is sound.

## In one sentence

**Thales is foundational because the same internal econometric state
(level + log-vol + regime + factor loadings + per-category drivers)
exposes four distinct product surfaces (nowcast / forecast / regime /
transmission), and each surface is consumed by *different downstream
systems* (trading risk / CFO planning / regime-API / industry P&L) —
not by humans reading a chart, but by other people's models.**

## Q3 — how would we update the models? once a month? a quarter?

Different cadences for different things, layered:

### Daily — **forecasts** update
- Tier 1 nowcast updates daily as new Truflation observations land.
  No re-fitting; just running the existing fitted model forward with
  today's data.
- Tier 2 multi-horizon forecasts update daily on the same cadence.
- Tier 3a regime probability updates daily from new data.

### Monthly — **hyperparameters** re-fit
- After each BLS / BEA monthly release lands (~13 days into the next
  month), re-fit each archetype's hyperparameters on a sliding window
  (typically 24-60 months).
- Re-fit the Truflation→BLS bridge on the same window.
- Re-fit the CBDF cross-component residual covariance.
- This is automatic / scheduled.

### Quarterly — **architecture** review
- Walk-forward eval over the past 90 days: how did each archetype do
  per-component? Did regime detection track known events?
- Coverage check: are 80%/95% bands still calibrated?
- DM tests vs Cleveland Fed and SPF over the past quarter.
- If a component's archetype is materially mis-calibrated, swap it
  (e.g., move Health from MS-only to MS+SV if SV has come back online).

### Annually — **major version bump**
- Truflation publishes new annual category weights each February.
  Major version bumps the headline composition by adopting the new
  weights and re-running validation across the full archive.
- Add new components if Truflation extends coverage.
- Re-pre-register methodology if the architecture changed.

### Event-driven — **regime / structural break**
- If a sustained P(high) > 0.5 signal arrives or a structural-break
  detector trips, run an out-of-cycle review: "did our archetypes
  handle this transition correctly? Do we need to extend the
  training window or constrain σ_eta?"

### What stays static
- The 5 archetype model classes (commodity TVP / BSTS / UC-SV-MS /
  VECM / hierarchical housing). These are pre-registered architectural
  commitments; we don't swap them out without good reason.
- The composition layer (CBDF). Same.
- The cross-walk between Truflation taxonomy and BLS subindex.
- The vintage discipline rules.

In summary: **forecasts update daily, hyperparameters monthly,
architecture quarterly, major versions annually, regimes event-driven.**
The split lets us be responsive to new data without thrashing the
methodology.
