# Tier 3 — Regime indicator + per-industry transmission VARs

Two related products that share the regime-detection backbone but
serve different customers.

## 3a — "VIX for inflation" (regime indicator API)

### One-line definition

**A daily-updating probability that the inflation regime is in a
high-volatility state, with cross-target coherence (CPI, Core CPI,
PCE, Core PCE) and historical regime windows.**

The name is borrowed from VIX (the equity-volatility index) for the
same reason: it's a single number that summarizes the *volatility*
state of inflation rather than its level. Just as VIX going from 15 to
30 tells you "equity vol regime is now elevated" without telling you
where the S&P will close tomorrow, a Thales regime indicator going
from P(high) = 0.05 to P(high) = 0.95 tells you "inflation
volatility regime has shifted" without forecasting next month's CPI
print.

### What subscribers receive

For each of the 4 official inflation series:

- **Live regime probability** P(high-vol regime | data through today)
- **Regime-window history**: list of past high-vol periods (start
  date, end date, peak P(high))
- **Cross-target coherence**: same regime indicator on CPI / Core CPI
  / PCE / Core PCE simultaneously, so subscribers can see whether
  divergence between Headline and Core regimes is happening (it
  currently is — see today's empirical finding)
- **Historical reproducibility**: regime model is purely Markov-
  switching variance (Hamilton 1989 + Kim 1994 smoother), no
  black-box learning — every subscriber-flagged regime change can be
  audited

### Inputs

- Monthly BLS CPI / BEA PCE YoY series (via ALFRED)
- That's it. No exogenous inputs needed for the basic regime indicator.
  (Future: extended-frequency variant uses Truflation daily YoY.)

### Output snapshot — what subscribers see today

From `results/regime/PURE_MS_ALL_TARGETS_FINDINGS.md`:

```
                                       CPI    Core CPI    PCE    Core PCE
Oil price collapse 2014-15            0.92      0.00     0.96      0.00
COVID-19 onset 2020 (6 mo)            0.02      0.00     0.01      0.00
Post-COVID surge 2021-23              0.81      1.00     0.93      1.00
Disinflation 2024                     0.00      1.00     0.01      1.00
```

The current state (April 2026): **Headline measures back to low-vol;
Core measures STILL in high-vol regime (60+ months and counting).**
That's a tradable insight — Fed's "core inflation is sticky" narrative
is showing up directly in our model.

### Downstream tasks

- **Macro hedge funds**: regime-indicator triggers for breakeven /
  TIPS / commodity overlay strategies
- **Asset-allocation desks**: regime-conditional rebalancing rules
  ("rebalance toward duration when P(high) drops below 0.30")
- **Risk-management teams**: VaR multiplier — high-vol regime →
  larger inflation surprises → larger position sizing impact
- **News / explanatory**: "is inflation back to normal?" — regime
  probability gives a direct yes/no answer that *isn't just a level
  comparison*
- **API subscribers**: lightweight pricing tier for users who don't
  need the full nowcast — just want a daily regime probability number

### Pricing / distribution model (TBD)

The natural cheap subscription tier:
- $X / month for daily regime probability ping
- $XX / month for full regime probability + windows + per-target table

### Status: **Working on real BLS data, ready for production deployment.**

What's built ✅
- Pure MS (Hamilton + Kim) regime detector
- Validated across all 4 official YoY targets
- Cross-target coherence verified empirically
- Energy-sensitivity test passes (oil shocks fire on Headline only)

What's not yet ⏳
- Daily-updating endpoint (depends on monthly BLS publication
  cadence; can be a daily endpoint that re-runs on the fixed monthly
  series)
- Subscription / billing infrastructure (Truflation business team)

---

## 3b — Per-industry transmission VARs

### One-line definition

**Industry-specific Bayesian VARs (Vector Autoregression) with
Minnesota priors that take the Tier 1 nowcast + commodity futures
curves as inputs and produce industry-level cost-pass-through paths.**

### Difference from VIX-for-inflation

VIX-for-inflation is a regime indicator (one probability, daily). The
transmission VARs are full structural models per industry — they take
the Tier 1 / Tier 2 forecasts as inputs and produce industry-specific
forward P&L paths. VIX subscribers don't need transmission VARs;
transmission-VAR subscribers consume the regime indicator as one
input among many.

### Per-industry product variants (planned)

| Vertical | Cost structure | Status |
|----------|----------------|-------:|
| Logistics | Fuel 35%, labor 25%, maintenance 10%, insurance 5%, other 25% | Phase 3.1 |
| Restaurants | Food COGS 30%, labor 30%, rent 8%, utilities 4%, other 28% | Phase 3.2 |
| Mid-market retail | TBD | Phase 3.3 |
| Healthcare operators | TBD | Phase 3.3 |
| Real estate operators | TBD | Phase 3.3 |
| Manufacturing verticals | TBD | Phase 3.3 |

### Downstream tasks

- **Industry operators**: forward P&L planning under inflation
  scenarios. "If logistics fuel pass-through holds the historical
  pattern, what's my margin in 6 months?"
- **Equity analysts covering specific sectors**: cost-pressure
  forecasts for company-level earnings models
- **Hedging desks at industrial firms**: hedge-ratio sizing for
  fuel / food / metals exposure
- **Credit analysts**: industry-level credit spread forecasting
  (industries with worse cost-pass-through expectations get wider
  spreads)

### Status: **Phase 3 / not started.**

The Tier 1 nowcast + Tier 2 forecast must be in production before
Tier 3b becomes useful (transmission VARs need both as inputs).

---

## Cross-product relationship

```
Tier 3a — VIX for inflation:
    P(high) ← regime detector on Tier 1 inputs

Tier 3b — Transmission VAR:
    industry_cost_path ← Bayesian VAR(
        Tier_1_density_forecast,    ← from Tier 1
        commodity_futures,            ← exogenous
        industry_cost_structure,      ← per-vertical config
        regime_state ← Tier 3a        ← regime modulates VAR coefficients
    )
```

Both Tier 3 products consume Tier 1 internally; Tier 3b also
consumes Tier 3a's regime indicator as a covariate.
