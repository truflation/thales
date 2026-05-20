# MoM-first composition — Fix #5 (2026-04-25)

Resolves user feedback: "Model MoM first, then compose YoY."

**Headline: MoM-first AR(1) cuts headline-CPI YoY one-month forecast
RMSE by 35.9% vs YoY-direct AR(1) on the same n=132 OOS window.** This
is the single largest improvement from any of the six prioritized fixes.

## What shipped

1. **`thales.models.mom_composed`** — new module:
   - `mom_from_level()` / `yoy_from_level()` — log-MoM and log-YoY
     utilities (in pp).
   - `compose_yoy_one_step()` — closed-form identity
     `yoy[T+1] = yoy[T] + mom_pred[T+1] − mom[T−11]`.
   - `MoMComposedForecaster` — wraps any inner Forecaster trained on
     a MoM column; exposes a YoY forecast through composition.
     Bands translate 1-to-1 (linear shift).

2. **8 unit tests** including the algebraic identity proof and an
   "oracle MoM" test confirming that a perfect MoM forecast composes
   back to the realized YoY exactly.

## Headline result

Same panel (Headline CPI, 2011-02 → 2026-03), same train/calib (36/24),
same horizon (h=1), same band method (rolling-conformal):

| Model | n | RMSE | MAE | cov80 | cov95 | dir hit | vs floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ar1_yoy_direct`   | 133 | 0.3814 | 0.2728 | 85.0% | 94.7% | 82.7% | +40.69% |
| `ar1_mom_composed` | 132 | **0.2444** | **0.1757** | 82.6% | 93.2% | **87.1%** | **+61.95%** |

**RMSE Δ: −35.93%. MSE Δ: −58.96%.** Direction hit climbs from 82.7%
to 87.1%. Coverage stays calibrated under conformal.

## Why this works — the load-bearing diagnostic

| | YoY series | MoM series |
|---|---:|---:|
| AR(1) coefficient (lag-1 autocorr) | **+0.982** | +0.532 |

The YoY series is nearly a unit-root walk. A "vanilla AR(1) on YoY"
collapses to `yoy[T+1] ≈ φ·yoy[T] + α ≈ yoy[T]` — i.e., persistence
plus a tiny tilt toward the long-run mean. The model has nothing
useful to predict because **the YoY differencing already builds in
12-month autocorrelation**, which the AR(1) absorbs into φ.

MoM has AC1 = 0.532 — a healthy stationary process where AR(1)
can learn genuine mean reversion in monthly inflation innovations.

Composing forward then uses **two** sources of information instead of
one:

1. `yoy[T]` — the current trend
2. `mom_pred[T+1]` — the AR(1)-implied monthly innovation
3. `mom[T−11]` — the realized "rolling-out" month from a year ago

YoY-direct AR(1) only uses #1.

## Regime detectability — bonus diagnostic

Hamilton 2-state MS fit on each series:

| | σ_low | σ_high | σ_high/σ_low | % time in high regime |
|---|---:|---:|---:|---:|
| YoY | 0.793 | 3.935 | 4.96× | 21.5% |
| MoM | 0.144 | 0.452 | 3.13× | 23.2% |

Both have detectable regimes — the YoY ratio is larger only because
YoY is noisier overall. **The MoM regimes are the economically-
meaningful ones** (a high-MoM month is genuinely different from a
low-MoM month; a high-YoY month just has 12 months of high-MoM
behind it). This corroborates the user's intuition that regime
mechanisms get absorbed into the YoY trend layer.

## Architectural implication

The Phase 2.2 finding — "UC+SV+MS over-parameterizes monthly YoY; the
regime mechanism stays dormant" — is now better understood as a
specific instance of the YoY-vs-MoM problem:

- UC layer absorbs the AR(0.98) → no residual variance for SV/MS
- SV/MS layer fires on residuals that don't exist → dormant

The fix is not "drop UC for already-differenced targets" (the prior
recommendation in `docs/SESSION_2026-04-25.md`). The cleaner fix is
**fit UC+SV+MS on MoM, compose to YoY at the end.** UC absorbs the
0.53 MoM mean-reversion (real signal); SV captures monthly-vol
dynamics (real signal); MS finds regimes (real signal). The
composition then mechanically produces a YoY forecast with all three
ingredients.

This is the hook for Phase 2.2d (out of scope for this fix).

## Production decision

For Tier 1 / Tier 2 inflation forecasting:

1. **Default**: replace AR(1)-on-YoY with `MoMComposedForecaster(inner=AR1Baseline(target_col='bls_mom'))`.
2. **Bridge variant**: same wrap around `SameMonthBridgeNowcaster`,
   but with a Truflation MoM signal — work for the next session;
   needs Truflation level data not just YoY.
3. **All future archetypes** (BSTS, UC+SV+MS, VECM) should default to
   MoM-target unless there's a specific reason to use YoY.

## Caveats

1. **Horizon = 1 only.** Multi-step composition needs forecasting a
   sequence of MoMs and rolling them out. Closed form is straight-
   forward but residual covariance across the sequence matters for
   bands. Out of scope for this fix.

2. **Bands are conditional on the inner.** The wrapper translates
   inner bands 1-to-1 to YoY space. This is correct for the linear
   composition but only as good as the inner's own band calibration.
   The cov80/cov95 numbers above use rolling-conformal on the inner.

3. **Oracle test passes; real test improves but doesn't perfectly
   match YoY.** That's because AR(1) on MoM is a real model, not an
   oracle. The 35.93% gain is what AR(1) actually buys; an oracle
   MoM would buy more.

4. **Truflation-MoM extension untested.** This fix benchmarks
   MoM-first vs YoY-direct using only the BLS series. Adding
   Truflation MoM as a feature in the inner forecaster (the natural
   next step) is queued separately.

## Files

- `src/thales/models/mom_composed.py` (new)
- `tests/test_mom_composed.py` (new — 8 tests, including identity proof)
- `scripts/mom_first_vs_yoy_direct.py` (new)
- `results/baseline_eval/mom_first_vs_yoy_direct.csv`

## Glossary (stats terms)

- **YoY (year-over-year):** percent change vs same month one year
  prior. Headline inflation gauge.
- **MoM (month-over-month):** percent change vs prior month. The
  "true" monthly innovation.
- **AC1 (lag-1 autocorrelation):** corr(y_t, y_{t-1}). High AC1 →
  near-unit-root behavior; AR(1) fit will be near-persistence.
- **Composition identity:** for log returns,
  `yoy[T] = Σ_{k=0..11} mom[T−k]` exactly. Forecasting MoM and rolling
  is mathematically equivalent to forecasting YoY directly under
  perfect inner.
