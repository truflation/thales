# Cleveland Fed nowcast — NATIVE same-month frame eval

**Date:** 2026-04-25
**Script:** `scripts/eval_clevfed_native.py`
**Window:** 2015-01-31 → 2026-03-31 (134 monthly origins)
**Frame:** h = 0 — score `clev[T]` against `y[T]`

## Why this exists

The earlier `eval_official_baselines.py` run scored Cleveland Fed in our
+1m harness (using `clev[T]` to predict `y[T+1]`). That's a frame
mismatch — Cleveland Fed's actual product is a **same-month nowcast**:
they build up an estimate of month T's CPI continuously through month T,
finalize it at end-of-T, and BLS publishes T's actual ~13 days into T+1.

Comparing their same-month estimate to next-month's print inflates their
apparent error by 30-70%. This script fixes the alignment.

## Headline result

**Cleveland Fed dominates on Headline CPI in its native frame:**

| Target    |   n  | Clev RMSE | Last-Release RMSE | **Δ vs Last-Release** | Clev direction (base) |
|-----------|-----:|----------:|------------------:|----------------------:|----------------------:|
| CPI       | 134  |    0.1727 |            0.3821 |          **+54.80%**  |      88.8% (56.7%)    |
| Core CPI  | 134  |    0.1752 |            0.2318 |          **+24.43%**  |      68.7% (46.3%)    |
| PCE       | 134  |    0.2029 |            0.2640 |          **+23.13%**  |      76.1% (58.2%)    |
| Core PCE  | 134  |    0.2191 |            0.1764 |          **−24.24%**  |      65.7% (54.5%)    |

Direction columns: `dir_clev` is the rate at which `clev[T]` correctly
predicted whether the released `y[T]` would be **up or down vs the last
known print** `y[T-1]`. `base` is the unconditional rate that `y[T] >
y[T-1]` over the window.

## Three findings worth flagging

### 1. Cleveland Fed is the comparator to beat — at least on Headline CPI

54.8% RMSE reduction vs last-release on Headline CPI is a **strong**
result. 88.8% directional accuracy (vs base 56.7%, lift of +32 pp) is
genuinely useful. This is the bar Thales archetype models have to clear
on the official Headline CPI same-month nowcast frame.

### 2. Last-release dominates Cleveland Fed on Core PCE

Cleveland Fed loses to "remember last month's value" by 24% on Core PCE
RMSE. This isn't a bug — it's a property of the series:

- Core PCE is the smoothest YoY measure of all four (lowest period-over-
  period volatility ~0.18 pp).
- Modeling adds variance; persistence has zero variance contribution.
- When the signal-to-noise ratio is already this high for the trivial
  baseline, every model has a hard time beating it.

This matches Stock-Watson 2007's famous result: nobody beats persistence
on near-unit-root inflation series. Core PCE is the canonical example.

### 3. Direction matters more than RMSE for nowcasts

Even when Cleveland Fed loses on RMSE (Core PCE), its directional
accuracy (65.7%) crushes the trivial baseline (54.5%) by +11 pp. RMSE is
a single-number summary that hides asymmetric value: getting *direction*
right is what matters for an inflation trader. The next-version
`weekly_rollup.py` should weight direction more heavily for nowcast
products.

## Comparison table — same window, two frames

How does Cleveland Fed look in our two frames? Compiled from this run
plus the master `FINDINGS.md`:

| Target    | h = 0 RMSE | h = +1m RMSE | h = 0 vs h = +1m |
|-----------|-----------:|-------------:|-----------------:|
| CPI       | **0.1727** |       0.4658 |          −63.0%  |
| Core CPI  | **0.1752** |       0.3306 |          −47.0%  |
| PCE       | **0.2029** |       0.3375 |          −39.9%  |
| Core PCE  | **0.2191** |       0.2945 |          −25.6%  |

Native frame is 25-63% lower RMSE than the misaligned +1m frame — exactly
what we'd expect from fixing the one-period offset.

## What this means for the rest of the program

1. **Path A's +9.50% on CPI at +1m is not directly comparable to Cleveland
   Fed's +54.80% at h=0.** They answer different product questions.
2. Anyone benchmarking Thales against "the Fed's nowcast" needs to pick
   a frame and stick with it. Per CLAUDE.md, our pre-registration commits
   us to the h=0 frame against Cleveland Fed because that's the
   canonical institutional question.
3. For the Phase 1 archetype rollout, every component model should be
   evaluated **at h=0 against last-release**, with Cleveland Fed as the
   research-frontier comparator. Beating last-release is the gate;
   approaching Cleveland Fed is the ambition.

## Files

- `clevfed_native_results.csv` — per-target metric block

## Follow-up

- Forward-month nowcast scrape (Cleveland Fed's `+1` forecast, distinct
  from `+0` nowcast) — would unlock a *fair* +1m comparator instead of
  the misaligned same-month one we used in `FINDINGS.md`.
- Adapt `walk_forward` to support `horizon=0` cleanly so this analysis
  flows through the harness like the others. Currently this is a
  one-shot script outside the harness.
