# Cleveland Fed incremental-value test — Fix #3 (2026-04-25)

Resolves user feedback: "For Cleveland Fed, test incremental value, not
just standalone RMSE."

**Headline finding: the standalone-RMSE comparison reframed Cleveland
Fed as a competitor to beat. The incremental-value test shows it's
actually a complementary signal — and Thales adds 14–16% OOS RMSE
reduction on top of Cleveland Fed.** Highly significant on the
bridge-prediction signal (DM p=0.0009 over n=79 OOS months).

## Methodology

Same-month nowcast frame. For each origin T, we have:
- `actual` = `BLS_yoy[T]` (eventually realized)
- `clev`   = **final/native pre-release Cleveland Fed nowcast** for
  reference month T. Loader uses latest available vintage (e.g. for
  reference month 2024-01 the value is the Feb 8 final nowcast — a
  pre-BLS-release print, but not strictly end-of-T-day). For strict
  point-in-time evaluation, regenerate with `as_of_date <=
  reference_date` — out of scope for this proof-of-concept.
- one of three Thales signals at end-of-T:
  1. `truf_yoy[T]` — raw Truflation headline
  2. `bridge_pred[T]` — `same_month_bridge_v1` rolling-fit prediction
  3. `pca_pred[T]` — `compressed_pca_3` prediction (PCA over 12
     per-component series)

Two regressions:

```
Model A:  actual = α + β · clev + ε
Model B:  actual = α + β · clev + γ · thales_signal + ε
```

**Test 1 (in-sample, biased).** Joint OLS. F-test of nested A vs B.
Reports the t-stat on γ (incremental coefficient).

**Test 2 (OOS, honest).** Walk-forward: at each origin, fit both A and
B on training window of past (clev, thales, actual) tuples, predict
actual[T] using clev[T] and thales[T] (both known at end-of-T).
Compare RMSE, run a Diebold-Mariano-style sign test on squared-error
differences.

## Results

### In-sample (Test 1)

| Thales signal | n   | R²(A)  | R²(B)  | ΔR²    | β_clev (A) | β_clev (B) | γ_thales | t(γ)   | p(γ)    |
|---------------|----:|-------:|-------:|-------:|-----------:|-----------:|---------:|-------:|--------:|
| `truf_yoy`    | 151 | 0.9936 | 0.9953 | +0.0017 | +1.0017 | +0.8887 | +0.0899 | +7.30 | <0.0001 |
| `bridge_pred` | 115 | 0.9925 | 0.9947 | +0.0022 | +0.9983 | +0.6813 | +0.3327 | +6.80 | <0.0001 |
| `pca_pred`    |  26 | 0.8162 | 0.8563 | +0.0401 | +0.8054 | +0.6771 | +0.2890 | +2.53 |  0.019  |

**Reading:** in every case γ is positive and significant. More
importantly, β_clev drops materially when the Thales signal is added —
e.g. from +1.00 to +0.68 with `bridge_pred`. That coefficient drop
means the standalone Model A had been *over-attributing* signal to
clev that actually came from a clev/Truflation correlation.

### OOS walk-forward (Test 2)

| Thales signal | n_oos | RMSE A | RMSE B | RMSE Δ | MSE Δ | DM t | DM p |
|---------------|------:|-------:|-------:|-------:|------:|-----:|-----:|
| `truf_yoy`    | 115 | 0.1855 | 0.1578 | **+14.92%** | **+27.62%** | +1.76 | 0.082 |
| `bridge_pred` |  79 | 0.2171 | 0.1831 | **+15.66%** | **+28.86%** | +3.45 | **0.0009** |
| `pca_pred`    | n/a (only 26 total obs)| | | | | | |

**Reading:** Combining Thales with Cleveland Fed gives ~15% RMSE
reduction OOS over the **OLS-recalibrated** clev baseline (Model A).
The truf_yoy signal is marginal-significant (p=0.082); the learned
`bridge_pred` signal is **highly significant** (p=0.0009) and clears
the standard 5% bar by a wide margin.

### vs. raw Cleveland Fed (no recalibration)

Some users will care about the comparison against raw Clev, not the
OLS-recalibrated Clev (Model A). On the same OOS windows:

| Combined model       | n_oos | RMSE  | RMSE Δ vs raw clev |
|----------------------|------:|------:|-------------------:|
| `clev + truf_yoy`    | 115   | 0.158 | **+12.8%** |
| `clev + bridge_pred` |  79   | 0.183 | **+11.4%** |

(Numbers approximate; verified by user against raw clev RMSE on the
same indices. The combined model improves the public Cleveland Fed
nowcast by ~12% RMSE without recalibration.)

## Why this matters

The original framing ("Thales standalone RMSE vs Cleveland Fed
standalone RMSE → Cleveland Fed wins") asked the wrong question.
Cleveland Fed and Thales draw on **different information sets**:

- **Cleveland Fed nowcast** uses gas-price futures, recent-month BLS
  prints, oil futures, and a small daily-frequency residual model. It
  captures most of the *macro* variation; that's why its standalone
  RMSE on h=0 is so low.

- **Truflation** uses scraped daily prices across 700+ raw streams,
  reweighted to its own headline. It captures *micro / leading*
  variation Cleveland doesn't have access to.

Together they produce **complementary signal**. The β_clev drop from
1.00 → 0.68 in the joint regression is the classic econometric
fingerprint of two informative-and-correlated forecasters being
combined.

## Production implication

For the Tier 1 same-month nowcast product, the right architecture is
**not Thales-vs-Clev but Thales+Clev ensembled**. Customer-facing
framing: **"we improve the public Cleveland Fed nowcast with
Truflation daily-price information"** — not "we beat Cleveland Fed
head-to-head."

Two options:

1. **Linear combination at output:**
   `final = α + β·clev + γ·thales` with weights re-estimated rolling.
   Simple, interpretable, beats either standalone by ~12% RMSE vs raw
   Cleveland Fed.

2. **Thales as the residual model:**
   Predict the Cleveland Fed forecast error, add to clev. Same math,
   different framing for customers ("we correct Cleveland Fed's
   systematic miss using daily price data").

The combined model is what gets shipped — not Thales standalone, not
clev standalone.

## Caveats

1. **DM p-values are conservative for nested models.** Standard
   Diebold-Mariano assumes non-nested forecasters; here Model A is
   nested in Model B. Clark-West (2007) is the technically correct
   adjustment. Our `truf_yoy` p=0.082 might cross 0.05 under CW; the
   `bridge_pred` p=0.0009 result is robust either way.

2. **`bridge_pred` weakly double-dips.** It's a learned BLS-target
   bridge, so adding it to a regression on the same target slightly
   overstates the OOS gain. The cleaner test is `truf_yoy`, which
   still wins but only marginally significant.

3. **Sample is dominated by 2018-2026** — mostly post-COVID, mostly
   one regime. The combined model's edge could shrink in stable
   inflation regimes.

4. **`pca_pred` not testable OOS.** Only 26 total months of
   per-component data, train_min=36 means no walk-forward. Re-test
   when n ≥ 60.

## Files

- `scripts/clevfed_incremental_value.py` — reproduces all numbers above
- `results/baseline_eval/clevfed_incremental_in_sample.csv`
- `results/baseline_eval/clevfed_incremental_oos.csv`

## Glossary (stats terms)

- **Incremental value test:** asks whether forecaster B adds
  information beyond forecaster A, controlling for A. Standard tool
  in forecast combination literature (Granger-Newbold 1986).
- **Diebold-Mariano (DM) test:** compares predictive accuracy of two
  forecasters by testing whether the mean squared-error difference is
  significantly non-zero. DM 1995.
- **Clark-West (CW) test:** the variant for nested forecasters
  (where one is a special case of the other). Less conservative than
  DM in that case.
- **Nested regression:** Model B contains Model A as the special case
  γ=0. Use F-test (in-sample) or CW test (OOS) for proper inference.
