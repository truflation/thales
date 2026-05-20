# Phase 2.1 — CBDF Composition Layer

**Date:** 2026-04-25
**Modules:**
- `src/thales/models/composition/weighted.py` (Phase 2.1a)
- `src/thales/models/composition/cbdf.py` (Phase 2.1b)
**Tests:** `tests/test_composition_weighted.py` (12/12) + `tests/test_composition_cbdf.py` (9/9) = 21 passing

## What's shipped

### 2.1a — `WeightedComposer`

The accounting-identity-respecting composition core. Given:
- Weights `w_r` over R top-level components (must sum to 1.0)
- Per-component forecasts (Forecast objects from any archetype)

Produces:
- Headline point forecast = Σ w_r · component_r (closed form)
- Headline density via Monte Carlo from per-component Gaussian/sample distributions
- Per-component attribution table sorted by |contribution|

**Key property — accounting identity preserved by construction.**
Whatever forecasts the archetypes produce, the composed headline equals
the published-weights-weighted sum. This is what makes the layer "CBDF"
in spirit: aggregation respects the BLS / Truflation accounting
structure, not just statistical fit.

### 2.1b — `CBDFComposer`

Extends `WeightedComposer` with **cross-component residual covariance**.
Without this, components are treated as independent and band coverage
mis-calibrates (typically too tight on positively correlated component
groups like utilities+food+transport during fuel shocks).

The fix:
1. `fit_residual_covariance(residual_panel)` estimates Σ_resid from
   historical per-component forecast errors. Ledoit-Wolf-style shrinkage
   to diagonal when n_origins < 2× n_components keeps the matrix PSD.
2. `compose()` draws joint Monte Carlo samples from
   `N(point_vec, Σ_resid)` and weighted-sums each draw, capturing the
   cross-component dependence.

**Validated** on synthetic data:
- Positive correlation widens composed bands by >10% vs independent baseline ✅
- Negative correlation narrows composed bands ✅
- PSD safety on short panels (n_origins < 2× n_components) ✅
- Falls back to independent draws when covariance not yet fit ✅

This is the Phase 2.1b MVP of O'Keeffe-Petrova 2025 (NY Fed SR 1152).
The full DFM-style estimation (jointly fitting components + factor with
EM) is Phase 2.1c — additive surgery on top of this; the multivariate
Gaussian residual approach is sufficient for first-pass band
calibration.

## Test coverage

Total **21 composition tests** across both files:

- Construction validation (weight sum, tolerance)
- Point composition (weighted sum, zero-weight handling, missing-component error)
- Density composition (Gaussian-from-band, explicit-samples preferred,
  samples returned for downstream)
- Identity preservation
- Real Truflation weights smoke test
- Cross-component correlation effects (positive widens, negative tightens)
- Validation (panel mismatch, short panel, missing components after fit)
- PSD safety on short panels

## What this enables

1. **Drop-in composition** for any model that produces per-component
   `Forecast` objects. The 5 Phase 1 archetype classes (commodity,
   BSTS-LLT, BSTS-LL, UC+MS, UC+SV+MS, hierarchical housing, VECM)
   all output `Forecast`-compatible data structures by design.

2. **Headline density forecasts** with proper cross-component
   correlation handling — the foundation for the public "Thales
   headline nowcast with bands" product.

3. **Component attribution** for every headline forecast — which
   categories moved the headline number this period and by how much.
   Drives Stefan-style explanations: "Headline is up 0.15pp because
   Utilities (+0.08), Food (+0.05), …".

## Outstanding for Phase 2.1c (full O'Keeffe-Petrova)

The current MVP fits residual covariance from EMPIRICAL component
forecast errors. The full O'Keeffe-Petrova spec jointly estimates:
- Component-level dynamics (each archetype's parameters)
- Cross-component shared factor F_t
- Loadings β_r tying components to F_t
- Idiosyncratic component errors λ_r,t

…all in one EM loop, with the accounting identity as a constraint.
This is Phase 2.1c — adds ~500 LoC and 1-2 days of careful debugging.
Worth it for the journal submission (O'Keeffe-Petrova ablation
"CBDF vs standard DFM"); not blocking for the first production
release.

## Files

(No CSV artifacts yet — those land when we wire real-data per-component
archetype fits through this layer.)

## Next

End-to-end demonstration: per-component persistence baselines on the
12 top-level Truflation categories → CBDFComposer with real weights →
composed headline forecast → compare to direct headline persistence on
Truflation YoY (should match by accounting identity). Then upgrade
per-component baselines to the actual archetypes (commodity → Utilities,
BSTS-LL → Recreation, UC+SV+MS → Health, etc.).
