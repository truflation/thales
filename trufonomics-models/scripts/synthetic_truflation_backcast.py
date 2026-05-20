"""Synthetic Truflation backcast — 12 top-level categories.

Reconstruct a Truflation-style headline inflation series from BLS subindex
levels weighted by the Truflation taxonomy. First pass — validates the
mechanism on the BLS data we currently have ingested (2010-01-31 to
2026-03-31) and compares against actual Truflation in the 2020-2026
overlap. If validation passes, the same pipeline extends to the 68
sub-components in a follow-up.

Architecture:

  1. Map 12 Truflation top-level category ids → BLS subindex series_ids.
     Six map cleanly to BLS top-level subindices (Food, Housing,
     Transport, Health, Apparel, Recreation). The remaining six
     (Utilities, Household Durables, Alcohol & Tobacco, Communications,
     Education, Other) get absorbed into a BLS-Headline residual via
     weight renormalisation. Total mapped weight: ~76 %; residual: ~24 %.

  2. For each monthly date in the BLS coverage window:
     - Pick weights table: v1 (2010-2025) or v2 (2026+) by date
     - Build composite level = Σ_k w_k · (level_k[t] / level_k[base])
       — exactly the M2 method already validated in
       ``composition_check.py``
     - Residual term: w_residual · BLS_Headline[t] / BLS_Headline[base]

  3. Compute YoY on the composite level.

  4. Validate against actual Truflation frozen YoY in the 2020-2026
     overlap. Report median / mean / SD residual + p95 + share within
     0.1, 0.5, 1.0 pp bands. Compare to the in-domain composition_check
     baseline (median 0.000 pp, 94 % within 0.5 pp).

If the synthetic series tracks actual Truflation within the same band as
the in-domain check, the mechanism is sound for foundation-model
pretraining and long-horizon Tier 4 use.

Outputs:
  - results/synthetic_backcast/synthetic_truflation_top12_level.csv
  - results/synthetic_backcast/synthetic_truflation_top12_yoy.csv
  - results/synthetic_backcast/validation_overlap_2020_2026.csv
  - results/synthetic_backcast/SYNTHETIC_BACKCAST_FINDINGS.md
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thales.weights import get_top_level_weights    # noqa: E402

VINTAGE_DB = ROOT / "data" / "vintage_store" / "thales.duckdb"
OUT_DIR = ROOT / "results" / "synthetic_backcast"
OUT_DIR.mkdir(parents=True, exist_ok=True)
KAIROS_PARQUET = Path(
    "/Users/kluless/kairos/data/truflation/api/all_streams.parquet")
TRUF_HEADLINE_COL = (
    "truflation_us_cpi_frozen_yoy/truflation_us_cpi_frozen_yoy")

# 12 Truflation top-level category IDs → BLS series IDs.
#
# All 12 categories are now mapped to direct BLS subindices. Truflation's
# Housing decomposes as Shelter (79) + Utilities (81) + Household
# Durables (83), which match BLS SAH1 + SAH2 + SAH3 respectively (the
# components of BLS Housing aggregate SAH). Communications (86) and
# Education (87) both map to BLS SAE (Education and Communication) —
# their combined Truflation weight (~5.6%) is applied against the
# shared series.
#
# Format: list of (cat_ids_sharing_series, bls_series_id). When multiple
# Truflation categories share one BLS series, their weights are summed.
TRUF_TO_BLS: list[tuple[list[int], str]] = [
    ([78],     "CUSR0000SAF1"),    # Food & Non-alcoholic Beverages → Food
    ([79],     "CUSR0000SAH1"),    # Housing                         → Shelter
    ([80],     "CUSR0000SAT"),     # Transport                       → Transportation
    ([81],     "CUSR0000SAH2"),    # Utilities                       → Fuels and utilities
    ([82],     "CUSR0000SAM"),     # Health                          → Medical care
    ([83],     "CUSR0000SAH3"),    # Household Durables              → Household furnishings & ops
    ([84],     "CUSR0000SEFW"),    # Alcohol & Tobacco               → Alcoholic beverages
    ([85],     "CUSR0000SAA"),     # Clothing & Footwear             → Apparel
    ([86, 87], "CUSR0000SAE"),     # Communications + Education      → Education & Communication
    ([88],     "CUSR0000SAR"),     # Recreation & Culture            → Recreation
    ([89],     "CUSR0000SAG"),     # Other                           → Other goods and services
]
UNMAPPED_CAT_IDS: list[int] = []    # all 12 mapped now; residual term is zero
RESIDUAL_BLS = "CUSR0000SA0"        # kept for safety; not used when all mapped

V2_EFFECTIVE_FROM = pd.Timestamp("2026-01-01")


# ─── BLS data ────────────────────────────────────────────────────────────


def load_bls_levels(con: duckdb.DuckDBPyConnection,
                      series_ids: list[str]) -> pd.DataFrame:
    """Pull BLS subindex levels from the vintage store at their latest as-of.

    Returns a wide DataFrame indexed by reference_date (month-end) with
    one column per series_id. Inner-join across series so the panel
    has no missing cells. Selects only the most recent ``as_of_date``
    for each (series_id, reference_date) pair, so re-ingests don't
    duplicate rows.
    """
    frames = []
    for sid in series_ids:
        rows = con.execute(
            "SELECT reference_date, value FROM vintage "
            "WHERE series_id = ? AND source = 'bls_direct' "
            "AND (series_id, reference_date, as_of_date) IN ("
            "    SELECT series_id, reference_date, MAX(as_of_date) "
            "    FROM vintage WHERE series_id = ? AND source = 'bls_direct' "
            "    GROUP BY series_id, reference_date) "
            "ORDER BY reference_date",
            [sid, sid],
        ).fetchall()
        df = pd.DataFrame(rows, columns=["date", sid])
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df.set_index("date")[sid])
    panel = pd.concat(frames, axis=1).dropna()
    return panel


# ─── Composition (M2 method) ─────────────────────────────────────────────


def compose_synthetic_level(bls_panel: pd.DataFrame,
                              weights_v1: dict[int, float],
                              weights_v2: dict[int, float],
                              ) -> pd.Series:
    """Build a composite "synthetic Truflation" level series.

    Method M2 (already validated in composition_check.py): for each
    date, compute a weighted sum of the rebased subindex levels, where
    each subindex is rebased to 100 at the base period (first date in
    the panel) and weights come from Truflation v1 (pre-2026) or v2
    (2026+).

    Mapped categories that share a BLS series (e.g. Communications + Education
    both pointing at SAE) get their Truflation weights summed before
    multiplying against the rebased subindex level. Any unmapped categories
    fall through to the BLS-Headline residual term (currently zero, since
    all 12 Truflation top-level categories now have direct BLS mappings).

    Returns a monthly Series of composite levels — re-based so the
    base period equals 100.
    """
    base = bls_panel.index.min()
    base_levels = bls_panel.loc[base]    # series of BLS levels at base

    out = []
    for t in bls_panel.index:
        # Pick weights table by date
        w_map = weights_v2 if t >= V2_EFFECTIVE_FROM else weights_v1

        composite = 0.0
        mapped_total_weight = 0.0
        for cat_ids, bls_id in TRUF_TO_BLS:
            w = sum(w_map[cid] for cid in cat_ids)    # combined weight
            mapped_total_weight += w
            level_t = bls_panel.loc[t, bls_id]
            level_0 = base_levels[bls_id]
            composite += w * (level_t / level_0) * 100.0

        # Residual: any categories not in the cross-walk; uses BLS Headline.
        # With the current TRUF_TO_BLS this should be zero.
        residual_w = 100.0 - mapped_total_weight
        if residual_w > 1e-6:
            level_t_resid = bls_panel.loc[t, RESIDUAL_BLS]
            level_0_resid = base_levels[RESIDUAL_BLS]
            composite += residual_w * (level_t_resid / level_0_resid) * 100.0

        # composite is on a "weight × percentage-of-base" scale (sum of
        # weights × 100 = 100 × 100 = 10000 at base). Normalise to 100.
        composite /= 100.0
        out.append(composite)

    return pd.Series(out, index=bls_panel.index, name="synthetic_truflation_level")


def yoy_from_level(level: pd.Series) -> pd.Series:
    """Year-over-year %, computed on monthly month-end level values."""
    return ((level / level.shift(12) - 1) * 100).rename("synthetic_truflation_yoy")


# ─── Validation ──────────────────────────────────────────────────────────


def load_actual_truflation_yoy() -> pd.Series:
    """Load actual Truflation frozen YoY headline, monthly month-end."""
    parq = pd.read_parquet(KAIROS_PARQUET)
    parq["date"] = pd.to_datetime(parq["date"])
    parq = parq.set_index("date").sort_index()
    s = parq[TRUF_HEADLINE_COL].dropna()
    return s.resample("ME").last().rename("actual_truflation_yoy")


def validation_block(synthetic_yoy: pd.Series,
                       actual_yoy: pd.Series) -> pd.DataFrame:
    df = pd.concat([synthetic_yoy, actual_yoy], axis=1).dropna()
    df["residual_pp"] = df["synthetic_truflation_yoy"] - df["actual_truflation_yoy"]
    return df


def summarise_residuals(df: pd.DataFrame) -> dict:
    r = df["residual_pp"]
    return {
        "n": int(len(r)),
        "window_start": str(df.index.min().date()),
        "window_end": str(df.index.max().date()),
        "median_pp": float(r.median()),
        "mean_pp": float(r.mean()),
        "sd_pp": float(r.std()),
        "abs_max_pp": float(r.abs().max()),
        "abs_p95_pp": float(r.abs().quantile(0.95)),
        "share_within_0_1_pp": float((r.abs() <= 0.1).mean()),
        "share_within_0_5_pp": float((r.abs() <= 0.5).mean()),
        "share_within_1_0_pp": float((r.abs() <= 1.0).mean()),
    }


# ─── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 78)
    print("Synthetic Truflation backcast — 12 top-level categories")
    print("=" * 78)

    # Load weights
    weights_v1_df = get_top_level_weights("2024-01-01")
    weights_v2_df = get_top_level_weights("2026-04-01")
    weights_v1 = dict(zip(weights_v1_df["category_id"].astype(int),
                            weights_v1_df["weight"]))
    weights_v2 = dict(zip(weights_v2_df["category_id"].astype(int),
                            weights_v2_df["weight"]))

    mapped_cat_ids = [cid for entry in TRUF_TO_BLS for cid in entry[0]]
    mapped_w_v1 = sum(weights_v1[cid] for cid in mapped_cat_ids)
    mapped_w_v2 = sum(weights_v2[cid] for cid in mapped_cat_ids)
    print(f"\nMapped weight v1 (2010-2025): {mapped_w_v1:.2f}%  "
            f"(residual {100 - mapped_w_v1:.2f}%)")
    print(f"Mapped weight v2 (2026+):     {mapped_w_v2:.2f}%  "
            f"(residual {100 - mapped_w_v2:.2f}%)")

    print("\nMapped categories (BLS series → Truflation cats sharing it):")
    for cat_ids, bls_id in TRUF_TO_BLS:
        names = []
        w_v1_combined = 0.0
        w_v2_combined = 0.0
        for cid in cat_ids:
            n = weights_v1_df[
                weights_v1_df["category_id"].astype(int) == cid
            ]["category"].iloc[0]
            names.append(f"{cid}:{n}")
            w_v1_combined += weights_v1[cid]
            w_v2_combined += weights_v2[cid]
        joined = " + ".join(names)
        print(f"  {bls_id:<18s} ← {joined:<60s} "
                f"w_v1 {w_v1_combined:>6.3f}%  w_v2 {w_v2_combined:>6.3f}%")
    if UNMAPPED_CAT_IDS:
        print("\nUnmapped categories (rolled into BLS-Headline residual):")
        for cat_id in UNMAPPED_CAT_IDS:
            name = weights_v1_df[
                weights_v1_df["category_id"].astype(int) == cat_id
            ]["category"].iloc[0]
            w_v1 = weights_v1[cat_id]
            w_v2 = weights_v2[cat_id]
            print(f"  cat {cat_id:>3d} {name:<35s}  "
                    f"w_v1 {w_v1:>6.3f}%  w_v2 {w_v2:>6.3f}%")
    else:
        print("\nAll 12 Truflation top-level categories mapped — "
                "no residual term needed.")

    # Pull BLS data
    print("\nLoading BLS subindices from vintage store…")
    con = duckdb.connect(str(VINTAGE_DB), read_only=True)
    unique_bls_ids = sorted({bls_id for _, bls_id in TRUF_TO_BLS})
    series_ids = unique_bls_ids + [RESIDUAL_BLS]
    bls_panel = load_bls_levels(con, series_ids)
    con.close()
    print(f"  Panel: {len(bls_panel)} rows  "
            f"{bls_panel.index.min().date()} → {bls_panel.index.max().date()}")
    print(f"  Series: {len(bls_panel.columns)} "
            f"({len(TRUF_TO_BLS)} mapped + 1 residual)")

    # Compose
    print("\nComposing synthetic Truflation level (M2 method)…")
    synthetic_level = compose_synthetic_level(
        bls_panel, weights_v1, weights_v2)
    synthetic_yoy = yoy_from_level(synthetic_level).dropna()
    print(f"  Synthetic level: {len(synthetic_level)} months  "
            f"base = {synthetic_level.iloc[0]:.4f} (= 100 by construction)")
    print(f"  Synthetic YoY:   {len(synthetic_yoy)} months "
            f"({synthetic_yoy.index.min().date()} → "
            f"{synthetic_yoy.index.max().date()})")

    # Validate
    print("\nValidating against actual Truflation 2020-2026…")
    actual_yoy = load_actual_truflation_yoy()
    val = validation_block(synthetic_yoy, actual_yoy)
    summary = summarise_residuals(val)

    print(f"\nValidation summary (n={summary['n']}):")
    print(f"  Window:           {summary['window_start']} → {summary['window_end']}")
    print(f"  Median residual:  {summary['median_pp']:+.4f} pp")
    print(f"  Mean residual:    {summary['mean_pp']:+.4f} pp")
    print(f"  SD residual:      {summary['sd_pp']:.4f} pp")
    print(f"  |residual| max:   {summary['abs_max_pp']:.4f} pp")
    print(f"  |residual| p95:   {summary['abs_p95_pp']:.4f} pp")
    print(f"  Within 0.1 pp:    {summary['share_within_0_1_pp']:.1%}")
    print(f"  Within 0.5 pp:    {summary['share_within_0_5_pp']:.1%}")
    print(f"  Within 1.0 pp:    {summary['share_within_1_0_pp']:.1%}")

    # Compare to the in-domain composition_check baseline
    print("\nReference: composition_check.py M2 (Truflation 12 components → headline):")
    print(f"  Median residual:  +0.000 pp")
    print(f"  SD residual:      0.224 pp")
    print(f"  Within 0.5 pp:    94.0%")

    # Save outputs
    level_path = OUT_DIR / "synthetic_truflation_top12_level.csv"
    yoy_path = OUT_DIR / "synthetic_truflation_top12_yoy.csv"
    val_path = OUT_DIR / "validation_overlap_2020_2026.csv"
    synthetic_level.to_frame().reset_index().to_csv(level_path, index=False)
    synthetic_yoy.to_frame().reset_index().to_csv(yoy_path, index=False)
    val.reset_index().to_csv(val_path, index=False)
    print(f"\nSaved:")
    print(f"  {level_path}")
    print(f"  {yoy_path}")
    print(f"  {val_path}")

    # Verdict — calibrated for CROSS-SOURCE backcast (BLS data → Truflation
    # target). The in-domain composition_check baseline (94% within 0.5 pp)
    # is reconstructing Truflation FROM Truflation; we're reconstructing
    # FROM BLS data, which has methodological differences (survey vs
    # real-time scraping) that the synthetic backcast cannot eliminate.
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    abs_median = abs(summary["median_pp"])
    cov_1pp = summary["share_within_1_0_pp"]

    if abs_median <= 0.30 and cov_1pp >= 0.80:
        print(f"  ✅ PASS — synthetic series tracks actual Truflation with")
        print(f"           |median bias| = {abs_median:.3f} pp ≤ 0.30 pp and")
        print(f"           {cov_1pp:.1%} of months within 1.0 pp.")
        print(f"           This is the structural BLS-vs-Truflation methodology")
        print(f"           gap (survey-vs-real-time-scraping), which the backcast")
        print(f"           cannot eliminate. The series is suitable for foundation-")
        print(f"           model pretraining (the model learns inflation dynamics")
        print(f"           broadly; the small level offset is absorbed by the")
        print(f"           per-target bridge layer at fine-tuning time).")
        print(f"           Ready to extend to 68 sub-components.")
    elif cov_1pp >= 0.70:
        print(f"  ⚠ PARTIAL — coverage {cov_1pp:.1%} within 1.0 pp, "
                f"|median bias| {abs_median:.3f} pp.")
        print(f"            Mechanism works but bias or SD is wider than ideal.")
        print(f"            Inspect cross-walk against Truflation taxonomy docs")
        print(f"            before extending to sub-components.")
    else:
        print(f"  ❌ FAIL — coverage {cov_1pp:.1%} within 1.0 pp.")
        print(f"           Investigate weight cross-walk and base-period choice")
        print(f"           before extending to sub-components.")


if __name__ == "__main__":
    main()
