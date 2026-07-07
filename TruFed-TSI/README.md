# TruFed TSI Reference

This folder is a reference implementation for the TruFed Text Sentiment Index
component.

The goal is to give the data and engineering team a clear starting point for:

1. collecting historical Fed communication,
2. storing it with point-in-time metadata,
3. scoring hawkish and dovish policy language, and
4. aggregating that score into the quarterly TSI input for the TruFed Index.

The active framework formula is:

```text
TSI = (dovish_terms - hawkish_terms) / eligible_policy_terms
```

Positive TSI means more dovish Fed communication. Negative TSI means more
hawkish Fed communication.

## Folder Layout

```text
TruFed-TSI/
├── README.md
├── config/
│   ├── fed_policy_dictionary.json
│   └── source_catalog.json
├── docs/
│   ├── DATA_SCHEMA.md
│   ├── HISTORICAL_COLLECTION.md
│   ├── METHODOLOGY.md
│   └── SOURCE_MAP.md
└── scripts/
    ├── aggregate_tsi.py
    ├── collect_fed_links.py
    ├── extract_pdf_text.py
    └── score_documents.py
```

## Recommended Workflow

Create an inventory of official Fed documents. For the FOMC-only MVP, use:

```bash
python TruFed-TSI/scripts/collect_fed_links.py \
  --start-year 2007 \
  --end-year 2026 \
  --sources fomc \
  --out TruFed-TSI/data/fed_document_inventory.csv \
  --raw-text-dir TruFed-TSI/data/raw_text
```

Score the collected documents:

```bash
python TruFed-TSI/scripts/extract_pdf_text.py \
  --input TruFed-TSI/data/fed_document_inventory.csv \
  --output TruFed-TSI/data/fed_document_inventory_with_pdf_text.csv \
  --raw-text-dir TruFed-TSI/data/raw_text

python TruFed-TSI/scripts/score_documents.py \
  --input TruFed-TSI/data/fed_document_inventory_with_pdf_text.csv \
  --dictionary TruFed-TSI/config/fed_policy_dictionary.json \
  --output TruFed-TSI/data/scored_documents.csv
```

Aggregate into a quarterly TSI series. If SEP release dates are available, use
them as the as-of dates:

```bash
python TruFed-TSI/scripts/aggregate_tsi.py \
  --input TruFed-TSI/data/scored_documents.csv \
  --asof-dates TruFed-TSI/data/sep_release_dates.csv \
  --output TruFed-TSI/data/quarterly_tsi.csv
```

If no SEP release-date file is provided, the script falls back to calendar
quarters based on `published_at`. Calendar-quarter output is enough for a
quarterly TSI series. SEP-date alignment is only needed if the final TruFed
Index is calculated strictly on each quarterly Dot Plot release date instead of
quarter end.

## Validation Run

A full FOMC-only run was completed for 2007 through 2026.

Command shape:

```bash
python TruFed-TSI/scripts/collect_fed_links.py \
  --start-year 2007 \
  --end-year 2026 \
  --sources fomc \
  --out TruFed-TSI/data/fomc_inventory_2007_2026.csv \
  --raw-text-dir TruFed-TSI/data/raw_text \
  --sleep-seconds 0.05

python TruFed-TSI/scripts/extract_pdf_text.py \
  --input TruFed-TSI/data/fomc_inventory_2007_2026.csv \
  --output TruFed-TSI/data/fomc_inventory_2007_2026_pdftext.csv \
  --raw-text-dir TruFed-TSI/data/raw_text

python TruFed-TSI/scripts/score_documents.py \
  --input TruFed-TSI/data/fomc_inventory_2007_2026_pdftext.csv \
  --dictionary TruFed-TSI/config/fed_policy_dictionary.json \
  --output TruFed-TSI/data/fomc_scored_2007_2026.csv

python TruFed-TSI/scripts/aggregate_tsi.py \
  --input TruFed-TSI/data/fomc_scored_2007_2026.csv \
  --output TruFed-TSI/data/fomc_quarterly_tsi_2007_2026.csv
```

Results:

- 415 official FOMC documents collected and scored.
- 168 FOMC statements.
- 154 FOMC minutes.
- 93 Chair press conference transcripts.
- 0 missing publication dates.
- 0 missing extracted text files.
- 0 documents with zero eligible policy terms.
- 78 quarterly TSI rows from 2007Q1 through 2026Q2.
- No missing quarter labels.

Recent quarterly TSI values from the validation run:

```text
2025Q1: -0.02753108 hawkish
2025Q2: -0.03621730 hawkish
2025Q3: -0.02571166 hawkish
2025Q4:  0.01003584 dovish
2026Q1: -0.01186944 hawkish
2026Q2: -0.02545455 hawkish
```

This confirms the reference pipeline can collect official FOMC communication,
extract text, score documents, and produce a quarterly TSI dataset without any
database or backend integration.

## What This Is

This is a reference pipeline for the TSI component only. It is meant to be easy
to inspect and translate into production code.

It handles:

- official source inventory,
- point-in-time document metadata,
- dictionary-based hawkish/dovish scoring,
- eligible policy term calculation,
- document-level scores, and
- quarterly aggregation.

## What Still Needs Production Hardening

The team should harden these pieces before production use:

- FOMC voter mapping by historical year.
- Reserve Bank president speech collection across all Reserve Bank sites.
- Manual review and expansion of the Fed-specific dictionary.
- Tests against hand-labeled FOMC documents.
- Storage in the production data warehouse instead of local CSV outputs.

## Dictionary Improvement Plan

The starter dictionary is intentionally simple and auditable. Improve it in
four steps:

1. Create a review set of FOMC documents from major policy regimes: 2008,
   2011-2012, 2015-2018, 2020, 2022, and 2024-2026.
2. Hand-label paragraphs or sentences as hawkish, dovish, or neutral.
3. Compare model term hits against the labels and add missing Fed-specific
   phrases to `config/fed_policy_dictionary.json`.
4. Add phrase weights only after review. Start with all weights equal to 1,
   then raise weights for high-signal phrases such as `not ready to cut`,
   `upside inflation risks`, `labor market weakening`, and `appropriate to cut`.

Dictionary review should focus on Fed-specific meaning rather than generic
sentiment. For example, weak labor language is dovish for monetary policy, even
though it is negative in ordinary sentiment.

Useful checks:

- Review the most frequent hawkish and dovish hits by year.
- Inspect phrases that appear in both hiking and cutting regimes.
- Add negation-safe phrases explicitly, for example `not ready to cut`.
- Keep neutral policy terms broad enough to stabilize the denominator, but not
  so broad that non-policy text dilutes the score.
- Re-run the historical TSI after each dictionary change and check whether
  known regimes move in the expected direction.

## Source Discipline

Use official Fed and Reserve Bank sources first. Media summaries should only be
used as a fallback when official text is unavailable, and should be flagged as
non-official.

No look-ahead rule: a document can only enter a historical TSI value if
`published_at <= calculation_asof_date`.
