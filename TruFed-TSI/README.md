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

Create an inventory of official Fed documents:

```bash
python TruFed-TSI/scripts/collect_fed_links.py \
  --start-year 2007 \
  --end-year 2026 \
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
quarters based on `published_at`.

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

- PDF extraction for older documents and transcript PDFs.
- FOMC voter mapping by historical year.
- Reserve Bank president speech collection across all Reserve Bank sites.
- Manual review and expansion of the Fed-specific dictionary.
- Tests against hand-labeled FOMC documents.
- Storage in the production data warehouse instead of local CSV outputs.

## Source Discipline

Use official Fed and Reserve Bank sources first. Media summaries should only be
used as a fallback when official text is unavailable, and should be flagged as
non-official.

No look-ahead rule: a document can only enter a historical TSI value if
`published_at <= calculation_asof_date`.
