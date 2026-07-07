# TSI Methodology

## Objective

The Text Sentiment Index measures whether official Fed communication is shifting
more dovish or more hawkish.

It is one component of the TruFed Index. The active TruFed framework weights TSI
at 25% of the final index:

```text
TruFed Index =
  Dot Plot Index * 0.50
+ Text Sentiment Index * 0.25
+ Fed Funds Rate Projection * 0.25
```

This folder only covers the Text Sentiment Index.

## Formula

```text
TSI = (dovish_terms - hawkish_terms) / eligible_policy_terms
```

Interpretation:

- Positive TSI = more dovish communication.
- Negative TSI = more hawkish communication.
- Near-zero TSI = balanced or neutral communication.

## Eligible Policy Terms

`eligible_policy_terms` is not the total word count of a document. It is the
count of terms that are relevant to Fed policy scoring.

```text
eligible_policy_terms =
  dovish_term_count
+ hawkish_term_count
+ neutral_policy_term_count
```

Neutral policy terms are policy-relevant terms that provide the denominator but
do not push the score in either direction. Examples include `inflation`, `labor
market`, `federal funds rate`, `target range`, `employment`, `financial
conditions`, and `policy stance`.

This prevents long non-policy documents from diluting the score simply because
they contain many words.

## Document Score

Each document can be scored independently:

```text
document_score =
  (dovish_terms - hawkish_terms) / eligible_policy_terms
```

If `eligible_policy_terms = 0`, the document should be marked as unscored and
excluded from TSI aggregation.

## Quarterly TSI

For each quarterly SEP/Dot Plot release date, collect the official Fed
communication available during that quarter and aggregate counts:

```text
TSI_q =
  (sum(dovish_terms_q) - sum(hawkish_terms_q))
  / sum(eligible_policy_terms_q)
```

This count-based aggregation is preferred over a simple average of document
scores because it naturally weights documents by how much eligible policy
language they contain.

## Point-In-Time Rule

A document can only enter the historical TSI value if:

```text
published_at <= calculation_asof_date
```

This matters most for FOMC minutes because they are released after the meeting.
Minutes should not be inserted into the meeting-date TSI if they were not
available yet.

## Dictionary Approach

Use a Fed-specific hawkish/dovish phrase dictionary, not a generic sentiment
dictionary alone.

Reason: ordinary positive/negative sentiment often maps incorrectly in monetary
policy. For example, `weak labor market` is negative in generic sentiment, but
it is dovish for Fed policy.

Recommended dictionary structure:

- `dovish`: phrases that imply easier policy, lower rate bias, weaker labor,
  inflation cooling, or growth downside.
- `hawkish`: phrases that imply tighter policy, higher rate bias, inflation
  persistence, strong labor, or upside inflation risk.
- `neutral_policy`: policy-relevant denominator terms that do not determine
  direction by themselves.

## Matching Rule

The reference scorer uses phrase matching:

- lowercase text,
- phrase-level regular expression matching,
- longest-match-first de-duplication,
- separate counts for dovish, hawkish, and neutral policy terms.

The production scorer can be upgraded later with stemming, negation handling,
speaker weighting, or LLM-assisted classification, but the initial production
metric should stay interpretable and auditable.

## Quality Checks

Before production use, hand-label a validation set across different regimes:

- 2008 financial crisis,
- 2011-2012 zero-rate / forward-guidance period,
- 2015-2018 hiking cycle,
- 2020 COVID emergency easing,
- 2022 inflation hiking cycle,
- 2024-2026 hold/cut debate.

Compare dictionary output to human labels and adjust the dictionary before
publishing the index.
