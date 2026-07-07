# Data Schema

## Document Inventory

The document inventory should contain one row per source document.

Required columns:

```text
document_id
document_type
source_url
published_at
retrieved_at
meeting_date
speaker
speaker_role
is_fomc_voter
is_chair
title
raw_text_path
clean_text
source_page
date_quality
include_by_default
```

Recommended `document_type` values:

```text
fomc_statement
fomc_minutes
chair_press_conference
board_speech
board_testimony
reserve_bank_speech
voting_member_commentary
```

`published_at` must be the actual publication date, not necessarily the meeting
date.

`date_quality` examples:

```text
url_date
minutes_release_date
feed_pubdate
manual_verified
missing
```

Rows with `date_quality = missing` should not enter historical TSI aggregation.

## Scored Documents

The scorer appends these columns:

```text
dovish_terms
hawkish_terms
neutral_policy_terms
eligible_policy_terms
document_score
term_hits_json
scoring_dictionary_version
```

Formula:

```text
eligible_policy_terms =
  dovish_terms + hawkish_terms + neutral_policy_terms

document_score =
  (dovish_terms - hawkish_terms) / eligible_policy_terms
```

If `eligible_policy_terms = 0`, `document_score` should be blank.

## Quarterly TSI Output

The quarterly output should contain:

```text
period_label
asof_date
window_start
window_end
document_count
dovish_terms
hawkish_terms
neutral_policy_terms
eligible_policy_terms
tsi
interpretation
```

Formula:

```text
tsi =
  (sum(dovish_terms) - sum(hawkish_terms))
  / sum(eligible_policy_terms)
```

Suggested interpretation thresholds for review only:

```text
tsi > 0.01     dovish
-0.01 to 0.01  neutral
tsi < -0.01    hawkish
```

These thresholds should be calibrated against history before publishing.
