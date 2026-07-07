# Historical TSI Collection

## Historical Start

Build the historical TSI series from 2007 onward to align with the SEP/Dot Plot
history used elsewhere in the TruFed framework.

Coverage caveats:

- FOMC statements and minutes are available throughout the period.
- SEP materials start in 2007.
- Chair press conferences begin later and should not be backfilled into earlier
  periods.
- Fed funds rate projections in the SEP begin later than the SEP itself.

For TSI, missing document types should be flagged as missing coverage, not
estimated.

## Quarterly Corpus Definition

For each quarterly SEP/Dot Plot release date:

```text
window_start = prior SEP release date, exclusive
window_end = current SEP release date, inclusive
```

Include official Fed communication where:

```text
window_start < published_at <= window_end
```

If SEP release dates are not available yet, use calendar quarters temporarily:

```text
published_at within calendar quarter
```

The SEP-date version is preferred because it aligns TSI with the rest of the
TruFed Index calculation.

## Source Priority

Use documents in this order:

1. FOMC statements.
2. Chair press conference transcripts where available.
3. FOMC minutes once released.
4. Speeches and testimony by Board members.
5. Speeches by voting Reserve Bank presidents.
6. Voting-member commentary only when official text is unavailable.

## Document Rules

Store every source document with:

- document id,
- document type,
- source URL,
- publication date,
- retrieval timestamp,
- meeting date if relevant,
- speaker if relevant,
- speaker role,
- FOMC voter status if known,
- title,
- raw text path,
- clean text path or clean text,
- source hash.

## FOMC Materials

Use:

- recent FOMC calendar:
  `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm`
- historical FOMC pages:
  `https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm`

Extract:

- statement URL and meeting date,
- minutes URL and actual minutes release date,
- press conference transcript PDF URL where available,
- projection-material URL for alignment, not for TSI scoring.

The collector should prefer HTML links where available because HTML is easier to
parse and audit than PDFs. The exception is Chair press conferences, where the
official transcript is commonly linked as a PDF from the press conference page.

## Board Speeches and Testimony

Use:

- `https://www.federalreserve.gov/newsevents/speeches-testimony.htm`
- yearly speeches pages,
- yearly testimony pages,
- RSS feeds for ongoing updates.

Extract:

- title,
- speaker,
- speaker role,
- date,
- event or venue,
- URL,
- text.

Filter:

- include monetary-policy, inflation, labor, growth, and financial-conditions
  content;
- exclude purely ceremonial, consumer-protection, bank-supervision, payments, or
  crypto-regulation speeches unless they contain monetary-policy language.

The reference scorer can still score all documents; production should add a
policy-relevance filter.

## Reserve Bank Speeches

Use the relevant Reserve Bank sites for voting regional presidents. The New York
Fed president should always be included because that role is a permanent FOMC
voter.

Minimum production scope:

- New York Fed speeches:
  `https://www.newyorkfed.org/newsevents/speeches`
- other Reserve Bank speech pages for the four rotating annual voters.

Extract the same fields as Board speeches.

## Voter Mapping

Use:

- current FOMC page:
  `https://www.federalreserve.gov/monetarypolicy/fomc.htm`
- voter lists embedded in each FOMC statement,
- historical annual FOMC rotation rules.

The reference data can include speeches before voter filtering, but production
should tag:

```text
is_fomc_voter = true / false / unknown
```

This allows the team to test whether the final TSI should use voting members
only or all official Fed speakers.

## No Look-Ahead Controls

Required controls:

- Statements enter on the statement release date.
- Press conference transcripts enter on the transcript/page release date.
- Minutes enter on the minutes release date, not the meeting date.
- Speeches enter on the speech publication date.
- Testimony enters on the testimony publication date.

Any document without a reliable `published_at` should be excluded from the
historical quarterly aggregation until the date is verified.
