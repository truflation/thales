# Source Map

## FOMC Calendar

Source:

`https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm`

Use for:

- recent FOMC statements,
- recent FOMC minutes,
- Chair press conference pages,
- projection-material release dates.

Extract:

- `meeting_date`
- `statement_url`
- `minutes_url`
- `minutes_released_at`
- `press_conference_url`
- `projection_materials_url`

For TSI scoring, use statements, minutes, and press conference transcripts.
Projection materials are useful for aligning quarterly SEP dates, but they are
not part of TSI text scoring.

## Historical FOMC Pages

Source pattern:

`https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm`

Use for:

- FOMC statements before the recent-calendar coverage,
- FOMC minutes before the recent-calendar coverage,
- older press conference links where available.

Extract the same fields as the FOMC calendar.

## Chair Press Conferences

Source:

Press conference pages linked from FOMC meeting pages.

Extract:

- transcript PDF URL where available,
- Chair opening statement,
- Chair answers,
- date,
- meeting date,
- source URL.

Exclude reporter questions from the scored text if the transcript format allows
speaker separation. If not, flag this as `speaker_split_missing`.

## Board Speeches and Testimony

Source:

`https://www.federalreserve.gov/newsevents/speeches-testimony.htm`

Use for:

- Board Governor speeches,
- Chair and Vice Chair speeches outside FOMC press conferences,
- congressional testimony.

Extract:

- speaker,
- speaker role,
- title,
- event,
- date,
- URL,
- full text.

## Reserve Bank Speeches

Example source:

`https://www.newyorkfed.org/newsevents/speeches`

Use for:

- New York Fed president speeches,
- annual voting Reserve Bank president speeches.

Extract:

- speaker,
- Reserve Bank,
- voter status for that year,
- title,
- date,
- URL,
- full text.

## FOMC Membership

Source:

`https://www.federalreserve.gov/monetarypolicy/fomc.htm`

Use for:

- current member list,
- current voting members,
- FOMC rotation rules.

For historical scoring, also derive voting status from each FOMC statement
because statements list voters for that meeting.

## Sentiment Dictionary

Base source:

`https://sraf.nd.edu/loughranmcdonald-master-dictionary/`

Use for:

- generic financial-language context,
- negative/positive uncertainty words as optional features.

Do not use it directly as the final TSI dictionary. Fed policy tone requires a
separate hawkish/dovish phrase dictionary.
