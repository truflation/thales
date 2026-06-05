# LLM Token Price Index v1

This project tracks public list prices for LLM inference tokens. It is a price index, not a usage-weighted spend index.

For the methods/results report, see [TOKEN_PRICE_INDEX_METHODS_RESULTS.md](TOKEN_PRICE_INDEX_METHODS_RESULTS.md).

## Executive Summary

The LLM Token Price Index is designed to answer one question:

```text
Are public LLM token prices getting cheaper or more expensive over time?
```

The current version collects public model-price data from OpenRouter, Portkey, LiteLLM, and Simon Willison's llm-prices feed, normalizes prices to USD per 1 million tokens, filters to a core text-generation universe for the headline index, combines matching source aliases, and calculates token-type indices for input, output, reasoning, cache read, cache write, and one-hour cache write prices.

The first baseline snapshot was dated `2026-06-04`. The current v1 snapshot dated `2026-06-05` produced `12,838` raw observations and `8,780` cleaned core observations. Trend information remains limited because only two live snapshots exist so far.

## Token Types

All prices are normalized to USD per 1 million tokens.

- `input`: standard prompt/input tokens
- `output`: completion/generated tokens
- `reasoning`: internal reasoning or thinking tokens where separately priced
- `cache_read`: cached input read/hit tokens
- `cache_write`: cached input write/create tokens
- `cache_write_1h`: one-hour cache write/create variants where exposed

## Current Snapshot Results

Snapshot date: `2026-06-05`

Normalized observations:

```text
raw observations:           12,838
positive raw observations:  11,687
cleaned core observations:   8,780
sources:               OpenRouter, Portkey, LiteLLM, Simon llm-prices
```

Cleaned core observations by token type:

```text
input:          3,671
output:         3,636
cache_read:     1,018
cache_write:      297
cache_write_1h:   109
reasoning:         49
```

Median public list prices:

```text
input:          $0.50 / 1M tokens
output:         $1.50 / 1M tokens
reasoning:      $2.50 / 1M tokens
cache_read:     $0.175 / 1M tokens
cache_write:    $3.75 / 1M tokens
cache_write_1h: $6.75 / 1M tokens
```

The shape is directionally sensible: output tokens are more expensive than input tokens, cache reads are discounted, cache writes often carry a premium, and one-hour cache writes are priced above standard cache writes.

## Sources

- OpenRouter Models API: https://openrouter.ai/docs/guides/overview/models
- OpenRouter live model JSON: https://openrouter.ai/api/v1/models
- Portkey Models Database: https://github.com/Portkey-AI/models
- Portkey pricing API example: https://configs.portkey.ai/pricing/openai.json
- LiteLLM pricing JSON: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
- Simon Willison llm-prices: https://github.com/simonw/llm-prices
- Simon current price feed: https://www.llm-prices.com/current-v1.json

Official provider pages remain validation references:

- OpenAI: https://openai.com/api/pricing/
- Anthropic: https://docs.anthropic.com/en/docs/about-claude/pricing
- Google Gemini: https://ai.google.dev/gemini-api/docs/pricing
- AWS Bedrock: https://aws.amazon.com/bedrock/pricing/
- Mistral: https://mistral.ai/pricing/
- Together AI: https://docs.together.ai/docs/inference/pricing
- Groq: https://groq.com/pricing
- Fireworks AI: https://fireworks.ai/docs/faq-new/billing-pricing/how-much-does-fireworks-cost

## Run

```bash
python llm_token_price_index/scripts/build_token_price_index.py
```

Outputs are written under `llm_token_price_index/csv/`:

- [normalized_prices_latest.csv](../csv/normalized_prices_latest.csv): latest normalized source observations
- [normalized_prices_history.csv](../csv/normalized_prices_history.csv): appended snapshot history
- [core_prices_latest.csv](../csv/core_prices_latest.csv): latest cleaned core-universe prices
- [core_prices_history.csv](../csv/core_prices_history.csv): cleaned core-universe price history
- [price_index_latest.csv](../csv/price_index_latest.csv): headline core-universe token-type price indices and descriptive stats
- [all_price_index_latest.csv](../csv/all_price_index_latest.csv): broad all-catalog diagnostic index
- [summary_latest.json](../csv/summary_latest.json): run metadata and output paths

## Methodology

Raw source/model/provider/token-type observations are retained for audit. The headline index is calculated from cleaned core-universe rows. The index for each token type is calculated as:

```text
TPI(date) = geometric mean(price(date) / price(base_date)) * 100
```

Zero prices are retained in normalized tables but excluded from index and descriptive statistics because geometric price ratios cannot use zero values.

Each cleaned core row groups observations by canonical provider, canonical model id, and token type. When multiple sources report the same canonical series, the row uses the median source price and retains source counts, min/max source prices, and source model ids for audit.

Alongside the index, the pipeline emits descriptive statistics:

```text
median
mean
min / max
p10 / p25 / p75 / p90
model count
provider count
source count
```

The median and percentiles are more reliable than the mean because the raw catalogs contain extreme provider-specific prices and non-core model types.

## What We Learned

The source strategy is strong enough for a first shared index. OpenRouter, Portkey, LiteLLM, and Simon's feed together provide broad input and output pricing coverage, plus enough cache and reasoning-token data to track those specialized price categories separately.

Coverage is strongest for input and output tokens. Cache-read coverage is also meaningful. Cache-write and one-hour cache-write coverage is smaller but usable. Reasoning-token pricing is still sparse, so the reasoning index should be labeled as lower-coverage.

The raw source universe is broader than the headline index should be. The source catalogs include chat models, responses models, embeddings, image models, audio models, and some unusual hosted endpoints. The headline index therefore filters to core text-generation models while keeping the broader source catalog available for audit.

The first snapshot establishes the base date. The trend story begins once the collector is run repeatedly or more historical pricing is backfilled.

## Caveats

Tokens are not perfectly comparable across model families because tokenizers differ. The index should be labeled as a public list-price index and should not be interpreted as enterprise net pricing, usage-weighted market pricing, or aggregate LLM spend.

The index excludes private discounts, committed-use pricing, promotional credits, enterprise contracts, and any unpublished provider-specific terms.
