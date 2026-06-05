# LLM Token Price Index v1

This folder contains the first shared LLM Token Price Index package: method notes, source code, latest CSV outputs, taxonomy files, validation outputs, and plots.

The index tracks public list prices for LLM API tokens, normalized to USD per 1 million tokens. It is a price index, not a usage or spend index.

## Start Here

- [Method and results doc](docs/TOKEN_PRICE_INDEX_METHODS_RESULTS.md)
- [Short overview](docs/TOKEN_PRICE_INDEX.md)

## Primary CSVs

- [normalized_prices_latest.csv](csv/normalized_prices_latest.csv): all raw source price observations after unit conversion
- [core_prices_latest.csv](csv/core_prices_latest.csv): cleaned headline model/token prices
- [price_index_latest.csv](csv/price_index_latest.csv): headline token price index values
- [model_taxonomy_latest.csv](csv/model_taxonomy_latest.csv): model companies and model families
- [price_validation_latest.csv](csv/price_validation_latest.csv): cross-source agreement checks
- [summary_latest.json](csv/summary_latest.json): run metadata and output file map

## Plots

- [median_token_prices.png](assets/median_token_prices.png)
- [token_price_distribution.png](assets/token_price_distribution.png)
- [top_model_companies.png](assets/top_model_companies.png)

## Rebuild

From the repository root:

```bash
python llm_token_price_index/scripts/build_token_price_index.py --historical-backfill
```

Outputs are written back into `llm_token_price_index/csv/`, and plots are written to `llm_token_price_index/assets/`.
