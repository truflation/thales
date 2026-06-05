# Token Price Index Model Taxonomy

Snapshot date: `2026-06-05`

This taxonomy groups the cleaned core model universe by model developer/company and model family.

## 01.AI

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| 01.AI Yi | 11 | 3 | input,output | llava-yi-34b, nous-hermes-2-yi-34b, tongyi-deepresearch-30b-a3b, yi-34b, yi-34b-200k-capybara, yi-34b-chat, yi-6b, yi-large |

## AI2

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| OLMo | 8 | 3 | input,output | allenai/molmo-2-8b, allenai/olmo-2-0325-32b-instruct, allenai/olmo-3-32b-think, allenai/olmo-3-7b-instruct, allenai/olmo-3-7b-think, allenai/olmo-3.1-32b-instruct, allenai/olmo-3.1-32b-think, olmo-3-32b-think |

## AI21

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Jamba | 27 | 5 | input,output | ai21-jamba-1.5-large, ai21-jamba-1.5-mini, ai21.jamba-1-5-large-v1-0, ai21.jamba-1-5-mini-v1-0, ai21.jamba-instruct-v1-0, ai21/jamba-large-1.7, ai21/jamba-mini-1.7, jamba-1.5 |
| Other / Unknown | 5 | 2 | input,output | ai21.j2-mid-v1, ai21.j2-ultra-v1, j2-light, j2-mid, j2-ultra |

## Aion Labs

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Aion | 6 | 2 | cache_read,input,output | aion-1.0, aion-1.0-mini, aion-2.0, aion-labs/aion-1.0, aion-labs/aion-1.0-mini, aion-labs/aion-2.0 |

## Alibaba / Qwen

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Qwen3 | 145 | 24 | cache_read,cache_write,input,output,reasoning | ap-northeast-1/qwen.qwen3-coder-next, ap-south-1/qwen.qwen3-coder-next, ap-southeast-3/qwen.qwen3-coder-next, cogito-v1-preview-qwen-32b, eu-central-1/qwen.qwen3-coder-next, eu-south-1/qwen.qwen3-coder-next, eu-west-1/qwen.qwen3-coder-next, eu-west-2/qwen.qwen3-coder-next |
| Other / Unknown | 67 | 6 | cache_read,cache_write,input,output,reasoning | code-qwen-1p5-7b, cogito-v1-preview-qwen-14b, qwen-coder, qwen-coder-plus, qwen-coder-plus-2024-11-06, qwen-coder-plus-latest, qwen-coder-turbo, qwen-coder-turbo-2024-09-19 |
| Qwen2 | 37 | 4 | input,output | dolphin-2-9-2-qwen2-72b, qwen/qwen2-vl-72b-instruct, qwen/qwen2-vl-7b-instruct, qwen2-72b-instruct, qwen2-7b-instruct, qwen2-vl-2b-instruct, qwen2-vl-72b-instruct, qwen2-vl-7b-instruct |
| Qwen2.5 | 31 | 11 | input,output | qwen-2.5-72b-instruct, qwen-2.5-7b-instruct, qwen-2.5-coder-32b-instruct, qwen-2.5-vl-7b-instruct, qwen/qwen-2.5-72b-instruct, qwen/qwen2.5-32b-instruct, qwen/qwen2.5-72b-instruct, qwen/qwen2.5-7b-instruct |
| QwQ / QVQ | 15 | 9 | input,output | arliai/qwq-32b-arliai-rpr-v1, qvq-max, qvq-max-2025-03-25, qvq-max-2025-05-15, qvq-max-latest, qvq-plus, qvq-plus-2025-05-15, qvq-plus-latest |

## Amazon

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Nova | 24 | 4 | cache_read,cache_write,cache_write_1h,input,output | amazon-nova-lite, amazon-nova-micro, amazon-nova-premier, amazon-nova-pro, amazon/nova-2-lite-v1, amazon/nova-lite, amazon/nova-lite-v1, amazon/nova-micro |
| Titan | 9 | 1 | input,output | titan-text-express-v1, titan-text-lite-v1, titan-text-premier-v1-0, us-gov-east-1/amazon.titan-text-express-v1, us-gov-east-1/amazon.titan-text-lite-v1, us-gov-east-1/amazon.titan-text-premier-v1-0, us-gov-west-1/amazon.titan-text-express-v1, us-gov-west-1/amazon.titan-text-lite-v1 |
| MiniMax | 1 | 1 | input,output | minimax-m2.7 |

## Anthropic

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Claude 4 | 81 | 15 | cache_read,cache_write,cache_write_1h,input,output | anthropic/claude-opus-4, anthropic/claude-opus-4.5, anthropic/claude-sonnet-4, anthropic/claude-sonnet-4.5, au.anthropic.claude-haiku-4-5-20251001-v1-0, au.anthropic.claude-opus-4-6-v1, au.anthropic.claude-opus-4-7, au.anthropic.claude-opus-4-8 |
| Other / Unknown | 22 | 3 | cache_read,cache_write,input,output | ap-northeast-1/anthropic.claude-instant-v1, ap-northeast-1/anthropic.claude-v1, ap-northeast-1/anthropic.claude-v2-1, claude, claude-2.1, claude-instant, claude-instant-v1, claude-v1 |
| Claude 3.5 | 20 | 10 | cache_read,cache_write,cache_write_1h,input,output | anthropic-claude-3.5-haiku, anthropic-claude-3.5-sonnet, claude-3-5-haiku, claude-3-5-haiku-20241022, claude-3-5-haiku-20241022-v1-0, claude-3-5-haiku-latest, claude-3-5-sonnet, claude-3-5-sonnet-20240620 |
| Claude 3 | 13 | 8 | cache_read,cache_write,cache_write_1h,input,output | anthropic-claude-3-opus, claude-3-haiku, claude-3-haiku-20240307, claude-3-haiku-20240307-v1-0, claude-3-opus, claude-3-opus-20240229, claude-3-opus-20240229-v1-0, claude-3-opus-latest |
| Claude 3.7 | 11 | 11 | cache_read,cache_write,cache_write_1h,input,output | anthropic-claude-3.7-sonnet, claude-3-7-sonnet, claude-3-7-sonnet-20240620-v1-0, claude-3-7-sonnet-20250219, claude-3-7-sonnet-20250219-v1-0, claude-3-7-sonnet-latest, claude-3-7-sonnet-v2-20250219, claude-3.7-sonnet |

## Arcee AI

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Arcee | 13 | 3 | cache_read,input,output | arcee-ai/coder-large, arcee-ai/maestro-reasoning, arcee-ai/spotlight, arcee-ai/trinity-large-preview, arcee-ai/trinity-large-thinking, arcee-ai/trinity-mini, arcee-ai/virtuoso-large, coder-large |

## Baidu

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| ERNIE | 15 | 5 | input,output | baidu/ernie-4.5-21b-a3b, baidu/ernie-4.5-21b-a3b-thinking, baidu/ernie-4.5-300b-a47b-paddle, baidu/ernie-4.5-vl-28b-a3b, baidu/ernie-4.5-vl-28b-a3b-thinking, baidu/ernie-4.5-vl-424b-a47b, ernie-4.5-21b-a3b, ernie-4.5-21b-a3b-thinking |

## ByteDance

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| ByteDance Seed | 50 | 3 | cache_read,cache_write,input,output | bytedance-seed/seed-1.6, bytedance-seed/seed-1.6-flash, bytedance-seed/seed-2.0-lite, bytedance-seed/seed-2.0-mini, bytedance/seed-1.8, bytedance/seed-2.0-code, bytedance/seed-2.0-mini, bytedance/seed-2.0-pro |
| UI-TARS | 2 | 2 | cache_read,input,output | bytedance/ui-tars-1.5-7b, ui-tars-1.5-7b |

## Cohere

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Command | 28 | 8 | input,output | cohere-command-r-08-2024, cohere-command-r-plus-08-2024, cohere.command-a-03-2025, cohere.command-a-reasoning, cohere.command-a-reasoning-08-2025, cohere.command-a-translate-08-2025, cohere.command-a-vision, cohere.command-a-vision-07-2025 |

## Core42

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| JAIS | 2 | 2 | input,output | core42/jais-13b-chat, jais-30b-chat |

## Databricks

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| DBRX | 2 | 2 | input,output | databricks/dbrx-instruct, dbrx-instruct |
| MPT | 2 | 1 | input,output | databricks-mpt-30b-instruct, databricks-mpt-7b-instruct |

## DeepSeek

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| DeepSeek V3 | 35 | 24 | cache_read,cache_write,input,output | deepseek-ai/deepseek-v3, deepseek-ai/deepseek-v3-0324, deepseek-ai/deepseek-v3.1, deepseek-ai/deepseek-v3.2, deepseek-v3, deepseek-v3-0324, deepseek-v3-0324-fast, deepseek-v3-0324-turbo |
| DeepSeek R1 | 28 | 21 | cache_read,input,output,reasoning | deepseek-ai/deepseek-r1, deepseek-ai/deepseek-r1-0528, deepseek-ai/deepseek-r1-distill-qwen-1.5b, deepseek-ai/deepseek-r1-distill-qwen-14b, deepseek-ai/deepseek-r1-distill-qwen-32b, deepseek-ai/deepseek-r1-distill-qwen-7b, deepseek-r1, deepseek-r1-0528 |
| Other / Unknown | 20 | 9 | cache_read,input,output | ap-northeast-1/deepseek.v3.2, ap-south-1/deepseek.v3.2, ap-southeast-3/deepseek.v3.2, deepcogito/cogito-v2-preview-deepseek-671b, deepseek-llm-67b-chat, deepseek-prover-v2, deepseek-reasoner, deepseek-v2-lite-chat |
| DeepSeek Coder | 9 | 2 | cache_read,input,output | deepseek-coder, deepseek-coder-1b-base, deepseek-coder-33b-instruct, deepseek-coder-7b-base, deepseek-coder-7b-base-v1p5, deepseek-coder-7b-instruct-v1p5, deepseek-coder-v2-instruct, deepseek-coder-v2-lite-base |
| DeepSeek Chat | 6 | 2 | cache_read,input,output | deepseek-chat, deepseek-chat-v3-0324, deepseek-chat-v3.1, deepseek/deepseek-chat, deepseek/deepseek-chat-v3-0324, deepseek/deepseek-chat-v3.1 |

## Google

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Gemini 2.5 | 44 | 13 | cache_read,cache_write,input,output,reasoning | databricks-gemini-2-5-flash, databricks-gemini-2-5-pro, gemini-2.5-flash, gemini-2.5-flash-gt-128k, gemini-2.5-flash-image, gemini-2.5-flash-image-gt-128k, gemini-2.5-flash-image-lte-128k, gemini-2.5-flash-image-preview |
| Gemini 1 | 42 | 3 | input,output | gemini-1.0-pro, gemini-1.0-pro-001, gemini-1.0-pro-002, gemini-1.0-pro-vision, gemini-1.0-pro-vision-001, gemini-1.5-flash, gemini-1.5-flash-001, gemini-1.5-flash-001-gt-128k |
| Gemini 3 | 42 | 8 | cache_read,cache_write,input,output,reasoning | gemini-3-1-pro-preview, gemini-3-1-pro-preview-200k, gemini-3-flash-preview, gemini-3-flash-preview-gt-128k, gemini-3-flash-preview-lte-128k, gemini-3-pro, gemini-3-pro-image-preview, gemini-3-pro-image-preview-gt-128k |
| Other / Unknown | 30 | 6 | cache_read,cache_write,input,output,reasoning | chat-bison, chat-bison-001, deep-research-pro-preview-12-2025-gt-128k, deep-research-pro-preview-12-2025-lte-128k, gemini-exp-1206, gemini-flash-1.5, gemini-flash-1.5-8b, gemini-flash-latest |
| Gemma | 29 | 17 | input,output | codegemma-2b, codegemma-7b, databricks-gemma-3-12b, gemini-gemma-2-27b-it, gemini-gemma-2-9b-it, gemma-2-27b-it, gemma-2-2b-it, gemma-2-9b |
| Gemini 2 | 17 | 8 | cache_read,input,output | gemini-2.0-flash, gemini-2.0-flash-001, gemini-2.0-flash-001-gt-128k, gemini-2.0-flash-001-lte-128k, gemini-2.0-flash-exp, gemini-2.0-flash-exp-001, gemini-2.0-flash-exp-001-gt-128k, gemini-2.0-flash-exp-001-lte-128k |
| Palmyra | 5 | 4 | input,output | palmyra-x5, writer.palmyra-vision-7b, writer.palmyra-x4-v1-0, writer.palmyra-x5-v1-0, writer/palmyra-x5 |

## Hugging Face

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Zephyr | 2 | 3 | input,output | huggingfaceh4/zephyr-7b-beta, zephyr-7b-beta |

## Inflection

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Inflection | 4 | 2 | input,output | inflection-3-pi, inflection-3-productivity, inflection/inflection-3-pi, inflection/inflection-3-productivity |

## Liquid AI

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| LFM | 6 | 3 | input,output | lfm-2-24b-a2b, lfm-40b, lfm-7b, liquid/lfm-2-24b-a2b, liquid/lfm-2.2-6b, liquid/lfm2-8b-a1b |

## Meta

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Other / Unknown | 64 | 16 | input,output | -hf/thebloke/codellama-7b-instruct-awq, alfredpros/codellama-7b-instruct-solidity, code-llama-13b, code-llama-13b-instruct, code-llama-13b-python, code-llama-70b, code-llama-70b-instruct, code-llama-70b-python |
| Llama 3 | 62 | 20 | input,output | ap-south-1/meta.llama3-70b-instruct-v1-0, ap-south-1/meta.llama3-8b-instruct-v1-0, ca-central-1/meta.llama3-70b-instruct-v1-0, ca-central-1/meta.llama3-8b-instruct-v1-0, code-llama-34b, code-llama-34b-instruct, code-llama-34b-python, codellama-34b-instruct |
| Llama 3.1 | 58 | 30 | input,output | aion-labs/aion-rp-llama-3.1-8b, aion-rp-llama-3.1-8b, databricks-meta-llama-3-1-405b-instruct, databricks-meta-llama-3-1-8b-instruct, dobby-mini-unhinged-plus-llama-3-1-8b, hermes-3-llama-3.1-405b, hermes-3-llama-3.1-70b, llama-3.1-405b |
| Llama 2 | 33 | 13 | input,output | -cf/meta/llama-2-7b-chat-fp16, -cf/meta/llama-2-7b-chat-int8, databricks-llama-2-70b-chat, llama-2-13b-chat-hf, llama-2-70b-chat, llama-2-70b-chat-hf, llama-2-7b-chat, llama-2-7b-chat-hf |
| Llama 3.2 | 27 | 15 | input,output | llama-3.2-11b-vision-instruct, llama-3.2-11b-vision-instruct-turbo, llama-3.2-1b-instruct, llama-3.2-3b, llama-3.2-3b-instruct, llama-3.2-3b-instruct-turbo, llama-3.2-90b-vision-instruct, llama-3.2-90b-vision-instruct-turbo |
| Llama 4 | 24 | 22 | input,output | databricks-llama-4-maverick, deepcogito/cogito-v2-preview-llama-405b, llama-4-maverick, llama-4-maverick-17b-128e-instruct, llama-4-maverick-17b-128e-instruct-fp8, llama-4-maverick-17b-128e-instruct-maas, llama-4-maverick-17b-128e-instruct-turbo, llama-4-scout |
| Llama 3.3 | 23 | 26 | input,output | databricks-meta-llama-3-3-70b-instruct, deepseek-llama3.3-70b, dobby-unhinged-llama-3-3-70b-new, llama-3.3-70b, llama-3.3-70b-instruct, llama-3.3-70b-instruct-fast, llama-3.3-70b-instruct-maas, llama-3.3-70b-instruct-turbo |
| DeepSeek R1 | 7 | 14 | input,output | deepseek-ai/deepseek-r1-distill-llama-70b, deepseek-ai/deepseek-r1-distill-llama-8b, deepseek-r1-7b-qwen, deepseek-r1-8b, deepseek-r1-distill-llama-70b, deepseek-r1-distill-llama-8b, deepseek/deepseek-r1-distill-llama-70b |
| Qwen3 | 2 | 1 | input,output | qwen3-8b, qwen3-vl-8b |
| DeepSeek Coder | 1 | 1 | input,output | deepseek-coder-6.7b |
| Mistral Base/Instruct | 1 | 1 | input,output | mistral-7b-v0.3 |
| Nemotron | 1 | 1 | input,output | llama-v3p1-nemotron-70b-instruct |
| Phi | 1 | 1 | input,output | dolphin3-8b |
| Qwen2.5 | 1 | 1 | input,output | qwen2.5-coder-7b |

## Microsoft

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Phi | 29 | 8 | cache_read,input,output | chatdolphin, dolphin, microsoft/phi-4-mini-instruct, phi-2, phi-2-3b, phi-3-medium-128k-instruct, phi-3-medium-4k-instruct, phi-3-mini-128k-instruct |

## MiniMax

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| MiniMax | 55 | 16 | cache_read,cache_write,input,output | ap-northeast-1/minimax.minimax-m2.1, ap-northeast-1/minimax.minimax-m2.5, ap-south-1/minimax.minimax-m2.1, ap-south-1/minimax.minimax-m2.5, ap-southeast-2/minimax.minimax-m2.5, ap-southeast-3/minimax.minimax-m2.1, ap-southeast-3/minimax.minimax-m2.5, eu-central-1/minimax.minimax-m2.1 |

## Mistral

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Mistral Base/Instruct | 28 | 16 | input,output | -cf/mistral/mistral-7b-instruct-v0.1, eu-west-3/mistral.mistral-7b-instruct-v0-2, hermes-2-pro-mistral-7b, intfloat/e5-mistral-7b-instruct, llava-v1.6-mistral-7b-hf, mistral-7b, mistral-7b-instruct, mistral-7b-instruct-4k |
| Mixtral | 24 | 19 | cache_read,input,output | cognitivecomputations/dolphin-2.6-mixtral-8x7b, databricks-mixtral-8x7b-instruct, discoresearch/discolm-mixtral-8x7b-v2, dolphin-2p6-mixtral-8x7b, eu-west-3/mistral.mixtral-8x7b-instruct-v0-1, mistralai/mixtral-8x22b-instruct-v0.1, mistralai/mixtral-8x7b-instruct-v0.1, mixtral-8x22b |
| Mistral Small | 19 | 14 | cache_read,input,output | labs-mistral-small-creative, mistral-small, mistral-small-2402-v1-0, mistral-small-2409, mistral-small-24b-instruct-2501, mistral-small-2501, mistral-small-2503, mistral-small-2503-001 |
| Ministral | 18 | 9 | cache_read,input,output | ministral-14b-2512, ministral-14b-latest, ministral-3-14b-2512, ministral-3-14b-instruct, ministral-3-14b-instruct-2512, ministral-3-3b-2512, ministral-3-3b-instruct, ministral-3-3b-instruct-2512 |
| Mistral Large | 17 | 12 | cache_read,input,output | eu-west-3/mistral.mistral-large-2402-v1-0, mistral-large, mistral-large-2402, mistral-large-2402-v1-0, mistral-large-2407, mistral-large-2407-v1-0, mistral-large-2411, mistral-large-2411-001 |
| Other / Unknown | 17 | 13 | cache_read,input,output | mistral-nemo, mistral-nemo-2407, mistral-nemo-base-2407, mistral-nemo-instruct-2407, mistral-nemo-latest, mistral-saba, mistral-saba-24b, mistral-saba-latest |
| Devstral | 11 | 9 | cache_read,input,output | devstral-2-123b, devstral-2512, devstral-latest, devstral-medium, devstral-medium-2507, devstral-medium-latest, devstral-small, devstral-small-2505 |
| Magistral | 11 | 6 | input,output | magistral-medium, magistral-medium-1-2-2509, magistral-medium-2506, magistral-medium-2506-thinking, magistral-medium-2509, magistral-medium-latest, magistral-small, magistral-small-1-2-2509 |
| Mistral Medium | 11 | 8 | cache_read,input,output | mistral-medium, mistral-medium-2312, mistral-medium-2505, mistral-medium-2508, mistral-medium-3, mistral-medium-3-001, mistral-medium-3-1-2508, mistral-medium-3-5 |
| Codestral | 10 | 9 | cache_read,input,output | codestral, codestral-2, codestral-2-001, codestral-2405, codestral-2501, codestral-2508, codestral-latest, codestral-mamba-latest |
| Pixtral | 7 | 6 | input,output | mistralai/pixtral-12b-2409, pixtral-12b, pixtral-12b-2409, pixtral-large, pixtral-large-2411, pixtral-large-2502-v1-0, pixtral-large-latest |
| Hermes | 1 | 1 | input,output | nousresearch/deephermes-3-mistral-24b-preview |

## Moonshot AI

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Kimi | 48 | 23 | cache_read,input,output | ap-northeast-1/moonshotai.kimi-k2-thinking, ap-northeast-1/moonshotai.kimi-k2.5, ap-south-1/moonshotai.kimi-k2-thinking, ap-south-1/moonshotai.kimi-k2.5, ap-southeast-3/moonshotai.kimi-k2.5, eu-north-1/moonshotai.kimi-k2.5, kimi-dev-72b, kimi-k2 |
| Other / Unknown | 10 | 1 | input,output | moonshot-v1-128k, moonshot-v1-128k-0430, moonshot-v1-128k-vision-preview, moonshot-v1-32k, moonshot-v1-32k-0430, moonshot-v1-32k-vision-preview, moonshot-v1-8k, moonshot-v1-8k-0430 |

## NVIDIA

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Nemotron | 20 | 7 | cache_read,input,output | nemotron-3-nano-30b-a3b, nemotron-3-super-120b-a12b, nemotron-3-ultra-550b-a55b, nemotron-nano-9b-v2, nemotron-nano-v2-12b-vl, nvidia-nemotron-nano-12b-v2, nvidia-nemotron-nano-9b-v2, nvidia.nemotron-nano-12b-v2 |

## Nous Research

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Hermes | 10 | 7 | input,output | austism/chronos-hermes-13b, austism/chronos-hermes-13b-v2, chronos-hermes-13b-v2, hermes-4-405b, hermes-4-70b, hermes3-405b, hermes3-70b, hermes3-8b |
| Other / Unknown | 3 | 2 | input,output | nous-capybara-7b-v1p9, nousresearch/nous-capybara-7b-v1p9, togethercomputer/stripedhyena-nous-7b |

## OpenAI

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| GPT-5 | 81 | 11 | cache_read,input,output | databricks-gpt-5, databricks-gpt-5-1, databricks-gpt-5-mini, databricks-gpt-5-nano, eu/gpt-5-2025-08-07, eu/gpt-5-mini-2025-08-07, eu/gpt-5-nano-2025-08-07, eu/gpt-5.1 |
| Other / Unknown | 60 | 9 | cache_read,cache_write,input,output | ada-v2, babbage-002, chat-latest, chatgpt-image-latest, codex-mini, codex-mini-latest, davinci, davinci-002 |
| GPT-4o | 47 | 9 | cache_read,cache_write,input,output | chatgpt-4o-latest, eu/gpt-4o-2024-08-06, eu/gpt-4o-2024-11-20, eu/gpt-4o-mini-2024-07-18, eu/gpt-4o-mini-realtime-preview-2024-12-17, eu/gpt-4o-realtime-preview-2024-10-01, eu/gpt-4o-realtime-preview-2024-12-17, ft-gpt-4o |
| OpenAI o-series | 35 | 8 | cache_read,input,output,reasoning | eu/o1-2024-12-17, eu/o1-mini-2024-09-12, eu/o1-preview-2024-09-12, eu/o3-mini-2025-01-31, ft-o4-mini-2025-04-16, o1, o1-2024-12-17, o1-mini |
| GPT-4.1 | 20 | 8 | cache_read,input,output | ft-gpt-4.1, ft-gpt-4.1-2025-04-14, ft-gpt-4.1-mini, ft-gpt-4.1-mini-2025-04-14, ft-gpt-4.1-nano-2025-04-14, gpt-4-1106-preview, gpt-4-1106-vision-preview, gpt-4.1 |
| GPT OSS | 19 | 24 | cache_read,input,output | databricks-gpt-oss-120b, databricks-gpt-oss-20b, gpt-oss-120b, gpt-oss-120b-exacto, gpt-oss-120b-maas, gpt-oss-120b-turbo, gpt-oss-20b, gpt-oss-20b-maas |
| GPT-4 | 18 | 6 | cache_read,input,output | ft-gpt-4-0613, gpt-4, gpt-4-0125-preview, gpt-4-0314, gpt-4-0613, gpt-4-32k, gpt-4-32k-0314, gpt-4-32k-0613 |

## OpenChat

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| OpenChat | 3 | 3 | input,output | openchat-3p5-0106-7b, openchat/openchat-3.5, openchat/openchat-3.5-1210 |

## Other / Unknown

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Other / Unknown | 206 | 39 | cache_read,input,output | 100b, 16b, 21b, 4b, 70b, 7b, ada, airoboros-70b |

## Perplexity

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Sonar | 16 | 4 | input,output,reasoning | perplexity/sonar, perplexity/sonar-deep-research, perplexity/sonar-pro, perplexity/sonar-pro-search, perplexity/sonar-reasoning, perplexity/sonar-reasoning-pro, sonar, sonar-deep-research |
| Other / Unknown | 4 | 2 | input,output | pplx-70b-chat, pplx-70b-online, pplx-7b-chat, pplx-7b-online |

## Reka

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Reka | 10 | 3 | input,output | reka-core, reka-core-20240415, reka-core-20240501, reka-edge, reka-edge-20240208, reka-flash, reka-flash-20240226, reka-flash-3 |

## Technology Innovation Institute

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Falcon | 4 | 1 | input,output | togethercomputer/falcon-40b, togethercomputer/falcon-40b-instruct, togethercomputer/falcon-7b, togethercomputer/falcon-7b-instruct |

## Tencent

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Hunyuan | 2 | 2 | input,output | hunyuan-a13b-instruct, tencent/hunyuan-a13b-instruct |

## Twelve Labs

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Pegasus | 1 | 1 | output | twelvelabs.pegasus-1-2-v1-0 |

## Zhipu AI

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| GLM | 63 | 21 | cache_read,input,output | glm-4-32b, glm-4-32b-0414-128k, glm-4.5, glm-4.5-air, glm-4.5-airx, glm-4.5-x, glm-4.5v, glm-4.6 |

## xAI

| Family | Canonical models | Providers | Token types | Examples |
|---|---:|---:|---|---|
| Grok 4 | 66 | 9 | cache_read,input,output | grok-4, grok-4-0709, grok-4-1-fast, grok-4-1-fast-non-reasoning, grok-4-1-fast-non-reasoning-latest, grok-4-1-fast-reasoning, grok-4-1-fast-reasoning-latest, grok-4-128k |
| Grok 3 | 18 | 7 | cache_read,input,output | global/grok-3, global/grok-3-mini, grok-3, grok-3-beta, grok-3-fast, grok-3-fast-beta, grok-3-fast-latest, grok-3-latest |
| Grok | 7 | 6 | cache_read,input,output | grok-beta, grok-build-0.1, grok-code-fast, grok-code-fast-1, grok-code-fast-1-0825, grok-vision-beta, xai.grok-code-fast-1 |
| Grok 2 | 6 | 2 | input,output | grok-2, grok-2-1212, grok-2-latest, grok-2-vision, grok-2-vision-1212, grok-2-vision-latest |
