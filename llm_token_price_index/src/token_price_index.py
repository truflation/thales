"""Build normalized LLM token price tables and price indices.

The module keeps source observations separate, normalizes token prices to
USD per million tokens, then calculates source-aware price indices by token
type. It intentionally tracks public list prices, not usage-weighted spend.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


USER_AGENT = "kairos-token-price-index/0.1"

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
PORTKEY_PRICING_DIR_URL = (
    "https://api.github.com/repos/Portkey-AI/models/contents/pricing?ref=main"
)
PORTKEY_PRICING_URL = "https://configs.portkey.ai/pricing/{provider}.json"
LITELLM_PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
SIMON_CURRENT_PRICES_URL = "https://www.llm-prices.com/current-v1.json"
SIMON_HISTORICAL_PRICES_URL = "https://www.llm-prices.com/historical-v1.json"

TOKEN_TYPES = (
    "input",
    "output",
    "reasoning",
    "cache_read",
    "cache_write",
    "cache_write_1h",
)

CORE_PORTKEY_PROVIDERS = (
    "openai",
    "anthropic",
    "google",
    "vertex-ai",
    "azure-openai",
    "bedrock",
    "mistral",
    "together-ai",
    "fireworks-ai",
    "groq",
    "deepinfra",
    "deepseek",
    "cerebras",
    "cohere",
    "x-ai",
    "perplexity-ai",
    "openrouter",
)

NON_CORE_MODEL_PATTERNS = (
    "audio",
    "bge-",
    "clip",
    "code-interpreter",
    "computer-use",
    "dall-e",
    "diffusion",
    "embed",
    "embedding",
    "flux",
    "gte-",
    "image-generation",
    "image_generation",
    "imagen",
    "janus",
    "moderation",
    "nomic-embed",
    "ocr",
    "rerank",
    "sdxl",
    "search-api",
    "sora",
    "speech",
    "stable-diffusion",
    "text2vec",
    "transcribe",
    "tts",
    "video",
    "whisper",
)

PROVIDER_ALIASES = {
    "amazon": "bedrock",
    "anthropic-vertex": "vertex-ai",
    "azure": "azure-openai",
    "azure-ai": "azure-ai",
    "bedrock-converse": "bedrock",
    "fireworks-ai": "fireworks-ai",
    "google-vertexai": "vertex-ai",
    "together-ai": "together-ai",
    "vertex-ai": "vertex-ai",
    "xai": "x-ai",
}

OFFICIAL_PRICING_URLS = {
    "AI21": "https://docs.ai21.com/docs/pricing",
    "Alibaba / Qwen": "https://www.alibabacloud.com/help/en/model-studio/models",
    "Amazon": "https://aws.amazon.com/bedrock/pricing/",
    "Anthropic": "https://docs.anthropic.com/en/docs/about-claude/pricing",
    "Baidu": "https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Blfmc9dlf",
    "Cerebras": "https://inference-docs.cerebras.ai/resources/pricing",
    "Cohere": "https://cohere.com/pricing",
    "DeepSeek": "https://api-docs.deepseek.com/quick_start/pricing",
    "Google": "https://ai.google.dev/gemini-api/docs/pricing",
    "01.AI": "https://platform.lingyiwanwu.com/",
    "AI2": "https://allenai.org/olmo",
    "Aion Labs": "https://www.aionlabs.ai/",
    "Arcee AI": "https://www.arcee.ai/",
    "ByteDance": "https://www.volcengine.com/product/ark",
    "Core42": "https://www.core42.ai/",
    "Databricks": "https://www.databricks.com/product/machine-learning/model-serving",
    "Hugging Face": "https://huggingface.co/",
    "Inflection": "https://inflection.ai/",
    "Liquid AI": "https://www.liquid.ai/",
    "Meta": "https://llama.meta.com/",
    "Microsoft": "https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/",
    "MiniMax": "https://www.minimaxi.com/en/platform",
    "Mistral": "https://mistral.ai/pricing/",
    "MosaicML": "https://www.databricks.com/product/machine-learning/foundation-models",
    "Moonshot AI": "https://platform.moonshot.ai/docs/pricing",
    "Naver": "https://clova.ai/en/hyperclova",
    "NVIDIA": "https://build.nvidia.com/",
    "Nous Research": "https://nousresearch.com/",
    "OpenChat": "https://github.com/imoneoi/openchat",
    "OpenAI": "https://openai.com/api/pricing/",
    "Perplexity": "https://docs.perplexity.ai/guides/pricing",
    "Reka": "https://www.reka.ai/",
    "Snowflake": "https://www.snowflake.com/en/product/features/arctic/",
    "Technology Innovation Institute": "https://falconllm.tii.ae/",
    "Tencent": "https://cloud.tencent.com/product/hunyuan",
    "Twelve Labs": "https://www.twelvelabs.io/",
    "Writer": "https://writer.com/product/palmyra/",
    "xAI": "https://docs.x.ai/docs/models",
    "Zhipu AI": "https://bigmodel.cn/pricing",
}

MODEL_COMPANY_PATTERNS = (
    ("Anthropic", ("claude",)),
    ("OpenAI", ("gpt-", "chatgpt", "davinci", "babbage", "curie", "codex")),
    ("Google", ("gemini", "gemma", "palm", "medlm")),
    ("Meta", ("llama", "codellama", "meta-llama")),
    ("Mistral", ("mistral", "mixtral", "codestral", "ministral", "magistral", "devstral", "pixtral")),
    ("DeepSeek", ("deepseek",)),
    ("Alibaba / Qwen", ("qwen", "qwq", "qvq")),
    ("xAI", ("grok",)),
    ("Cohere", ("command", "aya", "cohere")),
    ("Amazon", ("nova", "titan")),
    ("Microsoft", ("phi",)),
    ("Perplexity", ("sonar", "pplx")),
    ("Moonshot AI", ("kimi", "moonshot")),
    ("Zhipu AI", ("glm", "zhipu")),
    ("AI21", ("jamba", "jurassic", "j2-")),
    ("01.AI", ("yi-", "yi-large", "yi-lightning", "01-ai")),
    ("AI2", ("olmo", "allenai")),
    ("Aion Labs", ("aion",)),
    ("Arcee AI", ("arcee", "trinity", "virtuoso", "maestro", "spotlight")),
    ("ByteDance", ("bytedance", "byteplus", "seed-", "ui-tars")),
    ("Core42", ("jais",)),
    ("Databricks", ("dbrx", "databricks")),
    ("Hugging Face", ("zephyr", "huggingfaceh4")),
    ("Inflection", ("inflection",)),
    ("Liquid AI", ("lfm", "liquid")),
    ("MiniMax", ("minimax",)),
    ("MosaicML", ("mpt-", "mosaicml")),
    ("Naver", ("hyperclova",)),
    ("NVIDIA", ("nemotron", "nvidia")),
    ("Nous Research", ("nous", "hermes")),
    ("OpenChat", ("openchat",)),
    ("Reka", ("reka",)),
    ("Snowflake", ("snowflake", "arctic")),
    ("Technology Innovation Institute", ("falcon",)),
    ("Twelve Labs", ("pegasus", "twelvelabs")),
    ("Writer", ("palmyra",)),
    ("Tencent", ("hunyuan",)),
    ("Baidu", ("ernie",)),
    ("Cerebras", ("cerebras",)),
)


@dataclass(frozen=True)
class PriceObservation:
    snapshot_date: str
    source: str
    source_url: str
    source_updated_at: str | None
    provider: str | None
    model_id: str
    model_name: str | None
    token_type: str
    price_usd_per_1m: float
    raw_price: str
    raw_unit: str
    pricing_plan: str
    price_dimension: str
    mode: str | None = None
    context_length: int | None = None
    tokenizer: str | None = None
    notes: str | None = None

    @property
    def series_id(self) -> str:
        provider = self.provider or ""
        return (
            f"{self.source}|{provider}|{self.model_id}|"
            f"{self.price_dimension}|{self.token_type}"
        )


def today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def fetch_json(url: str, timeout: int = 45, retries: int = 2) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url}: {last_error}") from last_error


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _usd_per_token_to_per_1m(value: Any) -> float | None:
    parsed = _as_float(value)
    if parsed is None:
        return None
    return parsed * 1_000_000


def _usd_per_1m(value: Any) -> float | None:
    parsed = _as_float(value)
    if parsed is None:
        return None
    return parsed


def _cents_per_token_to_usd_per_1m(value: Any) -> float | None:
    parsed = _as_float(value)
    if parsed is None:
        return None
    return parsed * 10_000


def _provider_from_openrouter_model(model_id: str) -> str | None:
    if "/" not in model_id:
        return None
    return model_id.split("/", 1)[0]


def normalize_provider(provider: Any) -> str | None:
    if provider is None or (isinstance(provider, float) and np.isnan(provider)):
        return None
    normalized = str(provider).strip().lower().replace(" ", "-").replace("_", "-")
    normalized = normalized.removeprefix("~")
    return PROVIDER_ALIASES.get(normalized, normalized)


def canonicalize_model_id(source: Any, provider: Any, model_id: Any, model_name: Any = None) -> str:
    del source
    del model_name
    value = str(model_id or "").strip().lower()
    value = value.replace("@", "-")
    value = value.replace(":", "-")
    value = value.replace("_", "-")
    value = value.replace(".", ".")

    region_prefix = re.compile(r"^(global|eu|us-gov-east-1|us-gov-west-1|[a-z]{2,3}(?:-[a-z]+)+-[0-9]+)/")
    while region_prefix.match(value):
        value = region_prefix.sub("", value, count=1)
    value = re.sub(r"^(global|eu|us|apac|usgov)\.", "", value)

    prefixes = (
        "azure/",
        "openai/",
        "anthropic/",
        "google/",
        "mistral/",
        "cohere/",
        "x-ai/",
        "xai/",
        "bedrock/",
        "vertex-ai/",
        "vertex_ai/",
        "together-ai/",
        "together_ai/",
        "fireworks-ai/",
        "fireworks_ai/",
        "openrouter/",
        "vercel_ai_gateway/",
        "vercel-ai-gateway/",
        "replicate/",
        "deepinfra/",
        "groq/",
        "alibaba/",
        "baidu/",
        "deepseek-ai/",
        "meta-llama/",
        "microsoft/",
        "mistralai/",
        "moonshotai/",
        "qwen/",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if value.startswith(prefix):
                value = value[len(prefix) :]
                changed = True

    value = value.replace("accounts/fireworks/models/", "")
    for dot_prefix in (
        "amazon.",
        "anthropic.",
        "cohere.",
        "deepseek.",
        "meta.",
        "mistral.",
        "moonshotai.",
        "qwen.",
    ):
        if value.startswith(dot_prefix):
            value = value[len(dot_prefix) :]
    if provider:
        normalized_provider = normalize_provider(provider) or ""
        provider_prefix = normalized_provider + "/"
        if value.startswith(provider_prefix):
            value = value[len(provider_prefix) :]

    while "//" in value:
        value = value.replace("//", "/")
    return value.strip("/ ")


def infer_model_company(model_id: Any, model_name: Any = None, provider: Any = None) -> str:
    text = " ".join(
        str(value).lower()
        for value in (model_id, model_name, provider)
        if value is not None and not (isinstance(value, float) and np.isnan(value))
    )
    if re.search(r"(^|[/\s-])(o1|o3|o4)([-\s/]|$)", text):
        return "OpenAI"
    for company, patterns in MODEL_COMPANY_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return company
    normalized_provider = normalize_provider(provider)
    provider_company = {
        "anthropic": "Anthropic",
        "cohere": "Cohere",
        "deepseek": "DeepSeek",
        "google": "Google",
        "mistral": "Mistral",
        "openai": "OpenAI",
        "x-ai": "xAI",
    }
    return provider_company.get(normalized_provider or "", "Other / Unknown")


def infer_model_family(model_id: Any, model_name: Any = None) -> str:
    text = " ".join(
        str(value).lower()
        for value in (model_id, model_name)
        if value is not None and not (isinstance(value, float) and np.isnan(value))
    )

    if re.search(r"(^|[/\s-])(o1|o3|o4)([-\s/]|$)", text):
        return "OpenAI o-series"

    family_rules = (
        ("Claude 4", ("claude-opus-4", "claude-sonnet-4", "claude-haiku-4", "claude-4", "claude opus 4", "claude sonnet 4")),
        ("Claude 3.7", ("claude-3.7", "claude-3-7")),
        ("Claude 3.5", ("claude-3.5", "claude-3-5")),
        ("Claude 3", ("claude-3",)),
        ("GPT-5", ("gpt-5",)),
        ("GPT-4.1", ("gpt-4.1", "gpt-4-1")),
        ("GPT-4o", ("gpt-4o",)),
        ("GPT-4", ("gpt-4",)),
        ("GPT OSS", ("gpt-oss",)),
        ("Gemini 3", ("gemini-3",)),
        ("Gemini 2.5", ("gemini-2.5", "gemini-2-5")),
        ("Gemini 2", ("gemini-2",)),
        ("Gemini 1", ("gemini-1",)),
        ("Gemma", ("gemma",)),
        ("Llama 4", ("llama-4", "llama4")),
        ("Llama 3.3", ("llama-3.3", "llama-3-3", "llama3-3", "llama3.3")),
        ("Llama 3.2", ("llama-3.2", "llama-3-2", "llama3-2", "llama3.2")),
        ("Llama 3.1", ("llama-3.1", "llama-3-1", "llama3-1", "llama3.1")),
        ("Llama 3", ("llama-3", "llama3")),
        ("Llama 2", ("llama-2", "llama2")),
        ("Mistral Large", ("mistral-large",)),
        ("Mistral Small", ("mistral-small",)),
        ("Mistral Medium", ("mistral-medium",)),
        ("Ministral", ("ministral",)),
        ("Mistral Base/Instruct", ("mistral-7b", "mistral-8b", "mistral.mistral-7b", "mistral.mistral-8b")),
        ("Mixtral", ("mixtral",)),
        ("Codestral", ("codestral",)),
        ("Magistral", ("magistral",)),
        ("Devstral", ("devstral",)),
        ("Pixtral", ("pixtral",)),
        ("DeepSeek R1", ("deepseek-r1",)),
        ("DeepSeek V3", ("deepseek-v3",)),
        ("DeepSeek Chat", ("deepseek-chat",)),
        ("DeepSeek Coder", ("deepseek-coder",)),
        ("Qwen3", ("qwen3", "qwen-3")),
        ("Qwen2.5", ("qwen2.5", "qwen-2.5", "qwen2-5")),
        ("Qwen2", ("qwen2", "qwen-2")),
        ("QwQ / QVQ", ("qwq", "qvq")),
        ("Grok 4", ("grok-4",)),
        ("Grok 3", ("grok-3",)),
        ("Grok 2", ("grok-2",)),
        ("Grok", ("grok",)),
        ("Command", ("command",)),
        ("Jamba", ("jamba",)),
        ("Nova", ("nova",)),
        ("Titan", ("titan",)),
        ("01.AI Yi", ("yi-", "yi-large", "yi-lightning")),
        ("Aion", ("aion",)),
        ("Arcee", ("arcee", "trinity", "virtuoso", "maestro", "spotlight")),
        ("ByteDance Seed", ("seed-",)),
        ("DBRX", ("dbrx",)),
        ("Falcon", ("falcon",)),
        ("HyperCLOVA", ("hyperclova",)),
        ("Inflection", ("inflection",)),
        ("JAIS", ("jais",)),
        ("LFM", ("lfm",)),
        ("MiniMax", ("minimax",)),
        ("Nemotron", ("nemotron",)),
        ("Hermes", ("hermes",)),
        ("MPT", ("mpt-",)),
        ("OLMo", ("olmo",)),
        ("OpenChat", ("openchat",)),
        ("Palmyra", ("palmyra",)),
        ("Pegasus", ("pegasus",)),
        ("Phi", ("phi",)),
        ("Reka", ("reka",)),
        ("Snowflake Arctic", ("arctic",)),
        ("Sonar", ("sonar",)),
        ("UI-TARS", ("ui-tars",)),
        ("Zephyr", ("zephyr",)),
        ("Kimi", ("kimi",)),
        ("GLM", ("glm",)),
        ("Hunyuan", ("hunyuan",)),
        ("ERNIE", ("ernie",)),
    )
    for family, patterns in family_rules:
        if any(pattern in text for pattern in patterns):
            return family
    return "Other / Unknown"


def is_multimodal_text_output(row: pd.Series) -> bool:
    mode = str(row.get("mode") or "").lower()
    if "->" not in mode:
        return False
    input_modalities, output_modalities = mode.split("->", 1)
    return (
        any(modality in input_modalities.split("+") for modality in ("image", "audio", "video", "file"))
        and "text" in output_modalities.split("+")
        and not any(modality in output_modalities.split("+") for modality in ("image", "audio", "video"))
    )


def _model_text_for_classification(row: pd.Series) -> str:
    fields = [
        row.get("model_id"),
        row.get("model_name"),
        row.get("mode"),
        row.get("price_dimension"),
    ]
    return " ".join(str(value).lower() for value in fields if pd.notna(value))


def classify_core_model(row: pd.Series) -> tuple[bool, str | None]:
    text = _model_text_for_classification(row)
    mode = str(row.get("mode") or "").lower()

    for pattern in NON_CORE_MODEL_PATTERNS:
        if pattern in text:
            return False, f"excluded_pattern:{pattern}"

    if "->" in mode:
        output_modalities = mode.split("->", 1)[1].split("+")
        if "text" not in output_modalities:
            return False, "non_text_output_modality"
        if any(modality in output_modalities for modality in ("image", "audio", "video")):
            return False, "mixed_non_text_output_modality"

    litellm_mode = str(row.get("mode") or "").lower()
    if row.get("source") == "litellm" and litellm_mode not in {"chat", "completion", "responses", "nan", ""}:
        return False, f"non_generation_mode:{litellm_mode}"

    return True, None


def annotate_universe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    annotated = df.copy()
    annotated["canonical_provider"] = annotated["provider"].map(normalize_provider)
    annotated["canonical_model_id"] = annotated.apply(
        lambda row: canonicalize_model_id(
            row.get("source"),
            row.get("canonical_provider"),
            row.get("model_id"),
            row.get("model_name"),
        ),
        axis=1,
    )
    annotated["model_company"] = annotated.apply(
        lambda row: infer_model_company(
            row.get("canonical_model_id"), row.get("model_name"), row.get("canonical_provider")
        ),
        axis=1,
    )
    annotated["model_family"] = annotated.apply(
        lambda row: infer_model_family(row.get("canonical_model_id"), row.get("model_name")),
        axis=1,
    )
    annotated["official_pricing_url"] = annotated["model_company"].map(OFFICIAL_PRICING_URLS)
    annotated["is_multimodal_text_output"] = annotated.apply(is_multimodal_text_output, axis=1)
    classifications = annotated.apply(classify_core_model, axis=1)
    annotated["is_core_model"] = [item[0] for item in classifications]
    annotated["core_exclusion_reason"] = [item[1] for item in classifications]
    annotated["canonical_series_id"] = (
        annotated["canonical_provider"].fillna("")
        + "|"
        + annotated["canonical_model_id"].fillna("")
        + "|"
        + annotated["token_type"].fillna("")
    )
    return annotated


def _price_value(price_record: Any) -> Any:
    if isinstance(price_record, dict):
        return price_record.get("price")
    return price_record


def collect_openrouter(snapshot_date: str) -> list[PriceObservation]:
    payload = fetch_json(OPENROUTER_MODELS_URL)
    rows: list[PriceObservation] = []
    mapping = {
        "prompt": "input",
        "completion": "output",
        "internal_reasoning": "reasoning",
        "input_cache_read": "cache_read",
        "input_cache_write": "cache_write",
    }

    for model in payload.get("data", []):
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if not model_id:
            continue
        pricing = model.get("pricing") or {}
        if not isinstance(pricing, dict):
            continue

        architecture = model.get("architecture") or {}
        for field, token_type in mapping.items():
            if field not in pricing:
                continue
            raw_price = pricing.get(field)
            price = _usd_per_token_to_per_1m(raw_price)
            if price is None:
                continue
            rows.append(
                PriceObservation(
                    snapshot_date=snapshot_date,
                    source="openrouter",
                    source_url=OPENROUTER_MODELS_URL,
                    source_updated_at=None,
                    provider=_provider_from_openrouter_model(model_id),
                    model_id=str(model_id),
                    model_name=model.get("name"),
                    token_type=token_type,
                    price_usd_per_1m=price,
                    raw_price=str(raw_price),
                    raw_unit="usd_per_token",
                    pricing_plan="lowest_available",
                    price_dimension=field,
                    mode=architecture.get("modality"),
                    context_length=model.get("context_length"),
                    tokenizer=architecture.get("tokenizer"),
                    notes="OpenRouter model-level lowest available price.",
                )
            )
    return rows


def list_portkey_providers(all_providers: bool) -> list[str]:
    if not all_providers:
        return list(CORE_PORTKEY_PROVIDERS)

    payload = fetch_json(PORTKEY_PRICING_DIR_URL)
    providers = [
        item["name"].removesuffix(".json")
        for item in payload
        if isinstance(item, dict) and item.get("name", "").endswith(".json")
    ]
    return sorted(providers)


def _collect_portkey_price(
    *,
    snapshot_date: str,
    source_url: str,
    provider: str,
    model_id: str,
    pay_as_you_go: dict[str, Any],
    field: str,
    token_type: str,
    notes: str | None = None,
) -> PriceObservation | None:
    if field not in pay_as_you_go:
        return None
    raw = _price_value(pay_as_you_go.get(field))
    price = _cents_per_token_to_usd_per_1m(raw)
    if price is None:
        return None
    return PriceObservation(
        snapshot_date=snapshot_date,
        source="portkey",
        source_url=source_url,
        source_updated_at=None,
        provider=provider,
        model_id=model_id,
        model_name=None,
        token_type=token_type,
        price_usd_per_1m=price,
        raw_price=str(raw),
        raw_unit="cents_per_token",
        pricing_plan="pay_as_you_go",
        price_dimension=field,
        notes=notes,
    )


def collect_portkey(
    snapshot_date: str, providers: Iterable[str] | None = None, all_providers: bool = True
) -> list[PriceObservation]:
    provider_list = list(providers) if providers is not None else list_portkey_providers(all_providers)
    rows: list[PriceObservation] = []

    primary_fields = {
        "request_token": "input",
        "response_token": "output",
        "reasoning_token": "reasoning",
        "cache_read_input_token": "cache_read",
        "cache_write_input_token": "cache_write",
    }
    fallback_fields = (
        ("request_text_token", "input", "request_token"),
        ("response_text_token", "output", "response_token"),
        ("cache_read_text_input_token", "cache_read", "cache_read_input_token"),
        ("cached_text_input_token", "cache_read", "cache_read_input_token"),
    )

    for provider in provider_list:
        source_url = PORTKEY_PRICING_URL.format(provider=provider)
        payload = fetch_json(source_url)
        if not isinstance(payload, dict):
            continue
        for model_id, record in payload.items():
            if model_id == "default" or not isinstance(record, dict):
                continue
            pricing_config = record.get("pricing_config") or {}
            pay_as_you_go = pricing_config.get("pay_as_you_go") or {}
            if not isinstance(pay_as_you_go, dict):
                continue

            for field, token_type in primary_fields.items():
                observation = _collect_portkey_price(
                    snapshot_date=snapshot_date,
                    source_url=source_url,
                    provider=provider,
                    model_id=str(model_id),
                    pay_as_you_go=pay_as_you_go,
                    field=field,
                    token_type=token_type,
                )
                if observation is not None:
                    rows.append(observation)

            for field, token_type, preferred_field in fallback_fields:
                if preferred_field in pay_as_you_go:
                    continue
                observation = _collect_portkey_price(
                    snapshot_date=snapshot_date,
                    source_url=source_url,
                    provider=provider,
                    model_id=str(model_id),
                    pay_as_you_go=pay_as_you_go,
                    field=field,
                    token_type=token_type,
                    notes=f"Fallback text-token field; {preferred_field} absent.",
                )
                if observation is not None:
                    rows.append(observation)

            additional_units = pay_as_you_go.get("additional_units") or {}
            if isinstance(additional_units, dict) and "cache_write_1h" in additional_units:
                raw = _price_value(additional_units["cache_write_1h"])
                price = _cents_per_token_to_usd_per_1m(raw)
                if price is not None:
                    rows.append(
                        PriceObservation(
                            snapshot_date=snapshot_date,
                            source="portkey",
                            source_url=source_url,
                            source_updated_at=None,
                            provider=provider,
                            model_id=str(model_id),
                            model_name=None,
                            token_type="cache_write_1h",
                            price_usd_per_1m=price,
                            raw_price=str(raw),
                            raw_unit="cents_per_token",
                            pricing_plan="pay_as_you_go",
                            price_dimension="additional_units.cache_write_1h",
                        )
                    )
    return rows


def collect_litellm(snapshot_date: str) -> list[PriceObservation]:
    payload = fetch_json(LITELLM_PRICES_URL)
    rows: list[PriceObservation] = []
    allowed_modes = {"chat", "completion", "responses", None}
    field_mapping = {
        "input_cost_per_token": "input",
        "output_cost_per_token": "output",
        "output_cost_per_reasoning_token": "reasoning",
        "cache_read_input_token_cost": "cache_read",
        "input_cost_per_token_cache_hit": "cache_read",
        "cache_creation_input_token_cost": "cache_write",
        "cache_creation_input_token_cost_above_1hr": "cache_write_1h",
    }

    for model_id, record in payload.items():
        if model_id == "sample_spec" or not isinstance(record, dict):
            continue
        mode = record.get("mode")
        if mode not in allowed_modes:
            continue
        provider = record.get("litellm_provider")
        for field, token_type in field_mapping.items():
            if field not in record:
                continue
            price = _usd_per_token_to_per_1m(record.get(field))
            if price is None:
                continue
            rows.append(
                PriceObservation(
                    snapshot_date=snapshot_date,
                    source="litellm",
                    source_url=LITELLM_PRICES_URL,
                    source_updated_at=None,
                    provider=str(provider) if provider is not None else None,
                    model_id=str(model_id),
                    model_name=None,
                    token_type=token_type,
                    price_usd_per_1m=price,
                    raw_price=str(record.get(field)),
                    raw_unit="usd_per_token",
                    pricing_plan="standard",
                    price_dimension=field,
                    mode=str(mode) if mode is not None else None,
                    context_length=record.get("max_input_tokens") or record.get("max_tokens"),
                    tokenizer=None,
                    notes="LiteLLM chat/completion/responses modes only.",
                )
            )
    return rows


def collect_simon_prices(snapshot_date: str) -> list[PriceObservation]:
    payload = fetch_json(SIMON_CURRENT_PRICES_URL)
    rows: list[PriceObservation] = []
    mapping = {
        "input": "input",
        "output": "output",
        "input_cached": "cache_read",
    }
    source_updated_at = payload.get("updated_at")

    for record in payload.get("prices", []):
        if not isinstance(record, dict) or not record.get("id"):
            continue
        for field, token_type in mapping.items():
            if field not in record:
                continue
            price = _usd_per_1m(record.get(field))
            if price is None:
                continue
            rows.append(
                PriceObservation(
                    snapshot_date=snapshot_date,
                    source="simon_llm_prices",
                    source_url=SIMON_CURRENT_PRICES_URL,
                    source_updated_at=str(source_updated_at) if source_updated_at else None,
                    provider=record.get("vendor"),
                    model_id=str(record["id"]),
                    model_name=record.get("name"),
                    token_type=token_type,
                    price_usd_per_1m=price,
                    raw_price=str(record.get(field)),
                    raw_unit="usd_per_1m_tokens",
                    pricing_plan="standard",
                    price_dimension=field,
                )
            )
    return rows


def collect_simon_historical_intervals() -> pd.DataFrame:
    payload = fetch_json(SIMON_HISTORICAL_PRICES_URL)
    rows: list[dict[str, Any]] = []
    mapping = {
        "input": "input",
        "output": "output",
        "input_cached": "cache_read",
    }

    for record in payload.get("prices", []):
        if not isinstance(record, dict) or not record.get("id"):
            continue
        for field, token_type in mapping.items():
            if field not in record:
                continue
            price = _usd_per_1m(record.get(field))
            if price is None:
                continue
            from_date = record.get("from_date")
            to_date = record.get("to_date")
            rows.append(
                {
                    "source": "simon_llm_prices",
                    "source_url": SIMON_HISTORICAL_PRICES_URL,
                    "provider": record.get("vendor"),
                    "model_id": str(record["id"]),
                    "model_name": record.get("name"),
                    "token_type": token_type,
                    "price_usd_per_1m": price,
                    "raw_price": str(record.get(field)),
                    "raw_unit": "usd_per_1m_tokens",
                    "price_dimension": field,
                    "from_date": from_date,
                    "to_date": to_date,
                    "history_quality": "dated_interval" if from_date or to_date else "undated_current_interval",
                }
            )

    intervals = pd.DataFrame.from_records(rows)
    if intervals.empty:
        return intervals
    return annotate_universe(intervals).sort_values(
        ["provider", "model_id", "token_type", "from_date"],
        na_position="last",
    ).reset_index(drop=True)


def build_simon_historical_backfill(
    intervals: pd.DataFrame, current: pd.DataFrame, snapshot_date: str
) -> pd.DataFrame:
    if intervals.empty:
        return intervals.copy()

    dated = intervals[intervals["history_quality"] == "dated_interval"].copy()
    if not dated.empty:
        dated["snapshot_date"] = dated["from_date"].fillna(dated["to_date"])
        dated["source_updated_at"] = None
        dated["pricing_plan"] = "historical_interval"
        dated["mode"] = None
        dated["context_length"] = None
        dated["tokenizer"] = None
        dated["notes"] = "Simon historical interval backfill row."
        dated["is_positive_price"] = dated["price_usd_per_1m"] > 0
        dated["series_id"] = (
            dated["source"].fillna("")
            + "|"
            + dated["provider"].fillna("")
            + "|"
            + dated["model_id"].fillna("")
            + "|"
            + dated["price_dimension"].fillna("")
            + "|"
            + dated["token_type"].fillna("")
        )
    else:
        dated = pd.DataFrame()

    current_simon = current[current["source"] == "simon_llm_prices"].copy()
    if not current_simon.empty:
        current_simon["history_quality"] = "current_snapshot"
        current_simon["from_date"] = None
        current_simon["to_date"] = None
        current_simon["snapshot_date"] = snapshot_date

    backfill = pd.concat([dated, current_simon], ignore_index=True, sort=False)
    if backfill.empty:
        return backfill
    dedupe_cols = [
        "snapshot_date",
        "source",
        "provider",
        "model_id",
        "token_type",
        "price_dimension",
    ]
    return backfill.drop_duplicates(dedupe_cols, keep="last").sort_values(
        ["snapshot_date", "provider", "model_id", "token_type"],
        na_position="last",
    ).reset_index(drop=True)


def collect_prices(
    snapshot_date: str | None = None,
    sources: Iterable[str] | None = None,
    portkey_providers: Iterable[str] | None = None,
    portkey_all_providers: bool = True,
) -> pd.DataFrame:
    snapshot = snapshot_date or today_utc()
    selected_sources = set(sources or ("openrouter", "portkey", "litellm", "simon"))
    rows: list[PriceObservation] = []

    if "openrouter" in selected_sources:
        rows.extend(collect_openrouter(snapshot))
    if "portkey" in selected_sources:
        rows.extend(
            collect_portkey(
                snapshot,
                providers=portkey_providers,
                all_providers=portkey_all_providers,
            )
        )
    if "litellm" in selected_sources:
        rows.extend(collect_litellm(snapshot))
    if "simon" in selected_sources or "simon_llm_prices" in selected_sources:
        rows.extend(collect_simon_prices(snapshot))

    records = []
    for row in rows:
        record = asdict(row)
        record["series_id"] = row.series_id
        record["is_positive_price"] = row.price_usd_per_1m > 0
        records.append(record)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    dedupe_cols = [
        "snapshot_date",
        "source",
        "provider",
        "model_id",
        "token_type",
        "price_dimension",
    ]
    df = df.drop_duplicates(dedupe_cols, keep="last")
    df = annotate_universe(df)
    return df.sort_values(
        ["source", "provider", "model_id", "token_type", "price_dimension"],
        na_position="last",
    ).reset_index(drop=True)


def merge_history(current: pd.DataFrame, history_path: Path) -> pd.DataFrame:
    if current.empty:
        return current
    if history_path.exists():
        history = pd.read_csv(history_path)
        history = history[history["snapshot_date"] != current["snapshot_date"].iloc[0]]
        combined = pd.concat([history, current], ignore_index=True)
    else:
        combined = current.copy()
    dedupe_cols = [
        "snapshot_date",
        "source",
        "provider",
        "model_id",
        "token_type",
        "price_dimension",
    ]
    combined = combined.drop_duplicates(dedupe_cols, keep="last")
    return combined.sort_values(
        ["snapshot_date", "source", "provider", "model_id", "token_type"],
        na_position="last",
    ).reset_index(drop=True)


def build_core_price_table(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    history = annotate_universe(history)

    core = history[(history["is_core_model"] == True) & (history["price_usd_per_1m"] > 0)].copy()
    if core.empty:
        return core

    group_cols = ["snapshot_date", "canonical_provider", "canonical_model_id", "token_type"]
    grouped = core.groupby(group_cols, dropna=False)
    deduped = grouped.agg(
        price_usd_per_1m=("price_usd_per_1m", "median"),
        min_source_price_usd_per_1m=("price_usd_per_1m", "min"),
        max_source_price_usd_per_1m=("price_usd_per_1m", "max"),
        observations=("price_usd_per_1m", "count"),
        source_count=("source", "nunique"),
        source_model_count=("model_id", "nunique"),
        model_company=("model_company", lambda values: sorted(set(map(str, values)))[0]),
        model_family=("model_family", lambda values: sorted(set(map(str, values)))[0]),
        official_pricing_url=("official_pricing_url", lambda values: sorted(set(map(str, values.dropna())))[0] if values.dropna().any() else None),
        is_multimodal_text_output=("is_multimodal_text_output", "max"),
        sources=("source", lambda values: ",".join(sorted(set(map(str, values))))),
        source_model_ids=("model_id", lambda values: ",".join(sorted(set(map(str, values)))[:12])),
    ).reset_index()

    deduped["source"] = "cleaned_core"
    deduped["provider"] = deduped["canonical_provider"]
    deduped["model_id"] = deduped["canonical_model_id"]
    deduped["model_name"] = deduped["canonical_model_id"]
    deduped["price_dimension"] = "median_across_sources"
    deduped["pricing_plan"] = "core_cleaned"
    deduped["raw_price"] = deduped["price_usd_per_1m"].astype(str)
    deduped["raw_unit"] = "usd_per_1m_tokens"
    deduped["source_url"] = "multiple"
    deduped["source_updated_at"] = None
    deduped["mode"] = None
    deduped["context_length"] = None
    deduped["tokenizer"] = None
    deduped["notes"] = "Core model universe cleaned across sources by canonical provider/model/token type."
    deduped["is_positive_price"] = True
    deduped["is_core_model"] = True
    deduped["core_exclusion_reason"] = None
    deduped["series_id"] = (
        deduped["canonical_provider"].fillna("")
        + "|"
        + deduped["canonical_model_id"].fillna("")
        + "|"
        + deduped["token_type"].fillna("")
    )
    deduped["canonical_series_id"] = deduped["series_id"]
    deduped["source_price_spread_usd_per_1m"] = (
        deduped["max_source_price_usd_per_1m"] - deduped["min_source_price_usd_per_1m"]
    )

    return deduped.sort_values(
        ["snapshot_date", "canonical_provider", "canonical_model_id", "token_type"],
        na_position="last",
    ).reset_index(drop=True)


def build_model_taxonomy(core_current: pd.DataFrame) -> pd.DataFrame:
    if core_current.empty:
        return pd.DataFrame()

    grouped = core_current.groupby(["model_company", "model_family"], dropna=False)
    taxonomy = grouped.agg(
        canonical_models=("canonical_model_id", "nunique"),
        price_rows=("price_usd_per_1m", "count"),
        providers=("canonical_provider", "nunique"),
        token_types=("token_type", lambda values: ",".join(sorted(set(map(str, values))))),
        official_pricing_url=("official_pricing_url", lambda values: sorted(set(map(str, values.dropna())))[0] if len(values.dropna()) else None),
        example_models=("canonical_model_id", lambda values: ", ".join(sorted(set(map(str, values)))[:8])),
        median_price_usd_per_1m=("price_usd_per_1m", "median"),
    ).reset_index()

    token_medians = (
        core_current.pivot_table(
            index=["model_company", "model_family"],
            columns="token_type",
            values="price_usd_per_1m",
            aggfunc="median",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    taxonomy = taxonomy.merge(token_medians, on=["model_company", "model_family"], how="left")
    return taxonomy.sort_values(
        ["model_company", "canonical_models", "model_family"], ascending=[True, False, True]
    ).reset_index(drop=True)


def build_validation_report(core_current: pd.DataFrame) -> pd.DataFrame:
    if core_current.empty:
        return pd.DataFrame()
    validation = core_current.copy()
    validation["relative_source_spread"] = np.where(
        validation["price_usd_per_1m"] > 0,
        validation["source_price_spread_usd_per_1m"] / validation["price_usd_per_1m"],
        np.nan,
    )

    def status(row: pd.Series) -> str:
        if row["source_count"] <= 1:
            return "single_source"
        if row["source_price_spread_usd_per_1m"] == 0:
            return "multi_source_match"
        if row["relative_source_spread"] <= 0.05:
            return "multi_source_close"
        return "multi_source_spread"

    validation["validation_status"] = validation.apply(status, axis=1)
    columns = [
        "snapshot_date",
        "model_company",
        "model_family",
        "canonical_provider",
        "canonical_model_id",
        "token_type",
        "price_usd_per_1m",
        "source_count",
        "sources",
        "min_source_price_usd_per_1m",
        "max_source_price_usd_per_1m",
        "source_price_spread_usd_per_1m",
        "relative_source_spread",
        "validation_status",
        "official_pricing_url",
    ]
    return validation[columns].sort_values(
        ["source_count", "source_price_spread_usd_per_1m", "model_company"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def render_taxonomy_markdown(taxonomy: pd.DataFrame, snapshot_date: str) -> str:
    lines = [
        "# Token Price Index Model Taxonomy",
        "",
        f"Snapshot date: `{snapshot_date}`",
        "",
        "This taxonomy groups the cleaned core model universe by model developer/company and model family.",
        "",
    ]
    if taxonomy.empty:
        return "\n".join(lines + ["No taxonomy rows available.", ""])

    for company, rows in taxonomy.groupby("model_company", sort=True):
        lines.extend([f"## {company}", ""])
        lines.append("| Family | Canonical models | Providers | Token types | Examples |")
        lines.append("|---|---:|---:|---|---|")
        for row in rows.sort_values(["canonical_models", "model_family"], ascending=[False, True]).itertuples(index=False):
            examples = str(row.example_models).replace("|", "/")
            lines.append(
                f"| {row.model_family} | {int(row.canonical_models)} | {int(row.providers)} | "
                f"{row.token_types} | {examples} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_token_price_plots(
    core_current: pd.DataFrame,
    core_index: pd.DataFrame,
    taxonomy: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    if core_current.empty:
        return {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    plot_dir = output_dir.parent / "assets"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}

    token_order = ["input", "output", "reasoning", "cache_read", "cache_write", "cache_write_1h"]
    token_labels = {
        "input": "Input",
        "output": "Output",
        "reasoning": "Reasoning",
        "cache_read": "Cache read",
        "cache_write": "Cache write",
        "cache_write_1h": "Cache write 1h",
    }

    medians = core_current.groupby("token_type")["price_usd_per_1m"].median().reindex(token_order).dropna()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar([token_labels.get(idx, idx) for idx in medians.index], medians.values, color="#2f6f73")
    ax.set_ylabel("USD per 1M tokens")
    ax.set_title("Median Token Price by Token Type")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "median_token_prices.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths["median_token_prices"] = str(path)

    percentiles = (
        core_current.groupby("token_type")["price_usd_per_1m"]
        .quantile([0.10, 0.50, 0.90])
        .unstack()
        .reindex(token_order)
        .dropna(how="all")
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(percentiles.index))
    ax.scatter(x, percentiles[0.50], color="#2f6f73", label="p50", zorder=3)
    ax.vlines(x, percentiles[0.10], percentiles[0.90], color="#6f7f85", linewidth=4, alpha=0.7, label="p10-p90")
    ax.set_yscale("log")
    ax.set_xticks(x, [token_labels.get(idx, idx) for idx in percentiles.index], rotation=25)
    ax.set_ylabel("USD per 1M tokens, log scale")
    ax.set_title("Token Price Distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / "token_price_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths["token_price_distribution"] = str(path)

    if not taxonomy.empty:
        top = taxonomy.groupby("model_company")["canonical_models"].sum().sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.barh(top.index[::-1], top.values[::-1], color="#596f9d")
        ax.set_xlabel("Canonical core models")
        ax.set_title("Top Model Companies in Core Universe")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = plot_dir / "top_model_companies.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths["top_model_companies"] = str(path)

    return plot_paths


def compute_price_stats(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    positive = history[history["price_usd_per_1m"] > 0].copy()
    if positive.empty:
        return pd.DataFrame()

    grouped = positive.groupby(["snapshot_date", "token_type"], dropna=False)
    stats = grouped["price_usd_per_1m"].agg(
        observations="count",
        min_usd_per_1m="min",
        median_usd_per_1m="median",
        mean_usd_per_1m="mean",
        max_usd_per_1m="max",
    )
    quantiles = grouped["price_usd_per_1m"].quantile([0.10, 0.25, 0.75, 0.90]).unstack()
    quantiles = quantiles.rename(
        columns={
            0.10: "p10_usd_per_1m",
            0.25: "p25_usd_per_1m",
            0.75: "p75_usd_per_1m",
            0.90: "p90_usd_per_1m",
        }
    )
    counts = grouped.agg(
        models=("model_id", "nunique"),
        providers=("provider", "nunique"),
        sources=("source", "nunique"),
    )
    result = stats.join(quantiles).join(counts).reset_index()
    return result.sort_values(["snapshot_date", "token_type"]).reset_index(drop=True)


def compute_price_index(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    positive = history[history["price_usd_per_1m"] > 0].copy()
    if positive.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for token_type in TOKEN_TYPES:
        token_rows = positive[positive["token_type"] == token_type]
        if token_rows.empty:
            continue
        base_date = str(token_rows["snapshot_date"].min())
        base_rows = token_rows[token_rows["snapshot_date"] == base_date]
        base_prices = base_rows.set_index("series_id")["price_usd_per_1m"]

        for snapshot_date, date_rows in token_rows.groupby("snapshot_date"):
            current = date_rows.set_index("series_id")["price_usd_per_1m"]
            common = base_prices.index.intersection(current.index)
            if len(common) == 0:
                index_value = np.nan
            else:
                ratios = current.loc[common] / base_prices.loc[common]
                index_value = float(np.exp(np.log(ratios).mean()) * 100)
            records.append(
                {
                    "snapshot_date": snapshot_date,
                    "token_type": token_type,
                    "base_date": base_date,
                    "index_value": index_value,
                    "matched_series": int(len(common)),
                    "eligible_series": int(date_rows["series_id"].nunique()),
                }
            )

    index = pd.DataFrame.from_records(records)
    if index.empty:
        return index
    stats = compute_price_stats(history)
    if not stats.empty:
        index = index.merge(stats, on=["snapshot_date", "token_type"], how="left")
    return index.sort_values(["snapshot_date", "token_type"]).reset_index(drop=True)


def write_outputs(
    current: pd.DataFrame,
    history: pd.DataFrame,
    all_index: pd.DataFrame,
    core_current: pd.DataFrame,
    core_history: pd.DataFrame,
    core_index: pd.DataFrame,
    output_dir: Path,
    snapshot_date: str,
    simon_intervals: pd.DataFrame | None = None,
    simon_backfill: pd.DataFrame | None = None,
    simon_backfill_index: pd.DataFrame | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    current_path = output_dir / f"normalized_prices_{snapshot_date}.csv"
    latest_path = output_dir / "normalized_prices_latest.csv"
    history_path = output_dir / "normalized_prices_history.csv"
    core_current_path = output_dir / f"core_prices_{snapshot_date}.csv"
    core_latest_path = output_dir / "core_prices_latest.csv"
    core_history_path = output_dir / "core_prices_history.csv"
    index_path = output_dir / f"price_index_{snapshot_date}.csv"
    latest_index_path = output_dir / "price_index_latest.csv"
    all_index_path = output_dir / f"all_price_index_{snapshot_date}.csv"
    all_latest_index_path = output_dir / "all_price_index_latest.csv"
    summary_path = output_dir / f"summary_{snapshot_date}.json"
    latest_summary_path = output_dir / "summary_latest.json"

    current.to_csv(current_path, index=False)
    current.to_csv(latest_path, index=False)
    history.to_csv(history_path, index=False)
    core_current.to_csv(core_current_path, index=False)
    core_current.to_csv(core_latest_path, index=False)
    core_history.to_csv(core_history_path, index=False)
    core_index.to_csv(index_path, index=False)
    core_index.to_csv(latest_index_path, index=False)
    all_index.to_csv(all_index_path, index=False)
    all_index.to_csv(all_latest_index_path, index=False)

    taxonomy = build_model_taxonomy(core_current)
    validation = build_validation_report(core_current)
    plot_paths = write_token_price_plots(core_current, core_index, taxonomy, output_dir)
    taxonomy_path = output_dir / f"model_taxonomy_{snapshot_date}.csv"
    taxonomy_latest_path = output_dir / "model_taxonomy_latest.csv"
    taxonomy_md_path = output_dir / "model_taxonomy_latest.md"
    validation_path = output_dir / f"price_validation_{snapshot_date}.csv"
    validation_latest_path = output_dir / "price_validation_latest.csv"
    taxonomy.to_csv(taxonomy_path, index=False)
    taxonomy.to_csv(taxonomy_latest_path, index=False)
    taxonomy_md_path.write_text(render_taxonomy_markdown(taxonomy, snapshot_date), encoding="utf-8")
    validation.to_csv(validation_path, index=False)
    validation.to_csv(validation_latest_path, index=False)

    historical_files: dict[str, str] = {}
    if simon_intervals is not None and not simon_intervals.empty:
        simon_intervals_path = output_dir / "simon_historical_intervals.csv"
        simon_intervals.to_csv(simon_intervals_path, index=False)
        historical_files["simon_intervals"] = str(simon_intervals_path)
    if simon_backfill is not None and not simon_backfill.empty:
        simon_backfill_path = output_dir / "simon_historical_backfill.csv"
        simon_backfill.to_csv(simon_backfill_path, index=False)
        historical_files["simon_backfill"] = str(simon_backfill_path)
    if simon_backfill_index is not None and not simon_backfill_index.empty:
        simon_backfill_index_path = output_dir / "simon_historical_price_index.csv"
        simon_backfill_index.to_csv(simon_backfill_index_path, index=False)
        historical_files["simon_backfill_index"] = str(simon_backfill_index_path)

    summary = {
        "snapshot_date": snapshot_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_observations": int(len(current)),
        "history_observations": int(len(history)),
        "core_current_observations": int(len(core_current)),
        "core_history_observations": int(len(core_history)),
        "positive_current_observations": int((current["price_usd_per_1m"] > 0).sum())
        if not current.empty
        else 0,
        "headline_universe": "cleaned_core",
        "sources": sorted(current["source"].dropna().unique().tolist()) if not current.empty else [],
        "token_types": sorted(current["token_type"].dropna().unique().tolist())
        if not current.empty
        else [],
        "output_files": {
            "current": str(current_path),
            "latest": str(latest_path),
            "history": str(history_path),
            "core_current": str(core_current_path),
            "core_latest": str(core_latest_path),
            "core_history": str(core_history_path),
            "index": str(index_path),
            "latest_index": str(latest_index_path),
            "all_index": str(all_index_path),
            "all_latest_index": str(all_latest_index_path),
            "model_taxonomy": str(taxonomy_path),
            "model_taxonomy_latest": str(taxonomy_latest_path),
            "model_taxonomy_markdown": str(taxonomy_md_path),
            "price_validation": str(validation_path),
            "price_validation_latest": str(validation_latest_path),
            "plots": plot_paths,
            **historical_files,
        },
        "methodology": {
            "price_unit": "USD per 1 million tokens",
            "headline_index": "Geometric mean of cleaned core-universe price ratios vs token-type base date.",
            "all_index": "Diagnostic geometric index over all source-aware catalog rows.",
            "zero_prices": "Retained in normalized tables, excluded from index and descriptive stats.",
            "source_combining": "Core prices are grouped by canonical provider, canonical model id, and token type; median source price is used.",
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    latest_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_token_price_index(
    output_dir: Path,
    snapshot_date: str | None = None,
    sources: Iterable[str] | None = None,
    portkey_providers: Iterable[str] | None = None,
    portkey_all_providers: bool = True,
    historical_backfill: bool = False,
) -> dict[str, Any]:
    snapshot = snapshot_date or today_utc()
    current = collect_prices(
        snapshot_date=snapshot,
        sources=sources,
        portkey_providers=portkey_providers,
        portkey_all_providers=portkey_all_providers,
    )
    history = merge_history(current, output_dir / "normalized_prices_history.csv")
    all_index = compute_price_index(history)

    core_history = build_core_price_table(history)
    core_current = core_history[core_history["snapshot_date"] == snapshot].copy()
    core_index = compute_price_index(core_history)

    simon_intervals = None
    simon_backfill = None
    simon_backfill_index = None
    if historical_backfill:
        simon_intervals = collect_simon_historical_intervals()
        simon_backfill = build_simon_historical_backfill(simon_intervals, current, snapshot)
        if simon_backfill is not None and not simon_backfill.empty:
            simon_backfill_core = build_core_price_table(simon_backfill)
            simon_backfill_index = compute_price_index(simon_backfill_core)

    return write_outputs(
        current,
        history,
        all_index,
        core_current,
        core_history,
        core_index,
        output_dir,
        snapshot,
        simon_intervals=simon_intervals,
        simon_backfill=simon_backfill,
        simon_backfill_index=simon_backfill_index,
    )


def parse_snapshot_date(value: str | None) -> str | None:
    if value is None:
        return None
    return date.fromisoformat(value).isoformat()
