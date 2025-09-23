from __future__ import annotations

from typing import Any, Mapping

from langchain_core.language_models import BaseLanguageModel
from langchain_community.llms import Ollama, VLLMOpenAI
from langchain_openai import ChatOpenAI

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_llm(settings: Settings) -> BaseLanguageModel:
    config = {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "api_key": settings.openai_api_key,
        "api_base": settings.openai_api_base,
        "base_url": settings.ollama_base_url,
        "vllm_api_base": settings.vllm_api_base,
    }
    return create_llm_from_config(settings, config)


def create_llm_from_config(settings: Settings, config: Mapping[str, Any]) -> BaseLanguageModel:
    provider = str(config.get("provider", settings.llm_provider)).lower()
    model_name = str(config.get("model", settings.llm_model))
    temperature = float(config.get("temperature", settings.llm_temperature))
    logger.info("Initialising %s provider with model %s", provider, model_name)

    if provider == "ollama":
        base_url = str(config.get("base_url") or settings.ollama_base_url)
        return Ollama(base_url=base_url, model=model_name, temperature=temperature)
    if provider == "vllm":
        api_base = str(config.get("api_base") or config.get("vllm_api_base") or settings.vllm_api_base)
        api_key = str(config.get("api_key") or settings.openai_api_key or "unused")
        return VLLMOpenAI(
            openai_api_base=api_base,
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
        )
    if provider == "openai":
        api_key = config.get("api_key") or settings.openai_api_key
        api_base = config.get("api_base") or settings.openai_api_base
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=api_base,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
