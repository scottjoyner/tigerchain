from __future__ import annotations

from langchain_core.language_models import BaseLanguageModel
from langchain_community.llms import Ollama, VLLMOpenAI
from langchain_openai import ChatOpenAI

from ..config import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_llm(settings: Settings) -> BaseLanguageModel:
    provider = settings.llm_provider.lower()
    logger.info("Using %s provider with model %s", provider, settings.llm_model)

    if provider == "ollama":
        return Ollama(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )
    if provider == "vllm":
        return VLLMOpenAI(
            openai_api_base=settings.vllm_api_base,
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key or "unused",
        )
    if provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
