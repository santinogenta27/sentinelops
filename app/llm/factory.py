"""LLM client factory. ``mock`` returns ``None`` → nodes use the deterministic engine only."""

from __future__ import annotations

import logging

from app.config import Settings
from app.llm.base import LLMClient
from app.llm.providers import GeminiClient, GroqClient

logger = logging.getLogger("sentinelops.llm")


def build_llm(settings: Settings) -> LLMClient | None:
    if settings.llm_provider == "gemini":
        key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
        if not key:
            logger.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is empty; using deterministic engine")
            return None
        return GeminiClient(
            api_key=key,
            model=settings.gemini_model,
            rpm=settings.llm_rpm,
            max_retries=settings.llm_max_retries,
            timeout_s=settings.llm_timeout_s,
        )
    if settings.llm_provider == "groq":
        key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
        if not key:
            logger.warning("LLM_PROVIDER=groq but GROQ_API_KEY is empty; using deterministic engine")
            return None
        return GroqClient(
            api_key=key,
            model=settings.groq_model,
            rpm=settings.llm_rpm,
            max_retries=settings.llm_max_retries,
            timeout_s=settings.llm_timeout_s,
        )
    return None
