from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.services.llm.anthropic import AnthropicProvider
from app.services.llm.base import LLMProvider, LLMResponse
from app.services.llm.gemini import GeminiProvider
from app.services.llm.openai import OpenAIProvider
logger = logging.getLogger(__name__)


def get_provider_by_name(name: str) -> LLMProvider:
    n = name.lower().strip()
    if n == "openai":
        return OpenAIProvider()
    if n == "anthropic":
        return AnthropicProvider()
    if n == "gemini":
        return GeminiProvider()
    raise ValueError(f"Unknown LLM provider: {name}")


class LLMFactory:
    """Select provider from settings or per-request overrides; optional fallback on errors."""

    def __init__(self) -> None:
        settings = get_settings()
        self._default_provider = settings.llm_provider.lower()
        self._fallback_provider = (settings.llm_fallback_provider or "").lower().strip() or None

    def resolve_provider(
        self,
        *,
        provider: str | None = None,
    ) -> LLMProvider:
        settings = get_settings()
        name = (provider or settings.llm_provider).lower().strip()
        return get_provider_by_name(name)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        settings = get_settings()
        prov_name = (provider or settings.llm_provider).lower().strip()
        resolved_model = model
        if resolved_model is None:
            if prov_name == "openai":
                resolved_model = "gpt-4o-mini"
            elif prov_name == "anthropic":
                resolved_model = "claude-3-5-haiku-20241022"
            else:
                resolved_model = settings.llm_model

        primary = self.resolve_provider(provider=provider)
        try:
            return primary.generate(messages, tools=tools, model=resolved_model, temperature=temperature)
        except Exception as exc:
            logger.warning("Primary LLM failed (%s): %s", primary.name, exc)
            if not self._fallback_provider or self._fallback_provider == primary.name:
                raise
            fb = get_provider_by_name(self._fallback_provider)
            settings = get_settings()
            fb_model = settings.llm_fallback_model or resolved_model
            return fb.generate(messages, tools=tools, model=fb_model, temperature=temperature)


__all__ = ["LLMFactory", "get_provider_by_name"]
