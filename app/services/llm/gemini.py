from __future__ import annotations

from typing import Any

from google import genai

from app.config import get_settings
from app.services.llm.base import LLMProvider, LLMResponse, LLMUsage


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key if api_key is not None else settings.gemini_api_key
        self._client = genai.Client(api_key=key) if key else None

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        if not self._client:
            raise RuntimeError("Gemini API key is not configured")

        settings = get_settings()
        use_model = model or settings.llm_model
        text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if tools:
            text += (
                "\n\n(Tools are defined but native Gemini tool binding is not used in this path; "
                "use JSON protocol in prompt.)"
            )

        response = self._client.models.generate_content(
            model=use_model,
            contents=text,
        )
        answer = getattr(response, "text", "") or ""
        usage = getattr(response, "usage_metadata", None)
        total = int(getattr(usage, "total_token_count", 0) or 0)
        out_t = int(getattr(usage, "candidates_token_count", 0) or 0)
        inp = max(0, total - out_t)

        from app.services.llm.pricing import estimate_cost_usd

        cost = estimate_cost_usd("gemini", use_model, inp, out_t)

        return LLMResponse(
            content=answer,
            tool_calls=[],
            usage=LLMUsage(input_tokens=inp, output_tokens=out_t, cost_usd=cost),
            provider="gemini",
            model=use_model,
        )
