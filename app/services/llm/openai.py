from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.config import get_settings
from app.services.llm.base import LLMProvider, LLMResponse, LLMUsage


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    raw = getattr(message, "tool_calls", None) or []
    out: list[dict[str, Any]] = []
    for tc in raw:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        args = getattr(fn, "arguments", "") or "{}"
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError:
            parsed = {}
        out.append({"id": getattr(tc, "id", ""), "name": fn.name, "arguments": parsed})
    return out


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key if api_key is not None else settings.openai_api_key
        self._client = OpenAI(api_key=key) if key else None

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        if not self._client:
            raise RuntimeError("OpenAI API key is not configured")

        settings = get_settings()
        use_model = model or settings.llm_model
        kwargs: dict[str, Any] = {
            "model": use_model,
            "temperature": temperature,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        usage = response.usage
        inp = int(usage.prompt_tokens or 0) if usage else 0
        out_t = int(usage.completion_tokens or 0) if usage else 0
        msg = response.choices[0].message
        content = msg.content or ""
        tcalls = _extract_tool_calls(msg)

        from app.services.llm.pricing import estimate_cost_usd

        cost = estimate_cost_usd("openai", use_model, inp, out_t)

        return LLMResponse(
            content=content,
            tool_calls=tcalls,
            usage=LLMUsage(input_tokens=inp, output_tokens=out_t, cost_usd=cost),
        )
