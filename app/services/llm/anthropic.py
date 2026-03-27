from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from app.config import get_settings
from app.services.llm.base import LLMProvider, LLMResponse, LLMUsage


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key if api_key is not None else settings.anthropic_api_key
        self._client = Anthropic(api_key=key) if key else None

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        if not self._client:
            raise RuntimeError("Anthropic API key is not configured")

        settings = get_settings()
        use_model = model or settings.llm_model
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        conv = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]

        kwargs: dict[str, Any] = {
            "model": use_model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": conv,
        }
        if system:
            kwargs["system"] = system

        if tools:
            anthropic_tools = []
            for t in tools:
                if t.get("type") == "function" and "function" in t:
                    fn = t["function"]
                    anthropic_tools.append(
                        {
                            "name": fn["name"],
                            "description": fn.get("description", ""),
                            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                        }
                    )
            kwargs["tools"] = anthropic_tools

        response = self._client.messages.create(**kwargs)

        inp = int(response.usage.input_tokens or 0)
        out_t = int(response.usage.output_tokens or 0)

        content = ""
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                content += getattr(block, "text", "") or ""
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": getattr(block, "id", ""),
                        "name": block.name,
                        "arguments": block.input if hasattr(block, "input") else {},
                    }
                )

        from app.services.llm.pricing import estimate_cost_usd

        cost = estimate_cost_usd("anthropic", use_model, inp, out_t)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=LLMUsage(input_tokens=inp, output_tokens=out_t, cost_usd=cost),
        )
