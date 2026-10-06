from __future__ import annotations

import asyncio
import json
import math
from typing import Any, cast

import httpx
from openai import AsyncOpenAI, OpenAI

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


class GroqCompletionError(ValueError):
    pass


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        settings = get_settings()
        key = api_key if api_key is not None else settings.groq_api_key
        url = base_url or "https://api.groq.com/openai/v1"
        self._key = key
        self._url = url

    async def _bounded_completion(
        self, kwargs: dict[str, Any], timeout: float, max_retries: int
    ) -> Any:
        # HTTP phase timeouts alone do not bound a slowly streaming response.
        async with httpx.AsyncClient(
            timeout=timeout, trust_env=False, follow_redirects=False
        ) as transport:
            async with AsyncOpenAI(
                # SDK 3.x accepts HTTPX at runtime but annotates only HTTPX2.
                api_key=self._key, base_url=self._url, http_client=cast(Any, transport),
                timeout=timeout, max_retries=max_retries,
            ) as client:
                async with asyncio.timeout(timeout):
                    return await client.chat.completions.create(**kwargs)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_completion_tokens: int | None = None,
        require_complete: bool = False,
    ) -> LLMResponse:
        if not self._key:
            raise RuntimeError("Groq API key is not configured")

        settings = get_settings()
        use_model = model or settings.llm_model
        if not use_model:
            use_model = "openai/gpt-oss-20b"

        kwargs: dict[str, Any] = {
            "model": use_model,
            "temperature": temperature,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        if timeout is not None:
            response = asyncio.run(self._bounded_completion(
                kwargs, timeout, max_retries if max_retries is not None else 2
            ))
        else:
            options: dict[str, Any] = {}
            if max_retries is not None:
                options["max_retries"] = max_retries
            with OpenAI(api_key=self._key, base_url=self._url, **options) as client:
                response = client.chat.completions.create(**kwargs)
        usage = response.usage
        if require_complete:
            if (
                len(response.choices) != 1
                or response.choices[0].finish_reason != "stop"
                or not usage
                or any(type(value) is not int or not 0 <= value <= 2**31 - 1 for value in (
                    usage.prompt_tokens, usage.completion_tokens
                ))
                or (max_completion_tokens is not None
                    and usage.completion_tokens > max_completion_tokens)
            ):
                raise GroqCompletionError("Invalid Groq completion")
        inp = int(usage.prompt_tokens or 0) if usage else 0
        out_t = int(usage.completion_tokens or 0) if usage else 0
        msg = response.choices[0].message
        content = msg.content or ""
        tcalls = _extract_tool_calls(msg)
        if require_complete and (
            not isinstance(msg.content, str) or not content.strip()
            or msg.role != "assistant" or msg.tool_calls
            or getattr(msg, "function_call", None) or getattr(msg, "refusal", None)
            or "\x00" in content
        ):
            raise GroqCompletionError("Invalid Groq completion")
        if require_complete:
            try:
                content.encode("utf-8")
            except UnicodeEncodeError:
                raise GroqCompletionError("Invalid Groq completion") from None

        from app.services.llm.pricing import estimate_cost_usd

        cost = estimate_cost_usd("groq", use_model, inp, out_t)
        if require_complete and (not math.isfinite(cost) or cost < 0):
            raise GroqCompletionError("Invalid Groq usage estimate")

        return LLMResponse(
            content=content,
            tool_calls=tcalls,
            usage=LLMUsage(input_tokens=inp, output_tokens=out_t, cost_usd=cost),
            provider="groq",
            model=use_model,
        )
