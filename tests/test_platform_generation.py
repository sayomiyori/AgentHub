import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from app.config import get_settings
from app.platform.config import PlatformSettings, get_platform_settings


def completion(content="Answer", finish="stop", usage=None):
    return {
        "id": "synthetic", "object": "chat.completion", "created": 1,
        "model": "llama-3.3-70b-versatile",
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": content}}],
        "usage": usage if usage is not None else {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def boundary(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("TELEGRAM_AI_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_AI_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("WEBHOOK_INTERNAL_URL", "http://webhook.test")
    for name, char in [("WEBHOOK_AGENT_INGRESS_KEY", "i"),
                       ("WEBHOOK_AGENT_SERVICE_KEY", "c"),
                       ("AGENT_WEBHOOK_REPLY_KEY", "r")]:
        monkeypatch.setenv(name, char * 32)
    get_settings.cache_clear()
    get_platform_settings.cache_clear()
    calls = []
    state = {"response": completion()}

    async def handler(request):
        calls.append(request)
        if "error" in state:
            raise state["error"]
        if state.get("blocked"):
            await asyncio.Event().wait()
        return httpx.Response(state.get("status", 200),
                              content=json.dumps(state["response"]).encode(),
                              headers={"Content-Type": "application/json"})

    original = httpx.AsyncClient

    class Client(original):
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False
            assert kwargs["follow_redirects"] is False
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    yield state, calls
    get_settings.cache_clear()
    get_platform_settings.cache_clear()


def generate():
    from app.platform.generation import generate_platform_answer
    return generate_platform_answer("Synthetic question", "llama-3.3-70b-versatile")


def test_only_groq_allowed():
    with pytest.raises(ValidationError):
        PlatformSettings(_env_file=None, provider="openai")


def test_sdk_options_bounded(boundary):
    _, calls = boundary
    result = generate()
    assert result.content == "Answer"
    assert result.provider == "groq"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.usage.cost_usd == pytest.approx(0.00000985)
    assert len(calls) == 1
    request = calls[0]
    assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
    body = json.loads(request.content)
    assert body["max_completion_tokens"] == 1024
    assert "tools" not in body and "tool_choice" not in body
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == "Synthetic question"
    assert request.extensions["timeout"]["read"] == 20


@pytest.mark.parametrize("kind", ["quota", "timeout", "server", "redirect"])
def test_no_fallback_on_timeout_or_quota(boundary, monkeypatch, kind):
    state, calls = boundary
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openai")
    if kind == "timeout":
        state["error"] = httpx.ReadTimeout("synthetic remote text")
    else:
        state["status"] = {"quota": 429, "server": 500, "redirect": 302}[kind]
        state["response"] = {"error": {"message": "synthetic remote text"}}
    with pytest.raises(Exception) as error:
        generate()
    assert str(error.value) == "Platform generation failed"
    assert len(calls) == 1


def test_total_deadline_cancels_one_request(boundary, monkeypatch):
    state, calls = boundary
    state["blocked"] = True
    original = asyncio.timeout

    def deadline(seconds):
        assert seconds == 20
        return original(0.5)

    monkeypatch.setattr(asyncio, "timeout", deadline)
    with pytest.raises(Exception, match="Platform generation failed"):
        generate()
    assert len(calls) == 1


@pytest.mark.parametrize("content,finish,usage", [
    ("", "stop", None), ("  ", "stop", None), ("partial", "length", None),
    ("text", "tool_calls", None),
    ("text", "stop", {"prompt_tokens": -1, "completion_tokens": 2}),
    ("text", "stop", {"prompt_tokens": float("inf"), "completion_tokens": 2}),
    ("text", "stop", {"prompt_tokens": 1.5, "completion_tokens": 2}),
    ("text", "stop", {"prompt_tokens": True, "completion_tokens": 2}),
    ("text", "stop", {"prompt_tokens": 1}),
])
def test_empty_nonfinite_or_truncated_output_rejected(boundary, content, finish, usage):
    from app.platform.generation import GenerationInvalidResponse
    state, calls = boundary
    state["response"] = completion(content, finish, usage)
    with pytest.raises(GenerationInvalidResponse):
        generate()
    assert len(calls) == 1


def test_answer_is_capped_before_storage(boundary):
    state, _ = boundary
    state["response"] = completion("x" * 5000)
    assert generate().content == "x" * 4095 + "…"


def test_missing_key_fails_before_http(boundary, monkeypatch):
    from app.platform.generation import GenerationConfigurationError
    _, calls = boundary
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    with pytest.raises(GenerationConfigurationError):
        generate()
    assert calls == []


@pytest.mark.parametrize("model,expected", [
    ("openai/gpt-oss-20b", 0.375), ("qwen/qwen3.8-27b", 4.8),
])
def test_groq_list_price_estimate_is_separate_from_free_billing(model, expected):
    from app.services.llm.pricing import estimate_cost_usd
    assert estimate_cost_usd("groq", model, 1_000_000, 1_000_000) == expected


def test_standalone_groq_factory_retains_optional_tools(monkeypatch):
    from app.services.llm.factory import LLMFactory
    monkeypatch.setenv("GROQ_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "")
    get_settings.cache_clear()
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        response = completion()
        response["choices"][0]["message"]["tool_calls"] = [{
            "id": "synthetic", "type": "function",
            "function": {"name": "calculator", "arguments": '{"x":1}'},
        }]
        response["choices"][0]["finish_reason"] = "tool_calls"
        return httpx.Response(200, json=response)

    original = httpx.Client

    class Client(original):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    from openai import OpenAI

    from app.services.llm import groq

    def sdk(**kwargs):
        return OpenAI(http_client=Client(), **kwargs)

    monkeypatch.setattr(groq, "OpenAI", sdk)
    try:
        result = LLMFactory().generate(
            [{"role": "user", "content": "Synthetic question"}], provider="groq",
            model="openai/gpt-oss-20b", tools=[{
                "type": "function", "function": {"name": "calculator"},
            }],
        )
        assert result.tool_calls[0]["arguments"] == {"x": 1}
        assert len(calls) == 1
        assert calls[0]["tools"][0]["function"]["name"] == "calculator"
        assert "max_completion_tokens" not in calls[0]
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_confirmed_permanent_rejection_has_static_kind(boundary, status):
    from app.platform.generation import GenerationRejectedError
    state, calls = boundary
    state["status"] = status
    state["response"] = {"error": {"message": "synthetic private text"}}
    with pytest.raises(GenerationRejectedError, match="Platform generation rejected"):
        generate()
    assert len(calls) == 1


@pytest.mark.parametrize("price", [float("inf"), float("nan"), -1.0])
def test_nonfinite_or_negative_estimate_is_not_a_result(boundary, monkeypatch, price):
    from app.platform.generation import GenerationInvalidResponse
    from app.services.llm.pricing import MODEL_PRICING
    monkeypatch.setitem(MODEL_PRICING["groq"], "llama-3.3-70b-versatile", (price, 0))
    with pytest.raises(GenerationInvalidResponse):
        generate()
    assert len(boundary[1]) == 1


@pytest.mark.parametrize("malformed", ["legacy_tool", "role", "surrogate", "nul", "huge_usage"])
def test_invalid_response_cannot_reach_storage(boundary, malformed):
    from app.platform.generation import GenerationInvalidResponse
    state, calls = boundary
    message = state["response"]["choices"][0]["message"]
    if malformed == "legacy_tool":
        message["function_call"] = {"name": "synthetic", "arguments": "{}"}
    elif malformed == "role":
        message["role"] = "user"
    elif malformed in {"surrogate", "nul"}:
        message["content"] = "\ud800" if malformed == "surrogate" else "a\x00b"
    else:
        state["response"]["usage"]["prompt_tokens"] = 2**31
    with pytest.raises(GenerationInvalidResponse):
        generate()
    assert len(calls) == 1
