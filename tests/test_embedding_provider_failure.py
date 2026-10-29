from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.services.rag.embedder import GeminiEmbedder


def embedder(provider):
    instance = GeminiEmbedder.__new__(GeminiEmbedder)
    instance.model = "synthetic"
    instance.client = SimpleNamespace(models=provider) if provider else None
    return instance


def response(values):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])


def test_no_key_preserves_offline_fallback():
    instance = embedder(None)
    assert instance.embed_texts(["sample"]) == [instance._fallback_embedding("sample")]
    assert instance.embed_query("sample") == instance._fallback_embedding("sample")


def test_provider_success_returns_values():
    vector = [0.5] * 1536
    instance = embedder(Mock(embed_content=Mock(return_value=response(vector))))
    assert instance.embed_texts(["sample"]) == [vector]
    assert instance.embed_query("sample") == vector


@pytest.mark.parametrize("value", [None, response([]), response([1.0]), response([float("nan")] * 1536),
                                   response([float("inf")] * 1536), response([0.0] * 1536),
                                   response([True] * 1536), response([1e100] * 1536),
                                   response([1e-50] * 1536), response([1e-23] * 1536),
                                   response([1e20] * 1536)])
def test_configured_provider_invalid_response_is_not_fallback(value):
    instance = embedder(Mock(embed_content=Mock(return_value=value)))
    with pytest.raises(RuntimeError, match="Embedding service unavailable"):
        instance.embed_texts(["sample"])
    with pytest.raises(RuntimeError, match="Embedding service unavailable"):
        instance.embed_query("sample")


def test_batch_failure_never_returns_mixed_vectors_or_provider_details():
    provider = Mock(embed_content=Mock(side_effect=[response([0.5] * 1536), ValueError("PRIVATE sentinel")]))
    with pytest.raises(RuntimeError, match="Embedding service unavailable") as error:
        embedder(provider).embed_texts(["one", "two"])
    assert "PRIVATE sentinel" not in str(error.value)
    assert error.value.__cause__ is None
    assert provider.embed_content.call_count == 2


def test_agent_knowledge_failure_does_not_generate_an_answer(monkeypatch):
    import json

    from app.services.agent.orchestrator import AgentOrchestrator
    from app.services.llm.base import LLMResponse, LLMUsage
    from app.services.rag.embedder import EmbeddingUnavailable

    agent = AgentOrchestrator.__new__(AgentOrchestrator)
    agent.llm_factory = Mock()
    agent.llm_factory.generate.return_value = LLMResponse(
        json.dumps({"type": "tool_call", "tool_name": "search_knowledge_base", "arguments": {"query": "x"}}),
        usage=LLMUsage(0, 0, 0), provider="synthetic", model="synthetic",
    )
    tool = Mock()
    tool.run.side_effect = EmbeddingUnavailable()
    monkeypatch.setattr(agent, "_build_tools", lambda *_: {"search_knowledge_base": tool})
    monkeypatch.setattr(agent, "_load_history", lambda *_: [])
    result = agent.run(db=Mock(), question="x", conversation_id=1)
    assert result["answer"] == "Knowledge base temporarily unavailable."
    assert result["sources"] == []
    assert result["tools_called"][0]["result"] == {"error": "Knowledge base temporarily unavailable"}
    assert agent.llm_factory.generate.call_count == 1


@pytest.mark.parametrize("use_agent", [False, True])
def test_query_reports_fixed_503_without_provider_details(monkeypatch, use_agent):
    from fastapi.testclient import TestClient

    from app.api.v1 import query
    from app.db.session import get_db
    from app.main import app
    from app.services.rag.embedder import EmbeddingUnavailable

    db = Mock()
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(query, "_get_or_create_conversation", lambda *_: SimpleNamespace(id=1))
    target = query.orchestrator if use_agent else query.pipeline
    method = "run" if use_agent else "ask"
    monkeypatch.setattr(target, method, Mock(side_effect=EmbeddingUnavailable()))
    try:
        result = TestClient(app).post("/api/v1/query", json={"question": "sample", "use_agent": use_agent})
        assert result.status_code == 503
        assert result.json() == {"detail": "Embedding service unavailable"}
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_modern_gemini_requests_database_dimensions_and_legacy_does_not():
    vector = [0.5] * 1536
    provider = Mock(embed_content=Mock(return_value=response(vector)))
    instance = embedder(provider)
    instance.model = "models/gemini-embedding-001"
    instance.embed_texts(["document"])
    instance.embed_query("query")
    for call in provider.embed_content.call_args_list:
        assert call.kwargs["config"].output_dimensionality == 1536
    provider.reset_mock()
    instance.model = "models/embedding-001"
    instance.embed_query("query")
    assert "config" not in provider.embed_content.call_args.kwargs
