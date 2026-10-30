import json

import httpx
import pytest
from google import genai
from google.genai import types

from app.services.rag.embedder import GeminiEmbedder


def test_sdk_serializes_scoped_embedding_dimensions_without_network():
    requests = []

    def handler(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"embeddings": [{"values": [0.5] * 1536}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = genai.Client(api_key="synthetic", http_options=types.HttpOptions(httpx_client=http_client))
        embedder = GeminiEmbedder.__new__(GeminiEmbedder)
        embedder.client = client
        embedder.model = "models/gemini-embedding-001"
        assert len(embedder.embed_texts(["synthetic document"])[0]) == 1536
        assert len(embedder.embed_query("synthetic question")) == 1536
        embedder.model = "models/embedding-001"
        assert len(embedder.embed_query("synthetic legacy")) == 1536

    assert len(requests) == 3
    assert all(path.endswith(":batchEmbedContents") for path, _ in requests)
    assert all(len(body["requests"]) == 1 for _, body in requests)
    assert [body["requests"][0].get("outputDimensionality") for _, body in requests] == [1536, 1536, None]


def test_embedding_constructor_sets_explicit_timeout_and_single_attempt(monkeypatch):
    from types import SimpleNamespace

    from app.services.rag import embedder as module

    options = []

    def client(**kwargs):
        options.append(kwargs.get("http_options"))
        return SimpleNamespace()

    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(
        gemini_api_key="synthetic", embedding_model="models/gemini-embedding-001"))
    monkeypatch.setattr(module.genai, "Client", client)
    module.GeminiEmbedder()
    assert options[0] is not None
    assert options[0].timeout == 20000
    assert options[0].retry_options.attempts == 1


@pytest.mark.parametrize("failure", [503, 429, "timeout"])
def test_real_sdk_does_not_repeat_failed_embedding_requests(monkeypatch, failure):
    from types import SimpleNamespace

    from app.services.rag import embedder as module

    calls = []

    def handler(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private provider detail", request=request)
        return httpx.Response(failure, json={"error": {"code": failure, "message": "private provider detail"}})

    original = genai.Client
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        def client(**kwargs):
            options = kwargs["http_options"].model_copy(update={"httpx_client": http_client})
            return original(api_key=kwargs["api_key"], http_options=options)

        monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(
            gemini_api_key="synthetic", embedding_model="models/gemini-embedding-001"))
        monkeypatch.setattr(module.genai, "Client", client)
        instance = module.GeminiEmbedder()
        with pytest.raises(module.EmbeddingUnavailable) as error:
            instance.embed_query("synthetic question")
        assert str(error.value) == "Embedding service unavailable"
        assert len(calls) == 1
