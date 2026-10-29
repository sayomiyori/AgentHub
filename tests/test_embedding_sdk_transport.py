import json

import httpx
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
