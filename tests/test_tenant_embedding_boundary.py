from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.services.agent.tools.knowledge_base import KnowledgeBaseTool
from app.services.rag.embedder import EmbeddingUnavailable, GeminiEmbedder
from app.services.rag_pipeline import RAGPipeline


def test_tenant_direct_search_without_provider_stops_before_embedding_and_database(monkeypatch):
    embedder = GeminiEmbedder.__new__(GeminiEmbedder)
    embedder.client = None
    embedder.model = "synthetic"
    fallback = Mock(side_effect=AssertionError("fallback used"))
    monkeypatch.setattr(embedder, "_fallback_embedding", fallback)
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.embedder = embedder
    db = Mock()
    with pytest.raises(EmbeddingUnavailable):
        pipeline.ask(db, "synthetic", tenant_id=uuid4())
    fallback.assert_not_called()
    db.query.assert_not_called()


def test_tenant_agent_search_without_provider_stops_before_embedding_and_retrieval(monkeypatch):
    embedder = GeminiEmbedder.__new__(GeminiEmbedder)
    embedder.client = None
    embedder.model = "synthetic"
    fallback = Mock(side_effect=AssertionError("fallback used"))
    monkeypatch.setattr(embedder, "_fallback_embedding", fallback)
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.embedder = embedder
    pipeline.retriever = Mock()
    tool = KnowledgeBaseTool(Mock(), pipeline, uuid4())
    with pytest.raises(EmbeddingUnavailable):
        tool.run(query="synthetic")
    fallback.assert_not_called()
    pipeline.retriever.retrieve.assert_not_called()


def test_retrieval_preserves_small_nonzero_vector_values():
    import json

    from app.services.rag.retriever import PgVectorRetriever

    db = Mock()
    db.execute.return_value.mappings.return_value.all.side_effect = [[{"chunk_id": 1}]]
    PgVectorRetriever().retrieve(db, [1e-9] * 1536, tenant_id=uuid4())
    vector = json.loads(db.execute.call_args.args[1]["query_embedding"])
    assert all(value != 0 for value in vector)
