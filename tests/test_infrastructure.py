"""Opt-in integration checks against disposable PostgreSQL and Redis."""
import os
from uuid import uuid4

import pytest
import redis
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

DATABASE_URL = os.getenv("AGENTHUB_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="Isolated database URL is required")


@pytest.fixture()
def real_db(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL or "")
    monkeypatch.setenv("REDIS_URL", os.environ["AGENTHUB_TEST_REDIS_URL"])
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "")
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "MCP_SERVERS"):
        monkeypatch.setenv(name, "")
    from app.config import get_settings
    get_settings.cache_clear()
    import app.models.chunk  # noqa: F401
    import app.models.conversation  # noqa: F401
    import app.models.document  # noqa: F401
    import app.models.llm_usage  # noqa: F401
    from app.db.session import Base
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    with engine.connect() as connection:
        transaction = connection.begin()
        with Session(connection, join_transaction_mode="create_savepoint") as db:
            yield db
        transaction.rollback()
    engine.dispose()
    get_settings.cache_clear()


@pytest.fixture()
def cache_client(monkeypatch):
    from app.cache import semantic_cache
    client = redis.from_url(os.environ["AGENTHUB_TEST_REDIS_URL"], decode_responses=True)
    prefix = f"verification:{uuid4()}:"
    monkeypatch.setattr(semantic_cache, "_client", client)
    monkeypatch.setattr(semantic_cache, "_cache_key", lambda bucket: prefix + bucket)
    yield client
    keys = list(client.scan_iter(prefix + "*"))
    if keys:
        client.delete(*keys)
    client.close()


def test_rag_http_cache_and_usage(real_db, cache_client, monkeypatch):
    from app.db.session import get_db
    from app.main import app
    from app.models.chunk import Chunk
    from app.models.document import ContentType, Document
    from app.models.llm_usage import LLMUsageRecord
    from app.services.llm.base import LLMResponse, LLMUsage
    from app.services.llm.gemini import GeminiProvider
    from app.services.rag.embedder import GeminiEmbedder
    calls = []

    def generate(self, *args, **kwargs):
        calls.append(1)
        return LLMResponse("Verified answer", usage=LLMUsage(10, 5, 0.0001), provider="gemini", model="test")

    monkeypatch.setattr(GeminiProvider, "generate", generate)
    monkeypatch.setattr(GeminiEmbedder, "embed_query", lambda self, text: [1.0] + [0.0] * 1535)
    document = Document(title="Evidence", content_type=ContentType.txt, file_path="unused")
    real_db.add(document)
    real_db.flush()
    real_db.add(Chunk(document_id=document.id, text="Verified knowledge", chunk_index=0,
                      embedding=[1.0] + [0.0] * 1535, meta={}))
    real_db.commit()
    app.dependency_overrides[get_db] = lambda: real_db
    try:
        client = TestClient(app)
        first = client.post("/api/v1/query", json={"question": "Evidence?", "use_agent": False})
        assert first.status_code == 200
        assert first.json()["answer"] == "Verified answer"
        assert first.json()["sources"][0]["document_title"] == "Evidence"
        second = client.post("/api/v1/query", json={"question": "Evidence?", "use_agent": False})
        assert second.status_code == 200
        assert len(calls) == 1
        assert second.json()["tokens_used"] == 0
        assert second.json()["cost_usd"] == 0
        rows = real_db.query(LLMUsageRecord).filter_by(conversation_id=first.json()["conversation_id"]).all()
        assert len(rows) == 1 and rows[0].total_tokens == 15 and rows[0].message_id is not None
        assert client.post("/api/v1/query", json={"question": "x", "top_k": 21}).status_code == 422
        assert client.get("/api/v1/documents/2147483647").status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_document_worker_duplicate_does_not_duplicate_chunks(real_db, monkeypatch, tmp_path):
    from app.models.chunk import Chunk
    from app.models.document import ContentType, Document, UploadStatus
    from app.workers import embed_worker
    path = tmp_path / "source.txt"
    path.write_text("Verified source", encoding="utf-8")
    document = Document(title="source", content_type=ContentType.txt, file_path=str(path))
    real_db.add(document)
    real_db.commit()
    monkeypatch.setattr(embed_worker, "SessionLocal", lambda: real_db)
    monkeypatch.setattr(real_db, "close", lambda: None)
    monkeypatch.setattr(embed_worker.GeminiEmbedder, "embed_texts", lambda self, texts: [[1.0] * 1536 for _ in texts])
    embed_worker.process_document.run(document.id)
    embed_worker.process_document.run(document.id)
    assert document.upload_status == UploadStatus.ready
    assert real_db.query(Chunk).filter_by(document_id=document.id).count() == document.chunk_count == 1


def test_worker_database_failure_marks_document_failed(real_db, monkeypatch, tmp_path):
    from app.models.document import ContentType, Document, UploadStatus
    from app.workers import embed_worker
    path = tmp_path / "source.txt"
    path.write_text("Verified source", encoding="utf-8")
    document = Document(title="source", content_type=ContentType.txt, file_path=str(path))
    real_db.add(document)
    real_db.commit()
    monkeypatch.setattr(embed_worker, "SessionLocal", lambda: real_db)
    monkeypatch.setattr(real_db, "close", lambda: None)
    monkeypatch.setattr(embed_worker.GeminiEmbedder, "embed_texts", lambda self, texts: [[1.0] for _ in texts])
    with pytest.raises(Exception):
        embed_worker.process_document.run(document.id)
    assert real_db.get(Document, document.id).upload_status == UploadStatus.failed


def test_failed_worker_cannot_overwrite_successful_concurrent_retry(real_db, monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, get_ident

    from app.models.chunk import Chunk
    from app.models.document import ContentType, Document, UploadStatus
    from app.workers import embed_worker

    engine = create_engine(DATABASE_URL)
    path = tmp_path / "retry-source.txt"
    path.write_text("Concurrent retry source", encoding="utf-8")
    with Session(engine) as db:
        document = Document(title="retry-source", content_type=ContentType.txt, file_path=str(path))
        db.add(document)
        db.commit()
        document_id = document.id
    rolled_back, retried = Event(), Event()
    first_thread = None

    class CoordinatedSession(Session):
        def rollback(self):
            super().rollback()
            if get_ident() == first_thread:
                rolled_back.set()
                assert retried.wait(timeout=10)

    def embed(self, texts):
        if get_ident() == first_thread:
            raise RuntimeError("temporary embedding failure")
        return [[1.0] * 1536 for _ in texts]

    def first_attempt():
        nonlocal first_thread
        first_thread = get_ident()
        with pytest.raises(RuntimeError, match="temporary embedding failure"):
            embed_worker.process_document.run(document_id)

    monkeypatch.setattr(embed_worker, "SessionLocal", lambda: CoordinatedSession(engine))
    monkeypatch.setattr(embed_worker.GeminiEmbedder, "embed_texts", embed)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(first_attempt)
            try:
                assert rolled_back.wait(timeout=10)
                embed_worker.process_document.run(document_id)
            finally:
                retried.set()
            first.result(timeout=10)
        with Session(engine) as db:
            assert db.get(Document, document_id).upload_status == UploadStatus.ready
            assert db.query(Chunk).filter_by(document_id=document_id).count() == 1
    finally:
        engine.dispose()


def test_provider_failure_marks_document_failed_without_chunks(real_db, monkeypatch, tmp_path):
    from app.models.chunk import Chunk
    from app.models.document import ContentType, Document, UploadStatus
    from app.services.rag.embedder import EmbeddingUnavailable
    from app.workers import embed_worker

    path = tmp_path / "source.txt"
    path.write_text("Synthetic test content", encoding="utf-8")
    document = Document(title="source", content_type=ContentType.txt, file_path=str(path))
    real_db.add(document)
    real_db.commit()
    monkeypatch.setattr(embed_worker, "SessionLocal", lambda: real_db)
    monkeypatch.setattr(real_db, "close", lambda: None)

    def fail(_self, _texts):
        raise EmbeddingUnavailable()

    monkeypatch.setattr(embed_worker.GeminiEmbedder, "embed_texts", fail)
    with pytest.raises(EmbeddingUnavailable):
        embed_worker.process_document.run(document.id)
    assert real_db.get(Document, document.id).upload_status == UploadStatus.failed
    assert real_db.query(Chunk).filter_by(document_id=document.id).count() == 0


def test_agent_embedding_failure_preserves_usage(real_db, monkeypatch):
    import json
    from unittest.mock import Mock

    from app.api.v1 import query
    from app.db.session import get_db
    from app.main import app
    from app.models.conversation import Message, MessageRole
    from app.models.llm_usage import LLMUsageRecord
    from app.services.llm.base import LLMResponse, LLMUsage
    from app.services.rag.embedder import EmbeddingUnavailable

    generation = Mock(return_value=LLMResponse(
        json.dumps({"type": "tool_call", "tool_name": "search_knowledge_base", "arguments": {"query": "x"}}),
        usage=LLMUsage(10, 5, 0.001), provider="synthetic", model="synthetic",
    ))
    monkeypatch.setattr(query.orchestrator.llm_factory, "generate", generation)
    monkeypatch.setattr(query.orchestrator.rag_pipeline.embedder, "embed_query",
                        Mock(side_effect=EmbeddingUnavailable()))
    app.dependency_overrides[get_db] = lambda: real_db
    try:
        result = TestClient(app).post("/api/v1/query", json={"question": "Synthetic question", "use_agent": True})
        assert result.status_code == 200
        payload = result.json()
        assert payload["answer"] == "Knowledge base temporarily unavailable."
        assert payload["sources"] == [] and payload["tokens_used"] == 15
        assert generation.call_count == 1
        usage = real_db.query(LLMUsageRecord).filter_by(request_id=payload["request_id"]).one()
        message = real_db.query(Message).filter_by(id=usage.message_id).one()
        assert usage.total_tokens == 15
        assert message.role == MessageRole.assistant and message.content == payload["answer"]
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_tenant_without_embedding_provider_fails_without_chunks(real_db, monkeypatch, tmp_path):
    from app.models.chunk import Chunk
    from app.models.document import ContentType, Document, UploadStatus
    from app.services.rag.embedder import EmbeddingUnavailable
    from app.workers import embed_worker

    path = tmp_path / "tenant-source.txt"
    path.write_text("Synthetic tenant content", encoding="utf-8")
    document = Document(title="source", content_type=ContentType.txt, file_path=str(path), tenant_id=uuid4())
    real_db.add(document)
    real_db.commit()
    monkeypatch.setattr(embed_worker, "SessionLocal", lambda: real_db)
    monkeypatch.setattr(real_db, "close", lambda: None)
    with pytest.raises(EmbeddingUnavailable):
        embed_worker.process_document.run(document.id)
    assert real_db.get(Document, document.id).upload_status == UploadStatus.failed
    assert real_db.query(Chunk).filter_by(document_id=document.id).count() == 0


def test_small_query_vector_remains_nonzero_in_real_pgvector(real_db):
    from app.models.chunk import Chunk
    from app.models.document import ContentType, Document, UploadStatus
    from app.services.rag.retriever import PgVectorRetriever

    tenant = uuid4()
    document = Document(title="synthetic", content_type=ContentType.txt, file_path="unused", tenant_id=tenant,
                        upload_status=UploadStatus.ready, chunk_count=1)
    real_db.add(document)
    real_db.flush()
    real_db.add(Chunk(document_id=document.id, text="synthetic", chunk_index=0,
                      embedding=[1e-9] * 1536, meta={"embedding": {
                          "provider": "gemini", "model": "synthetic", "dimensions": 1536}}))
    real_db.commit()
    result = PgVectorRetriever().retrieve(real_db, [1e-9] * 1536, tenant_id=tenant, embedding_model="synthetic")
    assert len(result) == 1
    assert result[0]["score"] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("provider_mode", ["offline", "gemini"])
def test_worker_records_source_and_embedding_provenance(real_db, monkeypatch, tmp_path, provider_mode):
    import hashlib

    from app.models.chunk import Chunk
    from app.models.document import ContentType, Document, UploadStatus
    from app.workers import embed_worker

    if provider_mode == "gemini":
        from types import SimpleNamespace

        from app.services.rag.embedder import GeminiEmbedder

        instance = GeminiEmbedder.__new__(GeminiEmbedder)
        instance.model = "models/gemini-embedding-001"
        instance.client = SimpleNamespace(models=SimpleNamespace(embed_content=lambda **_: SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.5] * 1536)])))
        monkeypatch.setattr(embed_worker, "GeminiEmbedder", lambda: instance)

    source = b"Synthetic source\r\nsecond line"
    path = tmp_path / "source.txt"
    path.write_bytes(source)
    document = Document(title="synthetic", content_type=ContentType.txt, file_path=str(path))
    real_db.add(document)
    real_db.commit()
    monkeypatch.setattr(embed_worker, "SessionLocal", lambda: real_db)
    monkeypatch.setattr(real_db, "close", lambda: None)
    embed_worker.process_document.run(document.id)
    row = real_db.query(Chunk).filter_by(document_id=document.id).one()
    assert document.upload_status == UploadStatus.ready
    assert row.meta["source_sha256"] == hashlib.sha256(source).hexdigest()
    expected = {"provider": "offline_sha256", "model": "sha256-v1", "dimensions": 1536}
    if provider_mode == "gemini":
        expected = {"provider": "gemini", "model": "models/gemini-embedding-001", "dimensions": 1536}
    assert row.meta["embedding"] == expected
    embed_worker.process_document.run(document.id)
    assert real_db.query(Chunk).filter_by(document_id=document.id).count() == 1


@pytest.mark.parametrize("change", ["replace", "remove"])
def test_source_change_during_indexing_prevents_ready_commit(real_db, monkeypatch, tmp_path, change):
    from app.models.chunk import Chunk
    from app.models.document import ContentType, Document, UploadStatus
    from app.workers import embed_worker

    path = tmp_path / "changing-source.txt"
    path.write_text("Synthetic original", encoding="utf-8")
    document = Document(title="source", content_type=ContentType.txt, file_path=str(path))
    real_db.add(document)
    real_db.commit()
    original = Chunk(document_id=document.id, text="Previously stored synthetic content",
                     chunk_index=0, embedding=[0.5] * 1536, meta={"synthetic": "original"})
    real_db.add(original)
    real_db.commit()
    monkeypatch.setattr(embed_worker, "SessionLocal", lambda: real_db)
    monkeypatch.setattr(real_db, "close", lambda: None)

    def embed(_self, texts):
        if change == "replace":
            path.write_text("Synthetic replacement", encoding="utf-8")
        else:
            path.unlink()
        return [[0.5] * 1536 for _ in texts]

    monkeypatch.setattr(embed_worker.GeminiEmbedder, "embed_texts", embed)
    message = ("Document source changed during indexing" if change == "replace"
               else "Document source unavailable after indexing")
    with pytest.raises(ValueError, match=message):
        embed_worker.process_document.run(document.id)
    assert real_db.get(Document, document.id).upload_status == UploadStatus.failed
    retained = real_db.query(Chunk).filter_by(document_id=document.id).one()
    assert retained.id == original.id and retained.meta == {"synthetic": "original"}
    assert retained.text == "Previously stored synthetic content"


@pytest.mark.parametrize("within_limit", [True, False])
def test_document_chunk_budget_precedes_provider_calls(real_db, monkeypatch, tmp_path, within_limit):
    from unittest.mock import Mock

    from app.models.chunk import Chunk
    from app.models.document import ContentType, Document, UploadStatus
    from app.services.rag.chunker import TextChunker
    from app.workers import embed_worker

    content = "synthetic content " * 100
    count = len(TextChunker(chunk_size=500, chunk_overlap=50).split(content))
    assert count > 1
    monkeypatch.setattr(embed_worker.settings, "max_document_chunks", count if within_limit else count - 1)
    path = tmp_path / "budget-source.txt"
    path.write_text(content, encoding="utf-8")
    document = Document(title="source", content_type=ContentType.txt, file_path=str(path))
    real_db.add(document)
    real_db.commit()
    monkeypatch.setattr(embed_worker, "SessionLocal", lambda: real_db)
    monkeypatch.setattr(real_db, "close", lambda: None)
    provider = Mock(side_effect=lambda texts: [[0.5] * 1536 for _ in texts])
    monkeypatch.setattr(embed_worker.GeminiEmbedder, "embed_texts", lambda self, texts: provider(texts))
    if within_limit:
        embed_worker.process_document.run(document.id)
        assert document.upload_status == UploadStatus.ready
        assert real_db.query(Chunk).filter_by(document_id=document.id).count() == count
        assert sum(len(call.args[0]) for call in provider.call_args_list) == count
    else:
        with pytest.raises(ValueError, match="Document exceeds embedding chunk limit"):
            embed_worker.process_document.run(document.id)
        assert document.upload_status == UploadStatus.failed
        provider.assert_not_called()
        assert real_db.query(Chunk).filter_by(document_id=document.id).count() == 0
