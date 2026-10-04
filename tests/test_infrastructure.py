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
