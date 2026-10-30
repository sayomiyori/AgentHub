"""Real PostgreSQL isolation; issuer and provider traffic stay controlled."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Event
from uuid import uuid4

import httpx
import pytest
import redis
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.api.v1 import conversations, documents, query, usage
from app.api.v1.platform_reads import get_access_client
from app.cache import semantic_cache
from app.config import get_settings
from app.db.platform import require_platform_schema
from app.db.session import get_db
from app.models.chunk import Chunk
from app.models.conversation import Conversation, Message, MessageRole
from app.models.document import ContentType, Document, UploadStatus
from app.models.llm_usage import LLMUsageRecord
from app.platform.config import PlatformSettings, get_platform_settings
from app.platform.tenant_access import TenantAccessClient
from app.services.agent.orchestrator import AgentOrchestrator
from app.services.llm.base import LLMResponse, LLMUsage
from app.services.llm.gemini import GeminiProvider
from app.services.rag.embedder import GeminiEmbedder
from app.services.rag.retriever import PgVectorRetriever
from app.services.rag_pipeline import RAGPipeline
from app.workers import embed_worker

URL = os.getenv("AGENTHUB_PLATFORM_TEST_DATABASE_URL")
VECTOR = [1.0] + [0.0] * 1535
pytestmark = pytest.mark.skipif(not URL, reason="Isolated migrated PostgreSQL is required")


@pytest.fixture()
def engine():
    engine = create_engine(URL)
    with engine.connect() as connection:
        name = connection.scalar(text("SELECT current_database()"))
        assert name.startswith("nexus_agent_ai_") and name.endswith("_test")
    require_platform_schema(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db(engine):
    with engine.connect() as connection:
        transaction = connection.begin()
        with Session(connection, join_transaction_mode="create_savepoint") as session:
            yield session
        transaction.rollback()


@pytest.fixture()
def cache(monkeypatch):
    client = redis.from_url(os.environ["AGENTHUB_TEST_REDIS_URL"], decode_responses=True)
    prefix = f"tenant-test:{uuid4()}:"
    monkeypatch.setattr(semantic_cache, "_client", client)
    monkeypatch.setattr(semantic_cache, "_cache_key", lambda bucket: prefix + bucket)
    yield client
    keys = list(client.scan_iter(prefix + "*"))
    if keys:
        client.delete(*keys)
    client.close()


def seed(db, tenant, name, *, ready=True, path="unused"):
    document = Document(tenant_id=tenant, title=name, content_type=ContentType.txt, file_path=str(path),
                        upload_status=UploadStatus.ready if ready else UploadStatus.pending,
                        chunk_count=1 if ready else 0)
    conversation = Conversation(tenant_id=tenant, title=name)
    db.add_all([document, conversation])
    db.flush()
    if ready:
        receipt = {"provider": "gemini", "model": get_settings().embedding_model, "dimensions": 1536}
        db.add(Chunk(document_id=document.id, text=name + " knowledge", chunk_index=0, embedding=VECTOR,
                     meta={"embedding": receipt}))
    db.add(Message(conversation_id=conversation.id, role=MessageRole.user, content=name + " history"))
    db.add(LLMUsageRecord(tenant_id=tenant, conversation_id=conversation.id, provider=name, model=name,
                          input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.01))
    db.commit()
    return document, conversation


def test_standalone_lists_keep_order_and_do_not_silently_truncate(db):
    baseline = datetime(2020, 1, 1, tzinfo=UTC)
    docs = [Document(title=f"standalone-{i}", content_type=ContentType.txt, file_path="unused",
                     created_at=baseline + timedelta(seconds=i)) for i in range(61)]
    convs = [Conversation(title=f"standalone-{i}", created_at=baseline + timedelta(seconds=i))
             for i in range(61)]
    db.add_all([*docs, *convs])
    db.flush()
    db.add_all([Message(conversation_id=convs[0].id, role=MessageRole.user, content=f"message-{i}")
                for i in range(61)])
    db.flush()

    doc_ids = {item.id for item in docs}
    conversation_ids = {item.id for item in convs}
    got_docs = [item.id for item in documents.list_documents(None, 0, 50, db) if item.id in doc_ids]
    got_convs = [item["id"] for item in conversations.list_conversations(None, 0, 50, db)
                 if item["id"] in conversation_ids]
    assert got_docs == [item.id for item in reversed(docs)]
    assert got_convs == [item.id for item in reversed(convs)]
    assert len(conversations.get_messages(convs[0].id, None, 0, 50, db)) == 61


@pytest.fixture()
def api(db, monkeypatch, tmp_path):
    tenant, other = uuid4(), uuid4()
    state = dict(status=200, role="owner", calls=[], reads=0)
    config = PlatformSettings(_env_file=None, PLATFORM_TENANT_ENABLED=True,
                              AUTHFORTRESS_BASE_URL="http://issuer.test")

    def handler(request):
        state["calls"].append(request)
        assert request.headers["authorization"] == "Bearer synthetic-token"
        assert request.url.path == f"/api/v1/tenants/{tenant}/authorize"
        permission = json.loads(request.content)["permission"]
        receipt = dict(user_id=str(uuid4()), tenant_id=str(tenant), role=state["role"],
                       permission=permission, allowed=True)
        receipt.update(state.get("receipt", {}))
        return httpx.Response(state["status"], json=receipt)

    def sessions():
        state["reads"] += 1
        yield db

    app = FastAPI()
    for router in (documents.router, conversations.router, query.router, usage.router):
        app.include_router(router, prefix="/api/v1/tenants/{tenant_id}")
    app.dependency_overrides[get_db] = sessions
    app.dependency_overrides[get_platform_settings] = lambda: config
    app.dependency_overrides[get_access_client] = lambda: TenantAccessClient(config, httpx.MockTransport(handler))
    monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(documents.process_document, "delay", lambda _: None)
    with TestClient(app) as client:
        yield client, f"/api/v1/tenants/{tenant}", state, tenant, other, config


def request(api, method, path, **kwargs):
    return api[0].request(method, api[1] + path, headers={"Authorization": "Bearer synthetic-token"}, **kwargs)


@pytest.fixture()
def generation(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(get_settings(), "llm_provider", "gemini")
    monkeypatch.setattr(get_settings(), "llm_fallback_provider", "")
    calls = []

    def generate(self, messages, **kwargs):
        calls.append(messages)
        return LLMResponse("Controlled answer", usage=LLMUsage(10, 5, 0.0001), provider="gemini", model="test")

    monkeypatch.setattr(GeminiProvider, "generate", generate)
    def initialize(self):
        self.model = get_settings().embedding_model
        self.client = SimpleNamespace()
    monkeypatch.setattr(GeminiEmbedder, "__init__", initialize)
    # These tests replace the embedding provider with controlled vectors.
    monkeypatch.setattr(GeminiEmbedder, "require_provider", lambda self: None)
    monkeypatch.setattr(GeminiEmbedder, "embed_query", lambda self, _: VECTOR)
    monkeypatch.setattr(GeminiEmbedder, "embed_texts", lambda self, texts: [VECTOR for _ in texts])
    return calls


def test_documents_conversations_and_usage_exclude_foreign_and_null_rows(api, db):
    own, own_conversation = seed(db, api[3], "own")
    foreign, foreign_conversation = seed(db, api[4], "foreign")
    legacy, legacy_conversation = seed(db, None, "legacy")
    assert [x["id"] for x in request(api, "GET", "/documents").json()] == [own.id]
    assert [x["id"] for x in request(api, "GET", "/conversations").json()] == [own_conversation.id]
    assert request(api, "GET", f"/documents/{own.id}").status_code == 200
    assert request(api, "GET", f"/conversations/{own_conversation.id}/messages").json()[0]["content"] == "own history"
    for doc, conv in ((foreign, foreign_conversation), (legacy, legacy_conversation)):
        assert request(api, "GET", f"/documents/{doc.id}").status_code == 404
        assert request(api, "DELETE", f"/documents/{doc.id}").status_code == 404
        assert request(api, "GET", f"/conversations/{conv.id}/messages").status_code == 404
    stats = request(api, "GET", "/usage/stats").json()
    assert stats["total_tokens"] == 15 and stats["cost_by_provider"] == {"own": 0.01}
    assert stats["cost_by_model"] == {"own": 0.01} and len(stats["cost_by_day"]) == 1
    assert db.get(Document, foreign.id) and db.get(Document, legacy.id)
    assert request(api, "GET", "/documents?limit=101").status_code == 422


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_issuer_denial_and_outage_never_access_db(api, status):
    api[2]["status"] = status
    result = request(api, "GET", "/documents")
    assert result.status_code == (503 if status == 500 else status)
    assert api[2]["reads"] == 0
    assert "synthetic-token" not in result.text


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Basic x"},
    [("Authorization", "Bearer synthetic-token"), ("Authorization", "Bearer other")]])
def test_missing_or_duplicate_bearer_never_contacts_issuer(api, headers):
    assert api[0].get(api[1] + "/documents", headers=headers).status_code == 401
    assert not api[2]["calls"] and api[2]["reads"] == 0


def test_permission_and_canonical_receipt_are_enforced(api):
    api[2]["role"] = "member"
    assert request(api, "GET", "/documents").status_code == 200
    # Even a faulty issuer's 200 must not authorize a member configuration write.
    assert request(api, "POST", "/documents", files={"file": ("a.txt", b"hello")}).status_code == 503
    assert json.loads(api[2]["calls"][-1].content) == {"permission": "ai.configure"}
    for bad in ({"tenant_id": str(api[4])}, {"permission": "ai.configure"}, {"allowed": False}):
        api[2]["receipt"] = bad
        assert request(api, "GET", "/documents").status_code == 503


def test_revocation_and_disabled_flag_take_effect_on_next_request(api):
    assert request(api, "GET", "/documents").status_code == 200
    api[2]["status"] = 401
    assert request(api, "GET", "/documents").status_code == 401
    api[5].tenant_enabled = False
    assert request(api, "GET", "/documents").status_code == 404
    assert len(api[2]["calls"]) == 2


def test_upload_uses_canonical_tenant_and_random_filename_and_delete_removes_file(api, db, tmp_path):
    result = request(api, "POST", "/documents", files={"file": ("../../other/secret.txt", b"hello")})
    assert result.status_code == 202
    row = db.get(Document, result.json()["document_id"])
    assert row.tenant_id == api[3]
    from pathlib import Path
    stored = Path(row.file_path)
    assert stored.parent == tmp_path and stored.suffix == ".txt" and "secret" not in stored.name
    assert stored.read_bytes() == b"hello"
    assert request(api, "DELETE", f"/documents/{row.id}").status_code == 200
    assert not stored.exists() and db.get(Document, row.id) is None


@pytest.mark.parametrize("name,content,status", [("a.exe", b"hello", 400), ("a.pdf", b"not PDF", 400),
    ("a.txt", b"\xff", 400), ("a.txt", b"", 400), ("a.txt", b"x" * (10 * 1024 * 1024 + 1), 413)],
    ids=["extension", "pdf-signature", "utf8", "empty", "oversize"])
def test_invalid_upload_creates_no_record_or_file(api, db, tmp_path, name, content, status):
    before = db.query(Document).filter(Document.tenant_id == api[3]).count()
    result = request(api, "POST", "/documents", files={"file": (name, content)})
    assert result.status_code == status
    assert db.query(Document).filter(Document.tenant_id == api[3]).count() == before
    assert list(tmp_path.iterdir()) == []


def test_scoped_rag_cache_usage_and_foreign_conversation_are_isolated(api, db, cache, generation):
    own, _ = seed(db, api[3], "own")
    _, foreign = seed(db, api[4], "foreign")
    _, legacy = seed(db, None, "legacy")
    for conv in (foreign, legacy):
        result = request(api, "POST", "/query", json={"question": "x", "conversation_id": conv.id})
        assert result.status_code == 404
    assert not generation
    assert request(api, "POST", "/query", json={"question": "x", "tenant_id": str(api[4])}).status_code == 422
    first = request(api, "POST", "/query", json={"question": "own?", "use_agent": False})
    assert first.status_code == 200 and first.json()["sources"][0]["document_title"] == "own"
    second = request(api, "POST", "/query", json={"question": "own?", "use_agent": False})
    assert second.json()["semantic_cache_hit"] is True and second.json()["tokens_used"] == 0
    assert len(generation) == 1
    records = db.query(LLMUsageRecord).filter(LLMUsageRecord.request_id == first.json()["request_id"]).all()
    assert len(records) == 1 and records[0].tenant_id == api[3] and records[0].message_id is not None
    pipeline = RAGPipeline()
    assert pipeline.ask(db, "own?", tenant_id=api[4])["sources"][0]["document_title"] == "foreign"
    assert len(generation) == 2
    assert pipeline.ask(db, "own?", tenant_id=uuid4())["sources"] == []
    assert PgVectorRetriever().retrieve(db, VECTOR, tenant_id=api[3],
                                       embedding_model=get_settings().embedding_model)[0]["document_id"] == own.id
    assert PgVectorRetriever().retrieve(db, VECTOR, tenant_id=None)[0]["document_title"] == "legacy"


def test_agent_history_and_knowledge_tool_use_scope_without_external_mcp(api, db, generation, monkeypatch):
    _, own = seed(db, api[3], "own")
    _, foreign = seed(db, api[4], "foreign")
    agent = AgentOrchestrator()
    assert [x["content"] for x in agent._load_history(db, own.id, api[3])] == ["own history"]
    assert agent._load_history(db, foreign.id, api[3]) == []
    monkeypatch.setattr("app.services.agent.orchestrator.mcp_client_manager.iter_agent_tools",
                        lambda: pytest.fail("Unscoped external MCP invoked"))
    tools = agent._build_tools(db, api[3])
    assert tools["search_knowledge_base"].run(query="x")["results"][0]["document_title"] == "own"
    assert request(api, "POST", "/query", json={"question": "x", "conversation_id": own.id}).status_code == 200
    assert "own history" in generation[0][0]["content"] and "foreign history" not in generation[0][0]["content"]


def test_embedding_and_delete_invalidate_cache_despite_redis_outage(api, db, cache, generation, monkeypatch, tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("New knowledge", encoding="utf-8")
    document, _ = seed(db, api[3], "own", ready=False, path=path)
    seed(db, api[4], "other")
    pipeline = RAGPipeline()
    assert pipeline.ask(db, "x", tenant_id=api[3])["sources"] == []
    assert pipeline.ask(db, "x", tenant_id=api[4])["semantic_cache_hit"] is False
    revision = document.content_revision
    monkeypatch.setattr(embed_worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)

    class Outage:
        def get(self, *args):
            raise redis.ConnectionError("offline")

    with monkeypatch.context() as interruption:
        interruption.setattr(semantic_cache, "_client", Outage())
        embed_worker.process_document.run(document.id)
    assert document.content_revision != revision and document.upload_status == UploadStatus.ready
    assert pipeline.ask(db, "x", tenant_id=api[3])["semantic_cache_hit"] is False
    assert pipeline.ask(db, "x", tenant_id=api[4])["semantic_cache_hit"] is True
    assert request(api, "DELETE", f"/documents/{document.id}").status_code == 200
    result = pipeline.ask(db, "x", tenant_id=api[3])
    assert result["semantic_cache_hit"] is False and result["sources"] == []


def test_old_concurrent_generation_cannot_poison_cache_after_committed_delete(engine, cache, generation, monkeypatch):
    tenant = uuid4()
    started, release = Event(), Event()
    original_generate = GeminiProvider.generate

    def generate(self, messages, **kwargs):
        if "old knowledge" in messages[-1]["content"]:
            started.set()
            assert release.wait(10)
        return original_generate(self, messages, **kwargs)

    monkeypatch.setattr(GeminiProvider, "generate", generate)
    with Session(engine) as db:
        old, conversation = seed(db, tenant, "old")
    pipeline = RAGPipeline()

    def old_request():
        with Session(engine) as db:
            return pipeline.ask(db, "x", tenant_id=tenant)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(old_request)
            try:
                assert started.wait(10)
                with Session(engine) as db:
                    db.delete(db.query(Document).filter(Document.tenant_id == tenant).one())
                    db.commit()
                    seed(db, tenant, "new")
                    new = pipeline.ask(db, "x", tenant_id=tenant)
                    assert new["sources"][0]["document_title"] == "new"
                    assert new["semantic_cache_hit"] is False
            finally:
                release.set()
            assert future.result(10)["sources"][0]["document_title"] == "old"
        with Session(engine) as db:
            current = pipeline.ask(db, "x", tenant_id=tenant)
            assert current["semantic_cache_hit"] is True and current["sources"][0]["document_title"] == "new"
    finally:
        with Session(engine) as db:
            for row in db.query(Document).filter(Document.tenant_id == tenant):
                db.delete(row)
            for row in db.query(Conversation).filter(Conversation.tenant_id == tenant):
                db.delete(row)
            db.query(LLMUsageRecord).filter(LLMUsageRecord.tenant_id == tenant).delete()
            db.commit()


@pytest.mark.parametrize("mode", ["telegram_ai_enabled", "read_enabled", "tenant_enabled"])
def test_global_http_and_direct_mcp_are_closed_in_each_platform_mode(monkeypatch, mode, engine):
    from app import main
    from app.mcp import server
    from app.platform import resource_access

    config = PlatformSettings(_env_file=None)
    setattr(config, mode, True)
    monkeypatch.setattr(main, "get_platform_settings", lambda: config)
    monkeypatch.setattr(main, "engine", engine)
    monkeypatch.setattr(resource_access, "get_platform_settings", lambda: config)
    with TestClient(main.app) as client:
        for path in ("/api/v1/documents", "/api/v1/query", "/api/v1/conversations/1/messages",
                     "/api/v1/usage/stats", "/metrics", "/mcp/sse", "/mcp/messages/", "/mcp"):
            assert client.get(path).status_code == 404
            assert client.post(path, json={"question": "x"}).status_code == 404
    callbacks = []

    class CaptureMCP:
        def __init__(self, **kwargs):
            pass

        def tool(self, **kwargs):
            return self.resource("")

        def resource(self, _):
            def capture(function):
                callbacks.append(function)
                return function
            return capture

        def sse_app(self):
            return None

    monkeypatch.setattr(server, "FastMCP", CaptureMCP)
    monkeypatch.setattr(server, "SessionLocal", lambda: pytest.fail("MCP touched database"))
    server.create_mcp_starlette_app()
    for callback, args in zip(callbacks, [("x",), (), ("1",)], strict=True):
        with pytest.raises(HTTPException) as error:
            callback(*args)
        assert error.value.status_code == 404


@pytest.mark.parametrize("receipt", [None, {}, [], "invalid",
    {"provider": "offline_sha256", "model": "sha256-v1", "dimensions": 1536},
    {"provider": "gemini", "model": "wrong", "dimensions": 1536},
    {"provider": "gemini", "model": "CURRENT", "dimensions": "1536"},
    {"provider": "gemini", "model": "CURRENT", "dimensions": 768}])
def test_tenant_search_rejects_incompatible_receipts_even_after_cache_hit(db, cache, generation, receipt):
    from app.services.agent.tools.knowledge_base import KnowledgeBaseTool
    from app.services.rag.embedder import EmbeddingUnavailable

    tenant = uuid4()
    document, _ = seed(db, tenant, "owned")
    pipeline = RAGPipeline()
    first = pipeline.ask(db, "question", tenant_id=tenant)
    assert first["sources"] and len(generation) == 1
    row = db.query(Chunk).filter_by(document_id=document.id).one()
    if isinstance(receipt, dict) and receipt.get("model") == "CURRENT":
        receipt = {**receipt, "model": pipeline.embedder.model}
    row.meta = {"embedding": receipt}
    db.commit()
    with pytest.raises(EmbeddingUnavailable):
        pipeline.ask(db, "question", tenant_id=tenant)
    with pytest.raises(EmbeddingUnavailable):
        KnowledgeBaseTool(db, pipeline, tenant).run(query="question")
    assert len(generation) == 1


def test_corpus_receipt_guard_checks_beyond_top_k_and_keeps_foreign_scope(db):
    from app.services.rag.embedder import EmbeddingUnavailable

    tenant, foreign = uuid4(), uuid4()
    owned, _ = seed(db, tenant, "owned")
    other, _ = seed(db, foreign, "foreign")
    row = db.query(Chunk).filter_by(document_id=other.id).one()
    row.meta = {}
    db.commit()
    retriever = PgVectorRetriever()
    model = get_settings().embedding_model
    assert retriever.retrieve(db, VECTOR, 1, tenant, model)[0]["document_id"] == owned.id
    assert retriever.retrieve(db, VECTOR, 1, uuid4(), model) == []
    db.add(Chunk(document_id=owned.id, text="invalid distant", chunk_index=1,
                 embedding=[0.0, 1.0] + [0.0] * 1534, meta={}))
    db.commit()
    with pytest.raises(EmbeddingUnavailable):
        retriever.retrieve(db, VECTOR, 1, tenant, model)


def test_tenant_corpus_scan_limit_rejects_partial_search(db, monkeypatch):
    from app.services.rag import retriever as module
    from app.services.rag.embedder import EmbeddingUnavailable

    tenant = uuid4()
    first, _ = seed(db, tenant, "first")
    seed(db, tenant, "second")
    monkeypatch.setattr(module, "MAX_TENANT_SEARCH_CHUNKS", 1)
    retriever = PgVectorRetriever()
    with pytest.raises(EmbeddingUnavailable):
        retriever.retrieve(db, VECTOR, 1, tenant, get_settings().embedding_model)
    monkeypatch.setattr(module, "MAX_TENANT_SEARCH_CHUNKS", 2)
    assert retriever.retrieve(db, VECTOR, 1, tenant, get_settings().embedding_model)


@pytest.mark.parametrize("status", [UploadStatus.pending, UploadStatus.processing, UploadStatus.failed])
def test_tenant_search_rejects_nonready_document_with_retained_chunks(db, cache, generation, status):
    from app.services.agent.tools.knowledge_base import KnowledgeBaseTool
    from app.services.rag.embedder import EmbeddingUnavailable

    tenant = uuid4()
    document, _ = seed(db, tenant, "owned")
    pipeline = RAGPipeline()
    first = pipeline.ask(db, "question", tenant_id=tenant)
    assert first["sources"] and len(generation) == 1
    document.upload_status = status
    db.commit()
    with pytest.raises(EmbeddingUnavailable):
        pipeline.ask(db, "question", tenant_id=tenant)
    with pytest.raises(EmbeddingUnavailable):
        KnowledgeBaseTool(db, pipeline, tenant).run(query="question")
    assert len(generation) == 1
    assert db.query(Chunk).filter_by(document_id=document.id).count() == 1


def test_corpus_readiness_guard_keeps_tenant_and_chunk_scope(db):
    tenant, foreign = uuid4(), uuid4()
    owned, _ = seed(db, tenant, "owned")
    other, _ = seed(db, foreign, "foreign")
    other.upload_status = UploadStatus.failed
    for status in (UploadStatus.pending, UploadStatus.processing, UploadStatus.failed):
        db.add(Document(tenant_id=tenant, title=status.value, content_type=ContentType.txt,
                        file_path="unused", upload_status=status))
    db.commit()
    results = PgVectorRetriever().retrieve(db, VECTOR, 1, tenant, get_settings().embedding_model)
    assert [row["document_id"] for row in results] == [owned.id]
