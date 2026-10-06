import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import create_engine, event, func, select, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.models.telegram_job import TelegramAIJob
from app.models.telegram_reply import TelegramReplyOutbox
from app.models.telegram_usage import TelegramAIUsage
from app.platform.config import get_platform_settings
from app.platform.jobs import admit_job
from app.platform.schemas import IngressEnvelope
from app.services.llm.base import LLMResponse, LLMUsage

URL = os.getenv("AGENTHUB_PLATFORM_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="Isolated migrated PostgreSQL is required")


@pytest.fixture(scope="module")
def sessions():
    url = make_url(URL)
    assert url.database.startswith("nexus_agent_ai_") and url.database.endswith("_test")
    name = f"nexus_agent_ai_worker_{uuid4().hex}_test"
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    with admin.connect() as db:
        assert db.scalar(text("SELECT current_database()")) == url.database
        db.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    target = url.set(database=name)
    env = dict(os.environ, DATABASE_URL=target.render_as_string(hide_password=False))
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"],
                   env=env, check=True, capture_output=True)
    engine = create_engine(target)
    yield sessionmaker(engine, expire_on_commit=False)
    engine.dispose()  # Retain synthetic records; no DROP/TRUNCATE or deletion.


@pytest.fixture
def setup(sessions, monkeypatch):
    for key, value in {
        "TELEGRAM_AI_ENABLED": "true", "TELEGRAM_AI_MODEL": "openai/gpt-oss-20b",
        "WEBHOOK_INTERNAL_URL": "http://webhook.test", "GROQ_API_KEY": "synthetic-key",
        "WEBHOOK_AGENT_INGRESS_KEY": "i" * 32, "WEBHOOK_AGENT_SERVICE_KEY": "c" * 32,
        "AGENT_WEBHOOK_REPLY_KEY": "r" * 32,
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    get_platform_settings.cache_clear()
    bot = uuid4()
    item = dict(event_id=str(uuid4()), event_type="telegram.message.received", schema_version=1,
                occurred_at=datetime.now(UTC).isoformat(), tenant_id=str(uuid4()), bot_id=str(bot),
                correlation_id=str(uuid4()), idempotency_key=f"telegram:{bot}:42",
                payload=dict(update_id=42, chat_id=43, message_id=1, question="Synthetic question"))
    with sessions() as db:
        job, _ = admit_job(db, IngressEnvelope.model_validate_json(json.dumps(item)), "a" * 64)
        db.commit()
        job_id = job.id
    state = {"context_status": 200, "provider_status": 200, "calls": 0}

    def handler(request):
        if request.url.host == "webhook.test":
            return httpx.Response(state["context_status"], json={
                "bot_id": item["bot_id"], "tenant_id": state.get("tenant", item["tenant_id"]),
                "telegram_bot_id": 123, "is_active": state.get("active", True),
            })
        assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
        state["calls"] += 1
        with sessions() as db:
            row = db.get(TelegramAIJob, job_id)
            assert row.state == "processing" and row.call_started_at is not None
            assert row.provider == "groq" and row.model == "openai/gpt-oss-20b"
        if state.get("timeout"):
            raise httpx.ReadTimeout("synthetic private text")
        if state.get("crash"):
            raise WorkerCrash()
        return httpx.Response(state["provider_status"], json={
            "id": "synthetic", "object": "chat.completion", "created": 1,
            "model": "openai/gpt-oss-20b", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": state.get("text", "Synthetic answer")}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    original = httpx.AsyncClient

    class Client(original):
        def __init__(self, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    yield sessions, job_id, state, item
    get_settings.cache_clear()
    get_platform_settings.cache_clear()


def row(sessions, job_id):
    with sessions() as db:
        return db.get(TelegramAIJob, job_id)


def run(setup):
    from app.platform.processor import process_job
    process_job(setup[1], factory=setup[0])


def claim(sessions, job_id):
    from app.platform.jobs import claim_job
    with sessions() as db:
        result = claim_job(db, job_id)
        db.commit()
        return result


def expire(sessions, job_id):
    with sessions() as db:
        db.execute(update(TelegramAIJob).where(TelegramAIJob.id == job_id).values(
            lease_until=func.clock_timestamp() - text("interval '1 second'")))
        db.commit()


def make_due(sessions, job_id):
    with sessions() as db:
        db.execute(update(TelegramAIJob).where(TelegramAIJob.id == job_id).values(
            next_attempt_at=func.clock_timestamp() - text("interval '1 second'")))
        db.commit()


def response():
    return LLMResponse("Synthetic answer", provider="groq", model="openai/gpt-oss-20b",
                       usage=LLMUsage(10, 5, 0.00000225))


def test_concurrent_claim_has_one_owner(setup):
    sessions, job_id, _, _ = setup
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(sessions, job_id), range(2)))
    assert sum(x is not None for x in results) == 1
    assert row(sessions, job_id).attempts == 1
    assert claim(sessions, job_id) is None


def test_atomic_completion_once_and_legacy_isolation(setup):
    from app.models.conversation import Conversation
    from app.models.llm_usage import LLMUsageRecord as LegacyUsage
    sessions, job_id, state, item = setup
    with sessions() as db:
        before = (db.scalar(select(func.count()).select_from(Conversation)),
                  db.scalar(select(func.count()).select_from(LegacyUsage)))
    run(setup)
    run(setup)
    assert state["calls"] == 1
    assert row(sessions, job_id).state == "completed"
    assert row(sessions, job_id).answer == "Synthetic answer"
    with sessions() as db:
        usage = db.get(TelegramAIUsage, job_id)
        assert usage.tenant_id == UUID(item["tenant_id"]) and usage.bot_id == UUID(item["bot_id"])
        reply = db.scalar(select(TelegramReplyOutbox).where(TelegramReplyOutbox.job_id == job_id))
        assert reply.envelope["payload"] == {
            "ingress_event_id": item["event_id"], "job_id": str(job_id), "text": "Synthetic answer"}
        assert reply.envelope["idempotency_key"] == f'telegram-answer:{item["event_id"]}'
        assert reply.envelope["correlation_id"] == item["correlation_id"]
        assert db.scalar(select(func.count()).select_from(TelegramReplyOutbox).where(
            TelegramReplyOutbox.job_id == job_id)) == 1
        assert before == (db.scalar(select(func.count()).select_from(Conversation)),
                          db.scalar(select(func.count()).select_from(LegacyUsage)))


@pytest.mark.parametrize("kind", ["inactive", "cross_tenant", "denied"])
def test_fresh_context_denial_prevents_generation(setup, kind):
    sessions, job_id, state, _ = setup
    if kind == "inactive":
        state["active"] = False
    elif kind == "cross_tenant":
        state["tenant"] = str(uuid4())
    else:
        state["context_status"] = 403
    run(setup)
    assert row(sessions, job_id).state == "failed"
    assert row(sessions, job_id).call_started_at is None
    assert state["calls"] == 0


def test_context_outage_backoff_and_max_five(setup):
    sessions, job_id, state, _ = setup
    state["context_status"] = 503
    for attempt in range(1, 6):
        run(setup)
        result = row(sessions, job_id)
        assert result.attempts == attempt
        assert result.state == ("failed" if attempt == 5 else "pending")
        assert result.call_started_at is None
        with sessions() as db:
            remaining = db.scalar(select(func.extract("epoch", TelegramAIJob.next_attempt_at
                - func.clock_timestamp())).where(TelegramAIJob.id == job_id))
        if attempt < 5:
            assert 2**attempt - 1 <= remaining <= 2**attempt
            run(setup)
            assert row(sessions, job_id).attempts == attempt
            make_due(sessions, job_id)
    assert state["calls"] == 0


@pytest.mark.parametrize("started", [False, True])
def test_expired_claim_recovery_respects_effect_marker(setup, started):
    from app.platform.jobs import mark_call_started
    from app.platform.processor import recover_jobs_once
    sessions, job_id, _, _ = setup
    old = claim(sessions, job_id)
    if started:
        with sessions() as db:
            assert mark_call_started(db, old)
            db.commit()
    expire(sessions, job_id)
    recover_jobs_once(factory=sessions, notify=lambda _: None)
    assert row(sessions, job_id).state == ("unknown" if started else "pending")
    if not started:
        make_due(sessions, job_id)
        new = claim(sessions, job_id)
        assert new.claim_id != old.claim_id
        with sessions() as db:
            assert not mark_call_started(db, old)
            db.commit()


def test_expired_fence_cannot_store_result(setup):
    from app.platform.jobs import complete_job, mark_call_started
    sessions, job_id, _, _ = setup
    owned = claim(sessions, job_id)
    with sessions() as db:
        assert mark_call_started(db, owned)
        db.commit()
    expire(sessions, job_id)
    with sessions() as db:
        assert not complete_job(db, owned, response())
        db.commit()
        assert db.get(TelegramAIUsage, job_id) is None
        assert db.scalar(select(TelegramReplyOutbox.id).where(TelegramReplyOutbox.job_id == job_id)) is None


@pytest.mark.parametrize("kind,expected", [("timeout", "unknown"), ("quota", "unknown"),
                                         ("auth", "failed"), ("empty", "failed")])
def test_after_started_failure_never_repeats(setup, kind, expected):
    sessions, job_id, state, _ = setup
    if kind == "timeout":
        state["timeout"] = True
    elif kind == "empty":
        state["text"] = ""
    else:
        state["provider_status"] = 429 if kind == "quota" else 401
    run(setup)
    run(setup)
    assert state["calls"] == 1
    assert row(sessions, job_id).state == expected
    with sessions() as db:
        assert db.get(TelegramAIUsage, job_id) is None


def test_sql_failure_after_generation_is_atomic_and_unknown(setup):
    sessions, job_id, state, _ = setup
    engine = sessions.kw["bind"]

    def fail(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO telegram_reply_outbox"):
            raise RuntimeError("synthetic private SQL error")

    event.listen(engine, "before_cursor_execute", fail)
    try:
        run(setup)
    finally:
        event.remove(engine, "before_cursor_execute", fail)
    run(setup)
    assert state["calls"] == 1
    assert row(sessions, job_id).state == "unknown"
    assert row(sessions, job_id).answer is None
    with sessions() as db:
        assert db.get(TelegramAIUsage, job_id) is None
        assert db.scalar(select(TelegramReplyOutbox.id).where(TelegramReplyOutbox.job_id == job_id)) is None


def test_missing_key_is_permanent_before_effect(setup, monkeypatch):
    sessions, job_id, state, _ = setup
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    run(setup)
    assert row(sessions, job_id).state == "failed"
    assert row(sessions, job_id).call_started_at is None and state["calls"] == 0


def test_disabled_worker_leaves_pending(setup, monkeypatch):
    sessions, job_id, state, _ = setup
    monkeypatch.setenv("TELEGRAM_AI_ENABLED", "false")
    get_platform_settings.cache_clear()
    run(setup)
    assert row(sessions, job_id).state == "pending" and state["calls"] == 0


def test_task_rejects_malformed_uuid_before_database(setup):
    from app.workers.telegram_worker import process_telegram_job
    process_telegram_job("not-a-uuid")
    assert process_telegram_job.soft_time_limit == 25 and process_telegram_job.time_limit == 30
    assert process_telegram_job.acks_late and process_telegram_job.reject_on_worker_lost


class WorkerCrash(BaseException):
    pass


def test_crash_after_marker_recovers_unknown_without_generation_retry(setup):
    from app.platform.processor import recover_jobs_once
    sessions, job_id, state, _ = setup
    state["crash"] = True
    with pytest.raises(WorkerCrash):
        run(setup)
    assert state["calls"] == 1 and row(sessions, job_id).state == "processing"
    expire(sessions, job_id)
    recover_jobs_once(factory=sessions, notify=lambda _: None)
    run(setup)
    assert state["calls"] == 1 and row(sessions, job_id).state == "unknown"


def test_marker_commit_receipt_loss_cannot_reset_to_pending(setup):
    from sqlalchemy.orm import Session

    from app.platform.processor import recover_jobs_once
    sessions, job_id, state, _ = setup
    engine = sessions.kw["bind"]
    marked = [False]

    def statement(connection, cursor, sql, parameters, context, executemany):
        if sql.startswith("UPDATE telegram_ai_jobs") and "call_started_at=" in sql:
            marked[0] = True

    def committed(session):
        if marked[0]:
            marked[0] = False
            raise RuntimeError("synthetic lost commit receipt")

    event.listen(engine, "after_cursor_execute", statement)
    event.listen(Session, "after_commit", committed)
    try:
        run(setup)
    finally:
        event.remove(engine, "after_cursor_execute", statement)
        event.remove(Session, "after_commit", committed)
    assert state["calls"] == 0
    assert row(sessions, job_id).state == "processing"
    assert row(sessions, job_id).call_started_at is not None
    expire(sessions, job_id)
    recover_jobs_once(factory=sessions, notify=lambda _: None)
    run(setup)
    assert row(sessions, job_id).state == "unknown" and state["calls"] == 0


def test_selected_model_stays_frozen_across_safe_retry(setup, monkeypatch):
    sessions, job_id, state, _ = setup
    state["context_status"] = 503
    run(setup)
    monkeypatch.setenv("TELEGRAM_AI_MODEL", "some-new-model")
    get_platform_settings.cache_clear()
    state["context_status"] = 200
    make_due(sessions, job_id)
    run(setup)
    assert row(sessions, job_id).model == "openai/gpt-oss-20b"
    assert row(sessions, job_id).state == "completed" and state["calls"] == 1


def test_broker_outage_scanner_keeps_pending_then_real_redis_resume(setup, monkeypatch, caplog):
    import redis

    from app.platform.processor import recover_jobs_once
    sessions, job_id, _, _ = setup
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    get_settings.cache_clear()
    assert recover_jobs_once(factory=sessions) == 0
    assert row(sessions, job_id).state == "pending"
    assert "synthetic-key" not in caplog.text and "Synthetic question" not in caplog.text
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:56381/14")
    # CI uses its explicitly supplied test Redis URL instead of the local host port.
    redis_url = os.getenv("AGENTHUB_TEST_REDIS_URL", "redis://127.0.0.1:56381/14")
    monkeypatch.setenv("REDIS_URL", redis_url)
    get_settings.cache_clear()
    client = redis.Redis.from_url(redis_url)
    try:
        count = client.llen("telegram_ai")
        submitted = recover_jobs_once(factory=sessions)
        assert submitted > 0 and client.llen("telegram_ai") == count + submitted
        messages = client.lrange("telegram_ai", 0, submitted - 1)
        import base64
        args = [json.loads(base64.b64decode(json.loads(msg)["body"]))[0] for msg in messages]
        assert [str(job_id)] in args
        assert all(len(value) == 1 and str(UUID(value[0])) == value[0] for value in args)
    finally:
        client.close()


def test_scanner_batch_is_bounded(setup):
    from app.platform.jobs import recover_due_jobs
    sessions, _, _, _ = setup
    with sessions() as db:
        for _ in range(105):
            bot = uuid4()
            db.add(TelegramAIJob(id=uuid4(), event_id=uuid4(), tenant_id=uuid4(), bot_id=bot,
                                update_id=42, envelope={}, digest="a" * 64))
        db.commit()
    with sessions() as db:
        assert len(recover_due_jobs(db, 100)) == 100
        with pytest.raises(ValueError):
            recover_due_jobs(db, 101)
        db.rollback()
