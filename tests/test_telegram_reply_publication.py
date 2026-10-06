import asyncio
import gzip
import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import func, select, text, update

from app.models.telegram_job import TelegramAIJob
from app.models.telegram_reply import TelegramReplyOutbox
from app.platform.config import get_platform_settings
from tests.test_telegram_worker import URL
from tests.test_telegram_worker import sessions as sessions

pytestmark = pytest.mark.skipif(not URL, reason="Isolated migrated PostgreSQL is required")


@pytest.fixture
def publication(sessions, monkeypatch):
    settings = {
        "TELEGRAM_AI_ENABLED": "true", "TELEGRAM_AI_MODEL": "openai/gpt-oss-20b",
        "WEBHOOK_INTERNAL_URL": "http://webhook.test", "WEBHOOK_AGENT_INGRESS_KEY": "i" * 32,
        "WEBHOOK_AGENT_SERVICE_KEY": "c" * 32, "AGENT_WEBHOOK_REPLY_KEY": "r" * 32,
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    get_platform_settings.cache_clear()
    scope = dict(tenant_id=uuid4(), bot_id=uuid4())
    job_id, original_event, answer_event, outbox_id = uuid4(), uuid4(), uuid4(), uuid4()
    envelope = dict(event_id=str(answer_event), event_type="telegram.answer.created", schema_version=1,
        occurred_at="2026-10-06T00:00:00+00:00", tenant_id=str(scope["tenant_id"]),
        bot_id=str(scope["bot_id"]), correlation_id=str(uuid4()),
        idempotency_key=f"telegram-answer:{original_event}",
        payload=dict(ingress_event_id=str(original_event), job_id=str(job_id), text="Synthetic answer"))
    with sessions() as db:
        db.add(TelegramAIJob(id=job_id, event_id=original_event, **scope, update_id=1, envelope={},
            digest="a" * 64, state="completed", answer="Synthetic answer", provider="groq", model="test"))
        db.flush()
        db.add(TelegramReplyOutbox(id=outbox_id, job_id=job_id, **scope, envelope=envelope))
        db.commit()
    state = {"status": 202, "calls": [], "records": {}, "delivery_id": str(uuid4())}

    def handler(request):
        assert str(request.url) == "http://webhook.test/internal/v1/telegram/answers"
        assert request.headers["X-Webhook-Signature"] == "sha256=" + hmac.new(
            b"r" * 32, request.content, hashlib.sha256).hexdigest()
        assert "X-Service-Key" not in request.headers
        state["calls"].append(request.content)
        if state.get("lost_once") and len(state["calls"]) == 1:
            state["records"][envelope["event_id"]] = state["delivery_id"]
            raise httpx.ReadTimeout("synthetic private error")
        status = state["status"]
        body = state.get("body", {
            "event_id": envelope["event_id"], "delivery_id": state["delivery_id"], "state": "pending"})
        if status in {200, 202}:
            state["records"][envelope["event_id"]] = state["delivery_id"]
        headers = {"Content-Type": "application/json", **state.get("headers", {})}
        return httpx.Response(status, content=state.get("raw", json.dumps(body).encode()), headers=headers)

    original = httpx.AsyncClient

    class Client(original):
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    yield sessions, outbox_id, state, envelope
    get_platform_settings.cache_clear()


def row(factory, outbox_id):
    with factory() as db:
        return db.get(TelegramReplyOutbox, outbox_id)


def publish(publication):
    from app.platform.replies import publish_reply
    publish_reply(publication[1], factory=publication[0])


def due(factory, outbox_id):
    with factory() as db:
        db.execute(update(TelegramReplyOutbox).where(TelegramReplyOutbox.id == outbox_id).values(
            next_attempt_at=func.clock_timestamp() - text("interval '1 second'")))
        db.commit()


@pytest.mark.parametrize("status", [200, 202])
def test_matching_admission_receipt_is_published_not_sent(publication, status):
    factory, outbox_id, state, envelope = publication
    state["status"] = status
    publish(publication)
    publish(publication)
    result = row(factory, outbox_id)
    assert result.state == "published" and str(result.delivery_id) == state["delivery_id"]
    assert len(state["calls"]) == 1
    assert json.loads(state["calls"][0]) == envelope
    assert "chat_id" not in envelope["payload"] and "token" not in envelope["payload"]


def test_lost_receipt_retries_same_immutable_identity(publication):
    factory, outbox_id, state, _ = publication
    state["lost_once"] = True
    publish(publication)
    assert row(factory, outbox_id).state == "pending"
    due(factory, outbox_id)
    state["status"] = 200
    publish(publication)
    assert row(factory, outbox_id).state == "published"

    assert len(state["records"]) == 1
    assert len(state["calls"]) == 2 and state["calls"][0] == state["calls"][1]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 415, 422])
def test_terminal_rejection_is_not_retried(publication, status, caplog):
    factory, outbox_id, state, _ = publication
    state.update(status=status, body={"detail": "synthetic private text"})
    publish(publication)
    due(factory, outbox_id)
    publish(publication)
    assert row(factory, outbox_id).state == "failed" and len(state["calls"]) == 1
    assert "synthetic private text" not in caplog.text


def test_readiness_conflict_is_retryable(publication):
    factory, outbox_id, state, _ = publication
    state.update(status=409, body={"detail": "ingress_publication_not_ready"})
    publish(publication)
    assert row(factory, outbox_id).state == "pending"
    due(factory, outbox_id)
    state.pop("body")
    state["status"] = 202
    publish(publication)
    assert row(factory, outbox_id).state == "published"


@pytest.mark.parametrize("kind", ["malformed", "large", "encoding", "mime", "duplicate"])
def test_only_exact_readiness_conflict_can_retry(publication, kind):
    factory, outbox_id, state, _ = publication
    state["status"] = 409
    if kind == "malformed":
        state["raw"] = b"not-json"
    elif kind == "large":
        state["raw"] = b" " * 65537
    elif kind == "encoding":
        state["headers"] = {"Content-Encoding": "gzip"}
        state["raw"] = gzip.compress(b'{"detail":"ingress_publication_not_ready"}')
    elif kind == "mime":
        state["headers"] = {"Content-Type": "text/plain"}
    else:
        state["raw"] = b'{"detail":1,"detail":"ingress_publication_not_ready"}'
    publish(publication)
    assert row(factory, outbox_id).state == "failed"


@pytest.mark.parametrize("kind", ["ids", "uuid", "state", "extra", "large", "encoding", "duplicate", "nan"])
def test_ambiguous_receipt_cannot_be_published(publication, kind):
    factory, outbox_id, state, envelope = publication
    body = {"event_id": envelope["event_id"], "delivery_id": state["delivery_id"], "state": "pending"}
    if kind == "ids":
        body["event_id"] = str(uuid4())
    elif kind == "uuid":
        body["delivery_id"] = True
    elif kind == "state":
        body["state"] = "arbitrary"
    elif kind == "extra":
        body["extra"] = 1
    elif kind == "large":
        state["raw"] = b" " * 65537
    elif kind == "encoding":
        state["headers"] = {"Content-Encoding": "gzip"}
    elif kind == "duplicate":
        state["raw"] = b'{"event_id":1,"event_id":2}'
    else:
        state["raw"] = b'{"event_id":NaN}'
    state["body"] = body
    publish(publication)
    assert row(factory, outbox_id).state == "pending" and row(factory, outbox_id).delivery_id is None


def test_retry_after_is_bounded_and_attempts_stop_at_ten(publication):
    factory, outbox_id, state, _ = publication
    state.update(status=429, headers={"Retry-After": "999999"})
    for attempt in range(1, 11):
        publish(publication)
        result = row(factory, outbox_id)
        assert result.attempts == attempt
        assert result.state == ("failed" if attempt == 10 else "pending")
        with factory() as db:
            remaining = db.scalar(select(func.extract("epoch", TelegramReplyOutbox.next_attempt_at
                - func.clock_timestamp())).where(TelegramReplyOutbox.id == outbox_id))
        assert 59 <= remaining <= 60
        due(factory, outbox_id)
    publish(publication)
    assert len(state["calls"]) == 10


def test_stale_claim_cannot_overwrite_delivery(publication):
    from app.platform.replies import claim_reply, finish_reply, recover_replies_once
    factory, outbox_id, _, _ = publication
    with factory() as db:
        old = claim_reply(db, outbox_id)
        db.commit()
    with factory() as db:
        db.execute(update(TelegramReplyOutbox).where(TelegramReplyOutbox.id == outbox_id).values(
            lease_until=func.clock_timestamp() - text("interval '1 second'")))
        db.commit()
    recover_replies_once(factory=factory, notify=lambda _: None)
    due(factory, outbox_id)
    publish(publication)
    assert row(factory, outbox_id).state == "published"
    with factory() as db:
        assert not finish_reply(db, old, state="failed", code="stale")
        db.commit()
    assert row(factory, outbox_id).state == "published"


def test_concurrent_notifications_have_one_live_claim(publication):
    from app.platform.replies import claim_reply
    factory, outbox_id, _, _ = publication

    def acquire():
        with factory() as db:
            claim = claim_reply(db, outbox_id)
            db.commit()
            return claim

    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(lambda _: acquire(), range(4)))
    assert sum(claim is not None for claim in claims) == 1
    assert row(factory, outbox_id).attempts == 1
    publish(publication)
    assert row(factory, outbox_id).state == "processing"


def test_scanner_broker_failure_preserves_pending_and_can_resume(publication):
    from app.platform.replies import recover_replies_once
    factory, outbox_id, _, _ = publication

    def unavailable(_):
        raise OSError("synthetic broker failure")

    assert recover_replies_once(factory=factory, notify=unavailable) == 0
    assert row(factory, outbox_id).state == "pending" and row(factory, outbox_id).attempts == 0
    ids = []
    assert recover_replies_once(factory=factory, notify=ids.append) >= 1
    assert outbox_id in ids and all(type(value) is type(outbox_id) for value in ids)
    publish(publication)
    assert row(factory, outbox_id).state == "published"


@pytest.mark.parametrize("status", [500, 502, 503])
def test_server_errors_retry_with_backoff(publication, status):
    factory, outbox_id, state, _ = publication
    state["status"] = status
    publish(publication)
    assert row(factory, outbox_id).state == "pending"
    publish(publication)
    assert len(state["calls"]) == 1


def test_disabled_publication_does_not_access_database(publication, monkeypatch):
    from app.platform.replies import publish_reply, recover_replies_once
    monkeypatch.setenv("TELEGRAM_AI_ENABLED", "false")
    get_platform_settings.cache_clear()

    def forbidden():
        raise AssertionError("Disabled publication opened a session")

    publish_reply(publication[1], factory=forbidden)
    assert recover_replies_once(factory=forbidden) == 0


@pytest.mark.parametrize("batch", [0, 101])
def test_scanner_batch_is_bounded(publication, batch):
    from app.platform.replies import recover_replies_once
    with pytest.raises(ValueError, match="batch"):
        recover_replies_once(batch, factory=publication[0])


@pytest.mark.parametrize("identity", ["invalid", "", None, 123])
def test_malformed_queue_uuid_does_not_open_database(monkeypatch, identity):
    from app.workers import telegram_reply_worker
    def forbidden(_):
        raise AssertionError("Malformed notification reached processing")
    monkeypatch.setattr(telegram_reply_worker, "publish_reply", forbidden)
    telegram_reply_worker.publish_telegram_reply.run(identity)


def test_lost_local_commit_receipt_cannot_overwrite_published(publication):
    from app.platform.replies import publish_reply
    factory, outbox_id, state, _ = publication
    count = 0

    def uncertain_factory():
        nonlocal count
        db = factory()
        count += 1
        if count == 2:
            original = db.commit
            def commit():
                original()
                raise OSError("Synthetic lost commit receipt")
            db.commit = commit
        return db

    publish_reply(outbox_id, factory=uncertain_factory)
    assert row(factory, outbox_id).state == "published"
    publish(publication)
    assert len(state["calls"]) == 1


def test_total_deadline_keeps_uncertain_publication_retryable(publication, monkeypatch):
    from app.platform import replies
    original_timeout = asyncio.timeout
    def shortened(seconds):
        assert seconds == 8
        return original_timeout(0.02)
    async def slow(request):
        await asyncio.sleep(1)
        raise AssertionError("Total deadline not enforced")
    original_client = httpx.AsyncClient.__bases__[0]
    class Client(original_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(slow), **kwargs)
    monkeypatch.setattr(replies.asyncio, "timeout", shortened)
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    publish(publication)
    assert row(publication[0], publication[1]).state == "pending"


def test_conflict_body_deadline_cannot_be_readiness(publication, monkeypatch):
    from app.platform import replies
    original_timeout = asyncio.timeout
    def shortened(seconds):
        assert seconds == 8
        return original_timeout(0.02)
    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(1)
            yield b'{"detail":"ingress_publication_not_ready"}'
    original_client = httpx.AsyncClient.__bases__[0]
    class Client(original_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(lambda request:
                httpx.Response(409, headers={"Content-Type": "application/json"}, stream=SlowBody())), **kwargs)
    monkeypatch.setattr(replies.asyncio, "timeout", shortened)
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    publish(publication)
    assert row(publication[0], publication[1]).state == "failed"
