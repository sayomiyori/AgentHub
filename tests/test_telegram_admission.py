"""Signed durable admission against isolated PostgreSQL; outbound HTTP is the boundary double."""

import hashlib
import hmac
import json
import os
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from app.api.internal.telegram import get_context_client, get_job_notifier, get_session_factory, router
from app.models.telegram_job import TelegramAIJob
from app.platform.bot_context import BotContextClient
from app.platform.config import PlatformSettings, get_platform_settings

URL = os.getenv("AGENTHUB_PLATFORM_TEST_DATABASE_URL")
KEY = "a" * 32


def envelope():
    bot = uuid4()
    return dict(
        event_id=str(uuid4()),
        event_type="telegram.message.received",
        schema_version=1,
        occurred_at=datetime.now(UTC).isoformat(),
        tenant_id=str(uuid4()),
        bot_id=str(bot),
        correlation_id=str(uuid4()),
        idempotency_key=f"telegram:{bot}:42",
        payload=dict(update_id=42, chat_id=43, message_id=1, question="Synthetic question"),
    )


def signed(body):
    return {
        "Content-Type": "application/json",
        "X-Webhook-Signature": "sha256=" + hmac.new(KEY.encode(), body, hashlib.sha256).hexdigest(),
    }


@pytest.fixture()
def admission():
    if not URL:
        pytest.skip("Isolated migrated PostgreSQL is required")
    engine = create_engine(URL)
    with engine.connect() as connection:
        name = connection.scalar(text("SELECT current_database()"))
        assert name.startswith("nexus_agent_ai_") and name.endswith("_test")
        transaction = connection.get_transaction()
        config = PlatformSettings(
            _env_file=None,
            TELEGRAM_AI_ENABLED=True,
            TELEGRAM_AI_MODEL="test",
            WEBHOOK_INTERNAL_URL="http://webhook_service:8000",
            WEBHOOK_AGENT_INGRESS_KEY=KEY,
            WEBHOOK_AGENT_SERVICE_KEY="b" * 32,
            AGENT_WEBHOOK_REPLY_KEY="c" * 32,
        )
        item = envelope()
        outbound = []
        status = [200]
        context = dict(bot_id=item["bot_id"], tenant_id=item["tenant_id"], telegram_bot_id=123, is_active=True)

        def handler(request):
            outbound.append(request)
            return httpx.Response(status[0], json=context)

        def sessions():
            return Session(connection, join_transaction_mode="create_savepoint", expire_on_commit=False)

        notifications = []
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_platform_settings] = lambda: config
        app.dependency_overrides[get_context_client] = lambda: BotContextClient(config, httpx.MockTransport(handler))
        app.dependency_overrides[get_session_factory] = lambda: sessions
        app.dependency_overrides[get_job_notifier] = lambda: notifications.append
        with TestClient(app) as client:
            yield client, item, context, status, outbound, notifications, connection, app
        transaction.rollback()
    engine.dispose()


def post(client, item):
    body = json.dumps(item).encode()
    return client.post("/internal/v1/telegram/updates", content=body, headers=signed(body))


def test_first_admission_replay_and_conflict(admission):
    client, item, _, _, outbound, notifications, connection, _ = admission
    first = post(client, item)
    assert first.status_code == 202
    assert set(first.json()) == {"event_id", "job_id", "state"}
    assert first.json()["state"] == "pending"
    assert post(client, item).json() == first.json()
    assert post(client, item).status_code == 200
    changed = json.loads(json.dumps(item))
    changed["payload"]["question"] = "Changed synthetic question"
    assert post(client, changed).status_code == 409
    assert connection.scalar(select(TelegramAIJob.id).where(TelegramAIJob.event_id == item["event_id"]))
    assert outbound[0].headers["X-Service-Key"] == "b" * 32
    assert notifications


@pytest.mark.parametrize("kind,code", [("tenant", 403), ("inactive", 403), ("missing", 403), ("outage", 503)])
def test_fresh_canonical_context_required(admission, kind, code):
    client, item, context, status, _, notifications, connection, _ = admission
    if kind == "tenant":
        context["tenant_id"] = str(uuid4())
    elif kind == "inactive":
        context["is_active"] = False
    else:
        status[0] = 404 if kind == "missing" else 503
    assert post(client, item).status_code == code
    assert not notifications
    assert connection.scalar(select(TelegramAIJob.id).where(TelegramAIJob.event_id == item["event_id"])) is None


def test_signature_precedes_parsing(admission):
    client, _, _, _, outbound, _, _, _ = admission
    assert (
        client.post(
            "/internal/v1/telegram/updates",
            content=b"malformed",
            headers={"Content-Type": "application/json", "X-Webhook-Signature": "sha256=" + "0" * 64},
        ).status_code
        == 401
    )
    assert not outbound


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("extra", 1),
        ("idempotency_key", "wrong"),
        ("occurred_at", "2026-10-06T00:00:00"),
    ],
)
def test_invalid_envelope_never_admitted(admission, field, value):
    client, item, _, _, outbound, _, _, _ = admission
    item[field] = value
    assert post(client, item).status_code == 422
    assert not outbound


@pytest.mark.parametrize(
    "field,value", [("update_id", True), ("chat_id", 2**63), ("message_id", 0), ("question", " "), ("extra", 1)]
)
def test_invalid_payload_never_admitted(admission, field, value):
    client, item, _, _, outbound, _, _, _ = admission
    item["payload"][field] = value
    assert post(client, item).status_code == 422
    assert not outbound


@pytest.mark.parametrize("body", [b'{"event_id":1,"event_id":2}', b'{"x":NaN}', b"[]"])
def test_invalid_json_is_sanitized(admission, body):
    client = admission[0]
    response = client.post("/internal/v1/telegram/updates", content=body, headers=signed(body))
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid Telegram event"}


def test_broker_outage_keeps_committed_job(admission):
    client, item, _, _, _, _, connection, app = admission

    def unavailable(job_id):
        assert connection.scalar(select(TelegramAIJob.id).where(TelegramAIJob.id == job_id))
        raise ConnectionError("synthetic broker outage")

    app.dependency_overrides[get_job_notifier] = lambda: unavailable
    assert post(client, item).status_code == 202
    assert connection.scalar(select(TelegramAIJob.state).where(TelegramAIJob.event_id == item["event_id"])) == "pending"


def test_body_size_boundary(admission):
    client, item, _, _, _, _, _, _ = admission
    body = json.dumps(item).encode()
    exact = body + b" " * (1024 * 1024 - len(body))
    assert client.post("/internal/v1/telegram/updates", content=exact, headers=signed(exact)).status_code == 202
    overflow = exact + b" "
    assert client.post("/internal/v1/telegram/updates", content=overflow, headers=signed(overflow)).status_code == 413


def test_disabled_endpoint_performs_no_context_lookup(admission):
    client, item, _, _, outbound, notifications, _, app = admission
    app.dependency_overrides[get_platform_settings] = lambda: PlatformSettings(_env_file=None)
    assert post(client, item).status_code == 503
    assert not outbound and not notifications


def test_duplicate_signature_headers_rejected(admission):
    client, item, _, _, outbound, _, _, _ = admission
    body = json.dumps(item).encode()
    headers = list(signed(body).items()) + [("X-Webhook-Signature", signed(body)["X-Webhook-Signature"])]
    assert client.post("/internal/v1/telegram/updates", content=body, headers=headers).status_code == 401
    assert not outbound


@pytest.mark.parametrize("header,value", [("Content-Type", "text/plain"), ("Content-Encoding", "gzip")])
def test_unsupported_content_rejected(admission, header, value):
    client, item, _, _, outbound, _, _, _ = admission
    body = json.dumps(item).encode()
    headers = signed(body)
    headers[header] = value
    assert client.post("/internal/v1/telegram/updates", content=body, headers=headers).status_code == 415
    assert not outbound


def test_sql_failure_returns_safe_503(admission):
    client, item, _, _, _, notifications, connection, app = admission
    engine = create_engine(URL)
    broken = engine.connect()
    broken.close()
    app.dependency_overrides[get_session_factory] = lambda: lambda: Session(broken)
    response = post(client, item)
    assert response.status_code == 503
    assert response.json() == {"detail": "Job storage unavailable"}
    assert not notifications
    assert connection.scalar(select(TelegramAIJob.id).where(TelegramAIJob.event_id == item["event_id"])) is None
    engine.dispose()


def test_concurrent_admission_has_one_committed_job():
    if not URL:
        pytest.skip("Isolated migrated PostgreSQL required")
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from app.platform.jobs import persist_admission
    from app.platform.schemas import IngressEnvelope

    engine = create_engine(URL)
    with engine.connect() as connection:
        name = connection.scalar(text("SELECT current_database()"))
        assert name.startswith("nexus_agent_ai_") and name.endswith("_test")
    item = IngressEnvelope.model_validate_json(json.dumps(envelope()))
    barrier = Barrier(2)

    def writer():
        barrier.wait(timeout=5)
        return persist_admission(lambda: Session(engine), item, "a" * 64)

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(writer) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert sum(created for receipt, created in results) == 1
    assert results[0][0] == results[1][0]
    with Session(engine) as db:
        assert len(list(db.scalars(select(TelegramAIJob).where(TelegramAIJob.event_id == item.event_id)))) == 1
    engine.dispose()


@pytest.mark.parametrize("mode,expected", [("disconnect", 400), ("overflow", 413)])
def test_stream_stops_on_disconnect_or_overflow(admission, mode, expected):
    import asyncio

    client, item, _, _, outbound, notifications, _, app = admission
    body = json.dumps(item).encode()
    chunks = [
        dict(type="http.request", body=b"{" if mode == "disconnect" else b" " * (1024 * 1024 + 1), more_body=True),
        dict(type="http.disconnect"),
    ]
    reads = []
    sent = []

    async def receive():
        reads.append(1)
        return chunks.pop(0)

    async def send(message):
        sent.append(message)

    scope = dict(
        type="http",
        asgi={"version": "3.0"},
        http_version="1.1",
        method="POST",
        path="/internal/v1/telegram/updates",
        raw_path=b"/internal/v1/telegram/updates",
        root_path="",
        scheme="http",
        query_string=b"",
        server=("testserver", 80),
        client=("127.0.0.1", 1),
        headers=[(k.lower().encode(), v.encode()) for k, v in signed(body).items()],
    )
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == expected
    assert len(reads) == (2 if mode == "disconnect" else 1)
    assert not outbound and not notifications


@pytest.mark.parametrize("kind", ["redirect", "encoded", "large", "duplicate", "malformed", "timeout"])
def test_invalid_context_response_never_admitted(admission, kind):
    client, item, context, _, _, notifications, connection, app = admission
    config = app.dependency_overrides[get_platform_settings]()

    def response(request):
        if kind == "timeout":
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        if kind == "redirect":
            return httpx.Response(302, headers={"Location": "https://example.com"})
        if kind == "encoded":
            return httpx.Response(200, headers={"Content-Encoding": "gzip"})
        if kind == "large":
            return httpx.Response(200, content=b" " * 65537, headers={"Content-Type": "application/json"})
        if kind == "duplicate":
            body = b'{"bot_id":1,"bot_id":2}'
        else:
            body = b'{"is_active":true}'
        return httpx.Response(200, content=body, headers={"Content-Type": "application/json"})

    app.dependency_overrides[get_context_client] = lambda: BotContextClient(config, httpx.MockTransport(response))
    assert post(client, item).status_code == 503
    assert not notifications
    assert connection.scalar(select(TelegramAIJob.id).where(TelegramAIJob.event_id == item["event_id"])) is None
