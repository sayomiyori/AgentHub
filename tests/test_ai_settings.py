"""Settings API and CAS against an explicitly isolated migrated PostgreSQL."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.v1.ai_settings import router
from app.api.v1.platform_reads import get_access_client, get_bot_client
from app.db.platform import require_platform_schema
from app.db.session import get_db
from app.models.ai_settings import AIModelSetting, AIModelSettingAudit
from app.models.telegram_job import TelegramAIJob
from app.platform.ai_settings import SettingsConflict, SettingsInput, effective_model, save_settings
from app.platform.bot_context import BotContextClient
from app.platform.config import PlatformSettings, get_platform_settings
from app.platform.jobs import claim_job
from app.platform.tenant_access import TenantAccessClient

URL = os.getenv("AGENTHUB_PLATFORM_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="Isolated migrated PostgreSQL is required")


@pytest.fixture()
def api():
    engine = create_engine(URL)
    with engine.connect() as connection:
        name = connection.scalar(text("SELECT current_database()"))
        assert name.startswith("nexus_agent_ai_") and name.endswith("_test")
    require_platform_schema(engine)
    config = PlatformSettings(
        _env_file=None,
        PLATFORM_READ_ENABLED=True,
        AUTHFORTRESS_BASE_URL="http://issuer.test",
        WEBHOOK_INTERNAL_URL="http://bots.test",
        WEBHOOK_AGENT_SERVICE_KEY="b" * 32,
        TELEGRAM_AI_ENABLED=False,
        TELEGRAM_AI_MODEL="platform-model",
    )
    tenant, bot, actor = uuid4(), uuid4(), uuid4()
    state = dict(
        calls=[],
        issuer_status=200,
        bot_status=200,
        queries=0,
        role="owner",
        context=dict(bot_id=str(bot), tenant_id=str(tenant), telegram_bot_id=123, is_active=True),
    )

    def handler(request):
        state["calls"].append(request)
        if request.url.host == "issuer.test":
            assert request.headers["authorization"] == "Bearer synthetic-settings"
            assert request.url.path == f"/api/v1/tenants/{tenant}/authorize"
            permission = json.loads(request.content)["permission"]
            receipt = dict(
                user_id=str(actor),
                tenant_id=state.get("receipt_tenant", str(tenant)),
                role=state["role"],
                permission=state.get("receipt_permission", permission),
                allowed=True,
            )
            return httpx.Response(state["issuer_status"], json=receipt)
        assert request.method == "GET" and request.url.path == f"/internal/v1/bots/{bot}/context"
        return httpx.Response(state["bot_status"], json=state["context"])

    def sessions():
        state["queries"] += 1
        with Session(engine) as session:
            yield session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_platform_settings] = lambda: config
    app.dependency_overrides[get_access_client] = lambda: TenantAccessClient(config, httpx.MockTransport(handler))
    app.dependency_overrides[get_bot_client] = lambda: BotContextClient(config, httpx.MockTransport(handler))
    app.dependency_overrides[get_db] = sessions
    with TestClient(app) as client:
        yield client, f"/api/v1/tenants/{tenant}", state, config, engine, tenant, bot, actor
    engine.dispose()


def call(api, method="GET", bot=False, **kwargs):
    return api[0].request(
        method,
        api[1] + (f"/bots/{api[6]}" if bot else "") + "/ai/settings",
        headers={"Authorization": "Bearer synthetic-settings"},
        **kwargs,
    )


def put(api, model, revision, bot=False):
    return call(api, "PUT", bot, json={"model": model, "expected_revision": revision})


def test_defaults_overrides_reset_and_restart_independent_persistence(api):
    initial = call(api).json()
    assert initial["saved"] == dict(model=None, revision=0, updated_at=None)
    assert initial["effective"] == dict(scope="platform", provider="groq", model="platform-model")
    assert initial["processing_enabled"] is False and initial["bot_id"] is None
    assert put(api, "tenant-model", 0).status_code == 200
    assert call(api, bot=True).json()["effective"]["model"] == "tenant-model"
    saved = put(api, "bot-model", 0, True)
    assert saved.status_code == 200 and saved.json()["effective"]["scope"] == "bot"
    assert saved.json()["saved"]["revision"] == 1
    assert put(api, None, 1, True).json()["effective"]["model"] == "tenant-model"
    assert put(api, None, 1).json()["effective"]["scope"] == "platform"
    with Session(api[4]) as db:
        assert effective_model(db, api[5], api[6], api[3]).model == "platform-model"
        assert (
            db.scalar(
                select(func.count()).select_from(AIModelSettingAudit).where(AIModelSettingAudit.tenant_id == api[5])
            )
            == 4
        )
        assert set(
            db.scalars(select(AIModelSettingAudit.updated_by).where(AIModelSettingAudit.tenant_id == api[5]))
        ) == {api[7]}
        assert effective_model(db, uuid4(), api[6], api[3]).scope == "platform"
        assert effective_model(db, api[5], uuid4(), api[3]).scope == "platform"


def test_stale_revision_conflicts_without_extra_audit_or_overwrite(api):
    assert put(api, "first", 0).status_code == 200
    for version in (0, 2):
        response = put(api, "stale", version)
        assert response.status_code == 409
    result = call(api).json()
    assert result["saved"]["model"] == "first" and result["saved"]["revision"] == 1
    with Session(api[4]) as db:
        assert (
            db.scalar(
                select(func.count()).select_from(AIModelSettingAudit).where(AIModelSettingAudit.tenant_id == api[5])
            )
            == 1
        )


@pytest.mark.parametrize("revision", [0, 1])
def test_two_writers_have_one_winner_and_one_atomic_audit_revision(api, revision):
    if revision:
        assert put(api, "initial", 0).status_code == 200
    barrier = Barrier(2)

    def writer(model):
        with Session(api[4]) as db:
            barrier.wait(timeout=10)
            try:
                save_settings(db, api[5], None, api[7], SettingsInput(model=model, expected_revision=revision))
                db.commit()
                return "saved"
            except SettingsConflict:
                db.rollback()
                return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(writer, ["first", "second"])) == ["conflict", "saved"]
    assert call(api).json()["saved"]["revision"] == revision + 1
    with Session(api[4]) as db:
        assert (
            db.scalar(
                select(func.count()).select_from(AIModelSettingAudit).where(AIModelSettingAudit.tenant_id == api[5])
            )
            == revision + 1
        )


def test_rollback_leaves_neither_settings_nor_audit(api):
    with Session(api[4]) as db:
        save_settings(db, api[5], None, api[7], SettingsInput(model="rolled-back", expected_revision=0))
        db.rollback()
    assert call(api).json()["saved"]["revision"] == 0
    with Session(api[4]) as db:
        assert (
            db.scalar(
                select(func.count()).select_from(AIModelSettingAudit).where(AIModelSettingAudit.tenant_id == api[5])
            )
            == 0
        )


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {},
        {"model": "x"},
        {"model": "", "expected_revision": 0},
        {"model": " x ", "expected_revision": 0},
        {"model": "<script>", "expected_revision": 0},
        {"model": "a" * 129, "expected_revision": 0},
        {"model": "x\n", "expected_revision": 0},
        {"model": "x", "expected_revision": True},
        {"model": "x", "expected_revision": "0"},
        {"model": "x", "expected_revision": -1},
        {"model": "x", "expected_revision": 2147483647},
        {"model": "x", "expected_revision": 0, "processing_enabled": True},
    ],
)
def test_bad_inputs_are_fixed_errors_before_database_access(api, bad):
    assert call(api, "PUT", json=bad).status_code == 400
    assert api[2]["queries"] == 0


@pytest.mark.parametrize(
    "body", ['{"model":"a","model":"b","expected_revision":0}', '{"model":NaN,"expected_revision":0}', "x" * 16385]
)
def test_duplicate_nonfinite_and_oversize_json_are_rejected(api, body):
    response = api[0].put(
        api[1] + "/ai/settings",
        content=body,
        headers={"Authorization": "Bearer synthetic-settings", "Content-Type": "application/json"},
    )
    assert response.status_code == 400 and api[2]["queries"] == 0
    assert response.json() == {"detail": "Invalid settings"}


@pytest.mark.parametrize("target", [False, True])
def test_member_reads_but_cannot_forge_successful_configure_receipt(api, target):
    api[2]["role"] = "member"
    assert call(api, bot=target).status_code == 200
    api[2]["queries"] = 0
    assert put(api, "x", 0, target).status_code == 503
    assert api[2]["queries"] == 0
    assert json.loads(api[2]["calls"][-1].content)["permission"] == "ai.configure"


@pytest.mark.parametrize(
    "change", ["revoked", "foreign", "permission", "inactive", "wrong_bot", "bot_outage", "disabled"]
)
def test_failed_authorization_never_opens_database(api, change):
    state = api[2]
    if change == "revoked":
        state["issuer_status"] = 401
    elif change == "foreign":
        state["receipt_tenant"] = str(uuid4())
    elif change == "permission":
        state["receipt_permission"] = "ai.read"
    elif change == "inactive":
        state["context"]["is_active"] = False
    elif change == "wrong_bot":
        state["context"]["bot_id"] = str(uuid4())
    elif change == "bot_outage":
        state["bot_status"] = 503
    else:
        api[3].read_enabled = False
    response = put(api, "x", 0, True)
    assert response.status_code in {401, 404, 503} and state["queries"] == 0


def test_claim_freezes_effective_model_and_retry_ignores_later_edits(api, monkeypatch):
    assert put(api, "tenant-model", 0).status_code == 200
    assert put(api, "bot-model", 0, True).status_code == 200
    # This exercises DB claims only; no process_job, queue or provider is invoked.
    config = api[3].model_copy(update={"telegram_ai_enabled": True})
    monkeypatch.setattr("app.platform.jobs.get_platform_settings", lambda: config)
    job = TelegramAIJob(
        id=uuid4(), event_id=uuid4(), tenant_id=api[5], bot_id=api[6], update_id=123, envelope={}, digest="a" * 64
    )
    with Session(api[4]) as db:
        db.add(job)
        db.commit()
        first = claim_job(db, job.id)
        db.commit()
        assert first.model == "bot-model"
        stored = db.get(TelegramAIJob, job.id)
        stored.state = "pending"
        stored.claim_id = None
        stored.lease_until = None
        db.commit()
        job_id = job.id
    assert put(api, "new-bot-model", 1, True).status_code == 200
    with Session(api[4]) as db:
        second = claim_job(db, job_id)
        db.commit()
        assert second.model == "bot-model"


@pytest.mark.parametrize("patch", [{"model": "bad\n"}, {"target": "tenant", "subject_id": uuid4()}, {"revision": 0}])
def test_database_constraints_enforce_scope_revision_and_identifier(api, patch):
    with Session(api[4]) as db:
        value = dict(tenant_id=api[5], target="tenant", subject_id=api[5], model="good", revision=1, updated_by=api[7])
        value.update(patch)
        with pytest.raises(IntegrityError):
            db.execute(AIModelSetting.__table__.insert().values(**value))
            db.commit()
