"""Read-only configuration through real issuer/context HTTP clients, without a DB."""

import json
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.platform_reads import get_access_client, get_bot_client, router
from app.db.session import get_db
from app.platform.bot_context import BotContextClient
from app.platform.config import PlatformSettings, get_platform_settings
from app.platform.tenant_access import TenantAccessClient


@pytest.fixture()
def api():
    config = PlatformSettings(_env_file=None, PLATFORM_READ_ENABLED=True,
                              AUTHFORTRESS_BASE_URL="http://issuer.test", WEBHOOK_INTERNAL_URL="http://bots.test",
                              WEBHOOK_AGENT_SERVICE_KEY="b" * 32, TELEGRAM_AI_ENABLED=False, TELEGRAM_AI_MODEL="")
    tenant, bot = uuid4(), uuid4()
    state = dict(calls=[], issuer_status=200, bot_status=200,
                 receipt=dict(user_id=str(uuid4()), tenant_id=str(tenant), role="owner",
                              permission="ai.read", allowed=True),
                 context=dict(bot_id=str(bot), tenant_id=str(tenant), telegram_bot_id=123, is_active=True))

    def handler(request):
        state['calls'].append(request)
        if request.url.host == 'issuer.test':
            assert request.method == 'POST'
            assert request.url.path == f'/api/v1/tenants/{tenant}/authorize'
            assert json.loads(request.content) == {'permission': 'ai.read'}
            assert request.headers['authorization'] == 'Bearer synthetic-configuration'
            return httpx.Response(state['issuer_status'], json=state['receipt'])
        assert request.method == 'GET'
        assert request.url.path == f'/internal/v1/bots/{bot}/context'
        assert request.headers['X-Service-Key'] == 'b' * 32
        return httpx.Response(state['bot_status'], json=state['context'])

    def no_database():
        raise AssertionError('Configuration must not access a database')

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_platform_settings] = lambda: config
    app.dependency_overrides[get_access_client] = lambda: TenantAccessClient(config, httpx.MockTransport(handler))
    app.dependency_overrides[get_bot_client] = lambda: BotContextClient(config, httpx.MockTransport(handler))
    app.dependency_overrides[get_db] = no_database
    with TestClient(app) as client:
        yield client, f'/api/v1/tenants/{tenant}/bots/{bot}/ai/configuration', state, config, tenant, bot


def read(api, **kwargs):
    return api[0].get(api[1], headers={'Authorization': 'Bearer synthetic-configuration'}, **kwargs)


@pytest.mark.parametrize('role', ['owner', 'member'])
@pytest.mark.parametrize('model', ['', 'llama-3.3-70b-versatile', 'org/model:version_1.2'])
def test_configuration_returns_only_authorized_shared_policy(api, role, model):
    _, _, state, config, tenant, bot = api
    state['receipt']['role'] = role
    config.model = model
    response = read(api)
    assert response.status_code == 200
    assert response.json() == dict(tenant_id=str(tenant), bot_id=str(bot), scope='platform', provider='groq',
                                   model=model or None, processing_enabled=False)
    assert len(state['calls']) == 2
    assert 'synthetic-configuration' not in response.text
    assert 'bbbbbbbb' not in response.text


def test_configuration_processing_flag_is_metadata_not_execution(api):
    api[3].telegram_ai_enabled = True
    api[3].model = 'synthetic-model'
    assert read(api).json()['processing_enabled'] is True
    assert len(api[2]['calls']) == 2


@pytest.mark.parametrize('model', [' ', ' private-secret ', '<script>', 'a' * 129, 'модель', 'line\nsecret'])
def test_configuration_invalid_model_fails_without_disclosing_settings(api, model):
    api[3].model = model
    response = read(api)
    assert response.status_code == 503
    assert response.json() == {'detail': 'Configuration unavailable'}


def test_configuration_disabled_and_missing_or_duplicate_bearer_precede_upstreams(api):
    client, path, state, config, *_ = api
    config.read_enabled = False
    assert read(api).status_code == 404
    config.read_enabled = True
    assert client.get(path).status_code == 401
    assert client.get(path, headers=[('Authorization', 'Bearer a'), ('Authorization', 'Bearer b')]).status_code == 401
    assert state['calls'] == []
    assert client.post(path).status_code == 405


@pytest.mark.parametrize('status,expected', [(401, 401), (403, 404), (404, 404), (503, 503)])
def test_configuration_issuer_denial_stops_before_bot_lookup(api, status, expected):
    api[2]['issuer_status'] = status
    response = read(api)
    assert response.status_code == expected
    assert len(api[2]['calls']) == 1


@pytest.mark.parametrize('change', ['tenant', 'bot', 'inactive', 'issuer_receipt', 'bot_outage'])
def test_configuration_rejects_foreign_inactive_or_unavailable_context(api, change):
    state = api[2]
    if change in {'tenant', 'bot'}:
        state['context'][f'{change}_id'] = str(uuid4())
    elif change == 'inactive':
        state['context']['is_active'] = False
    elif change == 'issuer_receipt':
        state['receipt']['tenant_id'] = str(uuid4())
    else:
        state['bot_status'] = 503
    response = read(api)
    assert response.status_code == (503 if change in {'issuer_receipt', 'bot_outage'} else 404)
    assert len(state['calls']) == (1 if change == 'issuer_receipt' else 2)


def test_configuration_reauthorizes_each_read_and_denies_revoked_session(api):
    assert read(api).status_code == 200
    api[2]['issuer_status'] = 401
    assert read(api).status_code == 401
    assert len(api[2]['calls']) == 3
