"""Tenant reads against PostgreSQL with issuer/context HTTP boundary doubles."""

import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.api.v1.platform_reads import get_access_client, get_bot_client, router
from app.db.platform import require_platform_schema
from app.db.session import get_db
from app.models.telegram_job import TelegramAIJob
from app.models.telegram_usage import TelegramAIUsage
from app.platform.bot_context import BotContextClient
from app.platform.config import PlatformSettings, get_platform_settings
from app.platform.tenant_access import TenantAccessClient

URL = os.getenv("AGENTHUB_PLATFORM_TEST_DATABASE_URL")


def settings(**overrides):
    values = dict(PLATFORM_READ_ENABLED=True, AUTHFORTRESS_BASE_URL="http://issuer.test",
                  WEBHOOK_INTERNAL_URL="http://webhook.test", WEBHOOK_AGENT_SERVICE_KEY="b" * 32)
    values.update(overrides)
    return PlatformSettings(_env_file=None, **values)


@pytest.fixture()
def api():
    if not URL:
        pytest.skip("Isolated migrated PostgreSQL is required")
    engine = create_engine(URL)
    with engine.connect() as connection:
        name = connection.scalar(text("SELECT current_database()"))
        assert name.startswith("nexus_agent_ai_") and name.endswith("_test")
        require_platform_schema(engine)
        transaction = connection.get_transaction()
        config = settings()
        tenant, bot = uuid4(), uuid4()
        state = dict(issuer_status=200, bot_status=200, calls=[], queries=0,
                     receipt=dict(user_id=str(uuid4()), tenant_id=str(tenant), role="owner",
                                  permission="ai.read", allowed=True),
                     context=dict(bot_id=str(bot), tenant_id=str(tenant), telegram_bot_id=123, is_active=True))

        def handler(request):
            state['calls'].append(request)
            if request.url.host == "issuer.test":
                assert request.method == "POST" and json.loads(request.content) == {"permission": "ai.read"}
                assert request.headers['authorization'] == 'Bearer synthetic-user-token'
                assert request.url.path == f'/api/v1/tenants/{tenant}/authorize'
                if state.get('timeout'):
                    raise httpx.ReadTimeout('private upstream error synthetic-user-token')
                return httpx.Response(state['issuer_status'],
                                      content=state.get('raw', json.dumps(state['receipt']).encode()),
                                      headers=state.get('headers', {'Content-Type': 'application/json'}))
            assert request.url.host == 'webhook.test'
            assert 'authorization' not in request.headers
            assert request.headers['X-Service-Key'] == 'b' * 32
            return httpx.Response(state['bot_status'], json=state['context'])

        def sessions():
            state['queries'] += 1
            with Session(connection, join_transaction_mode='create_savepoint') as db:
                yield db

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_platform_settings] = lambda: config
        app.dependency_overrides[get_access_client] = lambda: TenantAccessClient(config, httpx.MockTransport(handler))
        app.dependency_overrides[get_bot_client] = lambda: BotContextClient(config, httpx.MockTransport(handler))
        app.dependency_overrides[get_db] = sessions
        with TestClient(app) as client:
            yield client, f'/api/v1/tenants/{tenant}/bots/{bot}/ai', state, connection, tenant, bot, config
        transaction.rollback()
    engine.dispose()


def get(api, path='jobs', **kwargs):
    client, base, *_ = api
    return client.get(base + '/' + path, headers={'Authorization': 'Bearer synthetic-user-token'}, **kwargs)


def seed(connection, tenant, bot, number, at):
    item = dict(id=UUID(int=number), event_id=uuid4(), tenant_id=tenant, bot_id=bot,
                update_id=number, envelope={'private': 'private-message'}, digest='a' * 64,
                answer='private-answer', created_at=at)
    connection.execute(TelegramAIJob.__table__.insert().values(**item))
    connection.execute(TelegramAIUsage.__table__.insert().values(
        job_id=item['id'], tenant_id=tenant, bot_id=bot, provider='groq', model='test',
        input_tokens=10, output_tokens=5, estimated_cost_usd=Decimal('0.00000003'), created_at=at))


@pytest.mark.parametrize('role', ['owner', 'member'])
def test_scoped_pagination_and_usage(api, role):
    _, _, state, connection, tenant, bot, _ = api
    state['receipt']['role'] = role
    start = datetime(2026, 10, 1, tzinfo=UTC)
    seed(connection, tenant, bot, 1, start)
    seed(connection, tenant, bot, 2, start + timedelta(hours=1))
    seed(connection, tenant, bot, 3, start + timedelta(days=1))
    seed(connection, uuid4(), uuid4(), 4, start)
    seed(connection, tenant, uuid4(), 5, start)
    # Even an inconsistent historic tenant/bot pair cannot bleed into this scope.
    seed(connection, uuid4(), bot, 6, start)
    first = get(api, params={'limit': 2})
    assert first.status_code == 200
    assert [x['id'] for x in first.json()['items']] == [str(UUID(int=1)), str(UUID(int=2))]
    assert first.json()['next_cursor'] == str(UUID(int=2))
    second = get(api, params={'limit': 2, 'cursor': first.json()['next_cursor']})
    assert [x['id'] for x in second.json()['items']] == [str(UUID(int=3))]
    assert second.json()['next_cursor'] is None
    assert not any(word in first.text for word in ('private-message', 'private-answer', 'envelope', 'claim_id'))
    usage = get(api, 'usage', params={'start': start.isoformat(), 'end': (start + timedelta(days=1)).isoformat()})
    assert usage.status_code == 200
    result = usage.json()
    assert isinstance(result['estimated_cost_usd'], str)
    assert Decimal(result.pop('estimated_cost_usd')) == Decimal('0.00000006')
    assert result == dict(records=2, input_tokens=20, output_tokens=10)
    assert len(state['calls']) == 6  # Authorization and bot lookup are fresh for every read.


def test_empty_scope(api):
    assert get(api).json() == {'items': [], 'next_cursor': None}
    result = get(api, 'usage', params={'start': '2026-10-01T00:00:00Z', 'end': '2026-10-02T00:00:00Z'})
    assert result.status_code == 200
    assert result.json()['records'] == result.json()['input_tokens'] == result.json()['output_tokens'] == 0
    assert Decimal(result.json()['estimated_cost_usd']) == 0


@pytest.mark.parametrize('headers', [[], [('Authorization', 'Basic x')], [('Authorization', 'Bearer')],
    [('Authorization', 'Bearer x y')], [('Authorization', 'Bearer x'), ('Authorization', 'Bearer y')],
    [('Authorization', 'Bearer ' + 'x' * 8193)]])
def test_bad_bearer_never_reads_or_contacts_upstream(api, headers):
    client, base, state, *_ = api
    assert client.get(base + '/jobs', headers=headers).status_code == 401
    assert not state['calls'] and state['queries'] == 0


def test_disabled_requires_no_external_work(api):
    *_, config = api
    config.read_enabled = False
    assert get(api).status_code == 404
    assert api[2]['calls'] == [] and api[2]['queries'] == 0


@pytest.mark.parametrize('status,expected', [(401, 401), (403, 404), (404, 404), (429, 503),
                                           (500, 503), (302, 503)])
def test_issuer_denial_and_outage(api, status, expected):
    state = api[2]
    state['issuer_status'] = status
    result = get(api)
    assert result.status_code == expected
    assert len(state['calls']) == 1 and state['queries'] == 0
    assert 'synthetic-user-token' not in result.text


@pytest.mark.parametrize('case', ['tenant', 'allowed', 'boolean', 'role', 'permission', 'extra', 'missing',
                                 'oversize', 'duplicate', 'invalid_json', 'mime', 'encoding', 'timeout'])
def test_untrusted_receipt_fails_closed(api, case):
    state = api[2]
    if case == 'tenant':
        state['receipt']['tenant_id'] = str(uuid4())
    elif case == 'allowed':
        state['receipt']['allowed'] = False
    elif case == 'boolean':
        state['receipt']['allowed'] = 1
    elif case == 'role':
        state['receipt']['role'] = 'superadmin'
    elif case == 'permission':
        state['receipt']['permission'] = 'bot.read'
    elif case == 'extra':
        state['receipt']['token'] = 'private'
    elif case == 'missing':
        del state['receipt']['user_id']
    elif case == 'oversize':
        state['raw'] = b' ' * 65537
    elif case == 'duplicate':
        state['raw'] = json.dumps(state['receipt']).replace(
            '"allowed": true', '"allowed": false,"allowed": true').encode()
    elif case == 'invalid_json':
        state['raw'] = b'private upstream error'
    elif case == 'mime':
        state['headers'] = {'Content-Type': 'text/plain'}
    elif case == 'encoding':
        state['headers'] = {'Content-Type': 'application/json', 'Content-Encoding': 'gzip'}
    else:
        state['timeout'] = True
    response = get(api)
    assert response.status_code == 503
    assert response.json() == {'detail': 'Authorization unavailable'}
    assert state['queries'] == 0 and len(state['calls']) == 1


@pytest.mark.parametrize('case,expected', [('tenant', 404), ('bot', 404), ('inactive', 404),
                                        ('denied', 404), ('outage', 503)])
def test_bot_scope_denied_before_database(api, case, expected):
    state = api[2]
    if case in {'tenant', 'bot'}:
        state['context'][case + '_id'] = str(uuid4())
    elif case == 'inactive':
        state['context']['is_active'] = False
    else:
        state['bot_status'] = 403 if case == 'denied' else 500
    assert get(api).status_code == expected
    assert state['queries'] == 0


def test_revocation_is_observed_on_next_request(api):
    assert get(api).status_code == 200
    api[2]['issuer_status'] = 401
    assert get(api).status_code == 401
    assert api[2]['queries'] == 1


@pytest.mark.parametrize('params', [{'limit': 0}, {'limit': 101}, {'cursor': 'bad'}])
def test_invalid_page(api, params):
    assert get(api, params=params).status_code == 422


@pytest.mark.parametrize('start,end', [
    ('2026-10-01', '2026-10-02'), ('2026-10-01T00:00:00Z', '2026-10-01T00:00:00Z'),
    ('2026-10-02T00:00:00Z', '2026-10-01T00:00:00Z'),
    ('2026-10-01T00:00:00Z', '2026-12-01T00:00:00Z'),
])
def test_invalid_usage_interval(api, start, end):
    assert get(api, 'usage', params={'start': start, 'end': end}).status_code == 422


@pytest.mark.parametrize('overrides', [
    {'AUTHFORTRESS_BASE_URL': ''}, {'AUTHFORTRESS_BASE_URL': 'https://user:pass@issuer.test'},
    {'AUTHFORTRESS_BASE_URL': 'http://issuer.test/path'}, {'AUTHFORTRESS_BASE_URL': 'http://issuer.test:0'},
    {'AUTHFORTRESS_BASE_URL': 'http://issuer.test\n'}, {'WEBHOOK_AGENT_SERVICE_KEY': ''},
])
def test_read_configuration_rejects_invalid_origins_and_missing_key(overrides):
    with pytest.raises(ValidationError):
        settings(**overrides)


def test_reads_do_not_require_generation_secrets_or_model():
    config = settings()
    assert config.read_enabled and not config.telegram_ai_enabled
    assert not config.ingress_key.get_secret_value() and not config.model
