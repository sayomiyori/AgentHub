"""Fail closed on incomplete migration histories and global resource routes."""

from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from starlette.requests import Request

from app.db.platform import require_platform_schema


def test_resource_and_settings_preserve_their_distinct_issuer_denials():
    import asyncio

    from app.platform.config import PlatformSettings
    from app.platform.resource_access import _scope
    from app.platform.tenant_access import TenantAccessClient

    tenant = uuid4()
    config = PlatformSettings(_env_file=None, PLATFORM_TENANT_ENABLED=True,
                              AUTHFORTRESS_BASE_URL="http://issuer.test")
    issuer = TenantAccessClient(config, httpx.MockTransport(lambda _: httpx.Response(403)))
    request = Request({"type": "http", "path": "/", "method": "GET",
                       "headers": [(b"authorization", b"Bearer synthetic")],
                       "path_params": {"tenant_id": str(tenant)}})
    with pytest.raises(HTTPException) as resource:
        asyncio.run(_scope(request, config, issuer, "ai.configure"))
    assert resource.value.status_code == 403
    with pytest.raises(HTTPException) as settings:
        asyncio.run(issuer.authorize(tenant, SecretStr("synthetic"), "ai.configure"))
    assert settings.value.status_code == 404


@pytest.mark.parametrize(
    "revisions,accepted",
    [
        ([], False),
        (["003_tenant_resources"], False),
        (["003_ai_settings"], False),
        (["003_tenant_resources", "003_ai_settings"], False),
        (["004_merge_ai_resources"], True),
    ],
)
def test_platform_requires_the_single_merged_revision(revisions, accepted):
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"))
        for revision in revisions:
            connection.execute(text("INSERT INTO alembic_version VALUES (:revision)"), {"revision": revision})
    if accepted:
        require_platform_schema(engine)
    else:
        with pytest.raises(RuntimeError, match="Platform migrations are required"):
            require_platform_schema(engine)
    engine.dispose()


@pytest.mark.parametrize("mode", ["telegram_ai_enabled", "read_enabled", "tenant_enabled"])
def test_platform_modes_close_unscoped_routes_without_database(monkeypatch, mode):
    from app import main
    from app.platform.config import PlatformSettings

    config = PlatformSettings(_env_file=None)
    setattr(config, mode, True)
    monkeypatch.setattr(main, "get_platform_settings", lambda: config)
    client = TestClient(main.app)
    for path in ("/api/v1/documents", "/api/v1/query", "/api/v1/conversations/1/messages",
                 "/api/v1/usage/stats", "/metrics", "/mcp/sse", "/mcp/messages/", "/mcp"):
        assert client.get(path).status_code == 404
        assert client.post(path).status_code == 404
