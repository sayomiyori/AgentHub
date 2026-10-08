"""Platform startup is fail-closed without disturbing standalone data."""

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect


def test_legacy_metadata_excludes_platform_tables():
    import app.models.chunk  # noqa: F401
    import app.models.conversation  # noqa: F401
    import app.models.document  # noqa: F401
    import app.models.llm_usage  # noqa: F401
    import app.models.telegram_job  # noqa: F401
    import app.models.telegram_reply  # noqa: F401
    import app.models.telegram_usage  # noqa: F401
    from app.db.platform import PlatformBase
    from app.db.session import Base

    assert set(Base.metadata.tables) == {
        "documents",
        "chunks",
        "conversations",
        "messages",
        "llm_usage_records",
    }
    assert set(Base.metadata.tables).isdisjoint(PlatformBase.metadata.tables)


def test_enabled_startup_requires_migration_head():
    from app.db.platform import require_platform_schema

    engine = create_engine("sqlite://")
    with pytest.raises(RuntimeError, match="Platform migrations are required"):
        require_platform_schema(engine)
    assert inspect(engine).get_table_names() == []


def test_disabled_platform_requires_no_credentials():
    from app.platform.config import PlatformSettings

    settings = PlatformSettings(_env_file=None)
    assert settings.telegram_ai_enabled is False


def test_environment_flag_accepts_compose_false(monkeypatch):
    from app.platform.config import PlatformSettings

    monkeypatch.setenv("TELEGRAM_AI_ENABLED", "false")
    assert PlatformSettings(_env_file=None).telegram_ai_enabled is False


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"WEBHOOK_INTERNAL_URL": "https://example.com/path"},
        {"TELEGRAM_AI_PROVIDER": "openai"},
    ],
)
def test_enabled_platform_rejects_invalid_configuration(overrides):
    from app.platform.config import PlatformSettings

    with pytest.raises(ValidationError):
        PlatformSettings(_env_file=None, TELEGRAM_AI_ENABLED=True, **overrides)


def test_enabled_platform_accepts_independent_configuration():
    from app.platform.config import PlatformSettings

    settings = PlatformSettings(_env_file=None, TELEGRAM_AI_ENABLED=True,
                                TELEGRAM_AI_MODEL="demo-model", WEBHOOK_INTERNAL_URL="http://webhook_service:8000",
                                WEBHOOK_AGENT_INGRESS_KEY="a" * 32, WEBHOOK_AGENT_SERVICE_KEY="b" * 32,
                                AGENT_WEBHOOK_REPLY_KEY="c" * 32)
    assert settings.telegram_ai_enabled


@pytest.mark.parametrize("override", [
    {"WEBHOOK_INTERNAL_URL": "https://example.com/path"},
    {"WEBHOOK_INTERNAL_URL": "http://localhost:invalid"},
    {"WEBHOOK_INTERNAL_URL": "http://localhost:99999"},
    {"WEBHOOK_INTERNAL_URL": "http://localhost:"},
    {"WEBHOOK_INTERNAL_URL": "http://user:password@example.com"},
    {"WEBHOOK_AGENT_SERVICE_KEY": "a" * 32},
    {"AGENT_WEBHOOK_REPLY_KEY": "short"},
    {"TELEGRAM_AI_MODEL": " padded "},
])
def test_enabled_platform_rejects_invalid_field_with_other_fields_valid(override):
    from app.platform.config import PlatformSettings

    values = dict(TELEGRAM_AI_ENABLED=True, TELEGRAM_AI_MODEL="demo-model",
                  WEBHOOK_INTERNAL_URL="http://webhook_service:8000",
                  WEBHOOK_AGENT_INGRESS_KEY="a" * 32, WEBHOOK_AGENT_SERVICE_KEY="b" * 32,
                  AGENT_WEBHOOK_REPLY_KEY="c" * 32)
    values.update(override)
    with pytest.raises(ValidationError):
        PlatformSettings(_env_file=None, **values)


@pytest.mark.parametrize("enabled,read,reject", [
    (False, False, False), (True, False, False), (True, False, True),
    (False, True, False), (False, True, True),
])
def test_startup_validates_platform_before_legacy_schema(monkeypatch, enabled, read, reject):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app import main

    calls = []
    monkeypatch.setattr(main, "get_platform_settings",
                        lambda: SimpleNamespace(telegram_ai_enabled=enabled, read_enabled=read))
    monkeypatch.setattr(main.Base.metadata, "create_all", lambda **_: calls.append("legacy"))
    monkeypatch.setattr(main, "_refresh_storage_gauges", lambda: None)
    monkeypatch.setattr(main.mcp_client_manager, "startup", AsyncMock())
    monkeypatch.setattr(main.mcp_client_manager, "shutdown", AsyncMock())

    def validate(engine):
        calls.append("revision")
        if reject:
            raise RuntimeError("Platform migrations are required")

    monkeypatch.setattr(main, "require_platform_schema", validate)

    async def start():
        async with main.lifespan(main.app):
            calls.append("ready")

    if reject:
        with pytest.raises(RuntimeError, match="Platform migrations are required"):
            asyncio.run(start())
        assert calls == ["revision"]
    else:
        asyncio.run(start())
        assert calls == (["revision"] if enabled or read else []) + ["legacy", "ready"]
