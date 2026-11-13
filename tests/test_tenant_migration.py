"""Rehearse only in a newly created disposable PostgreSQL database."""

import os
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from app.config import get_settings
from app.db.platform import require_resource_schema

URL = os.getenv("AGENTHUB_PLATFORM_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="Isolated migrated PostgreSQL is required")


def test_upgrade_downgrade_upgrade_preserves_legacy_rows_without_assigning_tenants(monkeypatch):
    source = make_url(URL)
    assert source.database.startswith("nexus_agent_ai_") and source.database.endswith("_test")
    name = f"nexus_agent_ai_migration_{uuid4().hex}_test"
    admin = create_engine(source, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    target = source.set(database=name)
    engine = create_engine(target)
    monkeypatch.setenv("DATABASE_URL", target.render_as_string(hide_password=False))
    get_settings.cache_clear()
    config = Config("alembic.ini")
    try:
        command.upgrade(config, "002_telegram_ai")
        with pytest.raises(RuntimeError, match="Resource migrations are required"):
            require_resource_schema(engine)
        with engine.begin() as connection:
            connection.execute(text("""INSERT INTO documents
                (id,title,content_type,upload_status,chunk_count,file_path)
                VALUES (123,'legacy','txt','pending',0,'unused')"""))
            connection.execute(text("INSERT INTO conversations (id,title) VALUES (123,'legacy')"))
            connection.execute(text("""INSERT INTO llm_usage_records
                (id,conversation_id,provider,model,input_tokens,output_tokens,total_tokens,cost_usd)
                VALUES (123,123,'synthetic','test',1,2,3,0)"""))
        command.upgrade(config, "head")
        require_resource_schema(engine)
        with engine.connect() as connection:
            for table in ("documents", "conversations", "llm_usage_records"):
                assert connection.execute(text(f"SELECT id,tenant_id FROM {table}")).one() == (123, None)
            assert connection.scalar(text("SELECT content_revision FROM documents")) is not None
        command.check(config)
        command.downgrade(config, "002_telegram_ai")
        for table in ("documents", "conversations", "llm_usage_records"):
            assert "tenant_id" not in {x["name"] for x in inspect(engine).get_columns(table)}
        command.upgrade(config, "head")
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT title FROM documents WHERE id=123")) == "legacy"
            assert connection.scalar(text("SELECT tenant_id FROM documents WHERE id=123")) is None
        command.check(config)
    finally:
        engine.dispose()
        get_settings.cache_clear()
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE "{name}"'))
        admin.dispose()
