"""Rehearse both migration branches in newly created, retained test databases."""

import os
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from app.config import get_settings
from app.db.platform import require_platform_schema

URL = os.getenv("AGENTHUB_PLATFORM_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL or os.getenv("AGENTHUB_CREATE_ISOLATED_TEST_DATABASES") != "yes",
                                reason="Explicitly authorized isolated PostgreSQL is required")


@pytest.mark.parametrize("branch", ["base", "003_tenant_resources", "003_ai_settings"])
def test_unified_upgrade_preserves_each_branch_and_metadata(monkeypatch, branch):
    source = make_url(URL)
    assert source.database.startswith("nexus_agent_ai_") and source.database.endswith("_test")
    name = f"nexus_agent_ai_merge_{uuid4().hex}_test"
    admin = create_engine(source, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    target = source.set(database=name)
    engine = create_engine(target)
    monkeypatch.setenv("DATABASE_URL", target.render_as_string(hide_password=False))
    get_settings.cache_clear()
    config = Config("alembic.ini")
    tenant, actor, revision = uuid4(), uuid4(), uuid4()
    try:
        if branch != "base":
            command.upgrade(config, branch)
            with engine.begin() as connection:
                connection.execute(text("""INSERT INTO documents
                    (id,title,content_type,upload_status,chunk_count,file_path)
                    VALUES (123,'synthetic','txt','pending',0,'unused')"""))
                connection.execute(text("INSERT INTO conversations (id,title) VALUES (123,'synthetic')"))
                connection.execute(text("""INSERT INTO llm_usage_records
                    (id,conversation_id,provider,model,input_tokens,output_tokens,total_tokens,cost_usd)
                    VALUES (123,123,'synthetic','synthetic',1,2,3,0)"""))
                if branch == "003_tenant_resources":
                    connection.execute(text("UPDATE documents SET tenant_id=:tenant, content_revision=:revision"),
                                       {"tenant": tenant, "revision": revision})
                    for table in ("conversations", "llm_usage_records"):
                        connection.execute(text(f"UPDATE {table} SET tenant_id=:tenant WHERE id=123"),
                                           {"tenant": tenant})
                else:
                    connection.execute(text("""INSERT INTO ai_model_settings
                        (tenant_id,target,subject_id,model,revision,updated_by)
                        VALUES (:tenant,'tenant',:tenant,'synthetic-model',1,:actor)"""),
                        {"tenant": tenant, "actor": actor})
                    connection.execute(text("""INSERT INTO ai_model_settings_audit
                        (tenant_id,target,subject_id,model,revision,updated_by,updated_at)
                        SELECT tenant_id,target,subject_id,model,revision,updated_by,updated_at
                        FROM ai_model_settings WHERE tenant_id=:tenant"""), {"tenant": tenant})
            with pytest.raises(RuntimeError, match="Platform migrations are required"):
                require_platform_schema(engine)
        command.upgrade(config, "head")
        require_platform_schema(engine)
        with engine.connect() as connection:
            assert set(connection.execute(text("SELECT version_num FROM alembic_version")).scalars()) == {
                "004_merge_ai_resources"}
            if branch != "base":
                assert connection.scalar(text("SELECT title FROM documents WHERE id=123")) == "synthetic"
            if branch == "003_tenant_resources":
                row = connection.execute(text("SELECT tenant_id,content_revision FROM documents WHERE id=123")).one()
                assert tuple(row) == (tenant, revision)
                for table in ("conversations", "llm_usage_records"):
                    assert connection.scalar(text(f"SELECT tenant_id FROM {table} WHERE id=123")) == tenant
            if branch == "003_ai_settings":
                for table in ("ai_model_settings", "ai_model_settings_audit"):
                    row = connection.execute(text(f"SELECT model,revision,updated_by FROM {table}")).one()
                    assert tuple(row) == ("synthetic-model", 1, actor)
        for table in ("documents", "conversations", "llm_usage_records"):
            assert "tenant_id" in {column["name"] for column in inspect(engine).get_columns(table)}
        command.check(config)
        # Removing only the merge marker retains both branch schemas and records.
        command.downgrade(config, "003_tenant_resources")
        with engine.connect() as connection:
            assert set(connection.execute(text("SELECT version_num FROM alembic_version")).scalars()) == {
                "003_tenant_resources", "003_ai_settings"}
        with pytest.raises(RuntimeError, match="Platform migrations are required"):
            require_platform_schema(engine)
        command.upgrade(config, "head")
        require_platform_schema(engine)
        command.check(config)
        with engine.connect() as connection:
            if branch == "003_tenant_resources":
                assert tuple(connection.execute(text(
                    "SELECT tenant_id,content_revision FROM documents WHERE id=123")).one()) == (tenant, revision)
                for table in ("conversations", "llm_usage_records"):
                    assert connection.scalar(text(f"SELECT tenant_id FROM {table} WHERE id=123")) == tenant
            if branch == "003_ai_settings":
                for table in ("ai_model_settings", "ai_model_settings_audit"):
                    row = connection.execute(text(f"SELECT model,revision,updated_by FROM {table}")).one()
                    assert tuple(row) == ("synthetic-model", 1, actor)
    finally:
        engine.dispose()
        admin.dispose()
        get_settings.cache_clear()
