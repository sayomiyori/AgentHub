"""Constraint enforcement in an explicitly isolated, migrated PostgreSQL database."""
import os
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.orm import Session

from app.db.platform import require_platform_schema
from app.models.telegram_job import TelegramAIJob
from app.models.telegram_reply import TelegramReplyOutbox
from app.models.telegram_usage import TelegramAIUsage

URL = os.getenv("AGENTHUB_PLATFORM_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="Isolated migrated platform database is required")


@pytest.fixture()
def db():
    engine = create_engine(URL)
    with engine.connect() as connection:
        name = connection.scalar(text("SELECT current_database()"))
        assert name.startswith("nexus_agent_ai_") and name.endswith("_test")
        require_platform_schema(engine)
        transaction = connection.begin() if not connection.in_transaction() else connection.get_transaction()
        with Session(connection, join_transaction_mode="create_savepoint") as session:
            yield session
        transaction.rollback()
    engine.dispose()


def job(db):
    item = TelegramAIJob(id=uuid4(), event_id=uuid4(), tenant_id=uuid4(), bot_id=uuid4(),
                         update_id=42, envelope={}, digest="a" * 64)
    db.add(item)
    db.flush()
    return item


@pytest.mark.parametrize("duplicate", ["event_id", "bot_update"])
def test_replayed_update_cannot_create_second_job(db, duplicate):
    original = job(db)
    other = TelegramAIJob(id=uuid4(), event_id=original.event_id if duplicate == "event_id" else uuid4(),
                          tenant_id=original.tenant_id,
                          bot_id=original.bot_id if duplicate == "bot_update" else uuid4(),
                          update_id=42, envelope={}, digest="b" * 64)
    with pytest.raises(IntegrityError), db.begin_nested():
        db.add(other)
        db.flush()


@pytest.mark.parametrize("record", ["usage", "reply"])
def test_related_records_cannot_cross_tenant(db, record):
    original = job(db)
    scope = dict(job_id=original.id, tenant_id=uuid4(), bot_id=original.bot_id)
    item = (TelegramAIUsage(**scope, provider="groq", model="test", input_tokens=1, output_tokens=1,
                            estimated_cost_usd=Decimal("0")) if record == "usage"
            else TelegramReplyOutbox(id=uuid4(), **scope, envelope={}))
    with pytest.raises(IntegrityError), db.begin_nested():
        db.add(item)
        db.flush()


@pytest.mark.parametrize("cost", ["-1", "NaN", "Infinity"])
def test_usage_rejects_invalid_cost(db, cost):
    original = job(db)
    item = TelegramAIUsage(job_id=original.id, tenant_id=original.tenant_id, bot_id=original.bot_id,
                           provider="groq", model="test", input_tokens=1, output_tokens=1,
                           estimated_cost_usd=Decimal(cost))
    with pytest.raises(DataError if cost == "Infinity" else IntegrityError), db.begin_nested():
        db.add(item)
        db.flush()


def test_completed_job_requires_an_answer(db):
    original = job(db)
    with pytest.raises(IntegrityError), db.begin_nested():
        original.state = "completed"
        db.flush()


@pytest.mark.parametrize("record", ["usage", "reply"])
def test_job_cannot_have_duplicate_usage_or_reply(db, record):
    original = job(db)
    scope = dict(job_id=original.id, tenant_id=original.tenant_id, bot_id=original.bot_id)
    if record == "usage":
        values = dict(**scope, provider="groq", model="test", input_tokens=1,
                      output_tokens=1, estimated_cost_usd=Decimal("0"))
        table = TelegramAIUsage.__table__
    else:
        values = dict(id=uuid4(), **scope, envelope={})
        table = TelegramReplyOutbox.__table__
    db.execute(table.insert().values(**values))
    if record == "reply":
        values["id"] = uuid4()
    with pytest.raises(IntegrityError), db.begin_nested():
        db.execute(table.insert().values(**values))
