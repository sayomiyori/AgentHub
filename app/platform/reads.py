"""Read only platform metadata with an explicit tenant and bot filter."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.telegram_job import TelegramAIJob
from app.models.telegram_usage import TelegramAIUsage


class JobView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    event_id: UUID
    state: str
    attempts: int
    provider: str | None
    model: str | None
    created_at: datetime
    updated_at: datetime


class JobPage(BaseModel):
    items: list[JobView]
    next_cursor: UUID | None


class UsageView(BaseModel):
    records: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: Decimal


def list_jobs(db: Session, tenant_id: UUID, bot_id: UUID, cursor: UUID | None, limit: int) -> JobPage:
    statement = select(*[getattr(TelegramAIJob, field) for field in JobView.model_fields]).where(
        TelegramAIJob.tenant_id == tenant_id, TelegramAIJob.bot_id == bot_id,
    )
    if cursor is not None:
        statement = statement.where(TelegramAIJob.id > cursor)
    rows = db.execute(statement.order_by(TelegramAIJob.id).limit(limit + 1)).all()
    items = [JobView.model_validate(row) for row in rows[:limit]]
    return JobPage(items=items, next_cursor=items[-1].id if len(rows) > limit else None)


def read_usage(db: Session, tenant_id: UUID, bot_id: UUID, start: datetime, end: datetime) -> UsageView:
    count, inputs, outputs, cost = db.execute(select(
        func.count(TelegramAIUsage.job_id),
        func.coalesce(func.sum(TelegramAIUsage.input_tokens), 0),
        func.coalesce(func.sum(TelegramAIUsage.output_tokens), 0),
        func.coalesce(func.sum(TelegramAIUsage.estimated_cost_usd), 0),
    ).where(
        TelegramAIUsage.tenant_id == tenant_id, TelegramAIUsage.bot_id == bot_id,
        TelegramAIUsage.created_at >= start, TelegramAIUsage.created_at < end,
    )).one()
    return UsageView(records=count, input_tokens=inputs, output_tokens=outputs, estimated_cost_usd=cost)
