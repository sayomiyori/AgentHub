from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKeyConstraint, Index, Integer, Numeric, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.platform import PlatformBase


class TelegramAIUsage(PlatformBase):
    __tablename__ = "telegram_ai_usage"
    job_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    bot_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_cost_usd: Mapped[Decimal] = mapped_column(Numeric(14, 8), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        ForeignKeyConstraint(
            ["job_id", "tenant_id", "bot_id"],
            ["telegram_ai_jobs.id", "telegram_ai_jobs.tenant_id", "telegram_ai_jobs.bot_id"],
            name="fk_telegram_ai_usage_scope",
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "input_tokens >= 0 AND output_tokens >= 0 AND estimated_cost_usd >= 0 "
            "AND estimated_cost_usd < 'Infinity'::numeric",
            name="ck_telegram_ai_usage_nonnegative",
        ),
        Index("ix_telegram_ai_usage_scope", "tenant_id", "bot_id", "created_at"),
    )
