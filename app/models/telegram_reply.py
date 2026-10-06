from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKeyConstraint, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.platform import PlatformBase


class TelegramReplyOutbox(PlatformBase):
    __tablename__ = "telegram_reply_outbox"
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    job_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    tenant_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    bot_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    envelope: Mapped[dict] = mapped_column(JSONB, nullable=False)
    state: Mapped[str] = mapped_column(String(16), server_default="pending", nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    claim_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivery_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        UniqueConstraint("job_id", name="uq_telegram_reply_job"),
        ForeignKeyConstraint(
            ["job_id", "tenant_id", "bot_id"],
            ["telegram_ai_jobs.id", "telegram_ai_jobs.tenant_id", "telegram_ai_jobs.bot_id"],
            name="fk_telegram_reply_scope",
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "state IN ('pending','processing','published','failed','cancelled')", name="ck_telegram_reply_state"
        ),
        CheckConstraint("attempts >= 0", name="ck_telegram_reply_attempts"),
        CheckConstraint(
            "(state = 'processing' AND claim_id IS NOT NULL AND lease_until IS NOT NULL) "
            "OR (state <> 'processing' AND claim_id IS NULL AND lease_until IS NULL)",
            name="ck_telegram_reply_claim",
        ),
        CheckConstraint("state <> 'published' OR delivery_id IS NOT NULL", name="ck_telegram_reply_delivery"),
        Index("ix_telegram_reply_scope", "tenant_id", "bot_id", "id"),
        Index("ix_telegram_reply_due", "state", "next_attempt_at", "id"),
    )
