from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, CheckConstraint, DateTime, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.platform import PlatformBase


class TelegramAIJob(PlatformBase):
    __tablename__ = "telegram_ai_jobs"
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    event_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    tenant_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    bot_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    update_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    envelope: Mapped[dict] = mapped_column(JSONB, nullable=False)
    digest: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(String(16), server_default="pending", nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    claim_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    call_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    provider: Mapped[str | None] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(128))
    answer: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_telegram_ai_event"),
        UniqueConstraint("bot_id", "update_id", name="uq_telegram_ai_bot_update"),
        UniqueConstraint("id", "tenant_id", "bot_id", name="uq_telegram_ai_job_scope"),
        CheckConstraint(
            "state IN ('pending','processing','completed','failed','unknown')", name="ck_telegram_ai_state"
        ),
        CheckConstraint("attempts >= 0", name="ck_telegram_ai_attempts"),
        CheckConstraint("length(digest) = 64", name="ck_telegram_ai_digest"),
        CheckConstraint(
            "(state = 'processing' AND claim_id IS NOT NULL AND lease_until IS NOT NULL) "
            "OR (state <> 'processing' AND claim_id IS NULL AND lease_until IS NULL)",
            name="ck_telegram_ai_claim",
        ),
        CheckConstraint(
            "state <> 'completed' OR (answer IS NOT NULL AND length(answer) BETWEEN 1 AND 4096 "
            "AND provider IS NOT NULL AND model IS NOT NULL)",
            name="ck_telegram_ai_result",
        ),
        Index("ix_telegram_ai_scope", "tenant_id", "bot_id", "id"),
        Index("ix_telegram_ai_due", "state", "next_attempt_at", "id"),
    )
